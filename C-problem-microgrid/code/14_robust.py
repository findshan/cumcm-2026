"""
鲁棒优化补充：'期望最优（随机）' vs '最坏保证（鲁棒）' 双视角

问题3 鲁棒建模
    不确定集（box）：光伏预报误差 ε = Ŝ − S 落在经验分位区间内
        ε ∈ [ −Γ·q_{1−α}(ε),  +Γ·q_{1−α}(ε) ]
    单阶段鲁棒对等（最坏方向为实际低于预报）：
        g_t + v_t − u_t − w_t ≥ L_t − Ŝ_t + Γ·q_{1−α,t}
    等价于"把预报按 α 分位误差保守打折"：Ŝ^rob = max(Ŝ − Γ·q_{1−α}, 0)
    后续 6/12/18 点的调整仍按最新预报做滚动重优化（robust planning + rolling recourse）

问题4 鲁棒建模
    电价不确定集：c_t ∈ [ĉ_t, (1+ρ)ĉ_t]，目标用上界电价定价（保守出清）

评价方式：在同一批全年场景上比较两条策略的费用分布
    · 确定性策略（Γ=0）  -> 期望低、尾部高
    · 鲁棒策略（Γ>0）    -> 期望高、尾部低
    Γ 的增大即"用期望换最坏保证"，其斜率就是"鲁棒的价格"(price of robustness)
"""
import numpy as np
import pandas as pd
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import (load_attachments, solve_lp, forecast_day_aligned,
                        DT, N_T, DAYS, E_INIT, ETA, DATA_DIR)
from importlib import import_module
r12 = import_module("12_risk")

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
I0 = 31
CPS = ((0, 0), (36, 6), (72, 12), (108, 18))
NSC = 30          # 用于策略对比的场景数
ALPHA = 0.95      # 置信水平（取 95% 分位误差作不确定集边界）


def build_discount(d, alpha=ALPHA):
    """
    每个发布时刻 h0、每个时钟时段的 (1−alpha) 分位预报误差 q（正值表示"预报偏高"）。
    鲁棒策略将预报下调 Γ·q。
    """
    S = d["S"]
    disc = {}
    for h0 in (0, 6, 12, 18):
        Fh = np.stack([forecast_day_aligned(d, D, h0) for D in range(DAYS)])
        err = Fh - S                       # 预报 − 实际
        q = np.quantile(err, alpha, axis=0)   # 逐时段 alpha 分位
        q = np.maximum(q, 0.0)
        # 夜间无不确定性：实测光伏恒 0 处不打折
        q[r12.DARK] = 0.0
        disc[h0] = q
    return disc


def mpc_year(d, Fc, Sact, price, L, disc=None, gamma=0.0, settle=None):
    """
    滚动 MPC。
      price  : 用于【制定计划】的电价序列（鲁棒时为保守上界电价）
      settle : 用于【最终结算】的电价序列（默认同 price）
      disc/gamma : 按 Γ·disc 对光伏预报保守打折
    """
    cw = np.tile(price, (DAYS, 1))
    sw = np.tile(price if settle is None else settle, (DAYS, 1))
    tot = 0.0; emg_kwh = 0.0
    for D in range(DAYS):
        Ec = np.zeros(N_T + 1); Ec[0] = E_INIT
        plan = np.zeros(N_T); adj = np.zeros(N_T); uu = np.zeros(N_T); vv = np.zeros(N_T)
        for ci, (s, h0) in enumerate(CPS):
            e = CPS[ci + 1][0] if ci + 1 < len(CPS) else N_T
            f = Fc[h0][D, s:]
            if disc is not None and gamma > 0:
                f = np.maximum(f - gamma * disc[h0][s:], 0.0)
            r = solve_lp(price[s:], L[D, s:], f, E_start=Ec[s], E_end=E_INIT)
            if ci == 0:
                plan[s:] = r["g"]
            adj[s:e] = r["g"][:e - s]; uu[s:e] = r["u"][:e - s]; vv[s:e] = r["v"][:e - s]
            for k in range(s, e):
                Ec[k + 1] = Ec[k] + ETA * DT * uu[k] - DT / ETA * vv[k]
        sur = Sact[D] + adj + vv - L[D] - uu
        emg = np.maximum(-sur, 0.0)
        if D >= I0:
            tot += (plan * DT * sw[D]).sum()
            tot += ((0.5 * np.maximum(plan - adj, 0) + 1.5 * np.maximum(adj - plan, 0)) * DT * sw[D]).sum()
            tot += (emg * DT * sw[D]).sum() * 5
            emg_kwh += (emg * DT).sum()
    return tot, emg_kwh


if __name__ == "__main__":
    t0 = time.time()
    d = load_attachments()
    c = d["price_day"]; L, Sv = d["L"], d["S"]; c4 = d["C4"]
    rel0, relh, F0_all = r12.build_rel_matrices(d)
    disc = build_discount(d)
    print(f"[不确定集] 95% 分位误差（kW），正午附近示例："
          f"0:00发布 {disc[0][78]:.0f}，6:00发布 {disc[6][78]:.0f}，"
          f"12:00发布 {disc[12][78]:.0f}")
    print(f"  夜间时段折扣已置 0（无不确定性）")

    # ---------- 生成评测场景（与 12_risk 同法） ----------
    base_fc = {h: np.stack([forecast_day_aligned(d, D, h) for D in range(DAYS)]) for _, h in CPS}
    rng = np.random.default_rng(2026)
    scen = []
    for _ in range(NSC):
        Fc, Sact = r12.gen_year_scenario(d, rel0, relh, F0_all, rng)
        scen.append((Fc, Sact))

    # ---------- Γ 扫描 ----------
    gammas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    print("\n" + "=" * 84)
    print(f"问题3 鲁棒 vs 确定性：{NSC} 组全年场景上的费用分布（Γ 为不确定集放大系数）")
    print("=" * 84)
    print("   Γ    期望(万元)  标准差   VaR95(万元)  CVaR95(万元)  紧急购电(kWh)")
    rows = []
    for g in gammas:
        cs, es = [], []
        for Fc, Sact in scen:
            f, ek = mpc_year(d, Fc, Sact, c, L, disc, g)
            cs.append(f); es.append(ek)
        cs = np.array(cs); es = np.array(es)
        v95 = np.percentile(cs, 95); cvar = cs[cs >= v95].mean()
        rows.append((g, cs.mean(), cs.std(), v95, cvar, es.mean(), cs))
        print(f"  {g:4.1f}  {cs.mean()/1e4:10.1f}  {cs.std()/1e4:7.2f}  "
              f"{v95/1e4:11.1f}  {cvar/1e4:11.1f}  {es.mean():13,.0f}")
        print(f"        用时 {time.time()-t0:.0f}s")
    cs0 = rows[0][6]
    print("\n  【鲁棒的价格 / 收益】以 Γ=0（确定性）为基准：")
    for g, mu, sd, v95, cvar, ek, _ in rows[1:]:
        print(f"    Γ={g:.1f}：期望 +{100*(mu-cs0.mean())/cs0.mean():5.2f}%　"
              f"CVaR95 {100*(cvar-rows[0][4])/rows[0][4]:+6.2f}%　"
              f"标准差 {100*(sd-cs0.std())/cs0.std():+6.1f}%")

    # ---------- 问题4 鲁棒电价 ----------
    print("\n" + "=" * 84)
    print("问题4 鲁棒电价：计划按 (1+ρ)·ĉ 保守定价，结算按情景真实电价")
    print("=" * 84)
    c_hi = np.quantile(c4, 0.95, axis=0)
    print("   ρ    实际电价结算(万元)   95分位电价结算(万元)   高电价情景改善")
    r4 = []
    for rho in (0.0, 0.05, 0.10, 0.20):
        c_plan = c * (1 + rho)
        f_act, _ = mpc_year(d, base_fc, Sv, c_plan, L, None, 0.0, settle=c)
        f_hi, _ = mpc_year(d, base_fc, Sv, c_plan, L, None, 0.0, settle=c_hi)
        r4.append((rho, f_act, f_hi))
        print(f"  {rho:4.2f}   {f_act/1e4:15,.1f}   {f_hi/1e4:17,.1f}   "
              f"{(f_hi-r4[0][2])/r4[0][2]*100:+7.2f}%")
    print(f"\n  【鲁棒的代价 vs 收益】")
    for rho, fa, fh in r4[1:]:
        print(f"    ρ={rho:.2f}：实际电价下多付 {(fa-r4[0][1])/r4[0][1]*100:+5.2f}%，"
              f"最坏电价情景下 {(fh-r4[0][2])/r4[0][2]*100:+5.2f}%")

    np.savez(RES / "robust.npz",
             gammas=np.array([r[0] for r in rows]),
             means=np.array([r[1] for r in rows]), sds=np.array([r[2] for r in rows]),
             v95=np.array([r[3] for r in rows]), cvar=np.array([r[4] for r in rows]),
             emgs=np.array([r[5] for r in rows]), costs0=cs0,
             disc0=disc[0], disc6=disc[6])
    print(f"\n[OK] 已写出 {RES/'robust.npz'}　总耗时 {time.time()-t0:.0f}s")
