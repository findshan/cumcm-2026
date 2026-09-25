"""
问题3：光伏出力仅可得预报条件下的滚动优化（MPC）

口径与假设
----------
* 预报：附件3，每天 0:00 / 6:00 / 12:00 / 18:00 发布未来 24h 整点预报，
        用 PCHIP 保形插值降尺度到 10 分钟（144 点），并按发布时刻对齐到当天时段。
* 决策：0:00 用当天预报制定全天 144 点的"计划购电"；
        6:00 / 12:00 / 18:00 依最新预报对尚未执行的时段重优化，得到"调整购电"。
* 执行：购电量与充放电量按最近一次求解结果执行；实际光伏（附件2）到达后
        不足 -> 5 倍电价的紧急购电；有余 -> 弃光。
* 储能：采用与问题1 一致的日周期约束 E(0:00) = E(24:00) = 6000 kWh，
        消除滚动优化固有的"时域末端效应"。
        （备选口径"终端储电价值"结果仅差 0.2%，见 08_sensitivity.py）
* 费用：
    口径A（主）总 = Σ c·plan·Δt + Σ[0.5c(plan−adj)⁺ + 1.5c(adj−plan)⁺]Δt + Σ5c·emg·Δt
    口径B（备选）总 = Σ[c·min(plan,adj) + 0.5c(plan−adj)⁺ + 1.5c(adj−plan)⁺]Δt + Σ5c·emg·Δt
"""
import numpy as np
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import (load_attachments, solve_lp, forecast_day_aligned,
                        DT, N_T, DAYS, E_INIT, ETA)
from cumcm_pen import solve_lp_pen

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; RES = ROOT / "results"

CPS_FULL = ((0, 0), (36, 6), (72, 12), (108, 18))
CPS_PLAN = ((0, 0),)
I0 = 31                                    # 交付起点 2025-02-01


def simulate(d, checkpoints=CPS_FULL, e_end=E_INIT, penalized=True):
    c, L, S = d["price_day"], d["L"], d["S"]
    plan = np.zeros((DAYS, N_T)); adj = np.zeros((DAYS, N_T))
    uu = np.zeros((DAYS, N_T)); vv = np.zeros((DAYS, N_T))
    cur = np.zeros((DAYS, N_T)); emg = np.zeros((DAYS, N_T))
    Ecur = E_INIT
    for D in range(DAYS):
        fc = {h: forecast_day_aligned(d, D, h) for _, h in checkpoints}
        Ec = np.zeros(N_T + 1); Ec[0] = Ecur
        for ci, (s, h0) in enumerate(checkpoints):
            e = checkpoints[ci + 1][0] if ci + 1 < len(checkpoints) else N_T
            if ci == 0 or not penalized:
                # 0:00 制定日计划：目标就是计划购电费用本身（题面：按计划购电量计费）
                r = solve_lp(c[s:], L[D, s:], fc[h0][s:], E_start=Ec[s], E_end=e_end)
                gv = r["g"]
            else:
                # 6/12/18 调整：目标应为【调整购电量的相关费用】
                #   min Σ[0.5c(g⁰−a)⁺ + 1.5c(a−g⁰)⁺]Δt
                # 现方案在此处误用"min Σ c·a"（等于假定调整免费），是本模型原先的缺陷
                r = solve_lp_pen(c[s:], L[D, s:], fc[h0][s:], plan[D, s:],
                                 E_start=Ec[s], E_end=e_end)
                gv = r["a"]
            if ci == 0:
                plan[D, s:] = gv
            adj[D, s:e] = gv[:e - s]
            uu[D, s:e] = r["u"][:e - s]
            vv[D, s:e] = r["v"][:e - s]
            for k in range(s, e):
                Ec[k + 1] = Ec[k] + ETA * DT * uu[D, k] - DT / ETA * vv[D, k]
        Ecur = Ec[N_T]
        surplus = S[D] + adj[D] + vv[D] - L[D] - uu[D]
        emg[D] = np.maximum(-surplus, 0.0)
        cur[D] = np.maximum(surplus, 0.0)
    return dict(plan=plan, adj=adj, u=uu, v=vv, cur=cur, emg=emg)


def cost(rs, d, mode="A", sl=slice(I0, DAYS)):
    c = d["price_day"]; plan = rs["plan"]; adj = rs["adj"]; emg = rs["emg"]
    P, A, E = plan[sl], adj[sl], emg[sl]
    d_lo = np.maximum(P - A, 0); d_hi = np.maximum(A - P, 0)
    adj_fee = ((0.5 * d_lo + 1.5 * d_hi) * DT * np.tile(c, (P.shape[0], 1))).sum()
    if mode == "A":
        base = (P * DT * np.tile(c, (P.shape[0], 1))).sum()
    else:
        base = (np.minimum(P, A) * DT * np.tile(c, (P.shape[0], 1))).sum()
    emg_fee = (E * DT * np.tile(c, (P.shape[0], 1))).sum() * 5
    return dict(base=float(base), adj=float(adj_fee), emg=float(emg_fee),
                total=float(base + adj_fee + emg_fee))


def deterministic_daily(d, price=None):
    """完美信息 + 日周期 的逐日最优（问题3 的干净对照基线）"""
    c = d["price_day"] if price is None else price
    L, S = d["L"], d["S"]
    g = np.zeros((DAYS, N_T)); u = np.zeros((DAYS, N_T))
    v = np.zeros((DAYS, N_T)); w = np.zeros((DAYS, N_T))
    for D in range(DAYS):
        r = solve_lp(c, L[D], S[D], E_start=E_INIT, E_end=E_INIT)
        g[D], u[D], v[D], w[D] = r["g"], r["u"], r["v"], r["w"]
    return g, u, v, w


if __name__ == "__main__":
    d = load_attachments()
    t0 = time.time()
    print("=" * 82)
    print("问题3  预报条件下的滚动优化（MPC）")
    print("=" * 82)

    r_naive = simulate(d, CPS_FULL, penalized=False)      # 现方案（子问题目标为购电成本）
    r_full = simulate(d, CPS_FULL, penalized=True)        # MPC-II（子问题目标为调整罚金）
    cA = cost(r_full, d, "A"); cB = cost(r_full, d, "B")
    cN = cost(r_naive, d, "A")
    print(f"[四次预报] 求解 {DAYS*4} 次 LP，耗时 {time.time()-t0:.1f}s")
    print(f"\n  [缺陷修正] 子问题目标由「购电成本」改为「调整购电量的相关费用」：")
    print(f"     调整罚金 {cN['adj']:>12,.2f} → {cA['adj']:>12,.2f} 元"
          f"（{(cA['adj']-cN['adj'])/cN['adj']*100:+.1f}%）")
    print(f"     总费用   {cN['total']:>12,.2f} → {cA['total']:>12,.2f} 元"
          f"（省 {cN['total']-cA['total']:,.0f} 元，{(cN['total']-cA['total'])/cN['total']*100:.2f}%）")

    print("\n--- 费用构成（2025-02-01 ~ 2025-12-31，334 天）---")
    for nm, cc in (("口径A（主）", cA), ("口径B（备选）", cB)):
        print(f"  {nm}: 计划购电费 {cc['base']:>13,.2f} + 调整费 {cc['adj']:>11,.2f}"
              f" + 紧急购电费 {cc['emg']:>11,.2f} = 总费用 {cc['total']:>13,.2f} 元")

    E = r_full["emg"][I0:] * DT
    print("\n--- 关键指标 ---")
    print(f"  计划购电量        : {r_full['plan'][I0:].sum()*DT:14,.2f} kWh")
    print(f"  调整购电量        : {r_full['adj'][I0:].sum()*DT:14,.2f} kWh")
    print(f"  调整偏离总量      : {np.abs(r_full['adj']-r_full['plan'])[I0:].sum()*DT:14,.2f} kWh")
    print(f"  紧急购电量        : {E.sum():14,.2f} kWh（{int((E.sum(1)>1e-6).sum())} 天）")
    print(f"  弃光电量          : {r_full['cur'][I0:].sum()*DT:14,.2f} kWh")
    print(f"  储能 0:00/24:00   : {E_INIT:.0f} / {E_INIT:.0f} kWh（日周期约束）")

    # ---- 干净对照：完美信息 + 日周期 ----
    g_det, u_det, v_det, w_det = deterministic_daily(d)
    det_fee = float((g_det[I0:] * DT * d["price_day"]).sum())
    print("\n" + "-" * 82)
    print("【预报不确定性的代价】")
    print(f"  完美信息 + 日周期（对照基线）: {det_fee:14,.2f} 元")
    print(f"  仅得预报 + 日周期（问题3）  : {cA['total']:14,.2f} 元")
    print(f"  → 预报不确定性代价           : {cA['total']-det_fee:14,.2f} 元"
          f"（{(cA['total']-det_fee)/det_fee*100:.2f}%）")

    # ---- 是否需引入其他时刻预报 ----
    t1 = time.time()
    r_00 = simulate(d, CPS_PLAN)
    c0 = cost(r_00, d, "A")
    print("\n" + "-" * 82)
    print("【是否需引入其他时刻的预报？—— 预报的信息价值】")
    print(f"  仅用 0:00 预报（1 次/天）      : 总费用 {c0['total']:>13,.2f} 元"
          f"   紧急购电 {r_00['emg'][I0:].sum()*DT:>10,.2f} kWh")
    print(f"  0:00+6+12+18（4 次/天）        : 总费用 {cA['total']:>13,.2f} 元"
          f"   紧急购电 {E.sum():>10,.2f} kWh")
    print(f"  → 追加预报的边际价值           : {c0['total']-cA['total']:>13,.2f} 元"
          f"（{(c0['total']-cA['total'])/c0['total']*100:.2f}%）")
    print(f"  完美信息基准                   : {det_fee:>13,.2f} 元")
    print(f"  → 完美信息价值 EVPI（对0:00）  : {c0['total']-det_fee:>13,.2f} 元")
    print(f"  → 剩余缺口（需更好的预报/预测）: {cA['total']-det_fee:>13,.2f} 元")
    print(f"  [仿真耗时 {time.time()-t1:.1f}s]")

    # ---- 写出 result3.xlsx ----
    from cumcm_write import write_result3_like
    dates = d["dates"]
    Est = np.full(DAYS, E_INIT); Eed = np.full(DAYS, E_INIT)
    fee_plan = (r_full["plan"] * DT * np.tile(d["price_day"], (DAYS, 1))).sum(1)
    fee_adj = (r_full["adj"] * DT * np.tile(d["price_day"], (DAYS, 1))).sum(1)
    write_result3_like(RES / "result3.xlsx", dates[I0:],
                       r_full["plan"][I0:] * DT, r_full["adj"][I0:] * DT,
                       r_full["u"][I0:], r_full["v"][I0:],
                       Est[I0:], Eed[I0:], emergency=None,
                       fee_plan=fee_plan[I0:], fee_adj=fee_adj[I0:])
    print(f"\n[OK] 已写出 {RES/'result3.xlsx'}")

    np.savez(RES / "p3_solution.npz",
             plan=r_full["plan"], adj=r_full["adj"], u=r_full["u"], v=r_full["v"],
             cur=r_full["cur"], emg=r_full["emg"],
             plan_naive=r_naive["plan"], adj_naive=r_naive["adj"],
             u_naive=r_naive["u"], v_naive=r_naive["v"],
             emg_naive=r_naive["emg"], cur_naive=r_naive["cur"],
             g_det=g_det, u_det=u_det, v_det=v_det, w_det=w_det,
             costA=cA["total"], costB=cB["total"], cost0=c0["total"], det_fee=det_fee,
             cost_naive=cN["total"], pen_naive=cN["adj"], pen_mpc2=cA["adj"])

    # ---- 图（已移除：统一由 11_figs_final.py 出图，避免覆盖高级版） ----
    # 本脚本只负责求解与写出 result*.xlsx；论文插图一律跑 `python3 11_figs_final.py`

    print(f"总耗时 {time.time()-t0:.1f}s")
