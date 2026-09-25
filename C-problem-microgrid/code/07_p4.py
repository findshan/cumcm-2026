"""
问题4：外部电网电价实时波动

双轨建模
--------
轨一（完美信息 / 理论上界）：
  直接采用附件4 的实际电价序列做确定性优化，给出"若能预知电价"的最优下界。
  · 4-2：整年连续 LP（对应问题2 的设定 + 波动电价）
  · 4-3：滚动 MPC（对应问题3 的设定：光伏仅得附件3 预报 + 波动电价）

轨二（电价不可预知 / 可实现策略）：
  0:00 时无法预知当天电价，只能用历史分时均价（= 附件1 电价，已实测为附件4 的逐时段均值）
  作为电价预测来制定计划，实际按附件4 结算。
  轨二与轨一之差 = 电价不确定性的代价。

输出：results/result4-2.xlsx、results/result4-3.xlsx
"""
import numpy as np
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_pen import solve_lp_pen
from cumcm_core import (load_attachments, solve_lp, forecast_day_aligned,
                        DT, N_T, DAYS, E_INIT, ETA)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; RES = ROOT / "results"
I0 = 31
CPS = ((0, 0), (36, 6), (72, 12), (108, 18))


# ---------------------------------------------------------------- 4-2
def solve_42(d, c_plan=None):
    """波动电价下的问题2：整年连续 LP（c_plan 为制定计划时使用的电价，默认用实际电价）"""
    c_act = d["C4"]
    c_use = c_act if c_plan is None else np.tile(c_plan, (DAYS, 1))
    r = solve_lp(c_use.ravel(), d["L"].ravel(), d["S"].ravel(),
                 E_start=E_INIT, E_end_min=E_INIT)
    g = r["g"].reshape(DAYS, N_T); u = r["u"].reshape(DAYS, N_T)
    v = r["v"].reshape(DAYS, N_T); w = r["w"].reshape(DAYS, N_T)
    Ee = r["E"].reshape(DAYS, N_T)[:, -1]
    Es = np.concatenate([[E_INIT], Ee[:-1]])
    fee_day = (g * DT * c_act).sum(1)          # 按实际电价结算
    return dict(g=g, u=u, v=v, w=w, Es=Es, Ee=Ee, fee_day=fee_day, resid=r["resid"])


# ---------------------------------------------------------------- 4-3
def solve_43(d, c_plan=None, cps=CPS, e_end=E_INIT, penalized=True):
    """波动电价下的问题3：滚动 MPC（光伏仅得预报）

    penalized=True 时，6/12/18 点子问题的目标为【调整购电量的相关费用】
    （0.5c/1.5c 罚金），而非"购电成本"——与 06_p3.py 的 MPC-II 口径一致。
    """
    c_act = d["C4"]
    c_use = c_act if c_plan is None else np.tile(c_plan, (DAYS, 1))
    L, S = d["L"], d["S"]
    plan = np.zeros((DAYS, N_T)); adj = np.zeros((DAYS, N_T))
    uu = np.zeros((DAYS, N_T)); vv = np.zeros((DAYS, N_T))
    cur = np.zeros((DAYS, N_T)); emg = np.zeros((DAYS, N_T))
    for D in range(DAYS):
        fc = {h: forecast_day_aligned(d, D, h) for _, h in cps}
        Ec = np.zeros(N_T + 1); Ec[0] = E_INIT
        for ci, (s, h0) in enumerate(cps):
            e = cps[ci + 1][0] if ci + 1 < len(cps) else N_T
            if ci == 0 or not penalized:
                r = solve_lp(c_use[D, s:], L[D, s:], fc[h0][s:],
                             E_start=Ec[s], E_end=e_end)
                gv = r["g"]
                if ci == 0:
                    plan[D, s:] = gv
            else:
                r = solve_lp_pen(c_use[D, s:], L[D, s:], fc[h0][s:], plan[D, s:],
                                 E_start=Ec[s], E_end=e_end)
                gv = r["a"]
            adj[D, s:e] = gv[:e - s]
            uu[D, s:e] = r["u"][:e - s]
            vv[D, s:e] = r["v"][:e - s]
            for k in range(s, e):
                Ec[k + 1] = Ec[k] + ETA * DT * uu[D, k] - DT / ETA * vv[D, k]
        surplus = S[D] + adj[D] + vv[D] - L[D] - uu[D]
        emg[D] = np.maximum(-surplus, 0.0)
        cur[D] = np.maximum(surplus, 0.0)
    return dict(plan=plan, adj=adj, u=uu, v=vv, cur=cur, emg=emg)


def cost43(rs, d, mode="A", sl=slice(I0, DAYS)):
    c = d["C4"][sl]
    P, A, E = rs["plan"][sl], rs["adj"][sl], rs["emg"][sl]
    d_lo = np.maximum(P - A, 0); d_hi = np.maximum(A - P, 0)
    adj_fee = ((0.5 * d_lo + 1.5 * d_hi) * DT * c).sum()
    base = (P * DT * c).sum() if mode == "A" else (np.minimum(P, A) * DT * c).sum()
    emg_fee = (E * DT * c).sum() * 5
    return dict(base=float(base), adj=float(adj_fee), emg=float(emg_fee),
                total=float(base + adj_fee + emg_fee))


if __name__ == "__main__":
    d = load_attachments(); dates = d["dates"]; c4 = d["C4"]
    t0 = time.time()
    print("=" * 82)
    print("问题4  外部电网电价实时波动")
    print("=" * 82)

    # ---------- 电价波动的刻画 ----------
    ratio = c4.max(1) / np.maximum(c4.min(1), 1e-9)
    print("\n[附件4 电价特征]")
    print(f"  全年 min/max/均值 = {c4.min():.4f} / {c4.max():.4f} / {c4.mean():.4f} 元/kWh")
    print(f"  逐日日内峰谷比：中位数 {np.median(ratio):.3f}，最大 {ratio.max():.3f}")
    print(f"  > 1/η²={1/ETA**2:.3f}（套利阈值）的天数 = {int((ratio>1/ETA**2).sum())}/365，"
          f"即每天都存在套利空间")
    print(f"  与附件1（历史分时均价）的最大逐时段偏差 = {np.abs(c4.mean(0)-d['price_day']).max():.2e}"
          f"（附件1 电价即为附件4 的逐时段均值）")

    # ---------- 轨一：完美信息 ----------
    print("\n" + "-" * 82)
    print("轨一：完美信息（直接采用附件4 实际电价）")
    r42 = solve_42(d)
    f42 = r42["fee_day"][I0:].sum()
    print(f"  [4-2 整年连续 LP] 残差 {r42['resid']:.1e}  求解 {time.time()-t0:.1f}s")
    print(f"    交付期购电费 {f42:14,.2f} 元   购电量 {r42['g'][I0:].sum()*DT:14,.2f} kWh")
    print(f"    弃光 {(r42['w'][I0:].sum()*DT):14,.2f} kWh   储能年末 {r42['Ee'][-1]:.2f} kWh")

    t1 = time.time()
    r43 = solve_43(d)
    c43a = cost43(r43, d, "A"); c43b = cost43(r43, d, "B")
    print(f"  [4-3 滚动 MPC] 求解 {DAYS*4} 次 LP，{time.time()-t1:.1f}s")
    print(f"    口径A: 计划费 {c43a['base']:>12,.2f} + 调整费 {c43a['adj']:>10,.2f}"
          f" + 紧急费 {c43a['emg']:>10,.2f} = 总费用 {c43a['total']:>13,.2f} 元")
    print(f"    口径B: 总费用 {c43b['total']:>13,.2f} 元")
    print(f"    紧急购电 {r43['emg'][I0:].sum()*DT:12,.2f} kWh   弃光 {r43['cur'][I0:].sum()*DT:12,.2f} kWh")
    print(f"    储能 0:00/24:00 = {E_INIT:.0f} kWh（日周期约束）")

    # ---------- 轨二：电价不可预知 ----------
    print("\n" + "-" * 82)
    print("轨二：电价不可预知（用历史分时均价=附件1 作电价预测，按附件4 结算）")
    r42p = solve_42(d, c_plan=d["price_day"])
    f42p = r42p["fee_day"][I0:].sum()
    r43p = solve_43(d, c_plan=d["price_day"])
    c43p = cost43(r43p, d, "A")
    print(f"  [4-2 轨二] 购电费 {f42p:14,.2f} 元   （轨一 {f42:14,.2f} 元）")
    print(f"  [4-3 轨二] 总费用 {c43p['total']:14,.2f} 元   （轨一 {c43a['total']:14,.2f} 元）")
    print("\n【电价不确定性的代价】")
    print(f"  4-2：轨二 − 轨一 = {f42p-f42:14,.2f} 元（{(f42p-f42)/f42*100:.2f}%）")
    print(f"  4-3：轨二 − 轨一 = {c43p['total']-c43a['total']:14,.2f} 元"
          f"（{(c43p['total']-c43a['total'])/c43a['total']*100:.2f}%）")

    # ---------- 与固定电价对比 ----------
    p2 = np.load(RES / "p2_solution.npz")
    p3 = np.load(RES / "p3_solution.npz")
    print("\n" + "-" * 82)
    print("【波动电价 vs 固定电价（均为完美信息）】")
    print(f"  问题2 固定电价（附件1）整年连续 : {p2['fee_day'][I0:].sum():14,.2f} 元")
    print(f"  4-2   波动电价（附件4）整年连续 : {f42:14,.2f} 元")
    print(f"  → 波动电价的效应                : {f42-p2['fee_day'][I0:].sum():14,.2f} 元"
          f"（{(f42-p2['fee_day'][I0:].sum())/p2['fee_day'][I0:].sum()*100:.2f}%）")
    print(f"  问题3 固定电价（四次预报+日周期）: {p3['costA']:14,.2f} 元")
    print(f"  4-3   波动电价（四次预报+日周期）: {c43a['total']:14,.2f} 元")
    print(f"  → 波动电价的效应                : {c43a['total']-float(p3['costA']):14,.2f} 元"
          f"（{(c43a['total']-float(p3['costA']))/float(p3['costA'])*100:.2f}%）")

    # ---------- 写出结果 ----------
    from cumcm_write import write_result2_like, write_result3_like
    write_result2_like(RES / "result4-2.xlsx", dates[I0:], r42["g"][I0:] * DT,
                       r42["u"][I0:], r42["v"][I0:], r42["Es"][I0:], r42["Ee"][I0:],
                       emergency=None, tpl_name="result4-2.xlsx",
                       fee_day=r42["fee_day"][I0:])
    print(f"\n[OK] 已写出 {RES/'result4-2.xlsx'}")
    fp = (r43["plan"] * DT * c4).sum(1); fa = (r43["adj"] * DT * c4).sum(1)
    write_result3_like(RES / "result4-3.xlsx", dates[I0:],
                       r43["plan"][I0:] * DT, r43["adj"][I0:] * DT,
                       r43["u"][I0:], r43["v"][I0:],
                       np.full(DAYS, E_INIT)[I0:], np.full(DAYS, E_INIT)[I0:],
                       emergency=None, tpl_name="result4-3.xlsx",
                       fee_plan=fp[I0:], fee_adj=fa[I0:])
    print(f"[OK] 已写出 {RES/'result4-3.xlsx'}")

    np.savez(RES / "p4_solution.npz",
             g42=r42["g"], u42=r42["u"], v42=r42["v"], w42=r42["w"],
             Es42=r42["Es"], Ee42=r42["Ee"], fee42=r42["fee_day"],
             plan43=r43["plan"], adj43=r43["adj"], u43=r43["u"], v43=r43["v"],
             cur43=r43["cur"], emg43=r43["emg"],
             f42=f42, f42p=f42p, c43a=c43a["total"], c43b=c43b["total"], c43p=c43p["total"])

    # ---------- 图（已移除：统一由 11_figs_final.py 出图，避免覆盖高级版） ----------
    # 本脚本只负责求解与写出 result*.xlsx；论文插图一律跑 `python3 11_figs_final.py`

    print(f"总耗时 {time.time()-t0:.1f}s")
