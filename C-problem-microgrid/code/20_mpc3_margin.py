"""
MPC-III：在罚金感知基础上，再按 SDP 的闭式分位数规则标定【日计划的安全边际】

理论依据（见《模型类型评估与随机动态规划分析.md》）
------------------------------------------------
在题面的 take-or-pay 结构下（计划量全额计费，偏离付 0.5c / 1.5c），
以 net 记"必须从外网取得的电量"，g 记日计划，a 记调整量，则

    a*(g, net) = max( g,  q_{0.7}(net) )
    g*         = q_{0.727}(net)

即：① 调整量取"max(计划, 净需求的 70% 分位)"——因为 5 倍紧急电价使 30% 的缺电概率可接受；
    ② 日计划应设在净需求的 ~73% 分位，而不是朴素 MPC 所用的"点预测"（≈中位数）。

本脚本把 ② 落地为"对 0:00 预报按 ε 的 p 分位打折"，并扫描 p 标定最优值。
（① 已由 19_mpc_penalty.py 的子问题目标修正自动实现。）
"""
import numpy as np
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import (load_attachments, solve_lp, forecast_day_aligned,
                        DT, N_T, DAYS, E_INIT, ETA)
from cumcm_pen import solve_lp_pen
from importlib import import_module
r_sdp = import_module("18_sdp")

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CPS = ((0, 0), (36, 6), (72, 12), (108, 18))
I0 = 31
EMG = 5.0
SPLIT = 500.0


def build_quantile_table(d, bank=None):
    """每 (提前期 k, 预报水平箱) 的 ε 分位点表 → 供任意 p 查询"""
    if bank is None:
        bank, _ = r_sdp.build_eps_bank(d)
    return bank


def eps_quantile(bank, k, F, p):
    """给定提前期 k 与预报水平 F，取 ε 的 p 分位（在 9 个等概率分位点间线性插值）"""
    b = 0 if F < SPLIT else 1
    pts = bank[(int(np.clip(k, 1, 24)), b)]
    m = len(pts)
    x = p * m - 0.5
    x = np.clip(x, 0.0, m - 1.0)
    i0 = int(np.floor(x)); i1 = min(i0 + 1, m - 1); w = x - i0
    return float(pts[i0] * (1 - w) + pts[i1] * w)


def simulate(d, p_plan=0.5, penalized=True, bank=None):
    """
    p_plan : 0:00 建计划时，把预报按 ε 的 p_plan 分位保守化（0.5 = 中位数 ≈ 朴素点预测）
    """
    c, L, S = d["price_day"], d["L"], d["S"]
    plan = np.zeros((DAYS, N_T)); adj = np.zeros((DAYS, N_T))
    uu = np.zeros((DAYS, N_T)); vv = np.zeros((DAYS, N_T))
    cur = np.zeros((DAYS, N_T)); emg = np.zeros((DAYS, N_T))
    Ecur = E_INIT
    for D in range(DAYS):
        fc = {h: forecast_day_aligned(d, D, h) for _, h in CPS}
        # 0:00 建计划所用的"保守化预报"
        F0 = fc[0]
        k_all = np.clip(np.round((np.arange(1, N_T + 1)) / 6.0), 1, 24).astype(int)
        eq = np.array([eps_quantile(bank, k_all[i], F0[i], p_plan) for i in range(N_T)]) \
            if p_plan != 0.5 else np.zeros(N_T)
        F0_adj = np.maximum(F0 - eq, 0.0)
        Ec = np.zeros(N_T + 1); Ec[0] = Ecur
        for ci, (s, h0) in enumerate(CPS):
            e = CPS[ci + 1][0] if ci + 1 < len(CPS) else N_T
            if ci == 0:
                r = solve_lp(c[s:], L[D, s:], F0_adj[s:], E_start=Ec[s], E_end=E_INIT)
                gv = r["g"]
                plan[D, s:] = gv
            elif penalized:
                r = solve_lp_pen(c[s:], L[D, s:], fc[h0][s:], plan[D, s:],
                                 E_start=Ec[s], E_end=E_INIT)
                gv = r["a"]
            else:
                r = solve_lp(c[s:], L[D, s:], fc[h0][s:], E_start=Ec[s], E_end=E_INIT)
                gv = r["g"]
            adj[D, s:e] = gv[:e - s]
            uu[D, s:e] = r["u"][:e - s]; vv[D, s:e] = r["v"][:e - s]
            for k in range(s, e):
                Ec[k + 1] = Ec[k] + ETA * DT * uu[D, k] - DT / ETA * vv[D, k]
        Ecur = Ec[N_T]
        sur = S[D] + adj[D] + vv[D] - L[D] - uu[D]
        emg[D] = np.maximum(-sur, 0.0); cur[D] = np.maximum(sur, 0.0)
    return dict(plan=plan, adj=adj, u=uu, v=vv, cur=cur, emg=emg)


def cost(rs, d, sl=slice(I0, DAYS), cw=None):
    c = d["price_day"] if cw is None else cw
    cwx = np.tile(c, (sl.stop - sl.start, 1)) if c.ndim == 1 else c[sl]
    P, A, E = rs["plan"][sl], rs["adj"][sl], rs["emg"][sl]
    base = float((P * DT * cwx).sum())
    pen = float(((0.5 * np.maximum(P - A, 0) + 1.5 * np.maximum(A - P, 0)) * DT * cwx).sum())
    emgf = float((E * DT * cwx).sum() * EMG)
    return dict(base=base, pen=pen, emg=emgf, total=base + pen + emgf)


if __name__ == "__main__":
    d = load_attachments()
    t0 = time.time()
    bank = build_quantile_table(d)
    print("=" * 92)
    print("MPC-III  日计划安全边际标定（按 SDP 的闭式分位数规则）")
    print("=" * 92)
    print(f"  {'p':>6}{'计划购电(元)':>16}{'调整罚金(元)':>14}{'紧急购电(元)':>14}{'总费用(万元)':>14}")
    print("-" * 92)
    rows = []
    for p in (0.50, 0.60, 0.65, 0.70, 0.73, 0.80, 0.90):
        rs = simulate(d, p_plan=p, penalized=True, bank=bank)
        cc = cost(rs, d)
        rows.append((p, cc, rs))
        print(f"  {p:>6.2f}{cc['base']:>16,.0f}{cc['pen']:>14,.0f}{cc['emg']:>14,.0f}"
              f"{cc['total']/1e4:>14,.1f}")
    print("-" * 92)
    best = min(rows, key=lambda r: r[1]["total"])
    base_row = [r for r in rows if abs(r[0] - 0.50) < 1e-9][0]
    print(f"  最优 p = {best[0]:.2f}（理论值 0.727）  总费用 {best[1]['total']/1e4:,.1f} 万元")
    print(f"  相对 p=0.50（朴素点预测）：省 {base_row[1]['total']-best[1]['total']:,.0f} 元"
          f"（{(base_row[1]['total']-best[1]['total'])/base_row[1]['total']*100:.2f}%）")

    z = np.load(RES / "p3_solution.npz")
    zs = np.load(RES / "sdp_solution.npz")
    print("\n【四层模型最终对照（同一信息结构、同一口径A）】")
    print(f"  完美信息 + 日周期（不可达到的下界） : {z['det_fee']/1e4:>9,.1f} 万元")
    print(f"  SDP 随机动态规划（策略最优）        : {zs['cost_d4']/1e4:>9,.1f} 万元")
    print(f"  MPC-III 罚金感知 + 最优日计划分位    : {best[1]['total']/1e4:>9,.1f} 万元")
    print(f"  MPC-II  仅罚金感知                  : {base_row[1]['total']/1e4:>9,.1f} 万元"
          f"   ← 上面 p=0.50 那一行")
    print(f"  朴素 MPC（原方案）                  : {z['cost_naive']/1e4:>9,.1f} 万元")

    rs = best[2]
    np.savez(RES / "mpc3_solution.npz",
             plan=rs["plan"], adj=rs["adj"], u=rs["u"], v=rs["v"],
             emg=rs["emg"], cur=rs["cur"], p_plan=best[0],
             cost_base=best[1]["base"], cost_pen=best[1]["pen"],
             cost_emg=best[1]["emg"], cost_total=best[1]["total"])
    print(f"\n[OK] 已保存 {RES/'mpc3_solution.npz'}    总耗时 {time.time()-t0:.1f}s")
