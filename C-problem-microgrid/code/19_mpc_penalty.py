"""
MPC-II：罚金感知的滚动优化（修正现方案的一个真实建模缺陷）

缺陷所在
--------
题面明确："除紧急购电费用外，其他时间段的购电费用均按计划购电量计算"，且
    计划购电量高于调整购电量的部分，违约电价 = 交易时刻电价的 50%
    调整购电量高于计划购电量的部分，超出部分电价 = 交易时刻电价的 1.5 倍
即计划量是【承诺量、全额按 c 计费】，之后偏离它要付罚金。

但现方案（06_p3.py）在 6:00/12:00/18:00 的子问题里，目标函数是
    min  Σ c_t · a_t · Δt   （购电成本）
这在物理上等价于"调整是免费的、随意重优化即可"，**完全没有惩罚计划修订**。
正确目标应当是【调整购电量的相关费用】：
    min  Σ [ 0.5 c_t (g⁰_t − a_t)⁺ + 1.5 c_t (a_t − g⁰_t)⁺ ] Δt
其中 g⁰ 是 0:00 制定的日计划（承诺量），a 是本次调整后的实际购电量。

本脚本实现该修正（仍是 LP，仅增加 2 个辅助变量/时段），并量化其收益。
"""
import numpy as np
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import (load_attachments, solve_lp, forecast_day_aligned,
                        DT, N_T, DAYS, E_INIT, ETA)
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CPS = ((0, 0), (36, 6), (72, 12), (108, 18))
I0 = 31
EMG = 5.0
P_MAX, E_MIN, E_MAX = 5000.0, 1200.0, 10800.0


def solve_lp_pen(c, L, S, g0, E_start, E_end,
                 eta=ETA, dt=DT, E_min=E_MIN, E_max=E_MAX, P_max=P_MAX):
    """
    最小化【调整购电量的相关费用】，约束与 cumcm_core.solve_lp 一致。
    变量：a(n) u(n) v(n) w(n) E(n) d1(n) d2(n)，其中 d1=(g0−a)⁺、d2=(a−g0)⁺
    约束：
        (a) a − u + v − w = L − S              功率平衡（a = 实际购电）
        (b) E_i − E_{i−1} − ηΔt u_i + (Δt/η) v_i = 0
        (c) d1 − d2 + a = g0                   罚金分解恒等式
        (d) E_{n−1} = E_end
    """
    c = np.asarray(c, float); L = np.asarray(L, float)
    S = np.asarray(S, float); g0 = np.asarray(g0, float)
    n = len(c); N = 7 * n
    nA, nU, nV, nW, nE, nD1, nD2 = 0, n, 2 * n, 3 * n, 4 * n, 5 * n, 6 * n

    cobj = np.zeros(N)
    cobj[nD1:nD1 + n] = 0.5 * c * dt          # 计划高于调整 → 违约电价 50%
    cobj[nD2:nD2 + n] = 1.5 * c * dt          # 调整高于计划 → 超出部分 1.5 倍
    cobj[nU:nU + n] = 1e-6                    # 轻微 tie-break
    cobj[nV:nV + n] = 1e-6

    rows, cols, vals, rhs = [], [], [], []

    r = np.repeat(np.arange(n), 4)
    cc = np.concatenate([nA + np.arange(n), nU + np.arange(n),
                         nV + np.arange(n), nW + np.arange(n)]).reshape(4, n).T.ravel()
    rows.append(r); cols.append(cc); vals.append(np.tile([1., -1., 1., -1.], n))
    rhs.append(L - S)

    r2 = n + np.repeat(np.arange(n), 4)
    cc2 = np.concatenate([nE + np.arange(n), nE + np.arange(n) - 1,
                          nU + np.arange(n), nV + np.arange(n)]).reshape(4, n).T.ravel()
    vv2 = np.tile([1., -1., -eta * dt, dt / eta], n)
    cc2[1] = nE; vv2[1] = 0.0
    rows.append(r2); cols.append(cc2); vals.append(vv2)
    b2 = np.zeros(n); b2[0] = E_start; rhs.append(b2)

    r3 = 2 * n + np.repeat(np.arange(n), 3)
    cc3 = np.concatenate([nD1 + np.arange(n), nD2 + np.arange(n),
                          nA + np.arange(n)]).reshape(3, n).T.ravel()
    rows.append(r3); cols.append(cc3); vals.append(np.tile([1., -1., 1.], n))
    rhs.append(g0)

    rows.append(np.array([3 * n])); cols.append(np.array([nE + n - 1]))
    vals.append(np.array([1.0])); rhs.append(np.array([float(E_end)]))

    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(3 * n + 1, N)).tocsr()
    b = np.concatenate(rhs)
    bounds = ([(0, None)] * n + [(0, P_max)] * n + [(0, P_max)] * n
              + [(0, None)] * n + [(E_min, E_max)] * n
              + [(0, None)] * n + [(0, None)] * n)
    res = linprog(cobj, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    return dict(a=x[nA:nA + n], u=x[nU:nU + n], v=x[nV:nV + n],
                w=x[nW:nW + n], E=x[nE:nE + n],
                d1=x[nD1:nD1 + n], d2=x[nD2:nD2 + n], obj=res.fun)


def simulate(d, penalized=True):
    c, L, S = d["price_day"], d["L"], d["S"]
    plan = np.zeros((DAYS, N_T)); adj = np.zeros((DAYS, N_T))
    uu = np.zeros((DAYS, N_T)); vv = np.zeros((DAYS, N_T))
    cur = np.zeros((DAYS, N_T)); emg = np.zeros((DAYS, N_T))
    Ecur = E_INIT
    for D in range(DAYS):
        fc = {h: forecast_day_aligned(d, D, h) for _, h in CPS}
        Ec = np.zeros(N_T + 1); Ec[0] = Ecur
        for ci, (s, h0) in enumerate(CPS):
            e = CPS[ci + 1][0] if ci + 1 < len(CPS) else N_T
            if ci == 0 or not penalized:
                r = solve_lp(c[s:], L[D, s:], fc[h0][s:], E_start=Ec[s], E_end=E_INIT)
                if ci == 0:
                    plan[D, s:] = r["g"]
                adj[D, s:e] = r["g"][:e - s]
                uu[D, s:e] = r["u"][:e - s]; vv[D, s:e] = r["v"][:e - s]
            else:
                r = solve_lp_pen(c[s:], L[D, s:], fc[h0][s:], plan[D, s:],
                                 E_start=Ec[s], E_end=E_INIT)
                adj[D, s:e] = r["a"][:e - s]
                uu[D, s:e] = r["u"][:e - s]; vv[D, s:e] = r["v"][:e - s]
            for k in range(s, e):
                Ec[k + 1] = Ec[k] + ETA * DT * uu[D, k] - DT / ETA * vv[D, k]
        Ecur = Ec[N_T]
        sur = S[D] + adj[D] + vv[D] - L[D] - uu[D]
        emg[D] = np.maximum(-sur, 0.0); cur[D] = np.maximum(sur, 0.0)
    return dict(plan=plan, adj=adj, u=uu, v=vv, cur=cur, emg=emg)


def cost(rs, d, sl=slice(I0, DAYS)):
    c = d["price_day"]; cw = np.tile(c, (DAYS - I0, 1))
    P, A, E = rs["plan"][sl], rs["adj"][sl], rs["emg"][sl]
    base = float((P * DT * cw).sum())
    pen = float(((0.5 * np.maximum(P - A, 0) + 1.5 * np.maximum(A - P, 0)) * DT * cw).sum())
    emgf = float((E * DT * cw).sum() * EMG)
    return dict(base=base, pen=pen, emg=emgf, total=base + pen + emgf)


if __name__ == "__main__":
    d = load_attachments()
    t0 = time.time()
    print("=" * 88)
    print("MPC-II  罚金感知的滚动优化（修正子问题目标函数）")
    print("=" * 88)

    print("\n[1/2] 现方案：子问题最小化【购电成本】")
    r0 = simulate(d, penalized=False)
    c0 = cost(r0, d)
    print(f"  计划购电 {c0['base']:>14,.0f} + 调整罚金 {c0['pen']:>11,.0f}"
          f" + 紧急购电 {c0['emg']:>11,.0f} = {c0['total']:>15,.2f} 元")
    print(f"  调整偏离总量 {np.abs(r0['adj'] - r0['plan'])[I0:].sum()*DT:>12,.1f} kWh"
          f"   紧急 {r0['emg'][I0:].sum()*DT:>11,.1f} kWh"
          f"   弃光 {r0['cur'][I0:].sum()*DT:>14,.1f} kWh")

    print("\n[2/2] MPC-II：子问题最小化【调整购电量的相关费用】")
    r1 = simulate(d, penalized=True)
    c1 = cost(r1, d)
    print(f"  计划购电 {c1['base']:>14,.0f} + 调整罚金 {c1['pen']:>11,.0f}"
          f" + 紧急购电 {c1['emg']:>11,.0f} = {c1['total']:>15,.2f} 元")
    print(f"  调整偏离总量 {np.abs(r1['adj'] - r1['plan'])[I0:].sum()*DT:>12,.1f} kWh"
          f"   紧急 {r1['emg'][I0:].sum()*DT:>11,.1f} kWh"
          f"   弃光 {r1['cur'][I0:].sum()*DT:>14,.1f} kWh")

    print("\n" + "-" * 88)
    print("【改进】")
    print(f"  调整罚金： {c0['pen']:>12,.0f} → {c1['pen']:>12,.0f} 元"
          f"　（{(c1['pen']-c0['pen'])/c0['pen']*100:+.1f}%）")
    print(f"  总费用  ： {c0['total']/1e4:>12,.1f} → {c1['total']/1e4:>12,.1f} 万元"
          f"　（{(c1['total']-c0['total'])/c0['total']*100:+.2f}%，"
          f"省 {c0['total']-c1['total']:,.0f} 元）")

    z = np.load(RES / "sdp_solution.npz")
    print("\n【三层模型对照（同一信息结构、同一口径）】")
    print(f"  完美信息 + 日周期（下界）      : {z['det_fee']/1e4:>9,.1f} 万元")
    print(f"  动态规划 SDP（策略最优）       : {z['cost_d4']/1e4:>9,.1f} 万元")
    print(f"  MPC-II（罚金感知）             : {c1['total']/1e4:>9,.1f} 万元")
    print(f"  朴素 MPC（现方案）             : {c0['total']/1e4:>9,.1f} 万元")
    print(f"  → MPC-II 相对现方案            : {(c0['total']-c1['total'])/1e4:>+9,.1f} 万元"
          f"（{(c0['total']-c1['total'])/c0['total']*100:+.2f}%）")
    print(f"  → MPC-II 距 SDP 最优仍有       : {(c1['total']-z['cost_d4'])/1e4:>+9,.1f} 万元"
          f"（{(c1['total']-z['cost_d4'])/z['cost_d4']*100:+.2f}%）")

    np.savez(RES / "mpc2_solution.npz",
             plan=r1["plan"], adj=r1["adj"], u=r1["u"], v=r1["v"],
             emg=r1["emg"], cur=r1["cur"],
             cost_base=c1["base"], cost_pen=c1["pen"], cost_emg=c1["emg"],
             cost_total=c1["total"], cost_naive=c0["total"])
    print(f"\n[OK] 已保存 {RES/'mpc2_solution.npz'}    总耗时 {time.time()-t0:.1f}s")
