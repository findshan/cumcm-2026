"""
公共模块：罚金感知的 LP 子问题求解器（供 06_p3 / 07_p4 复用）

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
from cumcm_core import DT, ETA
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

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


