"""
问题3 的随机动态规划（SDP）对照实验
=====================================

目的
----
在【与滚动 MPC 完全相同的信息结构】下，求"期望费用最小"的最优反馈策略，
用 SDP 与 MPC 的差额度量"确定性等价（certainty-equivalent）近似"的代价，
并回答"本题是否需要随机动态规划"。

SDP 要素
--------
* 状态      x_t = E_t（储电量，kWh），[1200, 10800] 上 97 点均匀网格
* 信息      y_t = 当期预报（含提前期 k_t）；不确定量只有实发光伏 S_t
* 决策      (g_t, u_t, v_t) → 计划购电量、充/放电功率。因成本只通过 δ=u−v 影响
            功率平衡、且 x' 关于 δ 单调，可把动作压缩为一维 δ ∈ [−5000, 5000]
* 随机      ε_t = F_t − S_t 取自按 (提前期 k, 预报水平分箱) 的经验分布（7 个等概率分位点）
* 递推      V_t(x) = min_δ { c_t Δt [g + 罚(g, g_prev)]
                            + 5 c_t Δt · E[(net − g)⁺] + V_{t+1}(x′) }
            x′ = x + [η·max(δ,0) − max(−δ,0)/η]·Δt
* 终端      V_144(x) = 0 (x ≥ 6000)，+∞ (x < 6000)  ← 与 MPC 一致的日周期约束

关键性质（使 DP 可精确求解）
--------------------------
净需求 net = A + ε，其中 A = L_t + u_t − v_t − F_t 与状态 x 无关。
因此：① 最优计划 g* 与状态无关，可对每个 (t, δ) 一次性精确求出；
      ② 成本关于 g 是凸分段线性函数，其导数断点恰为 {0, g_prev} ∪ {A + ε_m}，
         故只需在这 m+2 个候选点上比较即得全局最优（无需搜索）。

用法
----
    python3 18_sdp.py
"""
import numpy as np
import sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import (load_attachments, forecast_day_aligned,
                        DT, N_T, DAYS, E_INIT, ETA, DATA_DIR)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

CPS4 = ((0, 0), (36, 6), (72, 12), (108, 18))     # 与 MPC 同：4 次预报
CPS1 = ((0, 0),)                                  # 仅 0:00 预报
I0 = 31                                           # 交付起点 2025-02-01

P_MAX, E_MIN, E_MAX = 5000.0, 1200.0, 10800.0
BIG = 1e12
EMG = 5.0          # 紧急购电倍率
PEN_DN, PEN_UP = 0.5, 1.5    # 计划偏离罚金系数（与 06_p3 口径A 一致）

# ---------------- 离散化 ----------------
NG = 97
E_GRID = np.linspace(E_MIN, E_MAX, NG)
EI = int(np.argmin(np.abs(E_GRID - E_INIT)))      # 6000 对应下标
ND = 41
DELTA = np.linspace(-P_MAX, P_MAX, ND)
STEP = (ETA * np.maximum(DELTA, 0.0) - np.maximum(-DELTA, 0.0) / ETA) * DT
X_NEW = E_GRID[:, None] + STEP[None, :]
X_OK = (X_NEW >= E_MIN - 1e-9) & (X_NEW <= E_MAX + 1e-9)

M = 9              # 每个池的等概率分位点数
F_TH = 30.0        # 预报低于此值视为夜间，无不确定性
F_SPLIT = 500.0    # 预报水平分箱阈值（kW）


# ============================================================
# 1. 建立 ε = F − S 的经验条件分布库
# ============================================================
def build_eps_bank(d):
    X2P = pd.read_excel(DATA_DIR / "附件2.xlsx", "光伏发电实际功率")
    a2P = X2P.iloc[:, 1:].to_numpy(float)
    cols = list(X2P.columns[1:])
    hidx = {}
    for j, cc0 in enumerate(cols):
        if isinstance(cc0, str):
            hidx[24] = j
        elif cc0.minute == 0:
            hidx[cc0.hour if cc0.hour else 24] = j
    dates = pd.to_datetime(X2P.iloc[:, 0])
    actual = {(d0.normalize(), h % 24): a2P[i, j]
              for i, d0 in enumerate(dates) for h, j in hidx.items()}

    f3 = d["f3"]
    raw = {k: [] for k in range(1, 25)}
    for _, row in f3.iterrows():
        d0 = pd.to_datetime(row["date"]).normalize()
        h0 = int(row["h0i"])
        for k in range(1, 25):
            hh = h0 + k
            key = ((d0 + pd.Timedelta(days=hh // 24)).normalize(), hh % 24)
            if key in actual:
                raw[k].append((float(row[f"k{k}"]), float(actual[key])))

    qs = (np.arange(M) + 0.5) / M
    bank = {}
    n_used = 0
    for k in range(1, 25):
        arr = np.array(raw[k])
        F, A = arr[:, 0], arr[:, 1]
        eps = F - A
        for b in range(2):
            sel = (F >= F_TH) & (F < F_SPLIT) if b == 0 else (F >= F_SPLIT)
            if sel.sum() >= 30:
                bank[(k, b)] = np.quantile(eps[sel], qs)
                n_used += 1
            elif (F >= F_TH).sum() >= 30:
                bank[(k, b)] = np.quantile(eps[F >= F_TH], qs)
                n_used += 1
            else:
                bank[(k, b)] = np.zeros(M)
    return bank, qs


# ============================================================
# 2. 单时段的"最优计划"闭式解
# ============================================================
def best_plan(A, pts, gp, c, dt, has_prev):
    """
    A    : (ND,) 净需求中心 = L + δ − F
    pts  : (M,)  ε 的等概率分位点
    gp   : 上一版计划（标量，该时段）
    返回 (g*, stage_cost) 各 (ND,)
    """
    cand = np.concatenate([np.zeros((len(A), 1)), A[:, None] + pts[None, :]], axis=1)  # (ND, M+1)
    if has_prev:
        cand = np.concatenate([cand, np.full((len(A), 1), float(gp))], axis=1)
    cand = np.maximum(cand, 0.0)                                            # g ≥ 0
    g = cand                                                               # (ND, K)
    K = g.shape[1]
    if has_prev:
        pen = PEN_DN * np.maximum(gp - g, 0.0) + PEN_UP * np.maximum(g - gp, 0.0)
    else:
        pen = np.zeros_like(g)
    # E[(A + ε − g)⁺] = (1/M) Σ_m max(A + pts_m − g, 0)
    d = A[:, None, None] + pts[None, None, :] - g[:, :, None]              # (ND,K,M)
    ec = np.maximum(d, 0.0).mean(axis=2)                                   # (ND,K)
    cost = c * dt * (g + pen) + EMG * c * dt * ec
    j = np.argmin(cost, axis=1)
    return g[np.arange(len(A)), j], cost[np.arange(len(A)), j]


# ============================================================
# 3. 后向递推 + 前向执行
# ============================================================
def sdp_backward(s, h0, F, L, c, gprev, has_prev, bank):
    """返回 Vlist (N_T+1, NG) 与策略 argmin 下标 Pilist (N_T+1, NG)"""
    V = np.where(E_GRID >= E_INIT - 1e-9, 0.0, BIG)
    Vlist = np.empty((N_T + 1, NG)); Vlist[N_T] = V
    Pilist = np.zeros((N_T + 1, NG), dtype=np.int16)
    for i in range(N_T - 1, s - 1, -1):
        k = int(np.clip(round((i + 1) / 6.0 - h0), 1, 24))
        b = 0 if F[i] < F_SPLIT else 1
        pts = bank[(k, b)]
        A = L[i] + DELTA - F[i]
        _, sc = best_plan(A, pts, gprev[i] if has_prev else 0.0, c[i], DT, has_prev)
        Vi = np.interp(X_NEW.ravel(), E_GRID, V).reshape(NG, ND)
        tot = sc[None, :] + Vi
        tot = np.where(X_OK, tot, BIG)
        j = np.argmin(tot, axis=1)
        V = tot[np.arange(NG), j]
        Vlist[i] = V
        Pilist[i] = j
    return Vlist, Pilist


def plan_of(Pilist, i, E, F, L, c, gprev, has_prev, bank):
    """给定状态 E，按策略取本时段动作，并返回最优计划 g、充放电、阶段成本"""
    e = int(np.clip(np.searchsorted(E_GRID, E), 0, NG - 1))
    if e > 0 and abs(E_GRID[e - 1] - E) < abs(E_GRID[e] - E):
        e -= 1
    j = int(Pilist[i, e])
    dd = DELTA[j]
    u = max(dd, 0.0); v = max(-dd, 0.0)
    k = None
    A = L[i] + dd - F[i]
    return u, v, A


def sdp_forward(Pilist, s, e, E, F, L, c, S_act, gprev, has_prev, bank, h0):
    """前向执行 [s, e)：返回 E 轨迹、计划 g、充放、实际紧急/弃光、费用"""
    n = e - s
    g = np.zeros(n); uu = np.zeros(n); vv = np.zeros(n)
    emg = np.zeros(n); cur = np.zeros(n); pen_a = np.zeros(n)
    cost = 0.0
    for t, i in enumerate(range(s, e)):
        idx = int(np.clip(np.searchsorted(E_GRID, E), 0, NG - 1))
        if idx > 0 and abs(E_GRID[idx - 1] - E) < abs(E_GRID[idx] - E):
            idx -= 1
        j = int(Pilist[i, idx])
        dd = DELTA[j]
        u, v = max(dd, 0.0), max(-dd, 0.0)
        A = L[i] + dd - F[i]
        k = int(np.clip(round((i + 1) / 6.0 - h0), 1, 24))
        b = 0 if F[i] < F_SPLIT else 1
        gg, _ = best_plan(np.array([A]), bank[(k, b)],
                          gprev[i] if has_prev else 0.0, c[i], DT, has_prev)
        gg = float(gg[0])
        pen = (PEN_DN * max(gprev[i] - gg, 0.0) + PEN_UP * max(gg - gprev[i], 0.0)) if has_prev else 0.0
        pen_a[t] = pen
        sur = S_act[i] + gg - u + v - L[i]
        emg[t] = max(-sur, 0.0); cur[t] = max(sur, 0.0)
        cost += c[i] * DT * (gg + pen) + EMG * c[i] * DT * emg[t]
        g[t] = gg; uu[t] = u; vv[t] = v
        E = E + ETA * DT * u - DT / ETA * v
    return E, g, uu, vv, emg, cur, pen_a, cost


def roll_plan(Pilist, s, e_end, E, F, L, c, gprev, has_prev, bank, h0):
    """从状态 E 出发、按策略纯滚动（不消耗实际光伏）得到 [s, e_end) 的计划序列"""
    out = np.zeros(N_T)
    for i in range(s, e_end):
        idx = int(np.clip(np.searchsorted(E_GRID, E), 0, NG - 1))
        if idx > 0 and abs(E_GRID[idx - 1] - E) < abs(E_GRID[idx] - E):
            idx -= 1
        j = int(Pilist[i, idx])
        dd = DELTA[j]
        u, v = max(dd, 0.0), max(-dd, 0.0)
        A = L[i] + dd - F[i]
        k = int(np.clip(round((i + 1) / 6.0 - h0), 1, 24))
        b = 0 if F[i] < F_SPLIT else 1
        out[i] = float(best_plan(np.array([A]), bank[(k, b)],
                                 gprev[i] if has_prev else 0.0, c[i], DT, has_prev)[0][0])
        E = E + ETA * DT * u - DT / ETA * v
    return out, E


def simulate_sdp(d, cps, bank, verbose=False):
    c = d["price_day"]; L = d["L"]; S = d["S"]
    G = np.zeros((DAYS, N_T)); U = np.zeros((DAYS, N_T)); V = np.zeros((DAYS, N_T))
    EM = np.zeros((DAYS, N_T)); CU = np.zeros((DAYS, N_T)); PN = np.zeros((DAYS, N_T))
    Ecur = E_INIT
    for D in range(DAYS):
        fc = {h: forecast_day_aligned(d, D, h) for _, h in cps}
        Est = np.zeros(N_T + 1); Est[0] = Ecur
        gprev = np.zeros(N_T); has_prev = False
        for ci, (s, h0) in enumerate(cps):
            e = cps[ci + 1][0] if ci + 1 < len(cps) else N_T
            Vl, Pl = sdp_backward(s, h0, fc[h0], L[D], c, gprev, has_prev, bank)
            # 前向执行 [s, e)
            Eend, gg, uu, vv, em, cu, pn, cc = sdp_forward(
                Pl, s, e, Ecur, fc[h0], L[D], c, S[D], gprev, has_prev, bank, h0)
            G[D, s:e] = gg; U[D, s:e] = uu; V[D, s:e] = vv
            EM[D, s:e] = em; CU[D, s:e] = cu; PN[D, s:e] = pn
            for t, i in enumerate(range(s, e)):
                Est[i + 1] = Est[i] + ETA * DT * uu[t] - DT / ETA * vv[t]
            Ecur = Eend
            # 本版计划：把 [e, N_T) 也按策略滚动出来，作为下一轮的 g_prev
            tail, _ = roll_plan(Pl, s, N_T, Est[s], fc[h0], L[D], c, gprev, has_prev, bank, h0)
            gprev = tail
            has_prev = True
        if D >= I0 and verbose:
            pass
        if D % 60 == 0:
            print(f"    第 {D:3d} 天  当日费用 {cc:>10,.0f} 元  储电量 {Ecur:8.1f} kWh")
    return dict(g=G, u=U, v=V, emg=EM, cur=CU, pen=PN)


def evaluate(rs, d, c, sl=slice(I0, DAYS)):
    plan = rs["g"][sl]; emg = rs["emg"][sl]; pen = rs["pen"][sl]
    cw = np.tile(c, (plan.shape[0], 1))
    base = float((plan * DT * cw).sum())
    pen_fee = float((pen * DT * cw).sum())
    emg_fee = float((emg * DT * cw).sum() * EMG)
    return dict(base=base, pen=pen_fee, emg=emg_fee,
                total=base + pen_fee + emg_fee)


if __name__ == "__main__":
    d = load_attachments()
    print("=" * 92)
    print("问题3  随机动态规划（SDP）对照实验")
    print("=" * 92)
    t0 = time.time()
    bank, qs = build_eps_bank(d)
    bank_ce = {key: np.zeros(M) for key in bank}   # ε≡0：与 MPC 同假设的"预报即真值"库
    print(f"[误差库] 24 个提前期 × 2 个预报水平箱，每箱 {M} 个等概率分位点")
    print("[对照库] bank_ce：ε≡0（预报即真值），用于分离『罚金感知』与『分布感知』两种改进")
    for k in (1, 6, 12, 24):
        print(f"   k={k:2d}  低出力箱 ε 分位点: "
              + " ".join(f"{v:8.1f}" for v in bank[(k, 0)]))
        print(f"          高出力箱 ε 分位点: "
              + " ".join(f"{v:8.1f}" for v in bank[(k, 1)]))

    VAR = [("SDP-CE  四次预报（把预报当真值，但优化中计入偏离罚金）", CPS4, bank_ce),
           ("SDP     四次预报（用预报误差分布）", CPS4, bank),
           ("SDP-CE  仅 0:00 预报", CPS1, bank_ce),
           ("SDP     仅 0:00 预报（用预报误差分布）", CPS1, bank)]
    out = {}
    for name, cps, bk in VAR:
        print("\n" + "-" * 92)
        print(f"[{name}]")
        t = time.time()
        rs = simulate_sdp(d, cps, bk)
        ev = evaluate(rs, d, d["price_day"])
        out[name] = (rs, ev)
        print(f"  → 计划购电 {ev['base']:>14,.0f} + 偏离罚金 {ev['pen']:>11,.0f}"
              f" + 紧急购电 {ev['emg']:>12,.0f} = 总费用 {ev['total']:>15,.2f} 元")
        print(f"     紧急购电 {rs['emg'][I0:].sum()*DT:>12,.2f} kWh"
              f"   弃光 {rs['cur'][I0:].sum()*DT:>14,.2f} kWh   耗时 {time.time()-t:.1f}s")

    # ---------------- 对比汇总 ----------------
    z = np.load(RES / "p3_solution.npz")
    c = d["price_day"]; cw = np.tile(c, (DAYS - I0, 1))
    mp = z["plan"][I0:]; ma = z["adj"][I0:]; me = z["emg"][I0:]
    m_dlo = np.maximum(mp - ma, 0); m_dhi = np.maximum(ma - mp, 0)
    m_base = float((mp * DT * cw).sum())
    m_pen = float(((0.5 * m_dlo + 1.5 * m_dhi) * DT * cw).sum())
    m_emg = float((me * DT * cw).sum() * EMG)
    m_tot = m_base + m_pen + m_emg

    print("\n" + "=" * 92)
    print("结果对比（完全相同的信息结构：附件3 预报 + 附件2 实际光伏结算，口径A）")
    print("=" * 92)
    print(f"{'方法':<46}{'总费用(万元)':>13}{'紧急购电(kWh)':>15}{'弃光(kWh)':>14}")
    print("-" * 92)
    print(f"{'完美信息 + 日周期（不可达到的下界）':<46}{z['det_fee']/1e4:13,.1f}{0:15,.0f}{0:14,.0f}")
    rows = [("MPC 确定性等价  仅 0:00 预报（现方案）", None, None,
             {'base': 0, 'pen': 0, 'emg': 0, 'total': float(z['cost0'])}, None)]
    print(f"{'MPC 确定性等价  仅 0:00 预报':<46}{z['cost0']/1e4:13,.1f}{'—':>15}{'—':>14}")
    print(f"{'MPC 确定性等价  四次预报（现交付方案）':<46}{m_tot/1e4:13,.1f}"
          f"{me.sum()*DT:15,.0f}{z['cur'][I0:].sum()*DT:14,.0f}")
    for name, cps, bk in VAR:
        rs, ev = out[name]
        print(f"{name:<46}{ev['total']/1e4:13,.1f}"
              f"{rs['emg'][I0:].sum()*DT:15,.0f}{rs['cur'][I0:].sum()*DT:14,.0f}")
    print("-" * 92)

    s_ce4 = out[VAR[0][0]][1]['total']; s_d4 = out[VAR[1][0]][1]['total']
    s_ce1 = out[VAR[2][0]][1]['total']; s_d1 = out[VAR[3][0]][1]['total']
    print("【改进分解 · 四次预报】（正 = 省下）")
    print(f"  ① 罚金感知（把预报当真值，但优化中计入计划偏离罚金）: {m_tot-s_ce4:>+13,.0f} 元"
          f"（{(m_tot-s_ce4)/m_tot*100:+.2f}%）")
    print(f"  ② 分布感知（在①之上，用预报误差分布替代点预测）    : {s_ce4-s_d4:>+13,.0f} 元"
          f"（{(s_ce4-s_d4)/s_ce4*100:+.2f}%）")
    print(f"  合计 SDP 相对现方案 MPC 的改进                      : {m_tot-s_d4:>+13,.0f} 元"
          f"（{(m_tot-s_d4)/m_tot*100:+.2f}%）")
    print("【改进分解 · 仅 0:00 预报】（正 = 省下）")
    print(f"  ① 罚金感知: {z['cost0']-s_ce1:>+13,.0f} 元（{(z['cost0']-s_ce1)/z['cost0']*100:+.2f}%）"
          f"    ② 分布感知: {s_ce1-s_d1:>+13,.0f} 元（{(s_ce1-s_d1)/s_ce1*100:+.2f}%）")
    print(f"  （单检查点下无计划修订，罚金恒为 0，故①的差异全部来自"
          f"'计划是否按分布中心修正'：ε≡0 时计划＝纯确定性最优）")
    print("【预报频次的价值（在 SDP 框架下）】")
    print(f"  SDP：1 次 → 4 次节省 {s_d1-s_d4:>+13,.0f} 元（{(s_d1-s_d4)/s_d1*100:.2f}%）"
          f"　（MPC 下为 {z['cost0']-z['costA']:,.0f} 元，{(z['cost0']-z['costA'])/z['cost0']*100:.2f}%）")
    print(f"  → SDP 距完美信息仍差 {s_d4-z['det_fee']:,.0f} 元（{(s_d4-z['det_fee'])/z['det_fee']*100:.2f}%），"
          f"MPC 为 {z['costA']-z['det_fee']:,.0f} 元（{(z['costA']-z['det_fee'])/z['det_fee']*100:.2f}%）")

    np.savez(RES / "sdp_solution.npz",
             g=out[VAR[1][0]][0]["g"], emg=out[VAR[1][0]][0]["emg"],
             cur=out[VAR[1][0]][0]["cur"], u=out[VAR[1][0]][0]["u"],
             v=out[VAR[1][0]][0]["v"], pen=out[VAR[1][0]][0]["pen"],
             g_ce=out[VAR[0][0]][0]["g"], g1=out[VAR[3][0]][0]["g"],
             cost_ce4=s_ce4, cost_d4=s_d4, cost_ce1=s_ce1, cost_d1=s_d1,
             cost_mpc4=m_tot, cost_mpc1=float(z['cost0']), det_fee=float(z['det_fee']))
    print(f"\n[OK] 已保存 {RES/'sdp_solution.npz'}")
    print(f"总耗时 {time.time()-t0:.1f}s")
