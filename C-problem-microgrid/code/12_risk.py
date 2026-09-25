"""
问题3/4 的风险维度补充（全年场景法）
1. 场景生成：基于实测残差的"异方差经验自举"（按提前期 × 预报水平分箱）
2. 全年场景法蒙特卡洛：每组场景跑一次完整滚动 MPC -> 年费用分布 -> VaR / CVaR
3. 问题4：经验分位电价构造高/低电价情景 -> 费用区间
"""
import numpy as np
import pandas as pd
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import (load_attachments, solve_lp, forecast_day_aligned,
                        DT, N_T, DAYS, E_INIT, ETA, DATA_DIR)
import style as S

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"; FIG = ROOT / "figures"
I0, SL = 31, slice(31, 365)
CPS = ((0, 0), (36, 6), (72, 12), (108, 18))
NSCEN = 50
# 实测光伏恒为 0 的"深层夜间"时钟小时：这些时段不存在出力不确定性，
# 必须屏蔽扰动，否则会把"黄昏稍有出力"的残差注入"深夜应为 0"的时段，
# 造成场景系统性偏乐观（实测曾使场景期望费用低估 1.9%）。
DARK_HOURS = frozenset([0, 1, 2, 3, 4, 19, 20, 21, 22, 23])
DARK = np.array([(i // 6) in DARK_HOURS for i in range(N_T)])


# ============================================================ 残差库
def build_residual_bank(d, nbins=5):
    X2P = pd.read_excel(DATA_DIR / "附件2.xlsx", "光伏发电实际功率")
    a2P = X2P.iloc[:, 1:].to_numpy(float)
    hidx = {}
    for j, cc in enumerate(X2P.columns[1:]):
        if isinstance(cc, str):
            hidx[24] = j
        elif cc.minute == 0:
            hidx[cc.hour if cc.hour else 24] = j
    dd = pd.to_datetime(X2P.iloc[:, 0])
    actual = {(x.normalize(), h % 24): a2P[i, j]
              for i, x in enumerate(dd) for h, j in hidx.items()}
    bank = {}
    for _, row in d["f3"].iterrows():
        d0 = pd.to_datetime(row["date"]).normalize(); h0 = int(row["h0i"])
        for k in range(1, 25):
            hh = h0 + k
            key = ((d0 + pd.Timedelta(days=hh // 24)).normalize(), hh % 24)
            if key in actual:
                fv, av = float(row[f"k{k}"]), actual[key]
                if fv > 0 or av > 0:
                    bank.setdefault(k, []).append((fv, fv - av))
    pools, edges = {}, {}
    for k, lst in bank.items():
        arr = np.array(lst); f, e = arr[:, 0], arr[:, 1]
        qs = np.quantile(f, np.linspace(0, 1, nbins + 1)); qs[0], qs[-1] = -np.inf, np.inf
        edges[k] = qs
        for b in range(nbins):
            m = (f >= qs[b]) & (f < qs[b + 1])
            if m.sum() > 5:
                pools[(k, b)] = e[m]
        pools.setdefault(("all", k), e)
    return pools, edges, nbins


def draw_resid(pools, edges, nbins, k, f_level, rng):
    qs = edges[k]
    b = int(np.clip(np.searchsorted(qs, f_level, side="right") - 1, 0, nbins - 1))
    pool = pools.get((k, b), pools[("all", k)])
    return rng.choice(pool)


def build_rel_matrices(d, floor=200.0):
    """
    预计算"相对误差形态"矩阵，用于整日分块自举。
      rel0[D][t]  = (0:00 预报 − 实际) / max(实际, floor)
      relh[h0][D][t] = (h0 预报 − 实际) / max(实际, floor)
    只在 max(实际,floor) 上有意义；移植整日形态可保留预报误差的
    日内相关性 —— 这是紧急购电的主要驱动因素，逐时段独立重抽样会
    人为抹平它（实测曾使场景期望费用低估 4.8%）。
    """
    S = d["S"]
    rel0 = np.zeros((DAYS, N_T))
    F0 = np.stack([forecast_day_aligned(d, D, 0) for D in range(DAYS)])
    rel0 = (F0 - S) / np.maximum(S, floor)
    relh = {}
    for h0 in (6, 12, 18):
        Fh = np.stack([forecast_day_aligned(d, D, h0) for D in range(DAYS)])
        r = np.zeros((DAYS, N_T))
        m = np.arange(N_T) >= h0 * 6
        r[:, m] = ((Fh - S) / np.maximum(S, floor))[:, m]
        relh[h0] = r
    return rel0, relh, F0


def gen_year_scenario(d, rel0, relh, F0_all, rng, thresh=200.0):
    """整日分块自举：每个场景日随机抽取一个历史"源日"，移植其日内相对误差形态"""
    Sact = np.zeros((DAYS, N_T))
    for D in range(DAYS):
        src = int(rng.integers(0, DAYS))
        F0 = F0_all[D]
        hi = F0 > thresh
        Sact[D] = np.where(hi, np.clip(F0 * (1.0 - rel0[src]), 0.0, None), F0)
    Fc = {0: F0_all}
    for h0 in (6, 12, 18):
        arr = np.zeros((DAYS, N_T))
        for D in range(DAYS):
            src = int(rng.integers(0, DAYS))
            m = np.arange(N_T) >= h0 * 6
            base = Sact[D]
            v = np.clip(base * (1.0 + relh[h0][src]), 0.0, None)
            arr[D] = np.where(m, v, 0.0)
        Fc[h0] = arr
    return Fc, Sact


def mpc_year(d, Fc, Sact, price, L):
    """给定预报场景与实际光伏，跑整年滚动 MPC，返回交付期总费用与紧急购电量"""
    p3 = np.load(RES / "p3_solution.npz")
    cw = np.tile(price, (DAYS, 1))
    tot = 0.0; emg_kwh = 0.0
    for D in range(DAYS):
        Ec = np.zeros(N_T + 1); Ec[0] = E_INIT
        plan = np.zeros(N_T); adj = np.zeros(N_T); uu = np.zeros(N_T); vv = np.zeros(N_T)
        for ci, (s, h0) in enumerate(CPS):
            e = CPS[ci + 1][0] if ci + 1 < len(CPS) else N_T
            f = Fc[h0][D, s:]
            r = solve_lp(price[s:], L[D, s:], f, E_start=Ec[s], E_end=E_INIT)
            if ci == 0:
                plan[s:] = r["g"]
            adj[s:e] = r["g"][:e - s]; uu[s:e] = r["u"][:e - s]; vv[s:e] = r["v"][:e - s]
            for k in range(s, e):
                Ec[k + 1] = Ec[k] + ETA * DT * uu[k] - DT / ETA * vv[k]
        sur = Sact[D] + adj + vv - L[D] - uu
        emg = np.maximum(-sur, 0.0)
        if D >= I0:
            tot += (plan * DT * cw[D]).sum()
            tot += ((0.5 * np.maximum(plan - adj, 0) + 1.5 * np.maximum(adj - plan, 0)) * DT * cw[D]).sum()
            tot += (emg * DT * cw[D]).sum() * 5
            emg_kwh += (emg * DT).sum()
    return tot, emg_kwh


def mpc_price_year(d, price):
    """给定电价序列跑滚动 MPC（光伏用基准预报），返回交付期总费用"""
    L = d["L"]
    Fc = {h: np.stack([forecast_day_aligned(d, D, h) for D in range(DAYS)]) for _, h in CPS}
    tot, _ = mpc_year(d, Fc, d["S"], price, L)
    return tot


if __name__ == "__main__":
    S.setup()
    t0 = time.time()
    d = load_attachments()
    c = d["price_day"]; L, Sv = d["L"], d["S"]; dates = pd.DatetimeIndex(d["dates"])
    base_fee = float(np.load(RES / "p3_solution.npz")["costA"])

    rel0, relh, F0_all = build_rel_matrices(d)
    print(f"[相对误差形态] 已建立整日分块自举所需的 {DAYS}×{N_T} 形态矩阵")

    # ---------- 全年场景法 ----------
    rng = np.random.default_rng(2026)
    costs, emgs = [], []
    for n in range(NSCEN):
        Fc, Sact = gen_year_scenario(d, rel0, relh, F0_all, rng)
        # 基准场景（首组用实测数据校验）
        if n == 0:
            Fc0 = {h: np.stack([forecast_day_aligned(d, D, h) for D in range(DAYS)]) for _, h in CPS}
            chk, _ = mpc_year(d, Fc0, Sv, c, L)
            print(f"[校验] 用实测光伏+实测预报重跑 -> {chk:,.2f} 元（应为 {base_fee:,.2f}）")
        f, ek = mpc_year(d, Fc, Sact, c, L)
        costs.append(f); emgs.append(ek)
        if (n + 1) % 10 == 0:
            print(f"  场景 {n+1:3d}/{NSCEN}  费用 {f:14,.0f} 元  用时 {time.time()-t0:5.0f}s")
    costs = np.array(costs); emgs = np.array(emgs)
    Fc_d, Sact_d = gen_year_scenario(d, rel0, relh, F0_all, np.random.default_rng(99))
    print(f"\n[场景一致性] 实测年发电量 {Sv.sum()*DT:,.0f} kWh　"
          f"场景年发电量 {Sact_d.sum()/DAYS*365*DT if False else Sact_d.sum()*DT:,.0f} kWh"
          f"（{(Sact_d.sum()/Sv.sum()-1)*100:+.2f}%）")

    print("\n" + "=" * 82)
    print(f"问题3 全年场景法风险评估（{NSCEN} 组年场景，交付期 334 天）")
    print("=" * 82)
    print(f"  基准（实测光伏）  : {base_fee:14,.2f} 元")
    print(f"  场景期望          : {costs.mean():14,.2f} 元"
          f"（相对基准 {(costs.mean()-base_fee)/base_fee*100:+.2f}%）")
    print(f"  标准差            : {costs.std():14,.2f} 元")
    for q in (50, 75, 90, 95):
        print(f"  VaR{q:2d}%           : {np.percentile(costs, q):14,.2f} 元")
    for q in (90, 95):
        th = np.percentile(costs, q)
        print(f"  CVaR{q:2d}%          : {costs[costs >= th].mean():14,.2f} 元"
              f"（超期望 {costs[costs>=th].mean()-costs.mean():,.2f} 元）")
    print(f"  紧急购电量 期望 {emgs.mean():12,.2f} kWh　95 分位 {np.percentile(emgs,95):,.2f} kWh")

    # ---------- 问题4 电价情景 ----------
    c4 = d["C4"]
    c_hi = np.quantile(c4, 0.95, axis=0); c_lo = np.quantile(c4, 0.05, axis=0)
    c_typ = c4.mean(0)
    c43 = float(np.load(RES / "p4_solution.npz")["c43a"])
    print("\n" + "=" * 82)
    print("问题4 电价情景（逐时段经验分位，作为费用区间参照）")
    print("=" * 82)
    print(f"  95 分位电价均值 {c_hi.mean():.4f} 元/kWh　典型 {c_typ.mean():.4f}　5 分位 {c_lo.mean():.4f}")
    t1 = time.time()
    cost_hi = mpc_price_year(d, c_hi)
    cost_lo = mpc_price_year(d, c_lo)
    print(f"  高电价情景（95 分位）总费用 : {cost_hi:14,.2f} 元")
    print(f"  低电价情景（5  分位）总费用 : {cost_lo:14,.2f} 元")
    print(f"  实际电价（4-3）            : {c43:14,.2f} 元")
    print(f"  → 电价风险区间宽度          : {cost_hi-cost_lo:14,.2f} 元"
          f"（占实际值 {(cost_hi-cost_lo)/c43*100:.1f}%）　耗时 {time.time()-t1:.0f}s")

    np.savez(RES / "risk.npz", costs=costs, emgs=emgs, base_fee=base_fee,
             cost_hi=cost_hi, cost_lo=cost_lo, c43=c43,
             c_hi=c_hi, c_lo=c_lo)
    print(f"\n[OK] 已写出 {RES/'risk.npz'}　总耗时 {time.time()-t0:.0f}s")
