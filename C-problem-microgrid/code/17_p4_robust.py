"""
问题4 鲁棒建模（修正版）

⚠️ 方法要点：把电价整体乘以 (1+ρ) 是【无效】的 —— 目标函数正比例缩放不改变 argmin，
   四个 ρ 会给出完全相同的调度（实测已验证）。
   正确的 box 不确定集鲁棒对等为【逐时段取上界电价】：
       min Σ c_t g_t  ,  c_t ∈ [c_t^min, c_t^max]
       ⇒  min Σ c_t^max g_t
   只有当上界电价的"形状"与名义电价不同时，调度才会改变。
   这里用逐时段经验分位作为上界，α 越高越保守。
"""
import numpy as np
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import load_attachments, solve_lp, forecast_day_aligned, DT, N_T, DAYS, E_INIT, ETA

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
I0 = 31
CPS = ((0, 0), (36, 6), (72, 12), (108, 18))


def year_cost(d, Fc, Sact, plan_price, settle_price, L):
    tot = 0.0
    for D in range(DAYS):
        Ec = np.zeros(N_T + 1); Ec[0] = E_INIT
        pl = np.zeros(N_T); ad = np.zeros(N_T); uu = np.zeros(N_T); vv = np.zeros(N_T)
        for ci, (s, h0) in enumerate(CPS):
            e = CPS[ci + 1][0] if ci + 1 < len(CPS) else N_T
            r = solve_lp(plan_price[s:], L[D, s:], Fc[h0][D, s:], E_start=Ec[s], E_end=E_INIT)
            if ci == 0:
                pl[s:] = r["g"]
            ad[s:e] = r["g"][:e - s]; uu[s:e] = r["u"][:e - s]; vv[s:e] = r["v"][:e - s]
            for k in range(s, e):
                Ec[k + 1] = Ec[k] + ETA * DT * uu[k] - DT / ETA * vv[k]
        su = Sact[D] + ad + vv - L[D] - uu
        em = np.maximum(-su, 0.0)
        if D >= I0:
            tot += (pl * DT * settle_price).sum()
            tot += ((0.5 * np.maximum(pl - ad, 0) + 1.5 * np.maximum(ad - pl, 0)) * DT * settle_price).sum()
            tot += (em * DT * settle_price).sum() * 5
    return tot


if __name__ == "__main__":
    t0 = time.time()
    d = load_attachments()
    c4 = d["C4"]; L, Sv = d["L"], d["S"]
    Fc = {h: np.stack([forecast_day_aligned(d, D, h) for D in range(DAYS)]) for _, h in CPS}
    c_hi = np.quantile(c4, 0.95, axis=0)
    c_lo = np.quantile(c4, 0.05, axis=0)
    c_nom = c4.mean(0)                     # = 附件1 电价

    print("=" * 88)
    print("问题4 鲁棒电价（逐时段上界电价 = box 鲁棒对等）")
    print("=" * 88)
    print("  α(分位)  计划电价均价   实际电价结算(万元)  95分位电价结算(万元)  紧急购电(kWh)")
    rows = []
    for a_ in (50, 75, 90, 95):
        cp = np.quantile(c4, a_ / 100, axis=0)
        f_act = year_cost(d, Fc, Sv, cp, c_nom, L)
        f_hi = year_cost(d, Fc, Sv, cp, c_hi, L)
        rows.append((a_, cp.mean(), f_act, f_hi))
        print(f"    {a_:2d}%    {cp.mean():.4f}      {f_act/1e4:14,.1f}    {f_hi/1e4:16,.1f}     用时{time.time()-t0:.0f}s")

    f50, f75, f90, f95 = [r[2] for r in rows]
    h50, h75, h90, h95 = [r[3] for r in rows]
    print("\n  【鲁棒的代价（实际电价下多付） / 收益（最坏电价下少付）】")
    for (a_, cm, fa, fh) in rows:
        print(f"    α={a_}%：实际电价结算 {fa/1e4:9,.1f} 万，"
              f"最坏电价结算 {fh/1e4:9,.1f} 万")

    # 交叉评估矩阵：不同计划 x 不同结算
    print("\n  【交叉评估】行=制定计划所用电价分位，列=结算情景")
    print("            实际电价(万元)   95分位电价(万元)")
    grid = {}
    for a_ in (50, 95):
        cp = np.quantile(c4, a_ / 100, axis=0)
        grid[a_] = (year_cost(d, Fc, Sv, cp, c_nom, L), year_cost(d, Fc, Sv, cp, c_hi, L))
        print(f"    α={a_:2d}%      {grid[a_][0]/1e4:10,.1f}      {grid[a_][1]/1e4:12,.1f}")
    print(f"\n  → α=95% 相对 α=50%（名义）: 实际电价下 {(grid[95][0]-grid[50][0])/grid[50][0]*100:+.2f}%，"
          f"最坏电价下 {(grid[95][1]-grid[50][1])/grid[50][1]*100:+.2f}%")
    print(f"  → 最坏/实际 的比值：名义计划 {grid[50][1]/grid[50][0]:.4f}，"
          f"鲁棒计划 {grid[95][1]/grid[95][0]:.4f}")

    np.savez(RES / "p4_robust.npz",
             alphas=np.array([r[0] for r in rows]),
             plan_price_mean=np.array([r[1] for r in rows]),
             cost_actual=np.array([r[2] for r in rows]),
             cost_high=np.array([r[3] for r in rows]),
             grid50=np.array(grid[50]), grid95=np.array(grid[95]))
    print(f"\n[OK] 已写出 {RES/'p4_robust.npz'}　总耗时 {time.time()-t0:.0f}s")
