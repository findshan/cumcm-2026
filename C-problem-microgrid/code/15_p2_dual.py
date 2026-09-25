"""
问题2 的对偶分析（年度尺度）
不重复问题1 的日内分析，聚焦年度尺度上的新信息：
  (a) 储能边际价值 μ_t 的年度演化（月度分位数带）
  (b) 月度峰谷比 vs 月度储能价值 -> 机制解释
  (c) 弃光与储能价值的关系（弃光由储能饱和触发）
  (d) 电能影子价格 σ_t = c_t 的时段占比（"谁在边际上供电"）
"""
import numpy as np
import pandas as pd
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import load_attachments, DT, N_T, DAYS, E_INIT, ETA
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import style as S
from style import C, FS, save, note, axgrid
S.setup()
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
I0 = 31
SL = slice(31, 365)


def solve_year_dual(c, L, S, E_start, eta=ETA, dt=DT,
                    E_min=1200., E_max=10800., P_max=5000., E_end_min=None):
    """整年 LP（与 cumcm_core.solve_lp 同构），额外返回等式约束的 Lagrange 乘子"""
    n = len(c); N = 5 * n
    nG, nU, nV, nW, nE = 0, n, 2 * n, 3 * n, 4 * n
    cobj = np.zeros(N)
    cobj[nG:nG + n] = c * dt
    cobj[nU:nU + n] = 1e-7
    cobj[nV:nV + n] = 1e-7

    rows, cols, vals, rhs = [], [], [], []
    r = np.repeat(np.arange(n), 4)
    cc = np.concatenate([nG + np.arange(n), nU + np.arange(n),
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

    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(2 * n, N)).tocsr()
    b = np.concatenate(rhs)
    bounds = ([(0, None)] * n + [(0, P_max)] * n + [(0, P_max)] * n
              + [(0, None)] * n + [(E_min, E_max)] * n)
    Au = bu = None
    if E_end_min is not None:
        Au = coo_matrix((np.array([-1.0]), (np.array([0]), np.array([nE + n - 1]))),
                        shape=(1, N)).tocsr()
        bu = np.array([-float(E_end_min)])
    res = linprog(cobj, A_ub=Au, b_ub=bu, A_eq=A, b_eq=b, bounds=bounds,
                  method="highs") if Au is not None else \
          linprog(cobj, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    mar = np.asarray(res.eqlin.marginals, float)
    return dict(g=x[nG:nG + n], u=x[nU:nU + n], v=x[nV:nV + n],
                w=x[nW:nW + n], E=x[nE:nE + n],
                lam_bal=mar[:n] / dt, lam_store=-mar[n:2 * n], obj=res.fun)


if __name__ == "__main__":
    t0 = time.time()
    d = load_attachments()
    c = np.tile(d["price_day"], DAYS); L = d["L"].ravel(); Sv = d["S"].ravel()
    dates = pd.DatetimeIndex(np.repeat(d["dates"], 1))     # 日
    r = solve_year_dual(c, L, Sv, E_INIT, E_end_min=E_INIT)
    print(f"[整年 LP 对偶] 变量 {5*DAYS*N_T:,}　求解 {time.time()-t0:.1f}s")
    mu = r["lam_store"].reshape(DAYS, N_T)
    sig = r["lam_bal"].reshape(DAYS, N_T)
    w = r["w"].reshape(DAYS, N_T); u = r["u"].reshape(DAYS, N_T); v = r["v"].reshape(DAYS, N_T)
    c2 = np.tile(d["price_day"], (DAYS, 1))
    S2 = d["S"]

    print(f"  储能边际价值 μ: [{mu.min():.4f}, {mu.max():.4f}] 元/kWh  均值 {mu.mean():.4f}")
    print(f"  电能影子价格 σ: [{sig.min():.4f}, {sig.max():.4f}]  均值 {sig.mean():.4f}")
    buy = r["g"].reshape(DAYS, N_T) > 1e-6
    print(f"  购电时段 |σ−c| 最大偏差 = {np.abs(sig[buy]-c2[buy]).max():.2e}（互补松弛验证）")
    print(f"  电网为边际电源的时段占比 = {buy.mean()*100:.1f}%")
    print(f"  充电 / 放电 / 闲置 时段占比: "
          f"{(u>1e-6).mean()*100:.1f}% / {(v>1e-6).mean()*100:.1f}% / "
          f"{((u<=1e-6)&(v<=1e-6)).mean()*100:.1f}%")

    # ---------------- 图 ----------------
    months = np.array([x.month for x in d["dates"]])
    mnames = [f"{m}月" for m in range(1, 13)]
    fig = plt.figure(figsize=(13.8, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=.36, wspace=.24)

    # (a) μ 的月度分布
    a = fig.add_subplot(gs[0, 0])
    data = [mu[months == m].ravel() for m in range(1, 13)]
    bp = a.boxplot(data, positions=range(1, 13), widths=.6, patch_artist=True,
                   showfliers=False, medianprops=dict(color=C["ink"], lw=1.4),
                   whiskerprops=dict(color=C["sub"], lw=.9),
                   capprops=dict(color=C["sub"], lw=.9),
                   boxprops=dict(facecolor=C["ess"], edgecolor=C["ess"], alpha=.55, lw=.8))
    a.axhline(c.mean(), color=C["price"], ls=(0, (5, 3)), lw=1.3)
    a.text(12.4, c.mean(), f" 年均电价 {c.mean():.3f}", fontsize=FS["note"], color=C["price"], va="center")
    a.set_xticks(range(1, 13)); a.set_xticklabels(mnames, fontsize=8.5)
    a.set_title("(a) 储能边际价值 μ 的年度演化")
    a.set_ylabel("元/kWh"); axgrid(a)

    # (b) 月度光伏渗透率 vs 月度储能价值（问题2 电价每日相同，峰谷比恒定，故改用渗透率）
    a = fig.add_subplot(gs[0, 1])
    pen = np.array([S2[months == m].sum() / L.reshape(DAYS, N_T)[months == m].sum()
                    for m in range(1, 13)])
    mmean = np.array([mu[months == m].mean() for m in range(1, 13)])
    sc = a.scatter(pen, mmean, s=110, c=range(12), cmap="twilight_shifted", zorder=4,
                   edgecolor="white", lw=.8)
    for m in range(12):
        a.annotate(mnames[m], (pen[m], mmean[m]), xytext=(5, 4),
                   textcoords="offset points", fontsize=FS["note"], color=C["sub"])
    rr_ = np.corrcoef(pen, mmean)[0, 1]
    xx = np.linspace(pen.min() * .97, pen.max() * 1.03, 20)
    k, b_ = np.polyfit((pen - pen.mean()) / pen.std(), mmean, 1)
    a.plot(xx, k * (xx - pen.mean()) / pen.std() + b_, "--", color=C["base"], lw=1.4,
           label=f"线性拟合 $r$={rr_:.3f}")
    a.axhline(c.mean(), color=C["price"], ls=(0, (5, 3)), lw=1.3)
    a.text(pen.min(), c.mean(), f" 年均电价 {c.mean():.3f}",
           fontsize=FS["note"], color=C["price"], va="bottom")
    a.set_title(f"(b) 光伏渗透率越高，储能边际价值越低（$r$={rr_:.3f}）")
    a.set_xlabel("月度光伏渗透率（光伏发电量 / 负载用电量）")
    a.set_ylabel("月度平均 $\\mu$ (元/kWh)")
    a.legend(loc="upper right", handlelength=1.4); axgrid(a)

    # (c) 弃光的触发条件：当日光伏富余量 vs 弃光量，按是否饱和着色
    a = fig.add_subplot(gs[1, 0])
    E_all = r["E"].reshape(DAYS, N_T)
    Sd = S2.reshape(DAYS, N_T); Ld = L.reshape(DAYS, N_T)
    surplus = (np.maximum(Sd - Ld, 0) * DT).sum(1)
    cur_d = (w * DT).sum(1)
    sat = (E_all > 10799.9).any(1)
    a.scatter(surplus[~sat], cur_d[~sat], s=24, color=C["base"], alpha=.45, lw=0,
              label=f"储能当日未触上限（{int((~sat).sum())} 天，弃光恒为 0）")
    a.scatter(surplus[sat], cur_d[sat], s=30, color=C["curtail"], alpha=.75, lw=0,
              label=f"储能当日触到上限（{int(sat.sum())} 天）")
    a.plot([0, surplus.max()], [0, surplus.max()], "--", color=C["price"], lw=1.3,
           label="若富余全部可吸纳（$y=x$）")
    a.set_title(f"(c) 弃光的触发条件：储能饱和且光伏富余（合计 {cur_d[I0:].sum():,.0f} kWh）")
    a.set_xlabel("当日光伏富余电量 $\\sum\\max(S_t-L_t,0)\\Delta t$ (kWh)")
    a.set_ylabel("当日弃光电量 (kWh)")
    a.legend(loc="upper left", handlelength=1.4, fontsize=FS["legend"] - 1); axgrid(a)

    # (d) "谁在边际供电"：σ/c 的比值分布
    a = fig.add_subplot(gs[1, 1])
    rr = sig / np.maximum(c2, 1e-9)
    bins = np.linspace(0, 1.02, 52)
    a.hist(rr.ravel(), bins=bins, color=C["buy"], alpha=.85, edgecolor="white", lw=.4)
    a.axvline(1.0, color=C["price"], lw=1.6)
    a.axvline(0.0, color=C["ess"], lw=1.6)
    a.text(1.0, a.get_ylim()[1] * .95, f" σ=c（电网边际）\n {buy.mean()*100:.1f}%",
           fontsize=FS["note"], color=C["price"], va="top", ha="right")
    a.text(0.02, a.get_ylim()[1] * .95, f"σ≈0（光伏边际）\n {(rr<0.02).mean()*100:.1f}%",
           fontsize=FS["note"], color=C["ess"], va="top", ha="left")
    a.set_title("(d) 边际电源的构成：电能影子价格 / 电价 的分布")
    a.set_xlabel("$\\sigma_t\\,/\\,c_t$"); a.set_ylabel("时段数（10 分钟）"); axgrid(a)
    fig.suptitle("图 9　问题2 的年度尺度对偶分析", fontsize=13.5, y=.98)
    save(fig, "fig_p2_dual.png")

    np.savez(RES / "p2_dual.npz", mu=mu, sig=sig, w=w, buy=buy.reshape(DAYS, N_T),
             pen=pen, mmean=mmean)
    print(f"[OK] 已写出 {RES/'p2_dual.npz'}　总耗时 {time.time()-t0:.1f}s")
