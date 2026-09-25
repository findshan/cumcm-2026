"""
图 8（重构）：风险维度
(a) 年费用分布 + 尾部风险标注
(b) 累计分布与 CVaR₉₅ 尾部
(c) 确定性 vs 鲁棒策略：同一场景集上的费用分布对比（"用期望换最坏保证"）
(d) 电价水平 -> 费用 的响应曲线（替换原三柱对比）
"""
import numpy as np
import pandas as pd
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import (load_attachments, solve_lp, forecast_day_aligned,
                        DT, N_T, DAYS, E_INIT, ETA)
import style as S
from style import C, FS, save, note, axgrid
S.setup()
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"; FIG = ROOT / "figures"
I0 = 31
CPS = ((0, 0), (36, 6), (72, 12), (108, 18))

r = np.load(RES / "risk.npz")
rb = np.load(RES / "robust.npz")
costs = r["costs"] / 1e4
base = float(r["base_fee"]) / 1e4
mu, va = costs.mean(), np.percentile(costs, 95)
cv = costs[costs >= va].mean()

gammas = rb["gammas"]; rb_means = rb["means"] / 1e4
rb_cvar = rb["cvar"] / 1e4; rb_v95 = rb["v95"] / 1e4
costs0 = rb["costs0"] / 1e4
print(f"确定性: 期望 {mu:,.1f} 万  CVaR95 {cv:,.1f} 万")
for g, m, c_ in zip(gammas, rb_means, rb_cvar):
    print(f"  Γ={g:.1f}  期望 {m:,.1f} 万  CVaR95 {c_:,.1f} 万")

# ---------------- (d) 电价水平 -> 费用 响应曲线 ----------------
d = load_attachments()
c4 = d["C4"]; L, Sv = d["L"], d["S"]
lv = [5, 25, 50, 75, 95]
cache = RES / "price_curve.npz"
if cache.exists():
    z = np.load(cache); levels, curve = z["levels"], z["curve"]
else:
    base_fc = {h: np.stack([forecast_day_aligned(d, D, h) for D in range(DAYS)]) for _, h in CPS}

    def cost_at(prices):
        tot = 0.0
        for D in range(DAYS):
            Ec = np.zeros(N_T + 1); Ec[0] = E_INIT
            pl = np.zeros(N_T); ad = np.zeros(N_T); uu = np.zeros(N_T); vv = np.zeros(N_T)
            for ci, (s, h0) in enumerate(CPS):
                e = CPS[ci + 1][0] if ci + 1 < len(CPS) else N_T
                rr = solve_lp(prices[s:], L[D, s:], base_fc[h0][D, s:],
                              E_start=Ec[s], E_end=E_INIT)
                if ci == 0:
                    pl[s:] = rr["g"]
                ad[s:e] = rr["g"][:e - s]; uu[s:e] = rr["u"][:e - s]; vv[s:e] = rr["v"][:e - s]
                for k in range(s, e):
                    Ec[k + 1] = Ec[k] + ETA * DT * uu[k] - DT / ETA * vv[k]
            su = Sv[D] + ad + vv - L[D] - uu
            em = np.maximum(-su, 0.0)
            if D >= I0:
                tot += (pl * DT * prices).sum()
                tot += ((0.5 * np.maximum(pl - ad, 0) + 1.5 * np.maximum(ad - pl, 0)) * DT * prices).sum()
                tot += (em * DT * prices).sum() * 5
        return tot

    t0 = time.time()
    levels, curve = [], []
    for q in lv:
        pq = np.quantile(c4, q / 100, axis=0)
        levels.append(pq.mean()); curve.append(cost_at(pq))
        print(f"  电价分位 {q:2d}%  均价 {pq.mean():.4f}  费用 {curve[-1]/1e4:,.1f} 万  用时 {time.time()-t0:.0f}s")
    levels = np.array(levels); curve = np.array(curve)
    np.savez(cache, levels=levels, curve=curve)

# ---------------- 绘图 ----------------
fig = plt.figure(figsize=(13.8, 9.2))
gs = fig.add_gridspec(2, 2, hspace=.36, wspace=.24)

a = fig.add_subplot(gs[0, 0])
a.hist(costs, bins=22, color=C["buy"], alpha=.85, edgecolor="white", lw=.7)
a.axvline(mu, color=C["ink"], lw=1.8, label=f"场景期望 {mu:,.1f} 万")
a.axvline(base, color=C["ess"], lw=1.8, ls="--", label=f"实测基准 {base:,.1f} 万")
a.axvline(va, color=C["price"], lw=1.6, ls=(0, (5, 3)), label=f"VaR$_{{95}}$ {va:,.1f} 万")
a.set_title("(a) 年费用分布（50 组全年场景，每组一次完整 MPC）")
a.set_xlabel("交付期总购电费 (万元)"); a.set_ylabel("场景数")
a.legend(loc="upper left", handlelength=1.6); axgrid(a)

a = fig.add_subplot(gs[0, 1])
xs = np.sort(costs); ys = np.arange(1, len(xs) + 1) / len(xs)
a.plot(xs, ys, color=C["buy"], lw=2.2)
a.fill_betweenx([0, 1], va, xs.max(), color=C["price"], alpha=.10, lw=0)
a.axvline(va, color=C["price"], ls=(0, (5, 3)), lw=1.6)
a.axhline(.95, color=C["neutral"], lw=.9, ls=":")
a.annotate(f"CVaR$_{{95}}$ = {cv:,.1f} 万\n（超期望仅 {cv-mu:,.1f} 万，占 {(cv-mu)/mu*100:.2f}%）",
           (va, .95), xytext=(xs.min() + (xs.max() - xs.min()) * .04, .74),
           fontsize=FS["note"], color=C["price"],
           arrowprops=dict(arrowstyle="-", color=C["price"], lw=.8))
a.set_xlim(xs.min() - (xs.max() - xs.min()) * .06, xs.max() + (xs.max() - xs.min()) * .06)
a.set_title("(b) 累计分布与尾部风险：储能+滚动调整自带风险平滑")
a.set_xlabel("交付期总购电费 (万元)"); a.set_ylabel("累计概率")
a.set_ylim(0, 1.02)
a.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
axgrid(a)

a = fig.add_subplot(gs[1, 0])
xsg = np.asarray(gammas, float)
best = int(np.argmin(rb_means))
a.axvspan(xsg[0] - .1, xsg[best], color=C["ess"], alpha=.07, lw=0)
a.plot(xsg, rb_means, "-o", color=C["buy"], lw=2.3, ms=6.5, label="期望费用")
a.plot(xsg, rb_cvar, "--s", color=C["price"], lw=2.0, ms=5.5,
       label="CVaR$_{95}$（最坏 5% 的均值）")
a.scatter([xsg[best]], [rb_means[best]], s=190, facecolor="white",
          edgecolor=C["ess"], lw=2.0, zorder=6)
a.axvline(xsg[best], color=C["ess"], ls=(0, (5, 3)), lw=1.3, zorder=0)
a.annotate("$\\Gamma$ 增大 = 对光伏预报保守化\n"
           "$\\Gamma$≤0.5：为「实际低于预报」的\n"
           "　　　　短缺尾部预留安全边际\n"
           "　　　　期望与尾部「同时」下降\n"
           "$\\Gamma$>0.5：转入经典权衡，期望 "
           f"+{(rb_means[-1]-rb_means[best])/rb_means[best]*100:.1f}%\n"
           f"　　　　仅换取尾部 {(rb_cvar[-1]-rb_cvar[best])/1e4:.1f} 万元的收窄",
           xy=(xsg[best], rb_means[best]), xytext=(xsg[0] - .04, 1652),
           fontsize=FS["note"] - .3, color=C["ink"], ha="left", va="top",
           arrowprops=dict(arrowstyle="-", color=C["ess"], lw=.9))
a.annotate(f"最优 $\\Gamma$≈{xsg[best]:.1f}", (xsg[best], rb_means[best]),
           xytext=(xsg[best] + .05, rb_means[best] - 34),
           fontsize=FS["note"], color=C["ess"],
           arrowprops=dict(arrowstyle="->", color=C["ess"], lw=.9))
a.set_title("(c) 鲁棒系数 $\\Gamma$ 的效果：非单调，存在最优保守水平")
a.set_xlabel("鲁棒系数 $\\Gamma$（不确定集放大倍数）"); a.set_ylabel("交付期总购电费 (万元)")
a.set_ylim(1420, 1660); a.set_xlim(xsg[0] - .1, xsg[-1] + .1)
a.legend(loc="upper right", handlelength=1.8, borderaxespad=.6); axgrid(a)

a = fig.add_subplot(gs[1, 1])
a.plot(levels, curve / 1e4, "-o", color=C["price"], lw=2.2, ms=6)
a.fill_between(levels, curve / 1e4, color=C["price"], alpha=.12, lw=0)
for lv_, c_, lab in zip(levels, curve / 1e4, ["5%", "25%", "50%", "75%", "95%"]):
    a.annotate(f"{lab}\n{c_:,.0f}", (lv_, c_), xytext=(0, 8), textcoords="offset points",
               ha="center", fontsize=FS["note"] - .5, color=C["ink"])
a.set_title(f"(d) 电价水平 → 总费用：区间宽度 {np.ptp(curve)/1e4:,.0f} 万元")
a.set_xlabel("电价情景的分位水平对应均价 (元/kWh)"); a.set_ylabel("交付期总购电费 (万元)")
a.set_ylim(900, 1880); a.margins(x=.09); axgrid(a)
fig.suptitle("图 8　风险维度：场景法蒙特卡洛、尾部风险与鲁棒权衡", fontsize=13.5, y=.98)
save(fig, "fig_risk.png")
