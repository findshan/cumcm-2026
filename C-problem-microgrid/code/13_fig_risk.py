"""风险分析图：年费用分布 / CDF+CVaR / 紧急购电分布 / 电价风险区间"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S
from style import C, FS, save, note, axgrid
S.setup()
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
r = np.load(RES / "risk.npz")
costs = r["costs"] / 1e4            # 万元
emgs = r["emgs"] / 1e4
base = float(r["base_fee"]) / 1e4
mu = costs.mean(); va = np.percentile(costs, 95)
cv = costs[costs >= va].mean()
c_hi, c_lo, c43 = float(r["cost_hi"]) / 1e4, float(r["cost_lo"]) / 1e4, float(r["c43"]) / 1e4

fig = plt.figure(figsize=(13.8, 9.2))
gs = fig.add_gridspec(2, 2, hspace=.36, wspace=.24)

a = fig.add_subplot(gs[0, 0])
a.hist(costs, bins=22, color=C["buy"], alpha=.85, edgecolor="white", lw=.7)
a.axvline(mu, color=C["ink"], lw=1.8, label=f"场景期望 {mu:,.1f} 万")
a.axvline(base, color=C["ess"], lw=1.8, ls="--", label=f"实测基准 {base:,.1f} 万")
a.axvline(va, color=C["price"], lw=1.6, ls=(0, (5, 3)), label=f"VaR$_{{95}}$ {va:,.1f} 万")
a.set_title("(a) 年费用分布（50 组全年场景）")
a.set_xlabel("交付期总购电费 (万元)"); a.set_ylabel("场景数")
a.legend(loc="upper left", handlelength=1.6); axgrid(a)

a = fig.add_subplot(gs[0, 1])
xs = np.sort(costs); ys = np.arange(1, len(xs) + 1) / len(xs)
a.plot(xs, ys, color=C["buy"], lw=2.0)
a.fill_betweenx([0, 1], va, xs.max(), color=C["price"], alpha=.10, lw=0)
a.set_xlim(xs.min() - (xs.max()-xs.min())*.06, xs.max() + (xs.max()-xs.min())*.06)
a.axvline(va, color=C["price"], ls=(0, (5, 3)), lw=1.6)
a.axhline(.95, color=C["neutral"], lw=.9, ls=":")
a.annotate(f"CVaR$_{{95}}$ = {cv:,.1f} 万\n（超期望 {cv-mu:,.1f} 万，占 {((cv-mu)/mu*100):.2f}%）",
           (va, .95), xytext=(xs.min() + (xs.max() - xs.min()) * .06, .80),
           fontsize=FS["note"], color=C["price"],
           arrowprops=dict(arrowstyle="-", color=C["price"], lw=.8))
a.set_title("(b) 累计分布与尾部风险")
a.set_xlabel("交付期总购电费 (万元)"); a.set_ylabel("累计概率")
a.set_ylim(0, 1.02); axgrid(a)

a = fig.add_subplot(gs[1, 0])
a.hist(emgs, bins=22, color=C["emg"], alpha=.85, edgecolor="white", lw=.7)
a.axvline(emgs.mean(), color=C["ink"], lw=1.8, label=f"期望 {emgs.mean():,.2f} 万 kWh")
a.axvline(np.percentile(emgs, 95), color=C["price"], lw=1.6, ls=(0, (5, 3)),
          label=f"95 分位 {np.percentile(emgs,95):,.2f} 万 kWh")
a.set_title("(c) 紧急购电量的分布（5 倍电价的风险敞口）")
a.set_xlabel("交付期紧急购电量 (万 kWh)"); a.set_ylabel("场景数")
a.legend(loc="upper left", handlelength=1.6); axgrid(a)

a = fig.add_subplot(gs[1, 1])
labels = ["低电价情景\n（5 分位）", "实际电价\n（4-3）", "高电价情景\n（95 分位）"]
vals = [c_lo, c43, c_hi]
cols = [C["ess"], C["buy"], C["price"]]
xb = np.arange(3)
a.bar(xb, vals, width=.56, color=cols, alpha=.92, zorder=3)
for x_, v_ in zip(xb, vals):
    a.text(x_, v_ + max(vals) * .022, f"{v_:,.1f}", ha="center", fontsize=FS["note"], color=C["ink"])
a.annotate("", xy=(2, c_hi), xytext=(0, c_lo),
           arrowprops=dict(arrowstyle="<|-|>", color=C["acc"], lw=1.6))
a.text(1, (c_hi + c_lo) / 2, f"电价风险区间\n{vals[2]-vals[0]:,.1f} 万\n（占实际 {((vals[2]-vals[0])/c43*100):.0f}%）",
       ha="center", va="center", fontsize=FS["note"], color=C["acc"],
       bbox=dict(fc="white", ec="none", alpha=.85, pad=2))
a.set_xticks(xb); a.set_xticklabels(labels, fontsize=8.5)
a.set_ylim(0, max(vals) * 1.18)
a.set_title("(d) 问题4 电价风险区间（逐时段经验分位）")
a.set_ylabel("交付期总购电费 (万元)"); axgrid(a, "y")
fig.suptitle("图 8　风险维度：场景法蒙特卡洛与尾部风险度量", fontsize=13.5, y=.98)
save(fig, "fig_risk.png")
print(f"  期望 {mu:,.1f} 万 | 基准 {base:,.1f} 万 | VaR95 {va:,.1f} 万 | CVaR95 {cv:,.1f} 万")
