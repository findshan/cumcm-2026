"""
图 7（论文门面图）：模型演进阶梯 —— 决策方法与信息质量各自的改进空间

为什么需要这张图
----------------
朴素 MPC → 罚金感知 MPC-II → 随机动态规划 SDP → 完美信息下界，
四者之间是"层层递进、逐层收敛"的关系。若只用文字叙述，评审扫一眼抓不住；
这张图一屏讲完三件事：
    ① 我们发现了子问题目标函数的缺陷（朴素 → MPC-II，−2.50%）
    ② 修正后距 SDP 最优基准只剩 0.16%（MPC-II → SDP）
    ③ 再往上只剩预报精度的空间（SDP → 完美信息，−13.27%）
结论：投资预报，而非投资算法。
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S
from style import C, FS, save, axgrid
S.setup()
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import FancyArrowPatch, Rectangle

RES = Path(__file__).resolve().parent.parent / "results"

p3 = np.load(RES / "p3_solution.npz")
zs = np.load(RES / "sdp_solution.npz")

V_DET = float(p3["det_fee"])        # 完美信息 + 日周期（理论下界）
V_SDP = float(zs["cost_d4"])        # 随机动态规划（策略最优）
V_M2 = float(p3["costA"])           # 罚金感知 MPC-II（交付）
V_M1 = float(p3["cost_naive"])      # 朴素滚动 MPC
V_M0 = float(p3["cost0"])           # 仅 0:00 预报
WAN = lambda v: v / 1e4

ROWS = [
    ("完美信息 + 日周期\n（不可达到的理论下界）", V_DET, C["base"], "white", "理论下界"),
    ("随机动态规划 SDP\n（同信息结构下的最优基准）", V_SDP, C["ess"], "white", "最优基准"),
    ("罚金感知 MPC-II\n（修正子问题目标 · 交付方案）", V_M2, C["buy"], "white", "交付方案"),
    ("朴素滚动 MPC\n（子问题目标＝购电成本）", V_M1, "#85B7EB", C["ink"], "—"),
    ("仅用 0:00 预报（1 次/天）", V_M0, C["curtail"], C["ink"], "—"),
]

XMIN, XMAX = 1150.0, 1690.0         # 万元，截断轴以放大差异
BRX = 1560.0                        # 右侧括号横坐标

fig = plt.figure(figsize=(11.6, 6.3))
gs = fig.add_gridspec(2, 1, height_ratios=[1, .17], hspace=.30,
                      left=.245, right=.985, top=.905, bottom=.055)
ax = fig.add_subplot(gs[0])

n = len(ROWS)
yy = np.arange(n)[::-1]                       # 最优在上
barh = .58

# 交付方案行的高亮底衬（止步于括号之前，避免压住右侧文字）
ax.add_patch(Rectangle((XMIN, yy[2] - barh / 2 - .10), BRX - 12 - XMIN, barh + .20,
                       fc="#FFF6E8", ec="none", zorder=0))

for i, (lab, v, col, tc, badge) in enumerate(ROWS):
    y = yy[i]
    ax.barh(y, WAN(v) - XMIN, left=XMIN, height=barh, color=col,
            zorder=3, edgecolor="white", lw=.8)
    ax.text(WAN(v) + 8, y, f"{WAN(v):,.1f}", va="center", ha="left",
            fontsize=FS["label"], color=C["ink"], weight="bold", zorder=5)
    ax.text(XMIN - 6, y, lab, va="center", ha="right",
            fontsize=FS["label"] - .5, color=C["ink"], zorder=5, linespacing=1.4)
    if badge != "—":
        ax.text(XMIN + 8, y, badge, va="center", ha="left", fontsize=8.5,
                color="white", weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=.26", fc=col, ec="none", alpha=.95))

ax.set_yticks([])
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(-.65, n - .35)
ax.set_xlabel("交付期总购电费（万元，轴自 1 150 万元起以放大差异）")
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
axgrid(ax, "x")
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)

# ---------------- 右侧两段括号：在 SDP 处对接 ----------------
def bracket(y1, y2, color, text):
    ax.annotate("", xy=(BRX, y1), xytext=(BRX, y2),
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.8,
                                shrinkA=0, shrinkB=0))
    for yv in (y1, y2):                       # 虚线从数值标签之后引出，避免穿字
        ax.plot([V_LAB_END[yv] + 4, BRX], [yv, yv],
                color=color, lw=.7, ls=(0, (3, 3)), alpha=.45, zorder=1)
    ax.text(BRX + 10, (y1 + y2) / 2, text, va="center", ha="left",
            fontsize=FS["note"] + .3, color=color, weight="bold", linespacing=1.5)

V_LAB_END = {yy[i]: WAN(ROWS[i][1]) + 56 for i in range(n)}   # 数值标签右端

g1 = (V_M1 - V_SDP) / V_M1 * 100
g2 = (V_SDP - V_DET) / V_DET * 100
bracket(yy[4], yy[2], C["buy"],
        f"决策方法的改进空间\n−{(V_M1-V_SDP)/1e4:,.1f} 万（−{g1:.2f}%）\n"
        f"├ 罚金感知 −2.50%\n└ 分布感知 −0.16%")
bracket(yy[2], yy[0], C["price"],
        f"预报精度的改进空间\n−{(V_SDP-V_DET)/1e4:,.1f} 万（−{g2:.2f}%）")

# ---------------- 底部结论通栏 ----------------
a2 = fig.add_subplot(gs[1]); a2.axis("off")
a2.add_patch(Rectangle((0, 0), 1, 1, fc="#FCEBEB", ec=C["price"], lw=1.1,
                       transform=a2.transAxes, zorder=0))
a2.text(.5, .52,
        f"朴素 MPC → 罚金感知 MPC-II：省 {(V_M1-V_M2)/1e4:,.1f} 万元（2.50%）；"
        f"→ SDP 最优基准：再省 {(V_M2-V_SDP)/1e4:,.1f} 万元（0.16%）；"
        f"再往上只能靠预报精度（13.27%）　——　投资预报，而非投资算法",
        ha="center", va="center", fontsize=FS["label"], color="#7A1F1F",
        weight="bold", transform=a2.transAxes, zorder=3)

fig.text(.035, .955, "模型演进阶梯：决策方法与信息质量各自的改进空间",
         ha="left", va="center", fontsize=13.5, color=C["ink"], weight="bold")
fig.text(.035, .915, "同一信息结构、同一费用口径（题面 take-or-pay 规则）下的逐层对照",
         ha="left", va="center", fontsize=9.5, color=C["sub"])

save(fig, "fig_model_ladder.png")
print(f"朴素 {V_M1/1e4:,.1f} | MPC-II {V_M2/1e4:,.1f} | SDP {V_SDP/1e4:,.1f} | 下界 {V_DET/1e4:,.1f}")
print(f"决策方法空间 −{g1:.2f}%　预报精度空间 −{g2:.2f}%")
