"""
论文插图统一视觉规范
- 语义化配色（同一含义全篇同色）
- 去顶右边框、浅网格、无框图例、直接标注
- 统一字号、线宽、留白，300 dpi 输出
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---------------- 语义化配色（全篇统一） ----------------
C = dict(
    price="#C0392B",     # 电价 / 成本
    pv="#E8A33D",        # 光伏
    load="#2C3E50",      # 负载
    buy="#2E6FAF",       # 购电量
    ess="#16A085",       # 储能
    ess_fill="#9FD9CC",
    base="#9AA0A6",      # 基线 / 参考
    acc="#7D5BA6",       # 强调（调整量、对偶）
    curtail="#D98C5F",   # 弃光
    emg="#B03A2E",       # 紧急购电
    info="#0F6E56",      # 信息价值
    neutral="#B4B2A9",
    ink="#2C2C2A",       # 主文字
    sub="#6B7280",       # 次文字
    grid="#E6E8EB",
)

FS = dict(title=12, label=11, tick=10, legend=9.5, note=9)


def setup():
    plt.rcParams.update({
        "font.sans-serif": ["Arial Unicode MS", "Heiti TC", "PingFang SC", "Songti SC"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 120,
        "font.size": FS["tick"],
        "axes.titlesize": FS["title"],
        "axes.titleweight": "medium",
        "axes.titlecolor": C["ink"],
        "axes.titlepad": 9,
        "axes.labelsize": FS["label"],
        "axes.labelcolor": C["ink"],
        "axes.edgecolor": C["neutral"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": C["sub"],
        "ytick.color": C["sub"],
        "xtick.labelsize": FS["tick"],
        "ytick.labelsize": FS["tick"],
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": FS["legend"],
        "legend.labelcolor": C["ink"],
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "axes.axisbelow": True,
    })


def axgrid(ax, axis="y"):
    ax.grid(True, axis=axis, color=C["grid"], lw=0.7)
    ax.grid(False, axis="x" if axis == "y" else "y")


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {name}")
    return p


def note(ax, x, y, s, color=None, ha="left", va="bottom", size=None, style="italic"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size or FS["note"],
            color=color or C["sub"], style=style, zorder=6)


def waterfall(ax, labels, deltas, base, ylab="总购电费（万元）",
              unit="万元", fmt="{:.1f}", up_color=None, dn_color=None,
              base_color=None, total_color=None, ylim=None,
              base_label="无储能基线", total_label="最终方案", scale=1e4):
    """
    成本归因瀑布图（输入以"元"为单位，内部统一换算为万元作图）。
      labels : 各增量阶段的名称
      deltas : 各阶段增量（正 = 成本上升）
      base   : 起始绝对量（元）
    """
    up_color = up_color or C["price"]
    dn_color = dn_color or C["ess"]
    base_color = base_color or C["base"]
    total_color = total_color or C["load"]
    base = base / scale
    deltas = [d / scale for d in deltas]
    if ylim:
        ylim = (ylim[0] / scale, ylim[1] / scale)
    n = len(deltas)
    run = base
    ax.bar(0, base, color=base_color, width=.62, zorder=3)
    ax.annotate(fmt.format(base), xy=(0, base), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=FS["note"], color=C["ink"], zorder=5)
    for i, d in enumerate(deltas):
        x = i + 1
        bottom = run if d > 0 else run + d
        col = up_color if d > 0 else dn_color
        ax.bar(x, abs(d), bottom=bottom, color=col, width=.62, alpha=.92, zorder=3)
        ax.plot([x - .69, x - .31], [run, run], lw=.9, ls=(0, (3, 3)),
                color=C["neutral"], zorder=2)
        ax.annotate(("+" if d > 0 else "−") + fmt.format(abs(d)),
                    xy=(x, bottom + abs(d)), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS["note"], color=col, zorder=5)
        run += d
    ax.plot([n + .31, n + .69], [run, run], lw=.9, ls=(0, (3, 3)),
            color=C["neutral"], zorder=2)
    ax.bar(n + 1, run, color=total_color, width=.62, zorder=3)
    ax.annotate(fmt.format(run), xy=(n + 1, run), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=FS["note"], color=C["ink"], zorder=5)
    lo = ylim[0] if ylim else 0
    hi = (ylim[1] if ylim else run * 1.16)
    ax.set_ylim(lo, hi)
    off = lo - (hi - lo) * .075
    ax.text(0, off, base_label, ha="center", va="top", fontsize=FS["note"] - .5, color=C["sub"])
    for i, lb in enumerate(labels):
        ax.text(i + 1, off, lb, ha="center", va="top", fontsize=FS["note"] - .5, color=C["sub"])
    ax.text(n + 1, off, total_label, ha="center", va="top",
            fontsize=FS["note"] - .5, color=C["sub"])
    ax.set_xticks([])
    ax.set_ylabel(f"{ylab}")
    ax.set_xlim(-.7, n + 1.7)
    axgrid(ax, "y")
    return run * scale
