"""
论文用示意图（结构性 / 流程性图）——统一高级视觉规范
S1 微网系统结构与能量流
S2 统一建模框架：四问的信息集递进与模型类型
S3 问题3 滚动 MPC 决策-信息时序
S4 问题4 双轨建模框架（2×2 矩阵式度量）
"""
import numpy as np
import matplotlib.transforms as mtrans
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S
from style import C, FS, save as ssave

S.setup()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def save(fig, name):
    return ssave(fig, name)


# ---------------- 示意图专用语义色（与正文图同源） ----------------
SC = dict(
    pv_fc="#FDF3E3", pv_ec="#D98C1F", pv_tc="#8A5A10", pv_sc="#B4741A",
    gr_fc="#EAF1F9", gr_ec="#2E6FAF", gr_tc="#1B4A80", gr_sc="#2E6FAF",
    ess_fc="#E4F5F0", ess_ec="#16A085", ess_tc="#0A6D5C", ess_sc="#16A085",
    ld_fc="#EAEEF3", ld_ec="#2C3E50", ld_tc="#1F2E3D", ld_sc="#44576B",
    bus_fc="#F8FAFC", bus_ec="#2C3E50", bus_tc="#1B2A38", bus_sc="#5A6B7C",
    neu_fc="#F1F3F5", neu_ec="#9AA0A6", neu_tc="#2C2C2A", neu_sc="#6B7280",
    hi_fc="#FCEBEB", hi_ec="#C0392B", hi_tc="#7A1F1F", hi_sc="#C0392B",
    in_fc="#E7F0FA", in_ec="#2E6FAF", in_tc="#1B4A80", in_sc="#2E6FAF",
)


# ============================================================
# 基础绘制元件
# ============================================================
def _rrect(ax, x, y, w, h, fc, ec, lw=1.25, r=0.085, z=2, alpha=1.0, ls="-"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha, linestyle=ls,
                       mutation_aspect=1.0)
    ax.add_patch(p)
    return p


def card(ax, x, y, w, h, title, sub="", fc="#FFFFFF", ec="#9AA0A6",
         tc="#2C2C2A", sc="#6B7280", fs=11.5, fss=9.5, accent=None,
         icon=None, shadow=True, tw="bold", sw="normal", lw=1.25, r=0.085,
         pad_t=0.66, sub_dy=None, alpha=1.0):
    """带阴影 / 左侧强调条 / 可选图标的圆角卡片"""
    if shadow:
        _rrect(ax, x + .032, y - .042, w, h, "#0F172A", "none", lw=0, r=r, z=1, alpha=.055)
        _rrect(ax, x + .016, y - .021, w, h, "#0F172A", "none", lw=0, r=r, z=1, alpha=.035)
    _rrect(ax, x, y, w, h, fc, ec, lw=lw, r=r, z=2, alpha=alpha)
    cx = x + w / 2
    if accent:
        _rrect(ax, x + .055, y + .085, .075, h - .17, accent, "none", lw=0, r=.035, z=3)
    tx = cx
    if icon:
        icon(ax, x + .30, y + h / 2, min(h * .58, .52), ec, z=5)
        tx = x + .48 + (w - .48 - .10) / 2
    if sub:
        dy = sub_dy if sub_dy is not None else 0.0
        ax.text(tx, y + h * pad_t + dy, title, ha="center", va="center",
                fontsize=fs, color=tc, weight=tw, zorder=6)
        ax.text(tx, y + h * (1 - pad_t) + dy, sub, ha="center", va="center",
                fontsize=fss, color=sc, zorder=6, linespacing=1.45)
    else:
        ax.text(tx, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=tc, weight=tw, zorder=6, linespacing=1.45)


def tag(ax, x, y, s, color, fs=9.0, fc="#FFFFFF", ec="none", pad=.30, z=7, weight="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color,
            zorder=z, weight=weight, linespacing=1.4,
            bbox=dict(boxstyle=f"round,pad={pad}", fc=fc, ec=ec, lw=.7, alpha=.97))


def flow(ax, p1, p2, color, lw=2.4, rad=0.0, style="-|>", ls="-", ms=15, glow=True, alpha=1.0, z=4):
    if glow:
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms + 3,
                                     color=color, lw=lw + 4.5, alpha=.085, zorder=z - 1,
                                     connectionstyle=f"arc3,rad={rad}", linestyle=ls,
                                     shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=z, alpha=alpha,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def clean(ax, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")


def title(ax, x, y, main, sub=None, fs=13, ss=9.5):
    ax.add_patch(Rectangle((x, y - .035), .055, .30, fc=C["price"], ec="none", zorder=6))
    ax.text(x + .155, y + .13, main, ha="left", va="center", fontsize=fs,
            color=C["ink"], weight="bold", zorder=6)
    if sub:
        ax.text(x + .155, y - .125, sub, ha="left", va="center", fontsize=ss,
                color=C["sub"], zorder=6)


# ---------------- 图标 ----------------
def ic_pv(ax, cx, cy, s, c, z=5):
    ax.add_patch(Circle((cx - .30 * s, cy + .34 * s), .16 * s, fc=c, ec="none", alpha=.85, zorder=z))
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.add_line(Line2D([cx - .30 * s + .24 * s * np.cos(a), cx - .30 * s + .34 * s * np.cos(a)],
                           [cy + .34 * s + .24 * s * np.sin(a), cy + .34 * s + .34 * s * np.sin(a)],
                           color=c, lw=1.0, alpha=.8, zorder=z, solid_capstyle="round"))
    x0, y0, w, h = cx - .18 * s, cy - .40 * s, .82 * s, .62 * s
    ax.add_patch(Rectangle((x0, y0), w, h, fc="white", ec=c, lw=1.35, zorder=z))
    for f in (1 / 3, 2 / 3):
        ax.add_line(Line2D([x0 + f * w, x0 + f * w], [y0, y0 + h], color=c, lw=.85, zorder=z + 1))
    ax.add_line(Line2D([x0, x0 + w], [y0 + h / 2, y0 + h / 2], color=c, lw=.85, zorder=z + 1))


def ic_grid(ax, cx, cy, s, c, z=5):
    ax.add_patch(Polygon([[cx - .34 * s, cy - .44 * s], [cx + .34 * s, cy - .44 * s],
                          [cx + .26 * s, cy + .14 * s], [cx - .26 * s, cy + .14 * s]],
                         closed=True, fc="white", ec=c, lw=1.35, zorder=z))
    for yy, ww in ((cy - .26 * s, .30 * s), (cy - .06 * s, .23 * s), (cy + .12 * s, .15 * s)):
        ax.add_line(Line2D([cx - ww, cx + ww], [yy, yy], color=c, lw=1.1, zorder=z + 1))
    ax.add_line(Line2D([cx, cx], [cy + .14 * s, cy + .46 * s], color=c, lw=1.2, zorder=z + 1))
    ax.add_patch(Circle((cx, cy + .50 * s), .075 * s, fc=c, ec="none", zorder=z + 1))


def ic_ess(ax, cx, cy, s, c, z=5):
    w, h = .82 * s, .62 * s
    x0, y0 = cx - w / 2 + .04 * s, cy - h / 2
    ax.add_patch(Rectangle((x0 + .30 * s, y0 + h), .22 * s, .10 * s, fc=c, ec="none", zorder=z))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=.06",
                                fc="white", ec=c, lw=1.4, zorder=z))
    for k, f in enumerate((.30, .58, .86)):
        ax.add_patch(Rectangle((x0 + .10 * s + k * .24 * s, y0 + .11 * s),
                               .17 * s, h - .22 * s, fc=c, ec="none",
                               alpha=.35 + .30 * k, zorder=z + 1))


def ic_load(ax, cx, cy, s, c, z=5):
    base = cy - .44 * s
    for dx, hh, ww in ((-.42, .52, .30), (-.05, .88, .34), (.35, .66, .30)):
        x0 = cx + dx * s - ww * s / 2
        ax.add_patch(Rectangle((x0, base), ww * s, hh * s, fc="white", ec=c, lw=1.2, zorder=z))
        for r_ in range(int(hh * 4)):
            for c_ in range(2):
                ax.add_patch(Rectangle((x0 + .06 * s + c_ * .12 * s, base + .10 * s + r_ * .19 * s),
                                       .07 * s, .09 * s, fc=c, ec="none", alpha=.55, zorder=z + 1))


# ============================================================
# S1  微网系统结构与能量流
# ============================================================
fig, ax = plt.subplots(figsize=(10.4, 6.1)); clean(ax, (0, 10.4), (0, 6.1))
title(ax, .30, 5.78, "微网系统结构、功率流与能量流",
      "外生电源（不可控）· 唯一可控设备为电池储能 · 无售电通道")

# 分区底衬
for (yy, hh, fc_) in ((4.28, 1.20, "#FCFDFE"), (2.50, 1.30, "#FBFCFD"), (0.86, 1.20, "#FCFDFE")):
    _rrect(ax, .22, yy, 9.96, hh, fc_, "#EEF1F4", lw=.8, r=.10, z=0)

for ymid, lb in ((4.88, "外生电源"), (3.15, "功率汇流"), (1.46, "需求与调节")):
    ax.text(.40, ymid, lb, rotation=90, ha="center", va="center",
            fontsize=8.8, color=C["sub"], zorder=6)

card(ax, .72, 4.38, 2.92, 1.00, "光伏阵列", "出力 $S_t$（不可控）",
     SC["pv_fc"], SC["pv_ec"], SC["pv_tc"], SC["pv_sc"], fs=12, icon=ic_pv, accent=SC["pv_ec"])
card(ax, 6.86, 4.38, 2.92, 1.00, "外部电网", "购电 $g_t$ · 电价 $c_t$",
     SC["gr_fc"], SC["gr_ec"], SC["gr_tc"], SC["gr_sc"], fs=12, icon=ic_grid, accent=SC["gr_ec"])
card(ax, 3.35, 2.60, 3.70, 1.10, "微网母线", "功率平衡　弃光 $w_t\\geq 0$",
     SC["bus_fc"], SC["bus_ec"], SC["bus_tc"], SC["bus_sc"], fs=13, lw=1.9,
     accent=SC["bus_ec"], shadow=True)
card(ax, .72, .96, 2.92, 1.00, "电池储能", "$E_t\\in[1200,\\,10800]$ kWh",
     SC["ess_fc"], SC["ess_ec"], SC["ess_tc"], SC["ess_sc"], fs=12, icon=ic_ess, accent=SC["ess_ec"])
card(ax, 6.86, .96, 2.92, 1.00, "小区负荷", "需求 $L_t$（刚性）",
     SC["ld_fc"], SC["ld_ec"], SC["ld_tc"], SC["ld_sc"], fs=12, icon=ic_load, accent=SC["ld_ec"])

# 禁止售电提示
_rrect(ax, 6.86, 5.52, 2.92, .32, SC["hi_fc"], SC["hi_ec"], lw=1.1, r=.06, z=4)
ax.text(8.32, 5.68, "✕　无售电通道：富余电量只能充电或弃光",
        ha="center", va="center", fontsize=9.5, color=SC["hi_tc"], weight="bold", zorder=6)

# 功率流
flow(ax, (2.18, 4.38), (4.30, 3.70), SC["pv_ec"], lw=2.6, rad=-.10)
tag(ax, 2.72, 4.14, "发电 $S_t$", SC["pv_tc"], fs=9)
flow(ax, (8.32, 4.38), (6.10, 3.70), SC["gr_ec"], lw=2.6, rad=.10)
tag(ax, 7.58, 4.14, "购电 $g_t$", SC["gr_tc"], fs=9)
flow(ax, (4.30, 2.60), (2.18, 1.96), SC["ess_ec"], lw=2.6, rad=-.10, style="<|-|>")
tag(ax, 2.72, 2.36, "双向充放 $u_t/v_t$", SC["ess_tc"], fs=9)
flow(ax, (6.10, 2.60), (8.32, 1.96), SC["ld_ec"], lw=2.6, rad=.10)
tag(ax, 7.54, 2.36, "供电 $\\geq$ 负荷", SC["ld_tc"], fs=9)

# 图例 + 参数条
_rrect(ax, .22, .10, 9.96, .62, "#F7F9FB", "#E3E7EB", lw=.9, r=.08, z=1)
lx = .52
flow(ax, (lx, .41), (lx + .34, .41), C["sub"], lw=2.0, glow=False, ms=12)
ax.text(lx + .42, .41, "功率流方向", fontsize=9, color=C["sub"], va="center", zorder=6)
lx = 2.42
flow(ax, (lx, .41), (lx + .34, .41), SC["ess_ec"], lw=2.0, style="<|-|>", glow=False, ms=12)
ax.text(lx + .42, .41, "储能双向", fontsize=9, color=C["sub"], va="center", zorder=6)
lx = 4.05
ax.add_line(Line2D([lx, lx + .34], [.41, .41], color=SC["hi_ec"], lw=1.8,
                   ls=(0, (3, 2)), zorder=6, solid_capstyle="round"))
ax.text(lx + .17, .41, "✕", fontsize=10.5, color=SC["hi_ec"], va="center",
        ha="center", zorder=8, weight="bold")
ax.text(lx + .42, .41, "禁止售电", fontsize=9, color=C["sub"], va="center", zorder=6)
ax.text(9.90, .41, "$\\eta=0.9$　$P_{\\max}=5000$ kW　$E_{0{:}00}=6000$ kWh　$\\Delta t=10$ min",
        fontsize=8.8, color=C["ink"], va="center", ha="right", zorder=6)
fig.tight_layout(pad=.2); save(fig, "fig_s1_system.png")

# ============================================================
# S2  统一建模框架：信息集递进 + 模型类型
# ============================================================
fig, ax = plt.subplots(figsize=(13.6, 5.5)); clean(ax, (0, 13.6), (0, 5.9))
title(ax, .18, 5.56, "四问的统一建模框架：信息集逐层收缩 → 模型由确定性优化升级为在线决策",
      "同一套决策变量与约束，仅在“已知信息”上逐问收紧", fs=12.2, ss=9.2)

HDR = [("问题", .20, .95), ("光伏出力信息", 1.28, 2.42), ("外网电价信息", 3.83, 2.42),
       ("储能边界条件", 6.38, 2.36), ("模型类型 / 不确定性处理", 8.88, 3.74)]
for txt, hx, hw in HDR:
    ax.text(hx + hw / 2, 5.06, txt, ha="center", va="center", fontsize=9.5,
            color=C["sub"], weight="bold", zorder=6)

ROWY = [4.10, 3.05, 2.00, 0.95]
RH = 0.82
R = [
    ("问题 1", "单日\n典型日", "已知 · 单日预测", "已知 · 每日相同", "日周期  $E(0{:}00){=}E(24{:}00)$",
     "确定性线性规划（单日全时段）", "不确定性处理：无（典型日均值日）", True, True, SC["in_ec"], SC["in_fc"]),
    ("问题 2", "全年\n逐日", "已知 · 全年实际", "已知 · 每日相同", "跨日连续 + 年末下界",
     "确定性线性规划（整年全时段）", "不确定性处理：无（完美信息上界）", True, True, SC["in_ec"], SC["in_fc"]),
    ("问题 3", "全年\n滚动", "仅得逐时预报 $F_t$", "已知 · 每日相同", "日周期（防末端效应）",
     "滚动时域优化 MPC（反馈型序贯决策）", "不确定性处理：预报驱动 + 场景法 / 鲁棒对等", False, True,
     SC["pv_ec"], "#FBF3E6"),
    ("问题 4", "全年\n滚动", "仅得逐时预报 $F_t$", "实时波动 · 不可预知", "日周期",
     "双轨：完美信息 LP（上界）+ 在线决策 MPC", "不确定性处理：经验分位电价情景 + box 鲁棒对等",
     False, False, SC["hi_ec"], "#FBEDED"),
]
for i, (a, a2, b, c_, d_, e_, f_, knownS, knownC, ec_, fc_) in enumerate(R):
    y = ROWY[i]
    card(ax, .20, y, .95, RH, a, a2, SC["neu_fc"], SC["neu_ec"], SC["neu_tc"], SC["neu_sc"],
         fs=10.5, fss=8.2, accent=SC["neu_ec"], shadow=False, pad_t=.70)
    card(ax, 1.28, y, 2.42, RH, b, "", "#E7F5F1" if knownS else "#FBF3E6",
         SC["ess_ec"] if knownS else SC["pv_ec"], SC["ess_tc"] if knownS else SC["pv_tc"],
         fs=10, shadow=False)
    if knownC:
        card(ax, 3.83, y, 2.42, RH, c_, "", "#E7F5F1", SC["ess_ec"], SC["ess_tc"],
             fs=10, shadow=False)
    else:
        _rrect(ax, 3.83, y, 2.42, RH, "#FBF3E6", SC["pv_ec"], lw=1.25, r=.085, z=2)
        nn = 12
        for j in range(nn):
            w = 2.42 / nn
            col = ["#FDF3E3", "#F7CE8A", "#E8A33D", "#B4741A"][j % 4]
            ax.add_patch(Rectangle((3.83 + j * w, y), w, RH, fc=col, ec=SC["pv_ec"],
                                   lw=.5, zorder=3))
        ax.add_patch(FancyBboxPatch((3.83, y), 2.42, RH, boxstyle="round,pad=0,rounding_size=.085",
                                    fc="none", ec=SC["pv_ec"], lw=1.25, zorder=4))
        ax.text(5.04, y + RH / 2, c_, ha="center", va="center", fontsize=10,
                color="#412402", weight="bold", zorder=6)
    card(ax, 6.38, y, 2.36, RH, d_, "", SC["neu_fc"], SC["neu_ec"], SC["neu_tc"],
         fs=9.8, shadow=False)
    card(ax, 8.88, y, 3.74, RH, e_, f_, fc_, ec_, ec_, ec_, fs=10.5, fss=8.4,
         shadow=True, pad_t=.68)

# 递进箭头
_rrect(ax, .20, .22, 12.44, .52, "#FCEBEB", SC["hi_ec"], lw=1.1, r=.08, z=2)
ax.text(6.42, .48, "信息集逐层收缩  $\\mathcal{I}_1\\supset\\mathcal{I}_2\\supset\\mathcal{I}_3\\supset\\mathcal{I}_4$"
                   "　→　模型由「确定性优化」升级为「在线决策」，不确定性处理逐层加码",
        ha="center", va="center", fontsize=10, color=SC["hi_tc"], zorder=6, weight="bold")

# 右侧单条长箭头示意"信息量递减"，避免压住任何卡片
ax.add_patch(FancyArrowPatch((12.86, 4.94), (12.86, 0.90), arrowstyle="-|>",
                             mutation_scale=17, color=SC["hi_ec"], lw=2.0, zorder=5, alpha=.9))
for yv in (4.51, 3.46, 2.41):
    ax.plot([12.74, 12.98], [yv, yv], color=SC["hi_ec"], lw=.9, alpha=.55, zorder=5)
ax.text(13.22, 2.92, "信息量递减", rotation=90, ha="center", va="center",
        fontsize=9, color=SC["hi_tc"], zorder=6, weight="bold")
fig.tight_layout(pad=.2); save(fig, "fig_s2_framework.png")

# ============================================================
# S3  问题3 滚动 MPC 决策-信息时序
# ============================================================
fig, ax = plt.subplots(figsize=(13.3, 5.6)); clean(ax, (0, 13.3), (0, 6.1))
title(ax, .18, 5.76, "问题3 的滚动 MPC：决策—信息时序与费用结算",
      "一天 4 次预报更新，每次只重优化“尚未执行”的时段；实际光伏到达后按 5 倍电价紧急购电",
      fs=12.2, ss=9.2)
X0, X1 = 1.60, 12.30


def tx(h):
    return X0 + (X1 - X0) * h / 24.0


ax.add_patch(FancyBboxPatch((X0, 4.72), X1 - X0, .46, boxstyle="round,pad=0,rounding_size=.06",
                            fc="#F3F5F7", ec="#DDE2E7", lw=.8, zorder=2))
for h in range(0, 25, 2):
    ax.plot([tx(h), tx(h)], [4.68, 5.22], color="#C3CAD1", lw=.8, zorder=3)
    ax.text(tx(h), 5.36, f"{h}", ha="center", fontsize=8.4, color=C["sub"], zorder=6)
ax.text(X0 - .12, 5.36, "时刻 (h)", ha="right", fontsize=9, color=C["sub"], zorder=6)
ax.text(X0 - .12, 4.95, "发布时刻", ha="right", va="center", fontsize=9.5, color=C["ink"], zorder=6)
for h in (0, 6, 12, 18):
    ax.add_patch(Circle((tx(h), 5.18), .055, fc=SC["hi_ec"], ec="white", lw=.9, zorder=7))

ax.text(12.52, 5.36, "提前期", ha="left", fontsize=9, color=C["sub"], zorder=6)
rows = [("0:00 预报到达 → 制定全天计划购电量 $g_t$", 0, 24, 4.05, "#1B4A80", "24–1 h"),
        ("6:00 预报更新 → 重优化其后时段", 6, 24, 3.42, "#2E6FAF", "18–1 h"),
        ("12:00 预报更新 → 重优化其后时段", 12, 24, 2.79, "#4E8FD0", "12–1 h"),
        ("18:00 预报更新 → 重优化其后时段", 18, 24, 2.16, "#7FB2E5", "6–1 h")]
for lab, a, b, y, col, ld in rows:
    _rrect(ax, tx(a), y, tx(b) - tx(a), .44, col, col, lw=1.0, r=.055, z=2, alpha=.20)
    _rrect(ax, tx(a), y + .03, .075, .38, col, "none", lw=0, r=.03, z=3)
    ax.text(tx(a) + .24, y + .22, lab, fontsize=10, color="#1F2E3D", va="center", zorder=6)
    ax.text(12.52, y + .22, ld, fontsize=9, color="#5A6B7C", va="center", ha="left", zorder=6)

_rrect(ax, X0, 1.16, X1 - X0, .60, "#E4F5F0", SC["ess_ec"], lw=1.2, r=.07, z=2)
ax.text(X0 + .22, 1.46, "实际光伏（附件2）到达 → 逐时段结算：不足则 5 倍电价紧急购电，有余则弃光",
        fontsize=10.2, color=SC["ess_tc"], va="center", zorder=6, weight="bold")
ax.text(X0 + .22, .84, "费用（口径A）：计划购电费 $\\sum c_tg_t\\Delta t$"
                       " ＋ 计划偏离罚金 $\\sum[0.5c_t(g_t-a_t)^+ + 1.5c_t(a_t-g_t)^+]\\Delta t$"
                       " ＋ 紧急购电费 $\\sum 5c_te_t\\Delta t$",
        fontsize=9.6, color=C["ink"], va="center", zorder=6)
for h, y in ((0, 4.05), (6, 3.42), (12, 2.79), (18, 2.16)):
    ax.plot([tx(h), tx(h)], [.45, 5.9], color="#C3CAD1", lw=.8, ls=(0, (2, 3)), zorder=0)
fig.tight_layout(pad=.2); save(fig, "fig_s3_timeline.png")

# ============================================================
# S4  问题4 双轨建模框架（2×2 矩阵）
# ============================================================
fig, ax = plt.subplots(figsize=(11.4, 6.2)); clean(ax, (0, 11.4), (0, 6.2))
title(ax, .25, 5.86, "问题4 双轨建模框架：电价不确定性的可分离度量",
      "电价不可预知 ≠ 电价波动。二者分离度量后，才能判断“预测电价”与“预测光伏”谁更值得投入",
      fs=12.4, ss=9.3)

# 基准芯片
_rrect(ax, .25, 5.02, 4.30, .40, SC["neu_fc"], SC["neu_ec"], lw=1.0, r=.07, z=3)
ax.text(2.40, 5.22, "基准参照 · 问题2（固定电价 + 完美信息）  1 222.9 万元",
        ha="center", va="center", fontsize=9.2, color=SC["neu_tc"], zorder=6)
ax.text(4.72, 5.22, "→ 电价水平波动  +55.3 万元（+4.5%）", ha="left", va="center",
        fontsize=9.2, color=SC["hi_tc"], zorder=6)

# 列头
COLS = [(2.62, 3.72, "轨一 · 完美信息", "理论上界（实际电价已知）", SC["in_fc"], SC["in_ec"], SC["in_tc"]),
        (6.58, 3.72, "轨二 · 电价不可预知", "可实现策略（历史分时均价预测）",
         SC["pv_fc"], SC["pv_ec"], SC["pv_tc"])]
for x, w, t1, t2, fc_, ec_, tc_ in COLS:
    card(ax, x, 4.16, w, .62, t1, t2, fc_, ec_, tc_, ec_, fs=11, fss=8.4,
         shadow=False, pad_t=.70)

# 行头
ROWS = [(3.06, "4-2", "整年连续 LP", "全时段一次求全局最优"),
        (1.86, "4-3", "滚动 MPC", "4 次预报滚动、反馈型决策")]
for y, t1, t2, t3 in ROWS:
    card(ax, .25, y, 2.20, .94, t1, t2, SC["neu_fc"], SC["neu_ec"], SC["neu_tc"], SC["neu_sc"],
         fs=12.5, fss=9.4, accent=SC["neu_ec"], shadow=False, pad_t=.68)
    ax.text(1.35, y + .17, t3, ha="center", va="center", fontsize=8, color=C["sub"], zorder=6)

VAL = {(0, 0): (1278.3, "—", "#2E6FAF", "理论上界"),
       (0, 1): (1292.2, "+13.9", "#8A5A10", "电价不可预知"),
       (1, 0): (1486.0, "—", "#2E6FAF", "含光伏预报不确定性"),
       (1, 1): (1503.1, "+17.1", "#8A5A10", "两类不确定性叠加")}
VMAX = 1503.1
for ri, (y, *_rest) in enumerate(ROWS):
    for ci, (x, w, *_c) in enumerate(COLS):
        v, dd, ac_, note = VAL[(ri, ci)]
        fc_, ec_ = ("#F4F9FF", SC["in_ec"]) if ci == 0 else ("#FEF8EE", SC["pv_ec"])
        _rrect(ax, x, y, w, .94, "#0F172A", "none", lw=0, r=.09, z=1, alpha=.05)
        _rrect(ax, x, y, w, .94, fc_, ec_, lw=1.3, r=.09, z=2)
        ax.text(x + .24, y + .60, "交付期总购电费", fontsize=8.6, color=C["sub"], va="center", zorder=6)
        ax.text(x + .24, y + .30, f"{v:,.1f}", fontsize=16.5, color=ac_, va="center",
                zorder=6, weight="bold")
        ax.text(x + 1.62, y + .32, "万元", fontsize=9, color=C["sub"], va="center", zorder=6)
        # 相对条
        bw = (w - .48) * (v / VMAX)
        _rrect(ax, x + .24, y + .13, w - .48, .055, "#E3E7EB", "none", lw=0, r=.028, z=3)
        _rrect(ax, x + .24, y + .13, bw, .055, ac_, "none", lw=0, r=.028, z=4)
        ax.text(x + w - .24, y + .60, note, fontsize=8.2, color=C["sub"],
                ha="right", va="center", zorder=6)
        if dd != "—":
            ax.text(x + w - .24, y + .30, dd, fontsize=10.5, color=ac_,
                    ha="right", va="center", zorder=6, weight="bold")

# 行差（电价不确定性成本）
for ri, (y, *_r) in enumerate(ROWS):
    _rrect(ax, 10.44, y + .22, .84, .50, "#FCEBEB", SC["hi_ec"], lw=1.1, r=.07, z=3)
    ax.text(10.86, y + .55, "电价不可预知", fontsize=7.4, color=SC["hi_tc"],
            ha="center", va="center", zorder=6)
    ax.text(10.86, y + .35, {"4-2": "+1.09%", "4-3": "+1.15%"}[ROWS[ri][1]],
            fontsize=8.6, color=SC["hi_tc"], ha="center", va="center", zorder=6, weight="bold")
    flow(ax, (10.34, y + .47), (10.46, y + .47), SC["hi_ec"], lw=1.8, ms=11, glow=False)
    ax.text(10.86, y + .12, "＝轨二 − 轨一", fontsize=7, color=SC["hi_sc"],
            ha="center", va="center", zorder=6)

# 列差（光伏预报不确定性成本）
_rrect(ax, 2.62, .72, 7.68, .50, "#E7F5F1", SC["ess_ec"], lw=1.1, r=.07, z=3)
ax.text(6.46, .97, "光伏预报不确定性成本　+207.7 万元（+16.2%）　/　+210.9 万元（+16.3%）",
        ha="center", va="center", fontsize=10, color=SC["ess_tc"], zorder=6, weight="bold")
flow(ax, (4.48, 1.82), (4.48, 1.26), SC["ess_ec"], lw=1.8, ms=12, glow=False)
flow(ax, (8.44, 1.82), (8.44, 1.26), SC["ess_ec"], lw=1.8, ms=12, glow=False)

# 结论条
_rrect(ax, .25, .06, 11.02, .52, "#FCF3F3", SC["hi_ec"], lw=1.1, r=.07, z=3)
ax.text(.45, .32, "结论：电价不确定性的总代价（+4.5% 水平 + 约 +1.1% 不可预知）仍不足光伏预报不确定性（+16.3%）的 1/3"
                  " ——　提升光伏预报精度的边际收益远高于预测电价",
        ha="left", va="center", fontsize=9.4, color=SC["hi_tc"], zorder=6)
fig.tight_layout(pad=.2); save(fig, "fig_s4_framework.png")
