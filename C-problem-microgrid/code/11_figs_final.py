"""
论文插图最终版（统一视觉规范）
F1 问题1 调度方案      F2 问题1 对偶分析       F3 问题2 全年策略
F4 预报误差标定        F5 问题3 信息价值        F6 问题4 波动电价
F7 灵敏度分析
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S
from style import C, FS, save, note, axgrid, waterfall
S.setup()

import matplotlib.pyplot as plt
from scipy.stats import probplot

from cumcm_core import load_attachments, DT, N_T, DAYS, E_INIT, E_MIN, E_MAX, ETA, DATA_DIR

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
I0 = 31
SL = slice(I0, DAYS)

d = load_attachments()
dates = pd.DatetimeIndex(d["dates"]); c_day = d["price_day"]; c4 = d["C4"]
L, Sv = d["L"], d["S"]

p1 = np.load(RES / "p1_solution.npz"); p2 = np.load(RES / "p2_solution.npz")
p3 = np.load(RES / "p3_solution.npz"); p4 = np.load(RES / "p4_solution.npz")
du = np.load(RES / "dual.npz"); fe = pd.read_csv(RES / "forecast_error.csv")
sens = pd.read_csv(RES / "sensitivity.csv")

t24 = np.arange(1, N_T + 1) / 6.0
base_2 = float((np.maximum(L - Sv, 0)[SL] * DT * c_day).sum())
cost_p2 = float(p2["fee_day"][SL].sum())
cost_det = float(p3["det_fee"]); cost_p3A = float(p3["costA"]); cost_p3_0 = float(p3["cost0"])
cost_roll0 = float(sens[(sens.param == "err_scale") & (sens.value == 0.0)]["year_fee"].iloc[0])
cost_42, cost_43 = float(p4["f42"]), float(p4["c43a"])
cost_42p, cost_43p = float(p4["f42p"]), float(p4["c43p"])
print(f"基线 {base_2:,.0f} | P2 {cost_p2:,.0f} | 完美+日周期 {cost_det:,.0f} | "
      f"零误差滚动 {cost_roll0:,.0f} | P3 {cost_p3A:,.0f}")
print(f"4-2 {cost_42:,.0f} | 4-2轨二 {cost_42p:,.0f} | 4-3 {cost_43:,.0f} | 4-3轨二 {cost_43p:,.0f}")
WAN = lambda v: f"{v/1e4:,.1f}万"


def forecast_errors():
    """返回 {k: (预报, 实际)} 的白天样本"""
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
    F, A = {}, {}
    for _, row in d["f3"].iterrows():
        d0 = pd.to_datetime(row["date"]).normalize(); h0 = int(row["h0i"])
        for k in range(1, 25):
            hh = h0 + k
            key = ((d0 + pd.Timedelta(days=hh // 24)).normalize(), hh % 24)
            if key in actual:
                F.setdefault(k, []).append(float(row[f"k{k}"]))
                A.setdefault(k, []).append(actual[key])
    F = {k: np.array(v) for k, v in F.items()}; A = {k: np.array(v) for k, v in A.items()}
    m = {k: (F[k] > 0) | (A[k] > 0) for k in F}
    return ({k: F[k][m[k]] for k in F}, {k: A[k][m[k]] for k in F})


# ======================================================== F1 问题1 调度
g, u, v, w, E = p1["g"], p1["u"], p1["v"], p1["w"], p1["E"]
ld, pvd = d["load_day"], d["pv_day_fc"]          # 附件1 的单日负载与光伏
pk = g * DT
fig, ax = plt.subplots(4, 1, figsize=(11.5, 11.6), sharex=True, gridspec_kw=dict(hspace=.16))

a = ax[0]
a.fill_between(t24, c_day, color=C["price"], alpha=.15, lw=0)
a.plot(t24, c_day, color=C["price"], lw=1.7)
hi = int(c_day.argmax()); lo = int(c_day.argmin())
a.annotate(f"最高 {c_day[hi]:.3f}", (t24[hi], c_day[hi]), xytext=(t24[hi] - 3.6, c_day[hi] * .90),
           fontsize=FS["note"], color=C["price"],
           arrowprops=dict(arrowstyle="-", color=C["price"], lw=.8))
a.annotate(f"最低 {c_day[lo]:.3f}", (t24[lo], c_day[lo]), xytext=(t24[lo] + 1.2, c_day[lo] - .16),
           fontsize=FS["note"], color=C["sub"], va="top",
           arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=.8))
a.set_ylabel("电价 (元/kWh)")
note(a, 0.4, 0.86, "峰谷比 2.92 ≫ 套利阈值 $1/\\eta^2$ = 1.235", C["price"], ha="left", va="center", style="normal")

a = ax[1]
a.fill_between(t24, pvd, ld, where=(ld > pvd), color=C["neutral"], alpha=.30, lw=0, label="净负荷缺口")
a.fill_between(t24, pvd, ld, where=(pvd >= ld), color=C["pv"], alpha=.30, lw=0, label="光伏余电")
a.plot(t24, ld, color=C["load"], lw=1.7, label="小区负载 $L_t$")
a.plot(t24, pvd, color=C["pv"], lw=1.7, label="光伏预测 $S_t$")
a.set_ylabel("功率 (kW)"); a.legend(ncol=4, loc="upper left", handlelength=1.4)

a = ax[2]
a.fill_between(t24, pk, color=C["buy"], alpha=.20, lw=0)
a.plot(t24, pk, color=C["buy"], lw=1.5, label="计划购电量")
a.bar(t24, u * DT, width=.15, color=C["ess"], label="充电")
a.bar(t24, -v * DT, width=.15, color=C["acc"], label="放电")
a.axhline(0, color=C["neutral"], lw=.8)
a.set_ylabel("电量 (kWh/10min)"); a.legend(ncol=3, loc="upper left", handlelength=1.4)

a = ax[3]
a.fill_between(t24, E, E_MIN, color=C["ess"], alpha=.16, lw=0)
a.plot(t24, E, color=C["ess"], lw=2.0, label="储电量 $E_t$")
a.axhline(E_MAX, ls=(0, (5, 4)), c=C["price"], lw=1.0)
a.axhline(E_MIN, ls=(0, (5, 4)), c=C["price"], lw=1.0)
a.text(23.9, E_MAX, f" 上限 {E_MAX:.0f} ", ha="right", va="bottom", fontsize=FS["note"], color=C["price"])
a.text(23.9, E_MIN, f" 下限 {E_MIN:.0f} ", ha="right", va="bottom", fontsize=FS["note"], color=C["price"])
a.set_ylabel("储电量 (kWh)"); a.set_xlabel("时刻 (h)")
a.legend(loc="upper left", handlelength=1.4)
a.set_xlim(0, 24); a.set_xticks(range(0, 25, 2))
for aa in ax:
    axgrid(aa, "y")
fig.suptitle("图 1　问题1 单日最优购电与储能调度策略", fontsize=13.5, y=.995)
save(fig, "fig_p1_dispatch.png")

# ======================================================== F2 对偶分析
fig = plt.figure(figsize=(13.5, 9.2))
gs = fig.add_gridspec(2, 2, hspace=.36, wspace=.24)

a = fig.add_subplot(gs[0, 0])
a.fill_between(t24, du["lam_bal"], c_day, color=C["buy"], alpha=.18, lw=0)
a.plot(t24, c_day, color=C["price"], lw=1.6, label="外网电价 $c_t$")
a.plot(t24, du["lam_bal"], color=C["buy"], lw=1.6, ls=(0, (5, 3)), label="电能影子价格 $\\sigma_t$")
a.set_title("(a) 电能影子价格 $\\sigma_t \\leq c_t$：边际电源的排序")
a.set_xlabel("时刻 (h)"); a.set_ylabel("元/kWh"); a.set_xlim(0, 24)
a.legend(loc="upper left", handlelength=1.6)
note(a, 12.2, 1.31, "购电时段 $\\sigma_t=c_t$（偏差 $1.1\\times10^{-16}$）\n非购电时段 $\\sigma_t<c_t$",
     C["ink"], ha="center", va="top", style="normal")
axgrid(a)

a = fig.add_subplot(gs[0, 1])
a.plot(t24, c_day, color=C["price"], lw=1.1, alpha=.42, label="电价 $c_t$")
a.plot(t24, du["lam_store"], color=C["acc"], lw=2.0, label="储能边际价值 $\\mu_t$")
ch = du["u"] > 1e-6; dis = du["v"] > 1e-6
a.scatter(t24[ch], du["lam_store"][ch], s=32, marker="^", color=C["ess"], zorder=5,
          label=f"充电 {ch.sum()} 段（均值 {du['lam_store'][ch].mean():.3f}）")
a.scatter(t24[dis], du["lam_store"][dis], s=32, marker="v", color=C["emg"], zorder=5,
          label=f"放电 {dis.sum()} 段（均值 {du['lam_store'][dis].mean():.3f}）")
a.set_title("(b) 储能边际价值 $\\mu_t$ 驱动充放电决策")
a.set_xlabel("时刻 (h)"); a.set_ylabel("元/kWh"); a.set_xlim(0, 24)
a.legend(loc="upper left", ncol=2, handlelength=1.2)
axgrid(a)

a = fig.add_subplot(gs[1, 0])
r = np.linspace(0, 6, 400); c_low = float(c_day.min())
gain = c_low * (r - 1 / ETA ** 2)
a.axvspan(0, 1 / ETA ** 2, color=C["price"], alpha=.07, lw=0)
a.plot(r, gain, color=C["buy"], lw=2.2)
a.axhline(0, color=C["neutral"], lw=.9)
a.axvline(1 / ETA ** 2, color=C["price"], ls=(0, (5, 3)), lw=1.5)
a.text(1 / ETA ** 2 + .08, -.62, f"套利阈值 $1/\\eta^2$={1/ETA**2:.3f}", fontsize=FS["note"], color=C["price"])
r1 = 2.922
r4m = float(np.median(c4.max(1) / np.maximum(c4.min(1), 1e-9)))
for rv, col, lab, mk in ((r1, C["ess"], f"附件1（平均日）{r1:.2f}", "o"),
                         (r4m, C["acc"], f"附件4（日内中位数）{r4m:.2f}", "s")):
    yv = c_low * (rv - 1 / ETA ** 2)
    a.plot([rv], [yv], mk, ms=9, color=col, zorder=5)
    a.annotate(lab, (rv, yv), xytext=(rv + .28, yv - .34), fontsize=FS["note"], color=col,
               arrowprops=dict(arrowstyle="-", color=col, lw=.8))
a.set_title("(c) 套利阈值判据：净收益 $=c_{low}\\,(r-1/\\eta^2)$")
a.set_xlabel("峰谷电价比 $r=c_{high}/c_{low}$"); a.set_ylabel("每 kWh 放电净收益 (元)")
a.set_xlim(0, 6); axgrid(a)

a = fig.add_subplot(gs[1, 1])
rr = c4.max(1) / np.maximum(c4.min(1), 1e-9)
a.hist(np.log10(rr), bins=40, color=C["buy"], alpha=.85, edgecolor="white", lw=.6)
top = a.get_ylim()[1]
a.axvline(np.log10(1 / ETA ** 2), color=C["price"], ls=(0, (5, 3)), lw=1.6)
a.axvline(np.log10(r4m), color=C["acc"], ls=":", lw=1.6)
a.text(np.log10(1 / ETA ** 2) + .05, top * .97, f"阈值 {1/ETA**2:.3f}", fontsize=FS["note"],
       color=C["price"], va="top")
a.text(np.log10(r4m) + .07, top * .82, f"中位数 {r4m:.2f}", fontsize=FS["note"],
       color=C["acc"], va="top")
a.set_title(f"(d) 附件4 日内峰谷比分布：{int((rr > 1/ETA**2).sum())}/365 天存在套利空间")
a.set_xticks([0, np.log10(2), np.log10(3), np.log10(5), np.log10(10), np.log10(20)])
a.set_xticklabels(["1", "2", "3", "5", "10", "20"])
a.set_xlabel("日内峰谷比（对数刻度）"); a.set_ylabel("天数"); axgrid(a)
fig.suptitle("图 2　问题1 的线性规划对偶分析", fontsize=13.5, y=.98)
save(fig, "fig_p1_dual.png")

# ======================================================== F3 问题2 全年
fig, ax = plt.subplots(3, 1, figsize=(13, 9.6), sharex=True, gridspec_kw=dict(hspace=.14))
a = ax[0]
a.plot(dates, np.maximum(L - Sv, 0).sum(1) * DT, color=C["base"], lw=.8, label="无储能基线")
a.plot(dates, p2["purch_kwh"].sum(1), color=C["buy"], lw=.9, label="全年最优计划")
a.axvspan(dates[0], dates[I0 - 1], color=C["neutral"], alpha=.22, lw=0)
a.set_ylabel("日购电量 (kWh)"); a.legend(loc="upper left", ncol=2, handlelength=1.6)
a.text(dates[15], a.get_ylim()[1] * .95, "预热期\n1 月", ha="center", va="top", fontsize=FS["note"], color=C["sub"])
note(a, dates[-2], a.get_ylim()[1] * .95, "交付期 2/1 – 12/31", C["buy"], ha="right", va="top", style="normal")
axgrid(a)

a = ax[1]
a.plot(dates, p2["fee_day"], color=C["price"], lw=.8, label="全年最优日购电费")
a.plot(dates, (np.maximum(L - Sv, 0) * DT * c_day).sum(1), color=C["base"], lw=.8, label="无储能日购电费")
mo = pd.Series(p2["fee_day"], index=dates).resample("MS").mean()
a.plot(mo.index, mo.values, color=C["ink"], lw=2.0, alpha=.75, label="月度均值")
a.axvspan(dates[0], dates[I0 - 1], color=C["neutral"], alpha=.22, lw=0)
a.set_ylabel("日购电费 (元)"); a.legend(loc="upper left", ncol=3, handlelength=1.6)
note(a, dates[150], a.get_ylim()[1] * .97,
     f"交付期合计 {WAN(cost_p2)}　储能较无储能节省 {(base_2 - cost_p2) / base_2 * 100:.1f}%",
     C["ess"], va="top", style="normal")
axgrid(a)

a = ax[2]
a.fill_between(dates, p2["E_end_day"], E_MIN, color=C["ess"], alpha=.16, lw=0)
a.plot(dates, p2["E_end_day"], color=C["ess"], lw=1.2, label="每日 24:00 储电量")
a.axhline(E_MAX, ls=(0, (5, 4)), c=C["price"], lw=1.0)
a.axhline(E_MIN, ls=(0, (5, 4)), c=C["price"], lw=1.0)
a.text(dates[2], E_MAX, f" 上限 {E_MAX:.0f}", fontsize=FS["note"], color=C["price"], va="bottom")
a.text(dates[2], E_MIN, f" 下限 {E_MIN:.0f}", fontsize=FS["note"], color=C["price"], va="bottom")
a.axvspan(dates[0], dates[I0 - 1], color=C["neutral"], alpha=.22, lw=0)
a.set_ylabel("储电量 (kWh)"); a.set_xlabel("日期"); a.legend(loc="upper left", handlelength=1.6)
axgrid(a)
fig.suptitle("图 3　问题2 全年最优购电策略（整年连续 LP：262 800 变量 / 105 121 约束 / 3.0 s）",
             fontsize=13.5, y=.995)
save(fig, "fig_p2_year.png")

# ======================================================== F4 预报误差标定
Fc, Ac = forecast_errors()
ks = fe["k"].to_numpy(); mae = fe["MAE"].to_numpy(); rmse = fe["RMSE"].to_numpy()
bias = fe["bias"].to_numpy(); sd = fe["sd"].to_numpy()
a_sd, b_sd = float(du["sigma_a"]), float(du["sigma_b"])
fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.2))

a = ax[0, 0]
a.plot(ks, rmse, "-", color=C["buy"], lw=1.8, marker="s", ms=3.4, label="RMSE")
a.plot(ks, mae, "-", color=C["pv"], lw=1.8, marker="o", ms=3.4, label="MAE")
a.plot(ks, np.abs(bias), "--", color=C["ess"], lw=1.5, marker="^", ms=3.2, label="|系统偏差|")
a.set_title("(a) 误差随提前期的增长（白天样本）")
a.set_xlabel("预报提前期 $k$ (h)"); a.set_ylabel("误差 (kW)")
a.legend(loc="upper left", ncol=3, handlelength=1.6)
note(a, 24, mae[-1], f"MAE 放大 {mae[-1]/mae[0]:.1f} 倍", C["pv"], ha="right", va="bottom", style="normal")
axgrid(a)

a = ax[0, 1]
a.plot(ks, sd, color=C["acc"], lw=1.8, marker="o", ms=3.4, label="实测 $\\sigma(k)$")
a.plot(ks, a_sd * ks ** b_sd, "--", color=C["price"], lw=1.8,
       label=f"拟合 $\\sigma(k)={a_sd:.0f}\\,k^{{{b_sd:.2f}}}$　$R^2$=0.991")
a.plot(ks, bias, color=C["ess"], lw=1.5, marker="^", ms=3.2, label="系统偏差 $\\mu(k)$")
a.axhline(0, color=C["neutral"], lw=.8)
a.set_title("(b) 不确定性的幂律标定")
a.set_xlabel("预报提前期 $k$ (h)"); a.set_ylabel("kW")
a.legend(loc="upper left", handlelength=1.6); axgrid(a)

a = ax[1, 0]
(osm, osr), _ = probplot((Fc[24] - Ac[24] - bias[23]) / sd[23], dist="norm")
a.plot(osm, osr, "o", ms=2.6, color=C["price"], alpha=.55, label="$k$=24 h（实测）")
i6 = int(np.where(ks == 6)[0][0])
(osm6, osr6), _ = probplot((Fc[6] - Ac[6] - bias[i6]) / sd[i6], dist="norm")
a.plot(osm6, osr6, "o", ms=2.6, color=C["pv"], alpha=.55, label="$k$=6 h（实测）")
a.plot([-3.6, 3.6], [-3.6, 3.6], "--", color=C["base"], lw=1.3, label="标准正态参考")
a.set_xlim(-3.6, 3.6); a.set_ylim(-3.6, 3.6)
a.set_title("(c) 标准化残差的 Q–Q 图：尾部略厚于正态")
a.set_xlabel("理论分位数"); a.set_ylabel("样本分位数")
a.legend(loc="upper left", handlelength=1.2); axgrid(a)

a = ax[1, 1]
for kk, col in ((6, C["pv"]), (24, C["price"])):
    a.scatter(Ac[kk], Fc[kk], s=5, alpha=.25, color=col, lw=0, label=f"$k$={kk} h")
lim = [0, 10600]
a.plot(lim, lim, "--", color=C["base"], lw=1.2, label="理想预报")
a.set_title("(d) 预报 vs 实际：高辐照时段误差更大（异方差）")
a.set_xlabel("实际光伏功率 (kW)"); a.set_ylabel("预报光伏功率 (kW)")
a.set_xlim(lim); a.set_ylim(lim); a.legend(loc="upper left", handlelength=1.4); axgrid(a)
fig.suptitle("图 4　光伏预报误差标定（附件3 vs 附件2，剔除夜间零样本，每提前期约 730 样本）",
             fontsize=13.5, y=.98)
save(fig, "fig_p3_forecast_calib.png")

# ======================================================== F5 问题3
fig = plt.figure(figsize=(13.8, 9.4))
gs = fig.add_gridspec(2, 2, hspace=.38, wspace=.22)

a = fig.add_subplot(gs[0, 0])
emg_d = (p3["emg"] * DT).sum(1); cur_d = (p3["cur"] * DT).sum(1)
a.bar(dates, emg_d, width=1.0, color=C["emg"], label="紧急购电量")
a.bar(dates, -cur_d, width=1.0, color=C["curtail"], label="弃光电量")
a.axhline(0, color=C["neutral"], lw=.8)
a.set_title(f"(a) 日前平衡偏差：紧急购电 {emg_d[SL].sum():,.0f} kWh，弃光 {cur_d[SL].sum():,.0f} kWh")
a.set_ylabel("kWh/天"); a.legend(loc="lower left", ncol=2, handlelength=1.2)
note(a, dates[5], -1500, "↑ 紧急购电　　↓ 弃光", C["sub"], ha="left", va="top", style="normal")
axgrid(a)

a = fig.add_subplot(gs[0, 1])
dev = np.abs(p3["adj"] - p3["plan"]).sum(1) * DT
a.fill_between(dates, dev, color=C["acc"], alpha=.18, lw=0)
a.plot(dates, dev, color=C["acc"], lw=.9, label="日调整偏差 $|a_t-g_t|$")
a.plot(dates, pd.Series(dev, index=dates).rolling(30, center=True).mean(),
       color=C["ink"], lw=1.8, label="30 日滑动平均")
a.set_title(f"(b) 计划与调整的偏离：累计 {dev[SL].sum():,.0f} kWh")
a.set_ylabel("kWh/天"); a.legend(loc="upper left", handlelength=1.6)
axgrid(a)

a = fig.add_subplot(gs[1, 0])
mo_plan = pd.Series((p3["plan"] * DT).sum(1), index=dates).resample("MS").mean().iloc[1:]
mo_adj = pd.Series((p3["adj"] * DT).sum(1), index=dates).resample("MS").mean().iloc[1:]
mo_det = pd.Series((p3["g_det"] * DT).sum(1), index=dates).resample("MS").mean().iloc[1:]
x = np.arange(len(mo_plan)); wd = .27
a.bar(x - wd, mo_plan.values, wd, color=C["buy"], label="计划购电量")
a.bar(x, mo_adj.values, wd, color=C["ess"], label="调整后执行量")
a.bar(x + wd, mo_det.values, wd, color=C["base"], label="完美信息最优")
a.set_xticks(x); a.set_xticklabels([f"{m.month}月" for m in mo_plan.index], fontsize=8.5)
a.set_title("(c) 逐月日均购电量：计划 / 执行 / 完美信息")
a.set_ylabel("kWh/天"); a.legend(loc="upper left", ncol=3, handlelength=1.2)
axgrid(a)

a = fig.add_subplot(gs[1, 1])
_d3 = [cost_det - base_2, cost_roll0 - cost_det, cost_p3A - cost_roll0]
_b3 = [base_2, cost_det, cost_roll0]
_l3 = ["储能套利价值", "滚动优化固有次优", "光伏预报误差"]
waterfall(a,
          [f"{t}\n{'−' if d < 0 else '+'}{abs(d)/b*100:.1f}%"
           for t, d, b in zip(_l3, _d3, _b3)],
          _d3, base_2, ylim=(0, cost_p3A * 1.22))
a.set_title("(d) 交付期总费用归因：储能价值 vs 预报不确定性代价", fontsize=FS["title"] - 0.5)
fig.suptitle("图 5　问题3 光伏预报不确定下的滚动优化：费用归因与信息价值", fontsize=13.5, y=.98)
save(fig, "fig_p3_rolling.png")

# ======================================================== F6 问题4
fig = plt.figure(figsize=(13.8, 9.4))
gs = fig.add_gridspec(2, 2, hspace=.40, wspace=.24)

# (a) 分时段电价箱线图：固定曲线只是均值，实际在其两侧大幅波动
a = fig.add_subplot(gs[0, 0])
ph = np.stack([c4[:, h * 6:(h + 1) * 6].mean(1) for h in range(24)], axis=1)  # (365,24)
bp = a.boxplot([ph[:, h] for h in range(24)], positions=np.arange(24) + .5,
               widths=.62, patch_artist=True, showfliers=False,
               medianprops=dict(color=C["ink"], lw=1.2),
               whiskerprops=dict(color=C["base"], lw=.8),
               capprops=dict(color=C["base"], lw=.8),
               boxprops=dict(facecolor=C["buy"], edgecolor=C["buy"], alpha=.42, lw=.7))
a.plot(np.arange(24) + .5, c_day.reshape(24, 6).mean(1), color=C["price"], lw=2.0,
       marker="o", ms=3.2)
a.set_title("(a) 外网电价的逐时分布：固定曲线掩盖了巨大波动")
a.set_xlabel("时刻 (h)"); a.set_ylabel("电价 (元/kWh)")
a.set_xticks([0, 4, 8, 12, 16, 20, 23])
a.set_xticklabels(["0", "4", "8", "12", "16", "20", "23"])
a.set_ylim(.05, 2.02); axgrid(a)
note(a, 0.15, 1.99, "箱体 = 365 天的四分位区间 (P25–P75)，须线 = 1.5×IQR",
     C["ink"], ha="left", va="top", style="normal")
note(a, 0.15, 1.82,
     f"全期电价 P5~P95 = {np.percentile(c4,5):.2f} ~ {np.percentile(c4,95):.2f} 元/kWh，"
     f"极值 {c4.min():.2f} ~ {c4.max():.2f}",
     C["price"], ha="left", va="top", style="normal")
a.annotate("附件1 固定日曲线（历史分时均价）", xy=(19.5, 1.33), xytext=(13.2, 1.62),
           fontsize=FS["note"] - .5, color=C["price"], ha="left", va="top",
           arrowprops=dict(arrowstyle="-", color=C["price"], lw=.8))

# (b) 日内峰谷比的累积分布：直接读出越过套利阈值的比例
a = fig.add_subplot(gs[0, 1])
rr = np.sort(c4.max(1) / np.maximum(c4.min(1), 1e-9))
ys = np.arange(1, len(rr) + 1) / len(rr)
a.plot(np.log10(rr), ys, color=C["buy"], lw=2.2)
th = np.log10(1 / ETA ** 2)
a.axvline(th, color=C["price"], ls=(0, (5, 3)), lw=1.6)
a.fill_betweenx([0, 1.04], th, 1.6, color=C["ess"], alpha=.10, lw=0)
a.text(1.16, .28, "套利可行区", fontsize=FS["note"], color=C["ess"], ha="center")
a.axhline(1.0, color=C["neutral"], lw=.8, ls=":")
a.annotate(f"套利阈值 $1/\\eta^2$ = {1/ETA**2:.3f}\n365/365 天全部越线",
           xy=(th, .86), xytext=(-.17, 1.03), ha="left", va="top",
           fontsize=FS["note"], color=C["price"],
           arrowprops=dict(arrowstyle="->", color=C["price"], lw=.9))
a.set_title("(b) 日内峰谷比的累积分布：全年每天都可套利")
a.set_xlabel("日内峰谷比（对数刻度）"); a.set_ylabel("累计概率")
a.set_xlim(-.2, 1.6); a.set_ylim(0, 1.06)
a.set_xticks([0, .5, 1.0, 1.5]); a.set_xticklabels(["1", "3.2", "10", "32"])
axgrid(a)

a = fig.add_subplot(gs[1, 0])
a.plot(dates, p4["fee42"], color=C["price"], lw=.8, label="4-2 波动电价日购电费")
a.plot(dates, p2["fee_day"], color=C["base"], lw=.8, alpha=.9, label="问题2 固定电价日购电费")
mo4 = pd.Series(p4["fee42"], index=dates).resample("MS").mean()
a.plot(mo4.index, mo4.values, color=C["ink"], lw=1.8, alpha=.7, label="月度均值")
a.set_title(f"(c) 日购电费：波动电价使费用上升 {(cost_42-cost_p2)/cost_p2*100:.2f}%")
a.set_ylabel("元/天"); a.set_xlabel("日期"); a.legend(loc="upper left", ncol=3, handlelength=1.2)
a.margins(y=.20)
axgrid(a)

a = fig.add_subplot(gs[1, 1])
_d4 = [cost_42 - cost_p2, cost_43 - cost_42, cost_43p - cost_43]
_b4 = [cost_p2, cost_42, cost_43]
_l4 = ["电价波动", "光伏预报不确定", "电价不可预知"]
waterfall(a,
          [f"{t}\n{'−' if d < 0 else '+'}{abs(d)/b*100:.1f}%"
           for t, d, b in zip(_l4, _d4, _b4)],
          _d4, cost_p2, ylim=(0, cost_43p * 1.24),
          base_label="问题2 固定电价\n完美信息", total_label="问题4-3 波动电价\n预报驱动")
a.set_title("(d) 交付期总费用归因：三类风险的边际贡献", fontsize=FS["title"] - 0.5)
fig.suptitle("图 6　问题4 波动电价下的双轨建模：电价波动与电价不可预知的分离", fontsize=13.5, y=.98)
save(fig, "fig_p4_price.png")

# ======================================================== F7 灵敏度
fig = plt.figure(figsize=(13.8, 9.4))
gs = fig.add_gridspec(2, 2, hspace=.36, wspace=.24)

eta_r = sens[sens.param == "eta"]; pm_r = sens[sens.param == "pmax"]; er_r = sens[sens.param == "err_scale"]

a = fig.add_subplot(gs[0, 0])
a.plot(eta_r.value, eta_r.year_fee / 1e4, "-o", color=C["price"], ms=5, lw=1.8)
i0_ = int(np.argmin(np.abs(eta_r.value.values - .9)))
a.plot([.9], [eta_r.year_fee.values[i0_] / 1e4], "o", ms=11, color=C["ink"], zorder=5)
a.annotate(f"基准 $\\eta$=0.90　{eta_r.year_fee.values[i0_]/1e4:,.1f} 万元",
           (0.9, eta_r.year_fee.values[i0_] / 1e4), xytext=(.858, eta_r.year_fee.values[i0_] / 1e4 - 42),
           fontsize=FS["note"], color=C["ink"],
           arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=.8))
a.margins(y=.12)
a.set_title("(a) 储能效率 η 的灵敏度"); a.set_xlabel("充放电效率 $\\eta$")
a.set_ylabel("全年最优购电费 (万元)"); axgrid(a)

a = fig.add_subplot(gs[0, 1])
a.plot(pm_r.value, pm_r.year_fee / 1e4, "-s", color=C["buy"], ms=5, lw=1.8)
ip = int(np.argmin(np.abs(pm_r.value.values - 5000)))
a.plot([5000], [pm_r.year_fee.values[ip] / 1e4], "o", ms=11, color=C["ink"], zorder=5)
a.annotate(f"题目给定 $P_{{max}}$=5000　{pm_r.year_fee.values[ip]/1e4:,.1f} 万元",
           (5000, pm_r.year_fee.values[ip] / 1e4), xytext=(3050, pm_r.year_fee.values[ip] / 1e4 + 22),
           fontsize=FS["note"], color=C["ink"],
           arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=.8))
a.set_title("(b) 储能功率上限 $P_{max}$ 的灵敏度（5000 kW 后收益骤减）")
a.set_xlabel("$P_{max}$ (kW)"); a.set_ylabel("全年最优购电费 (万元)"); axgrid(a)

a = fig.add_subplot(gs[1, 0])
a.plot(er_r.value, er_r.year_fee / 1e4, "-^", color=C["acc"], ms=5, lw=1.8)
for xv in (0.0, 1.0, 2.0):
    iv = int(np.argmin(np.abs(er_r.value.values - xv)))
    a.plot([xv], [er_r.year_fee.values[iv] / 1e4], "o", ms=6, color=C["ink"], zorder=5)
    a.annotate(f"{er_r.year_fee.values[iv]/1e4:,.1f}", (xv, er_r.year_fee.values[iv] / 1e4),
               xytext=(xv - .09, er_r.year_fee.values[iv] / 1e4 + 18), fontsize=FS["note"], color=C["ink"])
a.set_title("(c) 光伏预报误差幅度的灵敏度（问题3）")
a.set_xlabel("预报误差倍率（1.0 = 实际）"); a.set_ylabel("总费用 (万元)"); axgrid(a)

a = fig.add_subplot(gs[1, 1])
scen = [("无储能基线", base_2, C["base"]),
        ("问题2 固定电价\n完美信息", cost_p2, C["load"]),
        ("完美信息\n+日周期", cost_det, C["ess"]),
        ("4-2 波动电价\n完美信息", cost_42, C["price"]),
        ("问题3 光伏预报", cost_p3A, C["pv"]),
        ("4-3 波动电价\n+光伏预报", cost_43, C["acc"])]
names = [s[0] for s in scen]
vals = [s[1] / 1e4 for s in scen]          # 统一换算为万元
cols = [s[2] for s in scen]
yy = np.arange(len(scen))[::-1]
a.hlines(yy, 0, vals, color=C["grid"], lw=1.2, zorder=1)
a.scatter(vals, yy, s=110, color=cols, zorder=4)
for y_, v_ in zip(yy, vals):
    a.text(v_ + max(vals) * .028, y_, f"{v_:,.1f}", va="center", fontsize=FS["note"], color=C["ink"])
a.set_yticks(yy); a.set_yticklabels(names, fontsize=8.5)
a.set_xlim(0, max(vals) * 1.16)
a.set_title("(d) 六情景总购电费对比")
a.set_xlabel("交付期总购电费 (万元)")
a.grid(True, axis="x", color=C["grid"], lw=.7); a.grid(False, axis="y")
fig.suptitle("图 7　模型检验与灵敏度分析", fontsize=13.5, y=.98)
save(fig, "fig_sensitivity.png")
print("\n全部插图已按统一视觉规范重新生成。")
