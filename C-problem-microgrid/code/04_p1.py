"""
问题1：单日确定性最优购电策略（LP，含储能首末电量相等约束）
输入：附件1（电价 / 小区负载 / 光伏发电预测功率）
输出：控制台报告 + results/result1.xlsx + figures/fig_p1_*.png
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import (load_attachments, solve_lp, DT, N_T, E_INIT,
                        E_MIN, E_MAX, P_MAX, ETA, slot_label)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; RES = ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

d = load_attachments()
c, L, S = d["price_day"], d["load_day"], d["pv_day_fc"]

# ---------- 1. 最优解 ----------
r = solve_lp(c, L, S, E_start=E_INIT, E_end=E_INIT)
g, u, v, w, E = r["g"], r["u"], r["v"], r["w"], r["E"]

purch_kwh = g * DT
fee_opt = float((purch_kwh * c).sum())

# ---------- 2. 基线：无储能 ----------
net = np.maximum(L - S, 0.0)
base_kwh = float((net * DT).sum())
base_fee = float((net * DT * c).sum())

# ---------- 3. 基线：贪心阈值策略（阈值 1/eta^2 判据 + 朴素峰谷规则） ----------
thr = ETA ** 2
c_lo, c_hi = c.min(), c.max()
c_mid = float(np.median(c))
greedy = np.zeros(N_T); gu = np.zeros(N_T); gv = np.zeros(N_T); gE = np.zeros(N_T + 1)
gE[0] = E_INIT; gw = 0.0
for k in range(N_T):
    need = L[k] - S[k]
    Ek = gE[k]
    if c[k] <= c_mid and Ek < E_MAX - 1e-9:          # 低价：买电（满足负载 + 充电）
        room_in = min((E_MAX - Ek) / (ETA * DT), P_MAX)
        ch = room_in
        gk = max(need, 0.0) + ch
        gu[k] = ch
    elif c[k] > c_mid and Ek > E_MIN + 1e-9:         # 高价：放电顶负载
        avail_out = min((Ek - E_MIN) * ETA / DT, P_MAX)
        dis = min(max(need, 0.0), avail_out)
        gv[k] = dis
        gk = max(need - dis, 0.0)
    else:
        gk = max(need, 0.0)
    greedy[k] = gk
    gE[k + 1] = Ek + ETA * gu[k] * DT - gv[k] * DT / ETA
greedy_fee = float((greedy * DT * c).sum())

# ---------- 4. 报告 ----------
print("=" * 74)
print("问题1  单日确定性最优购电策略（附件1 = 全年平均日，储能首末电量均为 6000 kWh）")
print("=" * 74)
print(f"全天购电量        : {purch_kwh.sum():12.2f} kWh")
print(f"全天购电费        : {fee_opt:12.2f} 元")
print(f"充电量 / 放电量   : {u.sum()*DT:10.2f} / {v.sum()*DT:10.2f} kWh")
print(f"弃光电量          : {w.sum()*DT:12.2f} kWh")
print(f"储能 0:00 / 24:00 : {E_INIT:10.1f} / {E[-1]:10.2f} kWh")
print(f"储能电量区间      : [{E.min():.2f}, {E.max():.2f}] kWh")
print(f"同时充放最大重叠  : {float(np.minimum(u, v).max()):.3e} kW  (应为0)")
print("-" * 74)
print(f"[基线] 无储能 购电量 {base_kwh:11.2f} kWh   购电费 {base_fee:11.2f} 元")
print(f"[基线] 贪心阈值策略 购电量 {float((greedy*DT).sum()):11.2f} kWh   购电费 {greedy_fee:11.2f} 元")
print(f"[最优] 相对无储能省 {base_fee-fee_opt:10.2f} 元（{(base_fee-fee_opt)/base_fee*100:5.2f}%）")
print("-" * 74)

# 论文表1 指定时段
T1 = ["10:00-10:10", "12:00-12:10", "14:00-14:10",
      "16:00-16:10", "18:00-18:10", "20:00-20:10"]
print("论文表1  指定时段购电量 (kWh)")
for lab in T1:
    h, m = lab.split("-")[0].split(":")
    # 时段 k 覆盖 ((k-1)*10, k*10] 分钟；标签 "HH:MM" 是该时段的左端点
    k = (int(h) * 60 + int(m)) // 10 + 1
    print(f"   {lab:14s} -> 时段{k:3d}  购电量 {purch_kwh[k-1]:10.3f} kWh （电价 {c[k-1]:.4f}）")
print(f"   {'全天购电量':14s}          {purch_kwh.sum():10.3f} kWh")
print(f"   {'全天购电费':14s}          {fee_opt:10.3f} 元")

# 论文表2 充电/放电 分块
print("\n论文表2  储能设备充放电量 (kWh)")
blocks = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
for a, b in blocks:
    print(f"   {a//6:2d}:00-{b//6:2d}:00   充电 {u[a:b].sum()*DT:9.2f}   放电 {v[a:b].sum()*DT:9.2f}")
print(f"   0:00 储电量 {E_INIT:.2f} kWh    24:00 储电量 {E[-1]:.2f} kWh")

# 储电量轨迹极值时刻
print("\n储能轨迹关键点（每 2 小时）")
for h in range(0, 25, 2):
    k = min(max(h * 6, 1), N_T)
    print(f"   {h:2d}:00  {E[k-1]:9.2f} kWh", end="")
    if h % 6 == 4:
        print()
print()

# ---------- 5. 写出 result1.xlsx ----------
from cumcm_write import write_result1
write_result1(RES / "result1.xlsx", purch_kwh, u, v, E, E_INIT)
print(f"\n[OK] 已写出 {RES/'result1.xlsx'}")

# 保存中间结果
np.savez(RES / "p1_solution.npz", price=c, load=L, pv=S, g=g, u=u, v=v, w=w, E=E)

# ---------- 6. 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang SC", "Songti SC"]
plt.rcParams["axes.unicode_minus"] = False

tt = np.arange(1, N_T + 1) / 6.0
fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
ax[0].plot(tt, c, color="#c0392b", lw=1.6, label="电价 (元/kWh)")
ax[0].set_ylabel("电价 (元/kWh)"); ax[0].legend(loc="upper left"); ax[0].grid(alpha=.3)
ax[1].plot(tt, L, color="#2c3e50", lw=1.5, label="小区负载 (kW)")
ax[1].plot(tt, S, color="#e67e22", lw=1.5, label="光伏预测功率 (kW)")
ax[1].fill_between(tt, S, L, where=(L > S), color="#95a5a6", alpha=.25, label="净负荷缺口")
ax[1].set_ylabel("功率 (kW)"); ax[1].legend(loc="upper left", ncol=2); ax[1].grid(alpha=.3)
ax[2].plot(tt, purch_kwh, color="#2980b9", lw=1.4, label="计划购电量 (kWh/10min)")
ax[2].bar(tt, u * DT, width=.14, color="#27ae60", alpha=.8, label="充电量")
ax[2].bar(tt, -v * DT, width=.14, color="#8e44ad", alpha=.8, label="放电量")
ax[2].axhline(0, color="k", lw=.5)
ax[2].set_ylabel("电量 (kWh)"); ax[2].legend(loc="upper left", ncol=3); ax[2].grid(alpha=.3)
ax[3].plot(tt, E, color="#16a085", lw=2, label="储电量 (kWh)")
ax[3].axhline(E_MAX, ls="--", c="#c0392b", lw=1, label=f"上限 {E_MAX:.0f}")
ax[3].axhline(E_MIN, ls="--", c="#c0392b", lw=1, label=f"下限 {E_MIN:.0f}")
ax[3].axhline(E_INIT, ls=":", c="#7f8c8d", lw=1, label=f"初值 {E_INIT:.0f}")
ax[3].set_ylabel("储电量 (kWh)"); ax[3].set_xlabel("时刻 (h)")
ax[3].legend(loc="upper left", ncol=4); ax[3].grid(alpha=.3)
ax[3].set_xlim(0, 24); ax[3].set_xticks(range(0, 25, 2))
fig.suptitle("问题1  单日最优购电与储能调度策略（典型日）", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "fig_p1_dispatch.png", dpi=200, bbox_inches="tight")
print(f"[OK] 已保存 {FIG/'fig_p1_dispatch.png'}")
