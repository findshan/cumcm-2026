"""
问题2：全年逐日最优购电策略（确定性 LP，完美信息）
- 电价：附件1（每天相同）
- 负载/光伏：附件2 实际值（365 天）
- 储能跨日连续，年初 E=6000，年末 E ≥ 6000（可持续）
- 交付区间：2025-02-01 ~ 2025-12-31（334 天）—— 1 月为预热期
"""
import numpy as np
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import (load_attachments, solve_lp, DT, N_T, DAYS,
                        E_INIT, E_MIN, E_MAX, P_MAX, ETA)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; RES = ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

d = load_attachments()
c_day = d["price_day"]                      # (144,)  每天相同
L, S = d["L"], d["S"]                       # (365,144)
dates = d["dates"]

# ---------- 整年一次性求解 ----------
c_y = np.tile(c_day, DAYS)
L_y = L.ravel(); S_y = S.ravel()

t0 = time.time()
r = solve_lp(c_y, L_y, S_y, E_start=E_INIT, E_end_min=E_INIT, tie_break=1e-7)
print(f"[全年 LP] 变量 {5*DAYS*N_T:,} 约束 {2*DAYS*N_T+1:,}  求解耗时 {time.time()-t0:.1f}s")
print(f"          自检残差 {r['resid']:.2e}")

g = r["g"].reshape(DAYS, N_T); u = r["u"].reshape(DAYS, N_T)
v = r["v"].reshape(DAYS, N_T); w = r["w"].reshape(DAYS, N_T)

# 每日 0:00 / 24:00 储电量
E_end_day = r["E"].reshape(DAYS, N_T)[:, -1]
E_start_day = np.concatenate([[E_INIT], E_end_day[:-1]])

purch_kwh = g * DT                                  # (365,144)
fee_day = (purch_kwh * np.tile(c_day, (DAYS, 1))).sum(1)

# ---------- 交付区间：2/1 ~ 12/31 ----------
i0, i1 = 31, 365
sl = slice(i0, i1)
print("\n" + "=" * 78)
print(f"问题2  结果区间 {dates[i0].date()} ~ {dates[i1-1].date()}（{i1-i0} 天）")
print("=" * 78)
print(f"计划购电量合计   : {purch_kwh[sl].sum():14.2f} kWh")
print(f"计划购电费合计   : {fee_day[sl].sum():14.2f} 元")
print(f"充电/放电量      : {u[sl].sum()*DT:14.2f} / {v[sl].sum()*DT:.2f} kWh")
print(f"弃光电量         : {w[sl].sum()*DT:14.2f} kWh")
print(f"紧急购电量       : {0.0:14.2f} kWh  (完美信息下恒为 0)")
print(f"平均日购电费     : {fee_day[sl].mean():14.2f} 元/天")
print(f"储能年初/年末    : {E_INIT:9.1f} / {E_end_day[-1]:9.2f} kWh")
print(f"储能全年区间     : [{r['E'].min():.2f}, {r['E'].max():.2f}] kWh")
print(f"全年购电费(1/1起): {fee_day.sum():14.2f} 元")

# ---------- 与问题1 基准对比（关键：这是 Jensen 缺口，不是"精细化损失"） ----------
p1 = np.load(RES / "p1_solution.npz")
p1_fee = float((p1["g"] * DT * c_day).sum())
net = np.maximum(L - S, 0.0)
base_fee = float((net * DT * c_day).sum())
p1_extrap = p1_fee * DAYS

print("\n" + "-" * 78)
print("[对比 1] 平均日模型 vs 真实逐日最优")
print(f"   问题1 典型日(平均日) 日购电费      : {p1_fee:13.2f} 元/天")
print(f"   问题2 真实逐日最优  平均日购电费    : {fee_day.mean():13.2f} 元/天")
print(f"   二者相差 {(fee_day.mean()-p1_fee)/p1_fee*100:5.2f}%")
print( "       —— 购电费对(负荷,光伏)剖面是凸函数，平均剖面抹平了真实波动，")
print( "          故『平均日模型』会系统性低估真实日均成本，此即 Jensen 缺口。")
print(f"   全年口径：平均日外推 {p1_extrap:,.0f} 元  vs  真实逐日最优 {fee_day.sum():,.0f} 元")
print(f"             低估 {fee_day.sum()-p1_extrap:,.0f} 元（{(fee_day.sum()-p1_extrap)/fee_day.sum()*100:.2f}%）")
print(f"[对比 2] 储能的价值（同一真实数据口径）")
print(f"   无储能全年购电费 : {base_fee:14,.2f} 元")
print(f"   全年最优购电费   : {fee_day.sum():14,.2f} 元")
print(f"   储能节省         : {base_fee-fee_day.sum():14,.2f} 元（{(base_fee-fee_day.sum())/base_fee*100:.2f}%）")

# ---------- 月度汇总 ----------
print("\n月度汇总（2月起）")
import pandas as pd
df = pd.DataFrame({"date": dates, "fee": fee_day, "kwh": purch_kwh.sum(1),
                   "curtail": w.sum(1) * DT})
df["ym"] = df["date"].dt.strftime("%Y-%m")
mo = df[sl if False else (df.index >= i0)].groupby("ym")[["kwh", "fee", "curtail"]].sum()
print(mo.round(2).to_string())

np.savez(RES / "p2_solution.npz", g=g, u=u, v=v, w=w,
         E_end_day=E_end_day, E_start_day=E_start_day, fee_day=fee_day,
         purch_kwh=purch_kwh, E_all=r["E"].reshape(DAYS, N_T))

# ---------- 写出 result2.xlsx ----------
from cumcm_write import write_result2_like
write_result2_like(RES / "result2.xlsx", dates[i0:], purch_kwh[i0:],
                   u[i0:], v[i0:], E_start_day[i0:], E_end_day[i0:],
                   emergency=None, fee_day=fee_day[i0:])
print(f"\n[OK] 已写出 {RES/'result2.xlsx'}（{i1-i0} 天 × 144 时段）")

# ---------- 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang SC", "Songti SC"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(3, 1, figsize=(13, 9))
ax[0].plot(dates, purch_kwh.sum(1), lw=.8, color="#2980b9", label="每日计划购电量 (kWh)")
ax[0].plot(dates, np.maximum(L - S, 0).sum(1) * DT, lw=.8, color="#95a5a6",
           alpha=.8, label="无储能时的每日购电量")
ax[0].axvline(dates[i0], color="#c0392b", ls="--", lw=1, label="交付起始 2025-02-01")
ax[0].set_ylabel("日购电量 (kWh)"); ax[0].legend(loc="upper left", ncol=3, fontsize=9); ax[0].grid(alpha=.3)

ax[1].plot(dates, fee_day, lw=.8, color="#c0392b", label="每日购电费 (元)")
ax[1].plot(dates, (net * DT * c_day).sum(1), lw=.8, color="#95a5a6",
           alpha=.8, label="无储能每日购电费")
ax[1].axvline(dates[i0], color="#c0392b", ls="--", lw=1)
ax[1].set_ylabel("日购电费 (元)"); ax[1].legend(loc="upper left", ncol=2, fontsize=9); ax[1].grid(alpha=.3)

ax[2].plot(dates, E_end_day, lw=1, color="#16a085", label="每日 24:00 储电量 (kWh)")
ax[2].axhline(E_MAX, ls="--", c="#c0392b", lw=1, label=f"上限 {E_MAX:.0f}")
ax[2].axhline(E_MIN, ls="--", c="#c0392b", lw=1, label=f"下限 {E_MIN:.0f}")
ax[2].axvline(dates[i0], color="#c0392b", ls="--", lw=1)
ax[2].set_ylabel("储电量 (kWh)"); ax[2].set_xlabel("日期")
ax[2].legend(loc="upper left", ncol=3, fontsize=9); ax[2].grid(alpha=.3)
fig.suptitle("问题2  全年最优购电策略（预热期 1 月 + 交付期 2–12 月）", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "fig_p2_year.png", dpi=200, bbox_inches="tight")
print(f"[OK] 已保存 {FIG/'fig_p2_year.png'}")
