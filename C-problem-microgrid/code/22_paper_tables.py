"""
生成论文所需的全部结果表数字（题面表1 / 表2 / 表3 格式）

表1  微网在指定时间段的购电量及全天的购电量和购电费
     时段：10:00-10:10  12:00-12:10  14:00-14:10  16:00-16:10  18:00-18:10  20:00-20:10
表2  储能设备在指定时间段的充放电量及 0:00 和 24:00 的储电量
     时段：0:00-4:00  4:00-8:00  8:00-12:00  12:00-16:00  16:00-20:00  20:00-24:00
表3  微网在指定日期的紧急购电量（2025.3.20 / 6.21 / 9.23 / 12.21）
"""
import numpy as np
import pandas as pd
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import load_attachments, DT, N_T, DAYS, E_INIT

RES = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "论文" / "tables.json"
OUT.parent.mkdir(exist_ok=True)

d = load_attachments()
dates = pd.DatetimeIndex(d["dates"])

# 指定时段（10 分钟）对应的时段序号（1-based）
def slot_index(hh, mm):
    return hh * 6 + mm // 10

T1_SLOTS = [(10, 0), (12, 0), (14, 0), (16, 0), (18, 0), (20, 0)]
T1_IDX = [slot_index(h, m) for h, m in T1_SLOTS]          # 1-based 时段号
# 表2 的 4 小时段
T2_EDGES = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
# 表3 指定日期
SPEC_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
SPEC_IDX = [int(np.where(dates == pd.Timestamp(t))[0][0]) for t in SPEC_DATES]

out = {}

# ---------------------------------------------------------------- 问题1
p1 = np.load(RES / "p1_solution.npz")
g1, u1, v1, E1 = p1["g"], p1["u"], p1["v"], p1["E"]
pk1 = g1 * DT
out["P1"] = dict(
    t1=[float(pk1[k - 1]) for k in T1_IDX],
    day_purch=float(pk1.sum()),
    day_fee=float((g1 * DT * d["price_day"]).sum()),
    t2_chg=[float(u1[a:b].sum() * DT) for a, b in T2_EDGES],
    t2_dis=[float(v1[a:b].sum() * DT) for a, b in T2_EDGES],
    E0=float(E_INIT), E24=float(E1[-1]),
    emg={t: 0.0 for t in SPEC_DATES},
)

# ---------------------------------------------------------------- 通用：按日提取
def day_tables(plan_kwh, adj_kwh, u, v, emg_kwh, Est, Eed, fee_day, tag):
    """plan/adj/u/v 均为 (DAYS,144)，u/v 为功率 kW，emg 为功率 kW"""
    res = dict()
    for t, D in zip(SPEC_DATES, SPEC_IDX):
        key = {"2025-03-20": "0320", "2025-06-21": "0621",
               "2025-09-23": "0923", "2025-12-21": "1221"}[t]
        src = adj_kwh if adj_kwh is not None else plan_kwh
        res[key] = dict(
            date=t,
            t1=[float(src[D, k - 1]) for k in T1_IDX] if src.ndim == 2 and src.shape[1] == 144
                else [float(src[D * N_T + k - 1]) for k in T1_IDX],
            day_purch=float(np.asarray(src).reshape(DAYS, N_T)[D].sum()),
            day_fee=float(fee_day[D]),
            t2_chg=[float(u.reshape(DAYS, N_T)[D, a:b].sum() * DT) for a, b in T2_EDGES],
            t2_dis=[float(v.reshape(DAYS, N_T)[D, a:b].sum() * DT) for a, b in T2_EDGES],
            E0=float(np.atleast_1d(Est)[D]), E24=float(np.atleast_1d(Eed)[D]),
            emg=float(emg_kwh.reshape(DAYS, N_T)[D].sum()) if emg_kwh is not None else 0.0,
        )
    out[tag] = res

# ---------------------------------------------------------------- 问题2
p2 = np.load(RES / "p2_solution.npz")
g2 = p2["g"].reshape(DAYS, N_T) * DT          # kWh
u2 = p2["u"].reshape(DAYS, N_T); v2 = p2["v"].reshape(DAYS, N_T)
Es2 = p2["E_start_day"]; Ee2 = p2["E_end_day"]; fee2 = p2["fee_day"]
emg2 = np.zeros((DAYS, N_T))                   # 完美信息 → 紧急购电恒为 0（已验证）
day_tables(g2, None, u2, v2, emg2, Es2, Ee2, fee2, "P2")

# ---------------------------------------------------------------- 问题3（MPC-II 交付）
p3 = np.load(RES / "p3_solution.npz")
plan3 = p3["plan"] * DT; adj3 = p3["adj"] * DT
u3 = p3["u"]; v3 = p3["v"]; emg3 = p3["emg"] * DT
fee3 = (p3["plan"] * DT * np.tile(d["price_day"], (DAYS, 1))).sum(1) \
     + ((0.5 * np.maximum(p3["plan"] - p3["adj"], 0) + 1.5 * np.maximum(p3["adj"] - p3["plan"], 0))
        * DT * np.tile(d["price_day"], (DAYS, 1))).sum(1) \
     + (p3["emg"] * DT * np.tile(d["price_day"], (DAYS, 1))).sum(1) * 5
out["P3_plan_extra"] = dict(
    day_purch_plan=float(plan3[31:].sum()), day_purch_adj=float(adj3[31:].sum()),
    fee_plan=float((plan3[31:] * d["price_day"]).sum()),
    fee_adj=float((((0.5 * np.maximum(p3["plan"] - p3["adj"], 0) + 1.5 * np.maximum(p3["adj"] - p3["plan"], 0))
                   * DT * np.tile(d["price_day"], (DAYS, 1)))[31:]).sum()),
    fee_emg=float(((p3["emg"] * DT * np.tile(d["price_day"], (DAYS, 1)))[31:]).sum() * 5),
)
day_tables(plan3, adj3, u3, v3, emg3, np.full(DAYS, E_INIT), np.full(DAYS, E_INIT), fee3, "P3")

# ---------------------------------------------------------------- 问题4-2 / 4-3
p4 = np.load(RES / "p4_solution.npz")
g42 = p4["g42"].reshape(DAYS, N_T); u42 = p4["u42"].reshape(DAYS, N_T)
v42 = p4["v42"].reshape(DAYS, N_T); Es42 = p4["Es42"]; Ee42 = p4["Ee42"]
fee42 = p4["fee42"]
c4 = d["C4"]
day_tables(g42 * DT, None, u42, v42, np.zeros((DAYS, N_T)), Es42, Ee42, fee42, "P42")

plan43 = p4["plan43"]; adj43 = p4["adj43"]
u43 = p4["u43"]; v43 = p4["v43"]; emg43 = p4["emg43"]
fee43 = (plan43 * DT * c4).sum(1) \
      + ((0.5 * np.maximum(plan43 - adj43, 0) + 1.5 * np.maximum(adj43 - plan43, 0)) * DT * c4).sum(1) \
      + (emg43 * DT * c4).sum(1) * 5
day_tables(plan43 * DT, adj43 * DT, u43, v43, emg43 * DT,
           np.full(DAYS, E_INIT), np.full(DAYS, E_INIT), fee43, "P43")

# ---------------------------------------------------------------- 汇总层
out["_meta"] = dict(
    t1_slots=[f"{h:02d}:{m:02d}-{h:02d}:{m+10:02d}" for h, m in T1_SLOTS],
    t2_slots=[f"{a//6:02d}:00-{b//6:02d}:00" for a, b in T2_EDGES],
    spec_dates=SPEC_DATES,
    I0=31,
)

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] 已写出 {OUT}")

# ---------------- 控制台速览 ----------------
for tag in ("P1", "P2", "P3", "P42", "P43"):
    print(f"\n===== {tag} =====")
    if tag == "P1":
        o = out["P1"]
        print("  表1 指定时段购电量(kWh):", " ".join(f"{v:9.3f}" for v in o["t1"]))
        print(f"       全天购电量 {o['day_purch']:.3f} kWh   全天购电费 {o['day_fee']:.3f} 元")
        print("  表2 充电量:", " ".join(f"{v:9.2f}" for v in o["t2_chg"]))
        print("       放电量:", " ".join(f"{v:9.2f}" for v in o["t2_dis"]))
        print(f"       0:00/24:00 储电量 {o['E0']:.1f} / {o['E24']:.1f} kWh")
    else:
        for k in ("0320", "0621", "0923", "1221"):
            o = out[tag][k]
            print(f"  {o['date']}  全天购电量 {o['day_purch']:11.2f} kWh   "
                  f"全天购电费 {o['day_fee']:10.2f} 元   紧急购电 {o['emg']:9.2f} kWh")
            print(f"        表1:" + " ".join(f"{v:8.2f}" for v in o["t1"]))
            print(f"        表2 充:" + " ".join(f"{v:8.1f}" for v in o["t2_chg"])
                  + "  放:" + " ".join(f"{v:8.1f}" for v in o["t2_dis"])
                  + f"   E {o['E0']:.0f}/{o['E24']:.0f}")
