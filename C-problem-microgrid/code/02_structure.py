"""
CUMCM 2026 C题  深层结构探查 (修正版)
(1) 验证"附件1 = 全年平均日"（附件1 数据只保留4位小数，故比对到 1e-4 量级）
(2) 光伏预报误差结构 —— 问题3 不确定性的物理来源
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "附件"
np.set_printoptions(precision=4, suppress=True)

a1 = pd.read_excel(DATA / "附件1.xlsx")
a1.columns = ["t", "price", "load", "pv"]
X2L = pd.read_excel(DATA / "附件2.xlsx", "小区负载")
X2P = pd.read_excel(DATA / "附件2.xlsx", "光伏发电实际功率")
a2L, a2P = X2L.iloc[:, 1:].to_numpy(float), X2P.iloc[:, 1:].to_numpy(float)
a4 = pd.read_excel(DATA / "附件4.xlsx").iloc[:, 1:].to_numpy(float)
cols = list(X2P.columns[1:])

# ---- 整点列索引：'0:00+1' 为 24:00 ----
hour_idx = {}
for j, c in enumerate(cols):
    if isinstance(c, str):
        hour_idx[24] = j
    elif c.minute == 0:
        hour_idx[c.hour if c.hour else 24] = j
print("整点列数 =", len(hour_idx), " (应为24)")

print("=" * 74)
print("【猜想验证】附件1 是否 = 附件2/附件4 的逐时段均值（保留4位小数）")
for nm, ref, got in [("电价", a4.mean(0), a1.price.to_numpy()),
                     ("负载", a2L.mean(0), a1.load.to_numpy()),
                     ("光伏", a2P.mean(0), a1.pv.to_numpy())]:
    d = np.abs(ref - got)
    bad = np.where(d > 1e-4)[0]
    print(f"  {nm}: max|Δ| = {d.max():.2e}  超差列数 = {len(bad)}/144  "
          f"=> {'✔ 就是平均日' if len(bad) == 0 else '检查 ' + str(bad[:10])}")
    if len(bad):
        print(f"       参考均值 {ref[bad][:6]}  附件1 {got[bad][:6]}")

print("=" * 74)
print("【附件3 预报 vs 附件2 实际】对齐与误差")
f3 = pd.read_excel(DATA / "附件3.xlsx")
f3.columns = ["date", "h0"] + [f"k{k}" for k in range(1, 25)]
f3["date"] = f3["date"].ffill()                       # 合并单元格向下填充
f3 = f3[f3["date"].notna()]
print("  有效行数 =", len(f3), "（365天×4次预报 =", 365 * 4, "）")

dates = pd.to_datetime(X2P.iloc[:, 0])
actual = {}
for i, d in enumerate(dates):
    for h, j in hour_idx.items():
        actual[(d.normalize(), h % 24)] = a2P[i, j]

err = {k: [] for k in range(1, 25)}
for _, r in f3.iterrows():
    d = pd.to_datetime(r["date"]).normalize()
    h0 = int(str(r["h0"]).split(":")[0])
    for k in range(1, 25):
        hh = h0 + k
        key = ((d + pd.Timedelta(days=hh // 24)).normalize(), hh % 24)
        if key in actual:
            err[k].append(float(r[f"k{k}"]) - actual[key])

print("\n  提前期k(h) |  样本 |  平均偏差 |    MAE   |   RMSE   | 归一化MAE(%额定)")
print("  " + "-" * 62)
for k in [1, 2, 3, 6, 9, 12, 18, 24]:
    e = np.array(err[k])
    print(f"     {k:2d}      | {len(e):5d} | {e.mean():9.2f} | {np.abs(e).mean():8.2f} | "
          f"{np.sqrt((e**2).mean()):8.2f} | {np.abs(e).mean()/10216.2*100:6.2f}%")
mae = {k: np.abs(np.array(err[k])).mean() for k in range(1, 25)}
print(f"\n  MAE(1h)={mae[1]:.0f} kW → MAE(24h)={mae[24]:.0f} kW，放大 {mae[24]/mae[1]:.2f}×")

print("=" * 74)
print("【分时刻误差】不同预报时刻发布时，其对当天剩余时段的可用性")
sub = f3.copy()
sub["h0i"] = sub["h0"].str.split(":").str[0].astype(int)
for h0 in [0, 6, 12, 18]:
    e1 = []; e6 = []
    for _, r in sub[sub.h0i == h0].iterrows():
        e1.append(abs(float(r["k1"]) - actual.get((pd.to_datetime(r["date"]).normalize(), (h0 + 1) % 24), np.nan)))
        e6.append(abs(float(r["k6"]) - actual.get((pd.to_datetime(r["date"]).normalize(), (h0 + 6) % 24), np.nan)))
    e1 = np.array(e1, float); e6 = np.array(e6, float)
    print(f"  {h0:02d}:00 发布 → 1h后 MAE={np.nanmean(e1):7.1f}   6h后 MAE={np.nanmean(e6):7.1f}")

print("=" * 74)
prof = a2P.mean(0)
idx = [hour_idx[h] for h in range(1, 25)]
print("【光伏平均出力曲线】(kW, 1..24时)")
print("  ", np.round(prof[idx], 1))
nz = np.where(prof[idx] > 1)[0]
print(f"  日出≈{nz[0]+1:02d}:00  日落≈{nz[-1]+1:02d}:00  正午峰值≈{prof[idx].max():.0f} kW")
print("\n【附件4 电价分时均值】")
print("  ", np.round(a4.mean(0)[idx], 4))
print("  附件4 日内电价序列的变异系数(逐日) = %.3f" % (a4.std(1) / a4.mean(1)).mean())
