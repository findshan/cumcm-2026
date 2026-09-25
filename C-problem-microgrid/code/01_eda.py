"""
CUMCM 2026 C题  数据探查脚本 (EDA)
目标：摸清附件1~4的规模、量纲、统计特征、预报误差结构
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "附件"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# ---------------- 附件1：典型日的电价 / 负载 / 光伏预测 ----------------
a1 = pd.read_excel(DATA / "附件1.xlsx")
a1.columns = ["t", "price", "load", "pv_fc"]
print("=" * 70)
print("[附件1] shape =", a1.shape)
print("  时间标签：首 =", a1["t"].iloc[0], " 末 =", a1["t"].iloc[-1])
print("  电价   min/max/mean = %.4f / %.4f / %.4f  元/kWh"
      % (a1.price.min(), a1.price.max(), a1.price.mean()))
print("  负载   min/max/mean = %.1f / %.1f / %.1f  kW"
      % (a1.load.min(), a1.load.max(), a1.load.mean()))
print("  光伏预测 min/max/mean = %.1f / %.1f / %.1f  kW"
      % (a1.pv_fc.min(), a1.pv_fc.max(), a1.pv_fc.mean()))
print("  光伏预测日发电量 = %.1f kWh" % (a1.pv_fc.sum() / 6))
print("  负载日用电量     = %.1f kWh" % (a1.load.sum() / 6))

# 电价的峰谷归属（取每小时的均值）
a1["h"] = [i // 6 for i in range(len(a1))]
hp = a1.groupby("h")[["price", "load", "pv_fc"]].mean()
print("\n  逐小时均价（元/kWh），标出峰谷：")
pk = hp.price.nlargest(4).index.tolist()
vl = hp.price.nsmallest(4).index.tolist()
print("    最高价4小时 =", sorted(pk), " 均价", round(hp.price[pk].mean(), 4))
print("    最低价4小时 =", sorted(vl), " 均价", round(hp.price[vl].mean(), 4))
print("    峰谷比 = %.3f" % (hp.price[pk].mean() / hp.price[vl].mean()))
eta = 0.9
print("    套利阈值 c_high/c_low > 1/eta^2 = %.3f  →  是否值得满充满放: %s"
      % (1 / eta ** 2, hp.price[pk].mean() / hp.price[vl].mean() > 1 / eta ** 2))

# 净负荷（负载 - 光伏）
a1["net"] = a1.load - a1.pv_fc
print("\n  净负荷(负载-光伏) min = %.1f kW, max = %.1f kW" % (a1.net.min(), a1.net.max()))
print("  光伏>负载的时段数 = %d / %d" % ((a1.net < 0).sum(), len(a1)))
print("  缺口(净负荷)日总量 = %.1f kWh；光伏日发电量 %.1f kWh"
      % (a1.net.clip(lower=0).sum() / 6, a1.pv_fc.sum() / 6))

# ---------------- 附件2：全年负载 & 光伏实际 ----------------
print("=" * 70)
x2 = pd.ExcelFile(DATA / "附件2.xlsx")
print("[附件2] sheets =", x2.sheet_names)
load = pd.read_excel(x2, "小区负载")
pv = pd.read_excel(x2, "光伏发电实际功率")
print("  shape =", load.shape, pv.shape)
print("  行首 =", load.iloc[0, 0], " 行末 =", load.iloc[-1, 0])
print("  列标签前5 =", list(load.columns[:5]))
print("  列标签后3 =", list(load.columns[-3:]))
L = load.iloc[:, 1:].to_numpy(float)
P = pv.iloc[:, 1:].to_numpy(float)
print("  负载 min/max/mean = %.1f / %.1f / %.1f kW" % (L.min(), L.max(), L.mean()))
print("  光伏 min/max/mean = %.1f / %.1f / %.1f kW" % (P.min(), P.max(), P.mean()))
print("  负载全年用电 = %.0f kWh" % (L.sum() / 6))
print("  光伏全年发电 = %.0f kWh" % (P.sum() / 6))
print("  全年净负荷 = %.0f kWh  → 缺口率 %.1f%%"
      % ((L - P).sum() / 6, (L - P).sum() / L.sum() * 100))
print("  单日光伏发电量 min/max = %.0f / %.0f kWh" % (P.sum(1).min() / 6, P.sum(1).max() / 6))

# ---------------- 附件3：光伏预报 ----------------
print("=" * 70)
f3 = pd.read_excel(DATA / "附件3.xlsx")
print("[附件3] shape =", f3.shape)
print("  列 =", list(f3.columns[:4]), "...", list(f3.columns[-2:]))
print("  预报时刻唯一值 =", sorted(f3.iloc[:, 1].unique()))
print("  天数 =", f3.iloc[:, 0].nunique(), "  总行数 =", len(f3))

# ---------------- 附件4：实时电价 ----------------
print("=" * 70)
p4 = pd.read_excel(DATA / "附件4.xlsx")
P4 = p4.iloc[:, 1:].to_numpy(float)
print("[附件4] shape =", p4.shape)
print("  电价 min/max/mean = %.4f / %.4f / %.4f 元/kWh" % (P4.min(), P4.max(), P4.mean()))
print("  电价 std = %.4f ；分位数 5%%/50%%/95%% = %.4f / %.4f / %.4f"
      % (P4.std(), np.percentile(P4, 5), np.percentile(P4, 50), np.percentile(P4, 95)))
daily_ratio = (P4.max(1) / np.maximum(P4.min(1), 1e-9))
print("  日内峰谷比 中位数 = %.3f ；>1/eta^2(1.234) 的天数 = %d / %d"
      % (np.median(daily_ratio), (daily_ratio > 1 / eta ** 2).sum(), len(daily_ratio)))
