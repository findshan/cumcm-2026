"""
CUMCM 2026 C题  核心模块
- 附件数据加载（含附件3 合并单元格、'0:00+1' 标签处理）
- 光伏预报降尺度（整点 24 点 -> 10 分钟 144 点）
- 储能-购电单日/多日线性规划求解器（scipy HiGHS）

时间约定（全模块统一）：
  一天 = 144 个 10 分钟时段，时段 k (k=1..144) 覆盖 ((k-1)*Δt, k*Δt]
  附件1/2/4 的时间标签 = 时段右端点；附件1 末行 '0:00+1' 即 24:00
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from scipy.interpolate import PchipInterpolator

# ---------------- 物理常数（附录1） ----------------
DT      = 1.0 / 6.0      # 时段长度 (h)
ETA     = 0.9            # 充放电效率（口径A：充0.9 且 放0.9 -> 往返0.81）
E_MIN   = 1200.0         # 储电量下限 (kWh)
E_MAX   = 10800.0        # 储电量上限 (kWh)
E_CAP   = 12000.0        # 最大容量
P_MAX   = 5000.0         # 最大充放电功率 (kW)
E_INIT  = 6000.0         # 2025-01-01 00:00 储电量 (kWh)
N_T     = 144            # 每天时段数
DAYS    = 365
PD_YEAR = 365 * N_T      # 全年时段数

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "附件"


# ============================================================
# 1. 数据加载
# ============================================================
def load_attachments(data_dir: Path | str = DATA_DIR) -> dict:
    """加载附件1~4，返回统一结构的字典。"""
    d = Path(data_dir)

    # 附件1：典型日（= 全年平均日）
    a1 = pd.read_excel(d / "附件1.xlsx")
    a1.columns = ["t", "price", "load", "pv_fc"]

    # 附件2：全年负载 / 光伏实际
    a2L = pd.read_excel(d / "附件2.xlsx", "小区负载")
    a2P = pd.read_excel(d / "附件2.xlsx", "光伏发电实际功率")
    dates = pd.to_datetime(a2P.iloc[:, 0]).dt.normalize()
    L = a2L.iloc[:, 1:].to_numpy(float)          # (365, 144) kW
    S = a2P.iloc[:, 1:].to_numpy(float)          # (365, 144) kW

    # 附件3：光伏预报（日期列为合并单元格，需 ffill）
    f3 = pd.read_excel(d / "附件3.xlsx")
    f3.columns = ["date", "h0"] + [f"k{k}" for k in range(1, 25)]
    f3["date"] = f3["date"].ffill()
    f3 = f3[f3["date"].notna()].copy()
    f3["date"] = pd.to_datetime(f3["date"]).dt.normalize()
    f3["h0i"] = f3["h0"].astype(str).str.split(":").str[0].astype(int)

    # 附件4：全年实时电价
    a4 = pd.read_excel(d / "附件4.xlsx")
    C4 = a4.iloc[:, 1:].to_numpy(float)          # (365, 144) 元/kWh

    return dict(
        a1=a1,
        price_day=a1["price"].to_numpy(float),   # (144,)
        load_day=a1["load"].to_numpy(float),
        pv_day_fc=a1["pv_fc"].to_numpy(float),
        dates=dates, L=L, S=S, C4=C4, f3=f3,
    )


def forecast_hourly(data: dict, day: int, h0: int) -> np.ndarray | None:
    """取第 day 天、h0 时发布的 24 小时整点预报值（长度 24）。"""
    f3 = data["f3"]
    m = (f3["date"] == data["dates"][day]) & (f3["h0i"] == h0)
    if not m.any():
        return None
    return f3.loc[m, [f"k{k}" for k in range(1, 25)]].to_numpy(float)[0]


def forecast_10min(data: dict, day: int, h0: int) -> np.ndarray:
    """
    把 h0 时发布的整点预报降尺度到 10 分钟粒度的 144 个时段。

    方法：PCHIP 保形三次插值。
    控制点 x = [h0, h0+1, ..., h0+24]，对应
      y = [anchor, f1, f2, ..., f24]
    其中 anchor 为 h0 时刻的光伏功率估计：优先取当天更早一次预报对 h0 的预测
    （h0 时已可获得，符合信息约束），h0=0 时取 0（夜间）。
    """
    vals = forecast_hourly(data, day, h0)
    if vals is None:
        return np.zeros(N_T)
    if h0 == 0:
        anchor = 0.0
    else:                                   # 用更早一次预报（提前 h0 小时）对 h0 的预测作锚
        prev = forecast_hourly(data, day, 0)
        anchor = float(prev[h0 - 1]) if prev is not None and h0 <= 24 else 0.0
    xs = np.arange(h0, h0 + 25, dtype=float)
    ys = np.concatenate([[anchor], vals])
    y_uni = np.maximum.accumulate(np.concatenate([[0.0], np.maximum(ys, 0.0)]))[1:]  # 防异常
    p = PchipInterpolator(xs, ys)
    tau = np.arange(1, N_T + 1) / 6.0        # 各时段右端点距 h0 的小时数
    fc = p(h0 + np.clip(tau, 0.0, 24.0))
    return np.maximum(fc, 0.0)


def forecast_day_aligned(data: dict, day: int, h0: int) -> np.ndarray:
    """
    返回"与当天 144 个时段对齐"的预报数组（长度 144）。

    forecast_10min 的输出是按"距发布时刻 h0 的小时数"索引的：
        fc[k]  对应 h0 之后 (k+1)/6 小时
    而当天时段 i (0-based) 覆盖 ((i)*10, (i+1)*10] 分钟，即 h0 之后 (i+1)/6 小时。
    因此仅当 (i+1)/6 > h0 时该时段在发布时刻之后，对应索引 i − 6·h0。

    对 h0 = 0 二者重合；对 h0 = 6/12/18 必须做这个移位，
    否则 6:00/12:00/18:00 的子问题会误用"次日"的预报（本模块早期版本的严重错误）。
    """
    fc = forecast_10min(data, day, h0)
    out = np.zeros(N_T)
    shift = 6 * h0                      # 已过去/正在进行的时段数
    if shift < N_T:
        out[shift:] = fc[:N_T - shift]
    return out


# ============================================================
# 2. LP 求解器
# ============================================================
def solve_lp(c, L, S, E_start,
             eta=ETA, dt=DT, E_min=E_MIN, E_max=E_MAX,
             P_max=P_MAX, E_end=None, E_end_min=None,
             terminal_value=0.0, tie_break=1e-6):
    """
    求解一个时段序列上的微网购电-储能优化 LP。

        min  Σ c_t g_t Δt  − terminal_value · E_last
        s.t. S_t + g_t + v_t − u_t − w_t = L_t
             E_t − E_{t−1} − η u_t Δt + (v_t Δt)/η = 0,  E_{−1} = E_start
             E_min ≤ E_t ≤ E_max,  0 ≤ u_t, v_t ≤ P_max
             g_t, w_t ≥ 0
    可选：E_end（终端等式）/ E_end_min（终端下界）/ terminal_value（终端储能价值）

    返回 dict(g,u,v,w,E) —— 各为长度 n 的数组（E_t 为时段 t 末储电量）
    """
    c = np.asarray(c, float); L = np.asarray(L, float); S = np.asarray(S, float)
    n = len(c)
    if not (len(L) == len(S) == n):
        raise ValueError("c/L/S 长度不一致")

    nG, nU, nV, nW, nE = 0, n, 2 * n, 3 * n, 4 * n
    N = 5 * n
    cobj = np.zeros(N)
    cobj[nG:nG + n] = c * dt
    cobj[nU:nU + n] = tie_break
    cobj[nV:nV + n] = tie_break
    if terminal_value:
        cobj[nE + n - 1] -= terminal_value

    # ---- 等式约束 ----
    # 注意：列索引必须按行"交错"排布（reshape(4,n).T.ravel()），
    #       否则 concat 得到的是分块顺序，与行的 repeat 模式对不齐，约束矩阵会整体错位。
    rows, cols, vals, rhs = [], [], [], []

    # (a) 功率平衡： g_i − u_i + v_i − w_i = L_i − S_i
    r = np.repeat(np.arange(n), 4)
    cc = np.concatenate([nG + np.arange(n), nU + np.arange(n),
                         nV + np.arange(n), nW + np.arange(n)]).reshape(4, n).T.ravel()
    vv = np.tile([1.0, -1.0, 1.0, -1.0], n)
    rows.append(r); cols.append(cc); vals.append(vv)
    rhs.append(L - S)

    # (b) 储能动态： E_i − E_{i−1} − ηΔt u_i + (Δt/η) v_i = 0 （i=0 时 E_{−1}=E_start 移项）
    r2 = n + np.repeat(np.arange(n), 4)
    cc2 = np.concatenate([nE + np.arange(n), nE + np.arange(n) - 1,
                          nU + np.arange(n), nV + np.arange(n)]).reshape(4, n).T.ravel()
    vv2 = np.tile([1.0, -1.0, -eta * dt, dt / eta], n)
    cc2[1] = nE                              # 行0的第2项 = i=0 的越界项 E_{−1}
    vv2[1] = 0.0                             # 系数置零，E_start 移到右端
    rows.append(r2); cols.append(cc2); vals.append(vv2)
    b2 = np.zeros(n); b2[0] = E_start
    rhs.append(b2)

    n_eq = 2 * n

    # ---- 终端条件 ----
    A_ub = A_eq = None
    b_ub = b_eq = None
    if E_end is not None:
        rows.append(np.array([2 * n]))
        cols.append(np.array([nE + n - 1]))
        vals.append(np.array([1.0]))
        rhs.append(np.array([float(E_end)]))
        n_eq += 1

    A = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n_eq, N)).tocsr()
    b = np.concatenate(rhs)

    bounds = ([(0, None)] * n + [(0, P_max)] * n + [(0, P_max)] * n
              + [(0, None)] * n + [(E_min, E_max)] * n)

    if E_end_min is not None:
        # 终端下界 E_{n-1} ≥ E_end_min  <=>  −E_{n-1} ≤ −E_end_min
        Au = coo_matrix((np.array([-1.0]), (np.array([0]), np.array([nE + n - 1]))),
                        shape=(1, N)).tocsr()
        res = linprog(cobj, A_ub=Au, b_ub=np.array([-float(E_end_min)]),
                      A_eq=A, b_eq=b, bounds=bounds, method="highs")
    else:
        res = linprog(cobj, A_eq=A, b_eq=b, bounds=bounds, method="highs")

    if not res.success:
        raise RuntimeError(f"LP 无解: {res.message}")

    x = res.x
    g_, u_, v_, w_, E_ = (x[nG:nG+n], x[nU:nU+n], x[nV:nV+n], x[nW:nW+n], x[nE:nE+n])

    # ---- 解的自检（防止约束装配错误导致"凭空造电"） ----
    E_rec = E_start + np.cumsum(eta * dt * u_ - dt / eta * v_)
    bal = S + g_ - u_ + v_ - w_ - L
    resid = max(float(np.abs(E_rec - E_).max()), float(np.abs(bal).max()))
    if resid > 1e-4:
        raise RuntimeError(f"约束自检未通过，最大残差 {resid:.3e}（检查约束矩阵装配）")

    return dict(g=g_, u=u_, v=v_, w=w_, E=E_, obj=res.fun,
                status=res.message, resid=resid)


def cost_breakdown(res, c, dt=DT, scale=1.0):
    """购电量/购电费统计（scale: 功率kW -> 电量kWh 的系数，默认 Δt）"""
    purch_kwh = res["g"] * dt
    return dict(
        total_kwh=float(purch_kwh.sum()),
        total_fee=float((purch_kwh * c).sum()),
        charge_kwh=float((res["u"] * dt).sum()),
        discharge_kwh=float((res["v"] * dt).sum()),
        curtail_kwh=float((res["w"] * dt).sum()),
    )


# ============================================================
# 3. 工具函数
# ============================================================
def hhmm(k: int) -> str:
    """第 k 个时段的右端点标签，k=1..144 -> '0:10' ... '24:00'"""
    m = k * 10
    h, mi = divmod(m, 60)
    return f"{h}:{mi:02d}"


def slot_label(k: int) -> str:
    """第 k 个时段标签（与 result 模板同格式），k=1..144 -> '0:00-0:10' ... '23:50-24:00'"""
    def f(m):
        h, mi = divmod(m, 60)
        return f"{h}:{mi:02d}"
    return f"{f((k-1)*10)}-{f(k*10)}"


def result1_slot_label(row_idx: int) -> str:
    """result1 模板第 row_idx 行(0-based, 0..143)的原始时段标签"""
    def f(m):
        h, mi = divmod(m, 60)
        tag = f"{h}:{mi:02d}"
        if m >= 1440:
            tag = f"0:{mi:02d}+1"
        return tag
    a, b = 10 + row_idx * 10, 20 + row_idx * 10
    return f"{f(a)}-{f(b)}"
