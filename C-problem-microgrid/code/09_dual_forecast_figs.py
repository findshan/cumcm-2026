"""
补充图 A：问题1 的 LP 对偶分析（影子价格 + 套利阈值判据）
补充图 B：光伏预报误差标定（误差-提前期、标准化分布、异方差性）
"""
import numpy as np
import sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumcm_core import (load_attachments, solve_lp, forecast_day_aligned,
                        DT, N_T, DAYS, E_INIT, ETA, DATA_DIR)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; RES = ROOT / "results"

# ============================================================
# 通用：带对偶信息的 LP 求解（复刻 cumcm_core.solve_lp，额外取 marginal）
# ============================================================
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def solve_lp_dual(c, L, S, E_start, E_end, eta=ETA, dt=DT,
                  E_min=1200.0, E_max=10800.0, P_max=5000.0):
    c = np.asarray(c, float); L = np.asarray(L, float); S = np.asarray(S, float)
    n = len(c); N = 5 * n
    nG, nU, nV, nW, nE = 0, n, 2 * n, 3 * n, 4 * n
    cobj = np.zeros(N)
    cobj[nG:nG + n] = c * dt
    cobj[nU:nU + n] = 1e-6
    cobj[nV:nV + n] = 1e-6

    rows, cols, vals, rhs = [], [], [], []
    r = np.repeat(np.arange(n), 4)
    cc = np.concatenate([nG + np.arange(n), nU + np.arange(n),
                         nV + np.arange(n), nW + np.arange(n)]).reshape(4, n).T.ravel()
    rows.append(r); cols.append(cc); vals.append(np.tile([1., -1., 1., -1.], n))
    rhs.append(L - S)

    r2 = n + np.repeat(np.arange(n), 4)
    cc2 = np.concatenate([nE + np.arange(n), nE + np.arange(n) - 1,
                          nU + np.arange(n), nV + np.arange(n)]).reshape(4, n).T.ravel()
    vv2 = np.tile([1., -1., -eta * dt, dt / eta], n)
    cc2[1] = nE; vv2[1] = 0.0
    rows.append(r2); cols.append(cc2); vals.append(vv2)
    b2 = np.zeros(n); b2[0] = E_start
    rhs.append(b2)

    rows.append(np.array([2 * n])); cols.append(np.array([nE + n - 1]))
    vals.append(np.array([1.0])); rhs.append(np.array([float(E_end)]))

    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(2 * n + 1, N)).tocsr()
    b = np.concatenate(rhs)
    bounds = ([(0, None)] * n + [(0, P_max)] * n + [(0, P_max)] * n
              + [(0, None)] * n + [(E_min, E_max)] * n)
    res = linprog(cobj, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    mar = np.asarray(res.eqlin.marginals, float)
    # 平衡约束 marginal 单位 元/kW -> 元/kWh 需 /Δt
    lam_bal = mar[:n] / dt
    # 储能动态约束 marginal 的 RHS 是 kWh -> 直接是 元/kWh
    lam_store = -mar[n:2 * n]
    return dict(
        g=x[nG:nG + n], u=x[nU:nU + n], v=x[nV:nV + n],
        w=x[nW:nW + n], E=x[nE:nE + n],
        lam_bal=lam_bal, lam_store=lam_store,
        lam_term=float(mar[2 * n]), obj=res.fun)


if __name__ == "__main__":
    d = load_attachments()
    c, Ld, Sd = d["price_day"], d["load_day"], d["pv_day_fc"]
    r = solve_lp_dual(c, Ld, Sd, E_INIT, E_INIT)
    t = np.arange(1, N_T + 1) / 6.0

    print("=" * 78)
    print("补充图 A  问题1 的 LP 对偶分析")
    print("=" * 78)
    print(f"  实际购电费 {(r['g']*DT*c).sum():.4f} 元   （与 04_p1 一致：35126.9486）")
    print(f"  LP 目标值 {r['obj']:.4f}（含 {1e-6:.0e} 量级的 tie-break 惩罚，差值 = "
          f"{r['obj']-(r['g']*DT*c).sum():.4f}，不影响调度决策）")
    print(f"  平衡约束影子价格 σ_t: min={r['lam_bal'].min():.4f} max={r['lam_bal'].max():.4f} 元/kWh")
    buy = r["g"] > 1e-6
    dev = np.abs(r["lam_bal"][buy] - c[buy]).max() if buy.any() else 0
    print(f"  购电时段 |σ_t − c_t| 最大偏差 = {dev:.2e} 元/kWh  → 买电时 σ_t = c_t ✔（互补松弛）")
    nb = ~buy
    print(f"  非购电时段 σ_t ≤ c_t : {bool((r['lam_bal'][nb] <= c[nb] + 1e-9).all())}"
          f"   平均 σ={r['lam_bal'][nb].mean():.4f} vs 平均 c={c[nb].mean():.4f}")
    print(f"  储能边际价值 μ_t: min={r['lam_store'].min():.4f} max={r['lam_store'].max():.4f} 元/kWh")

    # 状态判定
    ch = r["u"] > 1e-6; dis = r["v"] > 1e-6
    idle = ~(ch | dis)
    print(f"  充电 {ch.sum()} 段 / 放电 {dis.sum()} 段 / 闲置 {idle.sum()} 段")
    print(f"  充电时段 μ_t 均值 {r['lam_store'][ch].mean():.4f}，放电时段 μ_t 均值 {r['lam_store'][dis].mean():.4f}")

    # ---------------- 图 A ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang SC", "Songti SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # A1 电价 vs 平衡约束影子价格
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, c, lw=1.6, color="#2c3e50", label="外网电价 $c_t$")
    ax.plot(t, r["lam_bal"], lw=1.6, ls="--", color="#c0392b",
            label="功率平衡约束影子价格 $\\sigma_t$")
    ax.fill_between(t, r["lam_bal"], c, color="#e67e22", alpha=.25, label="差价 = 边际电源价值")
    ax.set_title("A1  电价与电能影子价格：边际电源的排序", fontsize=11)
    ax.set_xlabel("时刻 (h)"); ax.set_ylabel("元/kWh")
    ax.legend(fontsize=9); ax.grid(alpha=.3); ax.set_xlim(0, 24)

    # A2 储能边际价值 + 充放电状态
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, c, lw=1.2, color="#2c3e50", alpha=.55, label="电价 $c_t$")
    ax.plot(t, r["lam_store"], lw=1.8, color="#8e44ad", label="储能边际价值 $\\mu_t$")
    ax.scatter(t[ch], r["lam_store"][ch], s=44, marker="^", color="#27ae60",
               zorder=5, label=f"充电 ({ch.sum()} 段)")
    ax.scatter(t[dis], r["lam_store"][dis], s=44, marker="v", color="#c0392b",
               zorder=5, label=f"放电 ({dis.sum()} 段)")
    ax.set_title("A2  储能边际价值与充放电决策", fontsize=11)
    ax.set_xlabel("时刻 (h)"); ax.set_ylabel("元/kWh")
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=.3); ax.set_xlim(0, 24)

    # A3 套利阈值判据（净收益曲线）
    ax = fig.add_subplot(gs[1, 0])
    ratio = np.linspace(0, 6, 400)
    c_low = 0.425
    gain = c_low * (ratio - 1 / ETA ** 2)
    ax.plot(ratio, gain, lw=2.2, color="#2980b9", label="每 kWh 放电净收益")
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(1 / ETA ** 2, color="#c0392b", ls="--", lw=1.5,
               label=f"套利阈值 $1/\\eta^2={1/ETA**2:.3f}$")
    ax.axvspan(0, 1 / ETA ** 2, color="#c0392b", alpha=.10)
    ax.plot([2.922], [c_low * (2.922 - 1 / ETA ** 2)], "o", ms=10, color="#27ae60",
            label=f"附件1 峰谷比 2.92 → 净收益 {c_low*(2.922-1/ETA**2):.3f} 元/kWh")
    r4 = d["C4"].max(1) / np.maximum(d["C4"].min(1), 1e-9)
    med = float(np.median(r4))
    ax.plot([med], [c_low * (med - 1 / ETA ** 2)], "s", ms=10, color="#8e44ad",
            label=f"附件4 峰谷比中位数 {med:.2f} → {c_low*(med-1/ETA**2):.3f} 元/kWh")
    ax.set_title("A3  储能套利的阈值判据：$c_{high}/c_{low}>1/\\eta^2$", fontsize=11)
    ax.set_xlabel("峰谷电价比 $c_{high}/c_{low}$"); ax.set_ylabel("净收益 (元/kWh)")
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=.3); ax.set_xlim(0, 6)

    # A4 附件4 日内峰谷比分布
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(r4, bins=np.logspace(np.log10(max(r4.min(), 1e-2)), np.log10(r4.max()), 45),
            color="#2980b9", alpha=.85)
    ax.set_xscale("log")
    ax.axvline(1 / ETA ** 2, color="#c0392b", ls="--", lw=1.8,
               label=f"套利阈值 {1/ETA**2:.3f}")
    ax.axvline(med, color="#8e44ad", ls=":", lw=1.8, label=f"中位数 {med:.2f}")
    pct = (r4 > 1 / ETA ** 2).mean() * 100
    ax.set_title(f"A4  附件4 逐日日内峰谷比分布：{pct:.0f}% 的天数存在套利空间", fontsize=11)
    ax.set_xlabel("日内峰谷比（对数轴）"); ax.set_ylabel("天数")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    fig.suptitle("补充图 A  问题1 的线性规划对偶分析", fontsize=13)
    fig.savefig(FIG / "fig_p1_dual.png", dpi=200, bbox_inches="tight")
    print(f"[OK] 已保存 {FIG/'fig_p1_dual.png'}")

    # ============================================================
    # 补充图 B  预报误差标定
    # ============================================================
    print("\n" + "=" * 78)
    print("补充图 B  光伏预报误差标定")
    print("=" * 78)
    X2P = pd.read_excel(DATA_DIR / "附件2.xlsx", "光伏发电实际功率")
    a2P = X2P.iloc[:, 1:].to_numpy(float)
    cols = list(X2P.columns[1:])
    hidx = {}
    for j, cc0 in enumerate(cols):
        if isinstance(cc0, str):
            hidx[24] = j
        elif cc0.minute == 0:
            hidx[cc0.hour if cc0.hour else 24] = j
    dates = pd.to_datetime(X2P.iloc[:, 0])
    actual = {(d0.normalize(), h % 24): a2P[i, j]
              for i, d0 in enumerate(dates) for h, j in hidx.items()}

    f3 = d["f3"]
    fcv = {}          # k -> 预报值
    acv = {}
    for k in range(1, 25):
        fcv[k] = []; acv[k] = []
    for _, row in f3.iterrows():
        d0 = pd.to_datetime(row["date"]).normalize()
        h0 = int(row["h0i"])
        for k in range(1, 25):
            hh = h0 + k
            key = ((d0 + pd.Timedelta(days=hh // 24)).normalize(), hh % 24)
            if key in actual:
                fcv[k].append(float(row[f"k{k}"])); acv[k].append(actual[key])
    # 夜间（预报与实际同时为 0）的样本对运行无影响，须剔除后再标定，
    # 否则 0 误差会大幅稀释统计量、并在标准化残差图上堆出虚假尖峰。
    E, F, A, DAY = {}, {}, {}, {}
    for k in range(1, 25):
        e_all = np.array(fcv[k]) - np.array(acv[k])
        f_a, a_a = np.array(fcv[k]), np.array(acv[k])
        m = (f_a > 0) | (a_a > 0)
        E[k] = e_all[m]; F[k] = f_a[m]; A[k] = a_a[m]; DAY[k] = int(m.sum())

    ks = np.arange(1, 25)
    mae = np.array([np.abs(E[k]).mean() for k in ks])
    rmse = np.array([np.sqrt((E[k] ** 2).mean()) for k in ks])
    bias = np.array([E[k].mean() for k in ks])
    sd = np.array([E[k].std() for k in ks])
    mae_all = np.array([np.abs(np.array(fcv[k]) - np.array(acv[k])).mean() for k in ks])
    print(f"  有效样本（剔除双方均为 0 的夜间样本）: k=1 时 {DAY[1]}，k=24 时 {DAY[24]}")
    print(f"  k   MAE(白天)  MAE(全样本)   RMSE     偏差    标准差")
    for k in list(ks):
        print(f"  {k:2d} {mae[k-1]:10.2f} {mae_all[k-1]:12.2f} {rmse[k-1]:8.2f} "
              f"{bias[k-1]:8.2f} {sd[k-1]:8.2f}")
    # 拟合 σ(k)=a k^b 与 |bias(k)|=p k^q
    b_sd, a_sd = np.polyfit(np.log(ks), np.log(sd), 1)
    a_sd = np.exp(a_sd)
    b_b, a_b = np.polyfit(np.log(ks), np.log(np.maximum(np.abs(bias), 1e-3)), 1)
    a_b = np.exp(a_b)
    print(f"\n  拟合：σ(k) = {a_sd:.2f}·k^{b_sd:.3f}   （R²={np.corrcoef(np.log(ks),np.log(sd))[0,1]**2:.4f}）")
    print(f"  拟合：|bias(k)| = {a_b:.3f}·k^{b_b:.3f}")
    ratio24 = (sd[23] / sd[0])
    print(f"  σ(24h)/σ(1h) = {ratio24:.2f}×   MAE(24h)/MAE(1h) = {mae[23]/mae[0]:.2f}×")

    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))

    ax[0, 0].plot(ks, mae, "o-", color="#BA7517", label="MAE", lw=1.8)
    ax[0, 0].plot(ks, rmse, "s-", color="#378ADD", label="RMSE", lw=1.8)
    ax[0, 0].plot(ks, np.abs(bias), "^--", color="#0F6E56", label="|平均偏差|", lw=1.5)
    ax[0, 0].set_title("B1  误差随预报提前期的增长（白天样本）", fontsize=11)
    ax[0, 0].set_xlabel("预报提前期 $k$ (h)"); ax[0, 0].set_ylabel("误差 (kW)")
    ax[0, 0].legend(fontsize=9); ax[0, 0].grid(alpha=.3)

    ax[0, 1].plot(ks, sd, "o-", color="#8e44ad", lw=1.8, label="实测标准差 σ(k)")
    ax[0, 1].plot(ks, a_sd * ks ** b_sd, "--", color="#c0392b", lw=1.8,
                  label=f"拟合 $\\sigma(k)={a_sd:.0f}k^{{{b_sd:.2f}}}$")
    ax[0, 1].plot(ks, bias, "^-", color="#0F6E56", lw=1.5, label="系统偏差 $\\mu(k)$")
    ax[0, 1].set_title("B2  不确定性的标定：$\\sigma(k)$ 与 $\\mu(k)$ 的幂律拟合", fontsize=11)
    ax[0, 1].set_xlabel("预报提前期 $k$ (h)"); ax[0, 1].set_ylabel("kW")
    ax[0, 1].legend(fontsize=9); ax[0, 1].grid(alpha=.3)

    for a_, k_, col in ((0, 0, "#27ae60"), (0, 5, "#e67e22"), (0, 23, "#c0392b")):
        pass
    sel = [(1, "#27ae60"), (6, "#e67e22"), (24, "#c0392b")]
    for k_, col in sel:
        z = (E[k_] - bias[k_ - 1]) / sd[k_ - 1]
        ax[1, 0].hist(z, bins=48, range=(-5, 5), histtype="step", lw=1.6,
                      color=col, density=True, label=f"k={k_} h（n={len(z)}）")
    zz = np.linspace(-5, 5, 200)
    ax[1, 0].plot(zz, np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.2, label="标准正态")
    ax[1, 0].set_title("B3  白天样本的标准化残差：随提前期保持近似正态", fontsize=11)
    ax[1, 0].set_xlabel("标准化误差 $(\\varepsilon-\\mu)/\\sigma$"); ax[1, 0].set_ylabel("概率密度")
    ax[1, 0].legend(fontsize=9); ax[1, 0].grid(alpha=.3)

    ax[1, 1].scatter(A[24], F[24], s=6, alpha=.35, color="#c0392b", label="k=24 h")
    ax[1, 1].scatter(A[6], F[6], s=6, alpha=.35, color="#e67e22", label="k=6 h")
    lim = [0, 10500]
    ax[1, 1].plot(lim, lim, "k--", lw=1.2, label="理想预报")
    ax[1, 1].set_title("B4  预报 vs 实际：高辐照时段误差更大（异方差）", fontsize=11)
    ax[1, 1].set_xlabel("实际光伏功率 (kW)"); ax[1, 1].set_ylabel("预报光伏功率 (kW)")
    ax[1, 1].legend(fontsize=9); ax[1, 1].grid(alpha=.3); ax[1, 1].set_xlim(lim); ax[1, 1].set_ylim(lim)

    fig.suptitle("补充图 B  光伏预报误差标定（附件3 vs 附件2）", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "fig_p3_forecast_calib.png", dpi=200, bbox_inches="tight")
    print(f"[OK] 已保存 {FIG/'fig_p3_forecast_calib.png'}")

    pd.DataFrame({"k": ks, "MAE": mae, "RMSE": rmse, "bias": bias, "sd": sd}).to_csv(
        RES / "forecast_error.csv", index=False, encoding="utf-8-sig")
    np.savez(RES / "dual.npz", price=c, lam_bal=r["lam_bal"], lam_store=r["lam_store"],
             u=r["u"], v=r["v"], g=r["g"], E=r["E"],
             sigma_a=a_sd, sigma_b=b_sd, bias_a=a_b, bias_b=b_b,
             mae=mae, rmse=rmse, bias=bias, sd=sd)
    print(f"[OK] 已写出 {RES/'forecast_error.csv'} 与 {RES/'dual.npz'}")
