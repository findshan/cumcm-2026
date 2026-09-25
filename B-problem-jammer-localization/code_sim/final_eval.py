"""B题 最终评估：问题3（全向）/ 问题4（定向）演练测试 + 鲁棒性 + 误差场敏感性。

产出：
  results/eval_main.csv        主表：定向比例 × n 局
  results/eval_robust.csv      误差场模型 / 相关长度鲁棒性
  results/eval_detail.npz      逐局明细（供论文作图）
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

from jammer_sim import ErrorField, LocalTransport, RobotClient, Simulator, make_case
from strategy import Strategy

# 统一配置：内环 7 站 @1000 m（保证覆盖） + 外环 5 站 @1700 m（捕捉朝内定向源）
# 外环由"定向证据"越闸：全向场景下判据恒为假，自动退化为单环策略。
# 注意 loc_off 必须是本文主配置 "mdd"（中等深度双向探测）；
# "sml" 是基线策略，仅用于 order_ablation.py 的对照，误用会使结果显著变差。
CFG = dict(layout="dual", ring_m=7, ring_a=1000.0, ring_m2=5, ring_a2=1700.0,
           loc_min_dir=1, loc_off="mdd", max_loc=6, use_outer=True)

SEED_BASE = 20260000


def run_one(seed: int, dir_prob: float, err_mode: str = "smooth",
            corr: float = 1500.0, cfg: dict | None = None) -> dict:
    rng = np.random.default_rng(SEED_BASE + seed)
    case = make_case(rng, dir_prob=dir_prob, seed=seed)
    err = ErrorField(np.random.default_rng(5000 + seed), mode=err_mode, corr_len=corr)
    sim = Simulator(case, robot_id="2026123456", err=err)
    cli = RobotClient(LocalTransport(sim), "2026123456")
    st = Strategy(cfg or CFG)
    t0 = time.time()
    res = st.run(cli)
    return dict(seed=seed, N=case.N, n_dir_src=case.n_dir, cleared=res["n_cleared"],
                total=res["virtual_time"], mv=res["mv_total"], n_meas=res["n_meas"],
                n_clear=res["n_clear"], n_loc=res["n_loc"], n_det=res["n_det"],
                stations=res["stations"], has_dir=int(st.has_dir),
                err_mode=err_mode, corr=corr, dir_prob=dir_prob,
                wall=time.time() - t0)


def summarize(rows: list, tag: str) -> dict:
    N = np.array([r["N"] for r in rows], float)
    C = np.array([r["cleared"] for r in rows], float)
    T = np.array([r["total"] for r in rows], float)
    AV = np.array([r["total"] / max(r["cleared"], 1) for r in rows])
    R = C / N
    return dict(tag=tag, n=len(rows), rate=R.mean(), rate_min=R.min(),
                full=int((R > 0.9999).sum()),
                t_mean=T.mean(), t_med=float(np.median(T)), t_min=T.min(), t_max=T.max(),
                avg_mean=AV.mean(), avg_med=float(np.median(AV)),
                mv_mean=np.mean([r["mv"] for r in rows]),
                meas_mean=np.mean([r["n_meas"] for r in rows]),
                clear_mean=np.mean([r["n_clear"] for r in rows]),
                loc_mean=np.mean([r["n_loc"] for r in rows]),
                stations=rows[0]["stations"],
                t_p25=float(np.percentile(T, 25)), t_p75=float(np.percentile(T, 75)),
                t_p90=float(np.percentile(T, 90)))


def show(s: dict):
    print(f"{s['tag']:<34} 清除率={s['rate']*100:6.2f}% (最低{s['rate_min']*100:5.1f}%) "
          f"全清={s['full']:2d}/{s['n']:2d}  总时间 均值{s['t_mean']:6.0f} 中位{s['t_med']:6.0f} "
          f"[{s['t_min']:.0f},{s['t_max']:.0f}]  IQR[{s['t_p25']:.0f},{s['t_p75']:.0f}]  "
          f"平均每源={s['avg_mean']:5.0f}s  移动={s['mv_mean']:5.0f}m 测量={s['meas_mean']:5.0f}")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    os.makedirs("results", exist_ok=True)
    all_rows, main_tab, rob_tab = [], [], []

    print("=" * 132)
    print(f"B题 最终评估（n={N} 局/组）  统一配置：双环 7@1000+5@1700，定向证据越闸")
    print("=" * 132)

    print("\n【表1】定向源比例敏感性（问题 3 → 问题 4）")
    for dp in (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00):
        rows = [run_one(s, dp) for s in range(N)]
        all_rows += rows
        s = summarize(rows, f"定向源比例 {dp*100:4.0f}%")
        show(s)
        s["dir_prob"] = dp
        main_tab.append(s)

    print("\n【表2】误差场模型鲁棒性（定向源比例 35%）")
    for em, cc in (("smooth", 2500.0), ("smooth", 1200.0), ("smooth", 600.0),
                   ("iid", 0.0), ("grid", 300.0), ("grid", 150.0)):
        rows = [run_one(s, 0.35, em, cc) for s in range(N)]
        all_rows += rows
        s = summarize(rows, f"误差场 {em}(corr={cc:.0f})")
        show(s)
        s["dir_prob"] = 0.35
        rob_tab.append(s)

    print("\n【表3】误差场模型鲁棒性（全向，问题 3）")
    for em, cc in (("smooth", 600.0), ("iid", 0.0), ("grid", 200.0)):
        rows = [run_one(s, 0.0, em, cc) for s in range(N)]
        all_rows += rows
        s = summarize(rows, f"[全向] 误差场 {em}(corr={cc:.0f})")
        show(s)
        s["dir_prob"] = 0.0
        rob_tab.append(s)

    with open("results/eval_main.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(main_tab[0].keys()))
        w.writeheader()
        w.writerows(main_tab)
    with open("results/eval_robust.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rob_tab[0].keys()))
        w.writeheader()
        w.writerows(rob_tab)
    np.savez("results/eval_detail.npz",
             **{k: np.array([r[k] for r in all_rows])
                for k in ("seed", "N", "cleared", "total", "mv", "n_meas",
                          "n_clear", "n_loc", "n_det", "dir_prob", "corr")})
    print(f"\n已写出 results/eval_main.csv / eval_robust.csv / eval_detail.npz（共 {len(all_rows)} 局）")
