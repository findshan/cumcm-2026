"""B题 对照实验：单环 + 单向补测基线。

基线配置 = 本文策略去掉两项 Q4 增量：
  * use_outer=False  —— 不启用外环补扫（只有内环 7 站 @1000 m）
  * loc_off="sml"    —— 主动定位退回"单一方向、质心小偏置"（即 Q2 之前的老做法）

除这两项外，其余参数、随机种子、误差场与 final_eval.py 完全一致，
因此与 eval_main.csv 逐局可比。

产出：
  results/eval_baseline.csv    与 eval_main.csv 同构（含 dir_prob 列）
  results/eval_baseline.npz    逐局明细（逐局可比）
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

from jammer_sim import ErrorField, LocalTransport, RobotClient, Simulator, make_case
from strategy import Strategy

MAIN_CFG = dict(layout="dual", ring_m=7, ring_a=1000.0, ring_m2=5, ring_a2=1700.0,
                loc_min_dir=1, loc_off="mdd", max_loc=6, use_outer=True)
BASE_CFG = dict(MAIN_CFG, loc_off="sml", use_outer=False)

SEED_BASE = 20260000
DPS = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)


def run_one(seed: int, dir_prob: float, cfg: dict, err_mode: str = "smooth",
            corr: float = 1500.0) -> dict:
    rng = np.random.default_rng(SEED_BASE + seed)
    case = make_case(rng, dir_prob=dir_prob, seed=seed)
    err = ErrorField(np.random.default_rng(5000 + seed), mode=err_mode, corr_len=corr)
    sim = Simulator(case, robot_id="2026123456", err=err)
    cli = RobotClient(LocalTransport(sim), "2026123456")
    st = Strategy(cfg)
    res = st.run(cli)
    return dict(seed=seed, N=case.N, n_dir_src=case.n_dir, cleared=res["n_cleared"],
                total=res["virtual_time"], mv=res["mv_total"], n_meas=res["n_meas"],
                n_clear=res["n_clear"], n_loc=res["n_loc"], n_det=res["n_det"],
                stations=res["stations"], has_dir=int(st.has_dir),
                dir_prob=dir_prob, corr=corr, err_mode=err_mode)


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


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    os.makedirs("results", exist_ok=True)
    print("=" * 118)
    print(f"B题 对照实验：单环 + 单向补测基线（n={N} 局/组）  "
          f"use_outer=False, loc_off=sml")
    print("=" * 118)

    all_rows, tab = [], []
    for dp in DPS:
        rows = [run_one(s, dp, BASE_CFG) for s in range(N)]
        all_rows += rows
        s = summarize(rows, f"基线 定向源比例 {dp*100:4.0f}%")
        s["dir_prob"] = dp
        tab.append(s)
        print(f"{s['tag']:<28} 清除率={s['rate']*100:6.2f}% (最低{s['rate_min']*100:5.1f}%) "
              f"全清={s['full']:2d}/{s['n']:2d}  总时间 均值{s['t_mean']:6.0f} 中位{s['t_med']:6.0f} "
              f"[{s['t_min']:.0f},{s['t_max']:.0f}]  平均每源={s['avg_mean']:5.0f}s "
              f"移动={s['mv_mean']:5.0f}m")

    with open("results/eval_baseline.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(tab[0].keys()))
        w.writeheader()
        w.writerows(tab)
    np.savez("results/eval_baseline.npz",
             **{k: np.array([r[k] for r in all_rows])
                for k in ("seed", "N", "cleared", "total", "mv", "n_meas",
                          "n_clear", "n_loc", "n_det", "dir_prob", "corr")})
    print(f"\n已写出 results/eval_baseline.csv / eval_baseline.npz（共 {len(all_rows)} 局）")
