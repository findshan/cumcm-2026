"""参数扫描：站点布局 / 清剿阈值 / 主动定位上限。

在同一进程内跑，避免反复 import 与启动开销。
用法: python scan.py [--n 12] [--stage layout|thr|loc|all]
"""
from __future__ import annotations

import argparse
import itertools
import math
import numpy as np

from jammer_sim import ErrorField, LocalTransport, RobotClient, Simulator, make_case
from strategy import Strategy

SEED_BASE = 20260000


def run_one(seed: int, params: dict, dir_prob: float, err_mode: str, corr: float) -> dict:
    rng = np.random.default_rng(SEED_BASE + seed)
    case = make_case(rng, dir_prob=dir_prob, seed=seed)
    err = ErrorField(np.random.default_rng(1000 + seed), mode=err_mode, corr_len=corr)
    sim = Simulator(case, robot_id="t", err=err)
    cli = RobotClient(LocalTransport(sim), "t")
    st = Strategy(params)
    res = st.run(cli)
    return dict(N=case.N, cleared=res["n_cleared"], total=res["virtual_time"],
                mv=res["mv_total"], n_meas=res["n_meas"], n_clear=res["n_clear"],
                n_det=res["n_det"], n_loc=res["n_loc"], stations=res["stations"])


def evaluate(params: dict, n: int, dir_prob=0.0, err_mode="smooth", corr=1500.0) -> dict:
    rows = [run_one(s, params, dir_prob, err_mode, corr) for s in range(n)]
    ratio = np.mean([r["cleared"] / r["N"] for r in rows])
    tot = np.mean([r["total"] for r in rows])
    avg = np.mean([r["total"] / max(r["cleared"], 1) for r in rows])
    mv = np.mean([r["mv"] for r in rows])
    return dict(ratio=ratio, total=tot, avg=avg, mv=mv,
                full=int(sum(1 for r in rows if r["cleared"] == r["N"])),
                n_meas=np.mean([r["n_meas"] for r in rows]),
                n_clear=np.mean([r["n_clear"] for r in rows]),
                n_loc=np.mean([r["n_loc"] for r in rows]),
                stations=rows[0]["stations"])


def line(name: str, m: dict) -> str:
    return (f"{name:30s} 清除率={m['ratio']*100:6.2f}%  全清={m['full']:2d}/{_N:2d}  "
            f"总时间={m['total']:7.0f}s  平均={m['avg']:6.1f}s  移动={m['mv']:6.0f}m  "
            f"测量={m['n_meas']:5.1f} clear={m['n_clear']:5.1f} loc={m['n_loc']:4.1f}")


_N = 12

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--stage", type=str, default="all")
    ap.add_argument("--dir", type=float, default=0.0)
    args = ap.parse_args()
    _N = args.n

    print("=" * 120)
    print(f"参数扫描  n={_N}  定向比例={args.dir}")
    print("=" * 120)

    if args.stage in ("layout", "all"):
        print("\n【阶段1】圆环站点布局：站数 m × 半径 a")
        best = (1e18, None)
        for m, a in itertools.product((6, 7, 8, 9, 10, 11, 12),
                                      (850, 925, 1000, 1100, 1200)):
            mm = evaluate(dict(layout="ring", ring_m=m, ring_a=float(a)), _N, args.dir)
            print(line(f"  m={m:2d} a={a:4d}", mm))
            # 目标：全清优先，其次时间
            score = (1.0 - mm["ratio"]) * 1e6 + mm["total"]
            if score < best[0]:
                best = (score, (m, a, mm))
        print(f"\n  >>> 最优布局：m={best[1][0]} a={best[1][1]}  "
              f"清除率={best[1][2]['ratio']*100:.2f}% 时间={best[1][2]['total']:.0f}s")

    if args.stage in ("thr", "all"):
        base = dict(layout="ring", ring_m=8, ring_a=925.0)
        print("\n【阶段2】清剿/跳过阈值")
        for dr, ds, df in itertools.product((60.0, 100.0, 150.0, 200.0),
                                            (30.0, 45.0),
                                            (100.0, 120.0, 200.0)):
            mm = evaluate(dict(base, D_ready=dr, D_skip=ds, D_final=df), _N, args.dir)
            print(line(f"  D_ready={dr:5.0f} D_skip={ds:4.0f} D_final={df:5.0f}", mm))

    if args.stage in ("loc", "all"):
        base = dict(layout="ring", ring_m=8, ring_a=925.0)
        print("\n【阶段3】主动定位上限 max_loc / 清剿网格 grid_step")
        for ml, gs, mc in itertools.product((2, 3, 4, 6), (24.0, 36.0, 48.0), (40, 80)):
            mm = evaluate(dict(base, max_loc=ml, grid_step=gs, max_clear_try=mc), _N, args.dir)
            print(line(f"  max_loc={ml} grid={gs:4.0f} maxtry={mc:3d}", mm))
