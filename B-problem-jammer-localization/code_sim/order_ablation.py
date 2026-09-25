"""B题 消融实验：任务顺序（两趟式 vs 滚动重规划）与主动定位策略。

三组配置，除被消融项外其余参数、种子、误差场完全一致：
  A 两趟式       : two_pass=True,  use_outer=False, loc_off="sml"
  B 滚动·单环     : two_pass=False, use_outer=False, loc_off="sml"   ← 对照基线
  C 滚动·双环（本文）: two_pass=False, use_outer=True,  loc_off="mdd"

用法: python order_ablation.py [每档局数=50]
"""
from __future__ import annotations

import sys

import numpy as np

from jammer_sim import ErrorField, LocalTransport, RobotClient, Simulator, make_case
from strategy import Strategy

MAIN = dict(layout="dual", ring_m=7, ring_a=1000.0, ring_m2=5, ring_a2=1700.0,
            loc_min_dir=1, loc_off="mdd", max_loc=6, use_outer=True)
CONFIGS = [
    ("A 两趟式", dict(MAIN, two_pass=True,  use_outer=False, loc_off="sml")),
    ("B 滚动·单环", dict(MAIN, two_pass=False, use_outer=False, loc_off="sml")),
    ("C 滚动·双环", dict(MAIN, two_pass=False, use_outer=True,  loc_off="mdd")),
]


def run_one(seed, dp, cfg, err_mode="smooth", corr=1500.0):
    rng = np.random.default_rng(20260000 + seed)
    case = make_case(rng, dir_prob=dp, seed=seed)
    err = ErrorField(np.random.default_rng(5000 + seed), mode=err_mode, corr_len=corr)
    sim = Simulator(case, robot_id="2026123456", err=err)
    cli = RobotClient(LocalTransport(sim), "2026123456")
    st = Strategy(cfg)
    res = st.run(cli)
    R = res["n_cleared"] / case.N
    return dict(N=case.N, cleared=res["n_cleared"], rate=R, total=res["virtual_time"],
                mv=res["mv_total"], n_meas=res["n_meas"], n_loc=res["n_loc"])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    for dp in (0.0, 0.35):
        print("=" * 104)
        print(f"任务顺序 / 主动定位消融    定向源比例 = {dp*100:.0f}%    {n} 局/组")
        print("=" * 104)
        print(f"{'配置':<14}{'清除率':>9}{'全清':>8}{'总时间/s':>11}{'移动/m':>11}"
              f"{'检测次数':>10}{'主动定位次数':>13}")
        ref = None
        for tag, cfg in CONFIGS:
            rs = [run_one(s, dp, cfg) for s in range(n)]
            rate = np.mean([r["rate"] for r in rs]) * 100
            full = sum(1 for r in rs if r["rate"] > 0.9999)
            T = np.mean([r["total"] for r in rs])
            mv = np.mean([r["mv"] for r in rs])
            nm = np.mean([r["n_meas"] for r in rs])
            nl = np.mean([r["n_loc"] for r in rs])
            print(f"{tag:<14}{rate:>8.2f}%{full:>5d}/{n:<3d}{T:>11.0f}{mv:>11.0f}"
                  f"{nm:>10.1f}{nl:>13.1f}")
            if tag.startswith("A"):
                ref = (mv, T)
            elif ref is not None and tag.startswith("B"):
                print(f"{'':<14}→ 滚动重规划相对两趟式：移动 {ref[0]:.0f} → {mv:.0f} m "
                      f"（省 {ref[0] - mv:.0f} m，{100 * (1 - mv / ref[0]):.1f}%），"
                      f"时间 {ref[1]:.0f} → {T:.0f} s（省 {ref[1] - T:.0f} s）")
        print()
