"""
B题 演练测试（蒙特卡洛）：在自建等价模拟器上评估策略。

用法：
  python run_mc.py --n 30 --dir 0                 # 问题3（全向）
  python run_mc.py --n 30 --dir 0.4               # 问题4（40% 定向源）
  python run_mc.py --n 30 --err iid               # 误差场模型改为独立同分布
  python run_mc.py --n 30 --layout greedy         # 站点布局改为贪心覆盖
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from jammer_sim import ErrorField, LocalTransport, RobotClient, Simulator, make_case
from strategy import Strategy

# python3 run_mc.py --n 30 --dir 0

def run_one(seed: int, args) -> dict:
    rng = np.random.default_rng(20260000 + seed)
    case = make_case(rng, dir_prob=args.dir, seed=seed)
    err = ErrorField(np.random.default_rng(1000 + seed), mode=args.err, corr_len=args.corr)
    sim = Simulator(case, robot_id=args.team, err=err)
    cli = RobotClient(LocalTransport(sim), args.team)
    params = {}
    if args.layout != "ring":
        params["layout"] = args.layout
    if args.m is not None:
        params["ring_m"] = args.m
    if args.a is not None:
        params["ring_a"] = args.a
    if args.dskip is not None:
        params["D_skip"] = args.dskip
    if args.dready is not None:
        params["D_ready"] = args.dready
    if args.grid is not None:
        params["grid_step"] = args.grid
    st = Strategy(params)
    t0 = time.time()
    res = st.run(cli)
    wall = time.time() - t0
    return dict(N=case.N, n_dir=case.n_dir, cleared=res["n_cleared"], total=res["virtual_time"],
                mv=res["mv_total"], mv_break=res["mv"], n_meas=res["n_meas"],
                n_clear=res["n_clear"], stations=res["stations"], n_det=res["n_det"],
                n_loc=res.get("n_loc", 0),
                wall=wall, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dir", type=float, default=0.0, help="定向源比例（问题4）")
    ap.add_argument("--err", type=str, default="smooth", choices=["smooth", "iid", "grid"])
    ap.add_argument("--corr", type=float, default=1500.0, help="误差场相关长度(m) / iid 无意义 / grid 网格边长")
    ap.add_argument("--layout", type=str, default="ring", choices=["ring", "greedy"])
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--a", type=float, default=None)
    ap.add_argument("--dskip", type=float, default=None)
    ap.add_argument("--dready", type=float, default=None)
    ap.add_argument("--grid", type=float, default=None)
    ap.add_argument("--team", type=str, default="2026123456")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rows = []
    for s in range(args.n):
        r = run_one(s, args)
        rows.append(r)
        if not args.quiet:
            print(f"  局 {s:3d}: N={r['N']:2d} 定向={r['n_dir']:2d} 清除={r['cleared']:2d} "
                  f"比例={r['cleared']/r['N']*100:5.1f}%  总时间={r['total']:7.0f}s  "
                  f"平均={r['total']/max(r['cleared'],1):6.1f}s  移动={r['mv']:6.0f}m  "
                  f"测量={r['n_meas']:3d} /clear={r['n_clear']:3d}")

    N = np.array([r["N"] for r in rows], float)
    C = np.array([r["cleared"] for r in rows], float)
    T = np.array([r["total"] for r in rows], float)
    MV = np.array([r["mv"] for r in rows], float)
    ME = np.array([r["n_meas"] for r in rows], float)
    CL = np.array([r["n_clear"] for r in rows], float)
    avg = np.array([r["total"] / max(r["cleared"], 1) for r in rows])
    ratio = C / N
    nd = np.array([r["n_det"] for r in rows], float)

    print("\n" + "=" * 78)
    print(f"演练测试汇总  n={args.n}  定向比例={args.dir}  误差场={args.err}(corr={args.corr})  布局={args.layout}")
    print("=" * 78)
    print(f"  被清除干扰源个数的比例      平均 {ratio.mean()*100:6.2f}%   最小 {ratio.min()*100:6.2f}%   为100%的局数 {int((ratio>0.9999).sum())}/{args.n}")
    print(f"  平均定位清除时间            {avg.mean():8.1f} s   （中位 {np.median(avg):.1f}，最小 {avg.min():.1f}，最大 {avg.max():.1f}）")
    print(f"  定位清除总时间（虚拟）       {T.mean():8.1f} s   （中位 {np.median(T):.1f}）")
    print(f"  程序运行时间（本机墙体）      {np.mean([r['wall'] for r in rows]):8.2f} s")
    print("  --- 分解 ---")
    print(f"  移动                        {MV.mean():8.0f} m = {MV.mean()/5:6.0f} s  ({MV.mean()/5/T.mean()*100:.0f}% of total)")
    print(f"  检测次数                    {ME.mean():8.1f} 次 -> {ME.mean()*5:6.0f} s")
    print(f"  清除尝试                    {CL.mean():8.1f} 次")
    print(f"  主动定位(loc)次数            {np.mean([r['n_loc'] for r in rows]):8.1f} 次")
    brk = {k: np.mean([r['mv_break'][k] for r in rows]) for k in ("sweep", "clear", "refine")}
    print(f"  移动分解                    扫描站间 {brk['sweep']:.0f} m / 清剿 {brk['clear']:.0f} m / 主动定位 {brk['refine']:.0f} m")
    print(f"  识别到的频道数               {nd.mean():8.1f}  (真值 N 均值 {N.mean():.1f})")
    print(f"  漏检局数（清除<N）           {int((C<N-1e-9).sum())}/{args.n}")
    worst = sorted(rows, key=lambda r: r["cleared"] / r["N"])[:3]
    for r in worst:
        print(f"    最差局 seed={r['seed']}: N={r['N']} 清除={r['cleared']} 时间={r['total']:.0f}s "
              f"识别={r['n_det']} 移动={r['mv']:.0f}m")


if __name__ == "__main__":
    main()
