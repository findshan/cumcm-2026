"""B题 问题三：覆盖环几何与"oracle 下界"的独立复算脚本。

产出（全部可复现）：
  1) 保证覆盖所需的最少站数与环半径可行区间（等角单环）；
  2) 各 m 下的环周长与"自圆心入环"的巡回路程；
  3) oracle 下界：已知全部源位置时的最优路线（起点 + 全部源），
     以及并入保证覆盖环站后的合并路线（起点 + 全部源 + m 个环站）。

路线用最近邻 + 2-opt（多随机起点重启）求解；n ≤ 17 时通常即为最优，
故称"oracle 下界"是指"已知全部源位置"这一理想基准，而非松弛界。

用法: python q3_bounds.py [每档局数，默认 50]
"""
from __future__ import annotations

import math
import sys

import numpy as np

from jammer_sim import make_case

R_ARENA, R_MIN = 1800.0, 1000.0


# ---------------- 覆盖环几何 ----------------
def a_interval(m: int, r_cov: float = R_MIN):
    """等角 m 站环半径 a 的可行区间：同时满足
       (i) 圆心被覆盖  a ≤ r_cov；
       (ii) 竞技场边界最不利点（两站之间的角平分方向）被覆盖。"""
    c = math.cos(math.pi / m)
    A = 1.0
    B = -2 * R_ARENA * c
    C = R_ARENA ** 2 - r_cov ** 2
    disc = B * B - 4 * A * C
    if disc < 0:
        return None
    s = math.sqrt(disc)
    return ((-B - s) / 2, (-B + s) / 2)


def ring_perimeter(m: int, a: float) -> float:
    return 2 * m * a * math.sin(math.pi / m)


# ---------------- 路线求解 ----------------
def tour_len(pts, closed=False) -> float:
    p = np.asarray(pts, float)
    if len(p) < 2:
        return 0.0
    L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    if closed:
        L += float(np.linalg.norm(p[-1] - p[0]))
    return L


def two_opt(order, D, closed=False):
    n = len(order)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                a, b = order[i - 1], order[i]
                c, d = order[j], order[(j + 1) % n] if closed else order[j]
                if not closed and j == n - 1:
                    # 开放路径：只需比较内部翻转
                    continue
                before = D[a, b] + D[c, d]
                after = D[a, c] + D[b, d]
                if after < before - 1e-12:
                    order[i:j] = order[i:j][::-1]
                    improved = True
    return order


def tsp(pts, start=None, restarts=8, seed=0) -> float:
    """最近邻 + 2-opt（多起点）。start 给出则路径必须从 start 出发（开放路径）。"""
    p = np.asarray(pts, float)
    n = len(p)
    if n <= 2:
        return tour_len(p)
    D = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    rng = np.random.default_rng(seed)
    best = None
    cand_starts = [0] + list(rng.choice(np.arange(n), size=min(restarts, n), replace=False))
    for s0 in cand_starts:
        order = [int(s0)]
        left = set(range(n)) - {int(s0)}
        cur = int(s0)
        while left:
            nxt = min(left, key=lambda k: D[cur, k])
            left.remove(nxt)
            order.append(nxt)
            cur = nxt
        if start is not None:
            si = min(range(n), key=lambda k: np.linalg.norm(p[k] - np.asarray(start, float)))
            order.remove(si)
            order = [si] + order
        order = two_opt(order, D, closed=False)
        L = sum(D[order[i], order[i + 1]] for i in range(n - 1))
        if best is None or L < best:
            best = L
    return float(best)


if __name__ == "__main__":
    ncase = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("=" * 108)
    print("【1】保证覆盖的最少站数与环半径可行区间（等角单环，r_cov = 1000 m）")
    print("=" * 108)
    print(f"{'m':>3} {'a 可行区间 / m':>26} {'与 a≤1000 求交':>24} "
          f"{'环周长@a_min':>13} {'环周长@a=1000':>14} {'巡回路程(圆心入环)':>18}")
    rows = []
    for m in range(5, 17):
        iv = a_interval(m)
        if iv is None:
            print(f"{m:>3} {'∅（判别式 < 0）':>26} {'∅':>24} {'—':>13} {'—':>14} {'—':>18}")
            continue
        lo, hi = iv
        hi_cap = min(hi, R_MIN)
        feasible = lo <= hi_cap
        inter = f"[{lo:.1f}, {hi_cap:.1f}]" if feasible else "∅"
        if feasible:
            per_min = ring_perimeter(m, lo)
            per_1000 = ring_perimeter(m, R_MIN)
            sweep = R_MIN + per_1000
            print(f"{m:>3} {f'[{lo:.1f}, {hi:.1f}]':>26} {inter:>24} "
                  f"{per_min:>13.1f} {per_1000:>14.1f} {sweep:>18.1f}")
            rows.append((m, lo, hi_cap, per_min, per_1000, sweep))
        else:
            print(f"{m:>3} {f'[{lo:.1f}, {hi:.1f}]':>26} {inter:>24} {'—':>13} {'—':>14} {'—':>18}")

    print("\n  · m = 6 判别式 ≥ 0 但 [1122.9, 1994.8] 与 a ≤ 1000 无交集 → 无解（故最少 7 站）")
    print("  · 本文最终采用 m = 7, a = 1000 m")

    print("\n" + "=" * 108)
    print(f"【2】oracle 下界：已知全部源位置时的最优路线（每档 {ncase} 局，50 次不同 N）")
    print("=" * 108)
    ring7 = [R_MIN * math.cos(2 * math.pi * k / 7) for k in range(7)]
    ring7 = [(R_MIN * math.cos(2 * math.pi * k / 7), R_MIN * math.sin(2 * math.pi * k / 7))
             for k in range(7)]

    print(f"{'N':>4} {'局数':>5} {'oracle TSP(起点+源) / m':>26} {'折合时间 / s':>13} "
          f"{'合并覆盖环 7 站 / m':>20} {'折合时间 / s':>13}")
    summary = {}
    for target in (10, 13, 16):
        Ls, Ls2 = [], []
        s = 0
        got = 0
        while got < ncase and s < ncase * 12:
            case = make_case(np.random.default_rng(20260000 + s), dir_prob=0.0, seed=s)
            s += 1
            if case.N != target:
                continue
            got += 1
            pts = [(0.0, 0.0)] + [tuple(map(float, p)) for p in case.pos]
            Ls.append(tsp(pts, start=(0.0, 0.0)))
            pts2 = pts + ring7
            Ls2.append(tsp(pts2, start=(0.0, 0.0)))
        Ls, Ls2 = np.array(Ls), np.array(Ls2)
        summary[target] = (Ls.mean(), Ls2.mean())
        print(f"{target:>4} {got:>5} {Ls.mean():>26.1f} {Ls.mean()/5:>13.0f} "
              f"{Ls2.mean():>20.1f} {Ls2.mean()/5:>13.0f}")
    allL = np.array([v[0] for v in summary.values()]).mean()
    allL2 = np.array([v[1] for v in summary.values()]).mean()
    print(f"\n  · 三档平均：oracle TSP = {allL:.0f} m（{allL/5:.0f} s）；"
          f"并入覆盖环后 = {allL2:.0f} m（{allL2/5:.0f} s）")
    print(f"  · 站间扫描下界（7 站环＋源）：{allL2:.0f} m")
