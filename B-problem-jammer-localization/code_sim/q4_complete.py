"""B题 问题四：严格 100% 保证的兜底布局（按需启用）+ 单元顶点覆盖引理的数值复算。

核心引理（胞元顶点覆盖）
    设真值 G 落在胞元 conv(V) 内（V = 胞元顶点，同时是观测站），且
        rad(V) := max_{v in V} |v - G| <= r_min = 1000 m.
    则对任意辐射主方向 phi，必存在 v in V 使
        (v - G)·phi >= 0   且   |v - G| <= r_min <= r_k,
    即"源在 v 的覆盖半平面内且在有效接收半径内"，v 处一次 /measure 必能测到该源。
    证明：G 是 V 的凸组合 G = sum_i lam_i v_i（lam_i >= 0, sum lam_i = 1），
          故 sum_i lam_i (v_i - G) = 0；两边点乘 phi 得 sum_i lam_i (v_i-G)·phi = 0，
          各项非负加权求和为 0 => 至少一项 >= 0。取该顶点即得结论。  QED

紧常数（胞元内"最近可测顶点距离"的上确界 sup / h）
    正三角胞元：sup = h          => 边长上界 h <= r_min = 1000 m
    正方胞元  ：sup = (sqrt5/2)h => 边长上界 h <= 2 r_min/sqrt5 = 894 m
（正方胞元的最坏情形：源贴着胞元一侧边的中点、主方向指向胞元内部时，
 该侧边两顶点落在阴影里，只能用对侧两个顶点，距离 = sqrt(h^2 + (h/2)^2)。）

用法: python q4_complete.py        # 复算引理常数、给出可用的严格保证布局与巡回长度
"""
from __future__ import annotations

import math

import numpy as np

R_ARENA = 1800.0
R_MIN = 1000.0


# ---------------- 格点（三角格 / 方格，含界外顶点） ----------------
def tri_lattice(h: float) -> np.ndarray:
    pts = []
    ny = int((R_ARENA + h) / (h * math.sqrt(3) / 2)) + 2
    for j in range(-ny, ny + 1):
        y = j * h * math.sqrt(3) / 2
        for i in range(-2 * ny - 4, 2 * ny + 5):
            x = i * h + (j % 2) * h / 2
            if x * x + y * y <= (R_ARENA + h) ** 2:
                pts.append((x, y))
    return np.unique(np.round(np.array(pts), 6), axis=0)


def sq_lattice(h: float) -> np.ndarray:
    pts = []
    n = int(math.ceil(R_ARENA / h)) + 2
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            x, y = i * h, j * h
            if x * x + y * y <= (R_ARENA + 1.5 * h) ** 2:
                pts.append((x, y))
    return np.unique(np.round(np.array(pts), 6), axis=0)


# ---------------- 引理常数的数值复算 ----------------
def sup_ratio(verts, NG: int = 200, AN: int = 1440) -> float:
    """sup over (G in cell, phi) of min_{i: (vi-G)·phi>=0} |vi - G|，单位取胞元边长 h=1。"""
    V = np.asarray(verts, float)
    if len(V) == 4:
        gx, gy = np.meshgrid(np.linspace(0, 1, NG), np.linspace(0, 1, NG))
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
    else:
        a = np.linspace(0, 1, 260)
        A, B = np.meshgrid(a, a)
        m = A + B <= 1
        pts = V[0] + A[m][:, None] * (V[1] - V[0]) + B[m][:, None] * (V[2] - V[0])
    D = pts[:, None, :] - V[None, :, :]
    dist = np.linalg.norm(D, axis=2)
    best = 0.0
    for p in np.linspace(0, 2 * math.pi, AN, endpoint=False):
        phi = np.array([math.cos(p), math.sin(p)])
        dd = np.where(D @ phi >= -1e-12, dist, 1e18).min(axis=1)
        best = max(best, float(dd.max()))
    return best


# ---------------- 完整性验证（随机抽样，全向均匀） ----------------
def verify(S: np.ndarray, n: int = 200_000, seed: int = 1):
    rng = np.random.default_rng(seed)
    r = R_ARENA * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * math.pi, n)
    G = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    phi = rng.uniform(0, 2 * math.pi, n)
    fail, worst = 0, 0.0
    for k in range(n):
        D = S - G[k]
        d = np.linalg.norm(D, axis=1)
        dot = D[:, 0] * math.cos(phi[k]) + D[:, 1] * math.sin(phi[k])
        ok = (dot >= -1e-9) & (d <= R_MIN)
        if ok.any():
            worst = max(worst, float(d[ok].min()))
        else:
            fail += 1
    return fail / n, worst


# ---------------- 巡回（近邻 + 2-opt），自圆心出发 ----------------
def nn2opt(S: np.ndarray, start=(0.0, 0.0), rounds: int = 12):
    S = np.asarray(S, float)
    n = len(S)
    unv, cur, order = list(range(n)), np.array(start, float), []
    while unv:
        k = unv[int(np.argmin([np.linalg.norm(S[i] - cur) for i in unv]))]
        order.append(k)
        cur = S[k]
        unv.remove(k)
    P = S[order]

    def L(Q):
        return float(np.linalg.norm(np.diff(np.vstack([start, Q]), axis=0), axis=1).sum())

    best = L(P)
    for _ in range(rounds):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                Q = P.copy()
                Q[i:j + 1] = Q[i:j + 1][::-1]
                l = L(Q)
                if l < best - 1e-9:
                    best, P, improved = l, Q, True
        if not improved:
            break
    return best, P


if __name__ == "__main__":
    print("=" * 96)
    print("【1】胞元顶点覆盖引理的紧常数（数值复算，单位 = 胞元边长 h）")
    print("=" * 96)
    sq = sup_ratio([(0, 0), (1, 0), (1, 1), (0, 1)])
    tr = sup_ratio([(0, 0), (1, 0), (0.5, math.sqrt(3) / 2)])
    print(f"  正方胞元：sup = {sq:.4f} h  (解析 sqrt5/2 = {math.sqrt(5)/2:.4f})"
          f"  ->  h_max = {R_MIN/sq:.0f} m")
    print(f"  三角胞元：sup = {tr:.4f} h  (解析 1.0000)"
          f"        ->  h_max = {R_MIN/tr:.0f} m")

    print("\n" + "=" * 96)
    print("【2】严格保证布局：站数 / 覆盖率验证 / 巡回长度")
    print("=" * 96)
    print(f"{'布局':<14}{'站数':>6}{'失败率':>10}{'最近可测站距/m':>16}{'巡回/m':>10}{'巡回/s':>10}")
    for h, fn, name in ((950.0, tri_lattice, "三角格 h=950"),
                        (1000.0, tri_lattice, "三角格 h=1000"),
                        (850.0, sq_lattice, "方格 h=850"),
                        (600.0, sq_lattice, "方格 h=600")):
        S = fn(h)
        fr, worst = verify(S, 200_000)
        L, _ = nn2opt(S)
        print(f"{name:<14}{len(S):>6}{fr*100:>9.4f}%{worst:>16.1f}{L:>10.0f}{L/5:>10.0f}")

    print("\n【结论】三角格 h=950 m（31 站）即可以 0% 失败率覆盖全部 (位置,方向) 组合，")
    print("        巡回约 2.9e4 m ≈ 5.8e3 s，是求解$100\\%$保证时的最省已知布局；")
    print("        相对方格 h=600 m（60+ 站）站数与路程均降至约一半，故作为兜底方案的默认选择。")
