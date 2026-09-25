"""
B题 机器狗策略：主线扫描 + 动态顺路清剿（可对 LocalTransport / HttpTransport 双跑）。

设计依据见 ../B题_题目理解与建模方案.md：
  洞察1 集值估计 -> 用可行域直径作不确定度
  洞察2 每次检测给角度 + 径向双重约束（direction 蕴含 |PG|<=1500）
  洞察3 频道解耦、路径是唯一耦合 -> 动态 TSP（剩余扫描站 ∪ 已定位待清剿目标）
  洞察4 /clear 是廉价近距二值传感器 -> 直径压到阈值后转网格清剿
  洞察5 第二点偏角约 30°、距离约 1000 m（Q2 结论），精化阶段按 70° 偏置
  洞察6 定向源阴影 -> 怀疑清单（Q4 分支，见 suspect 相关代码）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

R_ARENA = 1800.0
R_RECV_MAX = 1500.0
R_RECV_MIN = 1000.0
EPS_DEG = 1.0
DISK_N = 72


# ---------------- 凸区域（半平面交） ----------------
class Region:
    """可行域：凸多边形。只用"楔形 + 圆盘"两类凸约束（no_signal 是凹约束，保守忽略）。"""

    def __init__(self, init_poly):
        self.poly = list(init_poly)

    @classmethod
    def arena(cls, n: int = 144):
        return cls([(R_ARENA * math.cos(2 * math.pi * k / n),
                     R_ARENA * math.sin(2 * math.pi * k / n)) for k in range(n)])

    def _clip(self, a, b, c):
        poly = self.poly
        if not poly:
            return
        out, n = [], len(poly)
        for i in range(n):
            P, Q = poly[i], poly[(i + 1) % n]
            fp, fq = a * P[0] + b * P[1] - c, a * Q[0] + b * Q[1] - c
            if fp <= 0:
                out.append(P)
            if (fp < 0 < fq) or (fq < 0 < fp):
                t = fp / (fp - fq)
                out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
        self.poly = out

    def add_wedge(self, S, theta_deg, half_deg=EPS_DEG):
        t = math.radians(theta_deg)
        d = math.radians(half_deg)
        p_in = np.array([S[0] + 300.0 * math.cos(t), S[1] + 300.0 * math.sin(t)])
        Sn = np.asarray(S, float)
        for tb in (t - d, t + d):
            n = np.array([-math.sin(tb), math.cos(tb)])
            if n @ (p_in - Sn) < 0:
                n = -n
            self._clip(-n[0], -n[1], -(n @ Sn))

    def add_disk(self, S, R, n: int = DISK_N):
        if R <= 0:
            self.poly = []
            return
        for k in range(n):
            mid = 2 * math.pi * (k + 0.5) / n
            nx, ny = math.cos(mid), math.sin(mid)
            self._clip(nx, ny, nx * S[0] + ny * S[1] + R)

    @property
    def empty(self) -> bool:
        return len(self.poly) < 3

    def hull(self):
        pts = sorted(set((round(x, 6), round(y, 6)) for x, y in self.poly))
        if len(pts) <= 2:
            return pts

        def cr(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lo = []
        for p in pts:
            while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
                lo.pop()
            lo.append(p)
        up = []
        for p in reversed(pts):
            while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
                up.pop()
            up.append(p)
        return lo[:-1] + up[:-1]

    def area(self) -> float:
        V = self.hull()
        if len(V) < 3:
            return 0.0
        p = np.array(V)
        return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                              - np.dot(p[:, 1], np.roll(p[:, 0], -1))))

    def diameter(self):
        V = self.hull()
        if len(V) < 2:
            return 0.0, None, None
        best = (-1.0, None, None)
        for i in range(len(V)):
            xi, yi = V[i]
            for j in range(i + 1, len(V)):
                d = math.hypot(V[j][0] - xi, V[j][1] - yi)
                if d > best[0]:
                    best = (d, V[i], V[j])
        return best

    def centroid(self):
        V = self.hull()
        if len(V) < 3:
            return np.array([0.0, 0.0])
        p = np.array(V)
        x, y = p[:, 0], p[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cr = x * yn - xn * y
        A = cr.sum() / 2.0
        if abs(A) < 1e-9:
            return p.mean(axis=0)
        return np.array([((x + xn) * cr).sum() / (6 * A), ((y + yn) * cr).sum() / (6 * A)])

    def inside(self, p) -> bool:
        V = self.hull()
        if len(V) < 3:
            return False
        for i in range(len(V)):
            a, b = V[i], V[(i + 1) % len(V)]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-9:
                return False
        return True


# ---------------- 每频道状态 ----------------
@dataclass
class Chan:
    reg: Region
    obs: list = field(default_factory=list)
    n_dir: int = 0
    n_nosig: int = 0
    cleared: bool = False
    fail: int = 0
    n_loc: int = 0
    nosig_pts: list = field(default_factory=list)
    suspect: bool = False          # Q4：疑似定向源（阴影）

    @classmethod
    def fresh(cls):
        return cls(reg=Region.arena())

    @property
    def D(self) -> float:
        return self.reg.diameter()[0]

    def mean_bearing(self, ref):
        if not self.obs:
            return None
        c = self.reg.centroid()
        return math.degrees(math.atan2(c[1] - ref[1], c[0] - ref[0])) % 360.0


# ---------------- 扫描站规划 ----------------
def tour(stations, start=(0.0, 0.0)):
    if not stations:
        return []
    S = [np.asarray(s, float) for s in stations]
    idx = list(range(len(S)))
    cur = np.asarray(start, float)
    order = []
    while idx:
        k = min(idx, key=lambda i: float(np.linalg.norm(S[i] - cur)))
        order.append(k)
        cur = S[k]
        idx.remove(k)
    P = [np.asarray(start, float)] + [S[i] for i in order]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(P) - 1):
            for j in range(i + 1, len(P) - 1):
                a, b, c, d = P[i - 1], P[i], P[j], P[j + 1]
                if (math.dist(a, c) + math.dist(b, d)) < (math.dist(a, b) + math.dist(c, d)) - 1e-9:
                    P[i:j + 1] = reversed(P[i:j + 1])
                    improved = True
    return [(float(p[0]), float(p[1])) for p in P[1:]]


def ring_stations(m: int = 8, a: float = 925.0, rot: float = 0.0):
    """均布圆环站点：m 站位于半径 a 的圆上。m=7,a=1000 或 m=8,a=925 可 100% 覆盖
    （覆盖半径 1000），且巡回长度最短（约 5900~6200 m）。"""
    return [(a * math.cos(rot + 2 * math.pi * k / m),
             a * math.sin(rot + 2 * math.pi * k / m)) for k in range(m)]


def plan_stations(r_cov: float = R_RECV_MIN, cand_step: float = 150.0,
                  grid_step: float = 60.0, margin: float = 0.0):
    """贪心最大覆盖：用半径 r_cov 的圆盘覆盖半径 (R_ARENA+margin) 的圆盘。"""
    G = R_ARENA + margin
    g = np.arange(-G, G + 1e-9, grid_step)
    XX, YY = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    pts = pts[np.linalg.norm(pts, axis=1) <= G + 1e-9]

    cg = np.arange(-G * 0.9, G * 0.9 + 1e-9, cand_step)
    CX, CY = np.meshgrid(cg, cg, indexing="ij")
    cand = np.stack([CX.ravel(), CY.ravel()], axis=1)
    cand = cand[np.linalg.norm(cand, axis=1) <= G + 1e-9]

    unc = np.ones(len(pts), dtype=bool)
    chosen = []
    while unc.any():
        d2 = ((cand[:, None, 0] - pts[None, :, 0]) ** 2 + (cand[:, None, 1] - pts[None, :, 1]) ** 2)
        cnt = ((d2 <= r_cov ** 2) & unc[None, :]).sum(axis=1)
        k = int(np.argmax(cnt))
        if cnt[k] == 0:
            break
        chosen.append(cand[k])
        unc &= ~(((cand[k, 0] - pts[:, 0]) ** 2 + (cand[k, 1] - pts[:, 1]) ** 2) <= r_cov ** 2)
    return tour(chosen)


# ---------------- 策略 ----------------
DEFAULT = dict(
    r_cov=1000.0,
    margin=0.0,
    D_skip=45.0,
    D_ready=100.0,
    D_final=120.0,      # 直径小于此值直接清剿（清剿比补测便宜，见洞察4）
    grid_step=24.0,
    max_clear_try=80,
    refine_off=70.0,
    max_iter=400,
    max_loc=6,
    loc_min_dir=1,
    # 主动定位候选点策略：
    #   "mdd" = 中等深度双向探测（本文，Q2 结论的工程落地）
    #   "sml" = 单一方向、质心小偏置补测（基线策略，用于对照实验）
    loc_off="mdd",
    # 两趟式（sweep-then-clear）消融开关：True 时先扫完所有覆盖站再回头清剿，
    # 用于对照"滚动重规划（边扫边清）"。默认 False。
    two_pass=False,
    use_outer=True,        # 是否允许在"证实存在定向源"后启动外环补扫
    layout="ring",       # ring | greedy
    ring_m=8,
    ring_a=925.0,
    ring_rot=0.0,
    n_max=16,
)


class Strategy:
    def __init__(self, params: dict | None = None, verbose: bool = False):
        self.P = dict(DEFAULT)
        if params:
            self.P.update(params)
        self.verbose = verbose
        self.n_meas = 0
        self.n_clear = 0
        self.n_det = 0
        self.mv = dict(sweep=0.0, clear=0.0, refine=0.0)
        self.n_loc = 0
        self.has_dir = False          # 是否已获得"存在定向源"的证据（Q4）
        self._last = np.array([0.0, 0.0])

    # ---- 观测更新 ----
    def _update(self, st: Chan, x, y, resp):
        res = resp.get("measure_result")
        if res == "direction":
            th = float(resp["svd_deg"])
            st.reg.add_wedge((x, y), th)
            st.reg.add_disk((x, y), R_RECV_MAX)
            st.obs.append((x, y, th))
            st.n_dir += 1
        elif res == "near":
            st.reg.add_disk((x, y), 5.0, n=24)
            st.n_dir += 1
        else:
            st.n_nosig += 1
            st.nosig_pts.append((float(x), float(y)))
            # 【定向源判据（Q4 关键）】全向源只要 |PG| <= r_k 就一定有信号，
            # 而 r_k >= 1000 m，故若能证明"整个可行域 R_k 都在 s 的 1000 m 内"，
            # 则此处 no_signal 只可能来自定向源的阴影半平面。
            #
            # 严格包络（不能用 D/2）：D/2 是"以质心为心的半径"的下界，
            # 真实上界可达 D/sqrt(3)（Jung），故 D/2 版本会误触发。
            # 可行域是凸多边形 -> max|g-C| 必在顶点取到，直接用凸包顶点算精确半径。
            if st.n_dir > 0:
                H = st.reg.hull()
                if len(H) >= 3:
                    C = st.reg.centroid()
                    rho_max = max(math.hypot(v[0] - C[0], v[1] - C[1]) for v in H)
                    if math.hypot(x - C[0], y - C[1]) + rho_max < R_RECV_MIN:
                        self.has_dir = True
        return res

    def _ready(self, st: Chan) -> bool:
        if st.cleared or st.n_dir == 0:
            return False
        thr = self.P["D_ready"] / (1.0 + 2.0 * st.fail)
        return st.D <= thr

    # ---- 扫描一个站 ----
    def _sweep(self, client, S, st, pos):
        order = [ch for ch in range(1, 21)
                 if not st[ch].cleared
                 and not (st[ch].n_dir > 0 and st[ch].D <= self.P["D_skip"])]
        self.mv["sweep"] += float(np.linalg.norm(np.asarray(S, float) - self._last))
        self._last = np.asarray(S, float)
        for ch in order:
            r = client.measure(S[0], S[1], ch)
            self.n_meas += 1
            res = self._update(st[ch], S[0], S[1], r)
            if res == "near":
                rr = client.clear(S[0], S[1], ch)
                self.n_clear += 1
                if rr.get("clear_result") == "success":
                    st[ch].cleared = True
        self.n_det = sum(1 for c in st.values() if c.n_dir > 0)
        return np.asarray(S, float)

    # ---- 精化 ----
    def _refine(self, client, ch, st: Chan, pos):
        for _ in range(5):
            if st.reg.empty or st.D <= self.P["D_final"]:
                break
            C = st.reg.centroid()
            mb = st.mean_bearing(pos)
            if mb is None:
                mb = math.degrees(math.atan2(C[1] - pos[1], C[0] - pos[0]))
            off = min(800.0, max(200.0, 2.0 * st.D))
            moved = False
            for sgn in (1, -1):
                ang = math.radians(mb + sgn * self.P["refine_off"])
                p = C + off * np.array([math.cos(ang), math.sin(ang)])
                self.mv["refine"] += float(np.linalg.norm(p - self._last)); self._last = p
                r = client.measure(p[0], p[1], ch)
                self.n_meas += 1
                res = self._update(st, p[0], p[1], r)
                pos = p
                if res in ("direction", "near"):
                    moved = True
                    break
                self.mv["refine"] += float(np.linalg.norm(C - self._last)); self._last = C
                r2 = client.measure(C[0], C[1], ch)
                self.n_meas += 1
                res2 = self._update(st, C[0], C[1], r2)
                pos = C
                if res2 in ("direction", "near"):
                    moved = True
                    break
            if not moved:
                break
        return pos

    # ---- 主动定位点（Q2/Q4 结论的工程实现） ----
    def _loc_points(self, st: Chan):
        """按优先序给出候选补测点。

        【Q2：全向源】单方位区域的形状是"从观测点 P0 出发、沿视线方向长约 t_max 的
        窄楔形"。补测点必须落在**中等深度** t≈(0.35~0.6)t_max 处并横向拉开 δ≈t/2，
        这样交会角 θ≈atan(δ/t)≈27°，可把 D 从 1300 m 压到数十米。

        【Q4：定向源】这是本题最容易翻车的地方：**阴影半平面以源为顶点**，
        当补测点越过源所在的深度后，(G−Q)·φ̂ 会由正变负，整条"远端"都收不到信号。
        实测反例（seed12/ch12：源在 (977,−760)，φ̂=(−0.721,−0.693)）：
        按质心末端取点 (1195,−1005) 得覆盖 dot=−13 → 必然 no_signal。
        因此：(a) 只在中等深度取点，不取视线末端；(b) **正反两侧都要试**——
        由于 φ̂ 先验均匀，阴影落在哪一侧等概率，单向尝试等价于 50% 失效率。
        """
        C = st.reg.centroid()
        V = st.reg.hull()
        if not st.obs or len(V) < 3:
            return [C]
        P0 = np.asarray(st.obs[-1][:2], float)
        pts = np.asarray(V, float)
        tmax = float(np.max(np.linalg.norm(pts - P0, axis=1)))
        v = C - P0
        n = float(np.linalg.norm(v))
        if n < 1e-6 or tmax < 1e-6:
            return [C]
        u = v / n
        w = np.array([-u[1], u[0]])
        if st.n_dir <= 1:
            off = min(250.0, max(60.0, 1.2 * st.D))
            if self.P["loc_off"] == "sml":
                # 【基线】单一方向、质心小偏置补测：不双向试探、不取中等深度。
                # 这正是 Q4 会失手的老做法（阴影半平面以源为顶点，单向尝试等价 50% 失效）。
                return [C + off * w]
            out = []
            for frac, sgn in ((0.35, 1.0), (0.35, -1.0), (0.60, 1.0), (0.60, -1.0)):
                t = frac * tmax
                p = P0 + t * u + sgn * (0.5 * t) * w
                if float(np.linalg.norm(p)) <= R_ARENA:      # 越界则退化为视线上的点
                    out.append(p)
                elif float(np.linalg.norm(P0 + t * u)) <= R_ARENA:
                    out.append(P0 + t * u)
            return out or [C]
        off = min(250.0, max(60.0, 1.2 * st.D))
        return [C + off * w, C - off * w]

    def _loc_point(self, st: Chan):
        return self._loc_points(st)[0]

    def _locate(self, client, ch, st: Chan, pos):
        """依次尝试候选补测点，取到第一个有效信号（direction/near）即停。"""
        for tgt in self._loc_points(st):
            self.mv["refine"] += float(np.linalg.norm(tgt - self._last)); self._last = tgt
            r = client.measure(tgt[0], tgt[1], ch)
            self.n_meas += 1
            res = self._update(st, tgt[0], tgt[1], r)
            st.n_loc += 1
            self.n_loc += 1
            pos = tgt
            if res in ("direction", "near"):
                break
            if st.reg.empty:
                break
        return pos

    # ---- 清剿 ----
    def _clear(self, client, ch, st: Chan, pos):
        pos = self._refine(client, ch, st, pos)
        V = st.reg.hull()
        cands = []
        if len(V) >= 3:
            p = np.array(V)
            xs = np.arange(p[:, 0].min(), p[:, 0].max() + 1e-9, self.P["grid_step"])
            ys = np.arange(p[:, 1].min(), p[:, 1].max() + 1e-9, self.P["grid_step"])
            for x in xs:
                for y in ys:
                    if st.reg.inside((x, y)):
                        cands.append((float(x), float(y)))
        if not cands:
            C = st.reg.centroid()
            cands = [(float(C[0]), float(C[1]))]
        cands.sort(key=lambda q: (q[0] - pos[0]) ** 2 + (q[1] - pos[1]) ** 2)
        for q in cands[:self.P["max_clear_try"]]:
            self.mv["clear"] += float(np.linalg.norm(np.array(q) - self._last)); self._last = np.array(q)
            r = client.clear(q[0], q[1], ch)
            self.n_clear += 1
            pos = np.array(q)
            if r.get("clear_result") == "success":
                st.cleared = True
                return True, pos
        C = st.reg.centroid()
        for k in range(12):
            ang = 2 * math.pi * k / 12
            q = C + 18.0 * np.array([math.cos(ang), math.sin(ang)])
            self.mv["clear"] += float(np.linalg.norm(q - self._last)); self._last = q
            r = client.clear(q[0], q[1], ch)
            self.n_clear += 1
            pos = q
            if r.get("clear_result") == "success":
                st.cleared = True
                return True, pos
        return False, pos

    # ---- 候选动作集：三类任务（扫描站 / 主动定位 / 清剿）----
    # 优先级规则（Q2 结论的工程落地）：
    #   1) 只要还有未扫的圆环站，就先扫站——相邻站基线 2a·sin(π/m)≈708 m，
    #      与 ~900 m 处的源构成 ~47° 交会角，单次观测即把 D 从 1300 m 压到 ~22 m，
    #      比"在质心附近偏 200 m 补测（交会角仅 ~12°）"高效得多；
    #   2) 只有"扫完全部站仍有 n_dir>=2 却 D>阈值"（几何病态）或 n_dir==1 的
    #      孤站源，才启用主动定位（沿垂直于视线的方向拉开基线）。
    def _actions(self, st, pending_st):
        acts = [(np.asarray(S, float), "st", -1) for S in pending_st]
        have_st = bool(pending_st)
        if self.P["two_pass"] and have_st:
            # 两趟式基线：覆盖站未扫完之前，绝不派发定位/清剿动作。
            return acts
        for ch in range(1, 21):
            c = st[ch]
            if c.cleared or c.n_dir == 0:
                continue
            if self._ready(c):
                acts.append((c.reg.centroid(), "cl", ch))
            elif c.n_loc < self.P["max_loc"] and (c.n_dir >= self.P["loc_min_dir"] or not have_st):
                acts.append((self._loc_point(c), "loc", ch))
        return acts

    # ---- 主流程：滚动重规划（每步只走 2-opt 巡回的第一站，随后按新信息重排）----
    def run(self, client, record=None):
        st = {ch: Chan.fresh() for ch in range(1, 21)}
        pos = np.array([0.0, 0.0])
        client.enter()
        inner, outer = None, None
        if self.P["layout"] == "dual":
            inner = ring_stations(self.P["ring_m"], self.P["ring_a"], self.P["ring_rot"])
            outer = ring_stations(self.P["ring_m2"], self.P["ring_a2"],
                                  self.P["ring_rot"] + math.pi / self.P["ring_m2"])
        elif self.P["layout"] == "ring":
            inner = ring_stations(self.P["ring_m"], self.P["ring_a"], self.P["ring_rot"])
        else:
            inner = plan_stations(self.P["r_cov"], margin=self.P["margin"])
        pending_st = [np.asarray(s, float) for s in tour(inner)]
        n_st = len(pending_st) + (len(outer) if outer else 0)
        n_cleared = 0
        it = 0
        while it < self.P["max_iter"]:
            it += 1
            acts = self._actions(st, pending_st)
            if not acts and not pending_st:
                # 第二阶段（外环补扫）：内环扫完且仍存在未识别频道时启用。
                # 依据：N∈[10,16]，若识别数已达 n_max 则全部源已找到，无需再扫；
                # 若仍有频道从未收到信号，则该源可能"朝圆心辐射"（内环恒为阴影），
                # 必须由 ρ'>ρ 的外环观测点才能进入其覆盖半平面（见报告 §Q4）。
                if outer and self.P["use_outer"] and self.has_dir \
                        and self.n_det < self.P["n_max"] \
                        and any(st[ch].n_dir == 0 for ch in range(1, 21)):
                    pending_st = [np.asarray(s, float) for s in tour(outer, start=pos)]
                    outer = None
                    acts = self._actions(st, pending_st)
            if not acts:
                break
            pts = [a[0] for a in acts]
            # 用 2-opt 巡回的前瞻值取下一步（滚动视界，只执行一步）
            first = np.asarray(tour(pts, start=pos)[0], float)
            k = min(range(len(acts)), key=lambda i: float(np.linalg.norm(pts[i] - first)))
            _, kind, ch = acts[k]

            if kind == "st":
                pos = self._sweep(client, pts[k], st, pos)
                pending_st = [S for S in pending_st
                              if float(np.linalg.norm(np.asarray(S, float) - pts[k])) > 1e-6]
            elif kind == "loc":
                pos = self._locate(client, ch, st[ch], pos)
            else:
                ok, pos = self._clear(client, ch, st[ch], pos)
                if ok:
                    n_cleared += 1
                else:
                    st[ch].fail += 1
            # 每步重算 n_det；达到题目上限 16 后不再派发扫描任务
            if pending_st and self.n_det >= self.P["n_max"]:
                pending_st = []
        client.exit()
        return dict(n_cleared=n_cleared, n_meas=self.n_meas, n_clear=self.n_clear,
                    virtual_time=client.virtual_time, stations=n_st,
                    n_det=self.n_det, mv=self.mv, n_loc=self.n_loc,
                    mv_total=sum(self.mv.values()))
