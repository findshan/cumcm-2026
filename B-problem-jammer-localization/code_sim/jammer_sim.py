"""
B题（2026 国赛）官方模拟器的 Python 等价实现。

严格依据：
- 题面 附录1（干扰源特征）、附录2（测向机原理与工作方式）
- 附件1《模拟器使用说明》（动作/指令表、虚拟世界计时规则）
- 附件2《模拟器通信接口说明及编程指南》（4 条指令、JSON 字段、响应语义）

设计目标：让"机器狗程序"用与官方完全一致的 HTTP+JSON 语义编写，
只需把 transport 从 LocalTransport 换成 HttpTransport 即可对真模拟器运行。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

# ---------------- 物理常数（附录1/2） ----------------
R_ARENA = 1800.0          # 目标区域半径
V_ROBOT = 5.0             # 机器狗速度 m/s
T_MEASURE = 5.0           # 检测动作耗时
T_SWITCH = 1.0            # 任意两频道切换耗时
T_CLEAR_FAIL = 3.0        # /clear 未发现：仅光学精确定位
T_CLEAR_OK = 5.0          # /clear 成功：精确定位 + 清除
R_NEAR = 5.0              # 距离过近阈值（返回 near，无示向度）
R_CLEAR = 20.0            # 清除半径
R_RECV_MIN = 1000.0       # 有效接收半径下界
R_RECV_MAX = 1500.0       # 有效接收半径上界
MAX_COORD = 2_000_000.0   # 坐标分量绝对值上限
VIRTUAL_LIMIT = 360000.0  # 虚拟世界限时
REAL_LIMIT = 1200.0       # /enter 后程序运行时间上限


def norm_deg(a: float) -> float:
    return a % 360.0


# ---------------- 示向度误差场（位置固定、值域 [-1,1]） ----------------
class ErrorField:
    """题面：'在一段时间内，同一地点的电磁环境干扰是固定的，所以重复检测不会改变检测误差。
    只有在不同地点、不同电磁环境下，误差才会呈现统计规律。从目标区域全局来看，
    所有这些误差在 [-1°, 1°] 范围内。'

    因此误差是**位置相关的确定性偏差**，不是白噪声。我们用三种模型做鲁棒性测试：
      - smooth : 平滑随机场（相关长度 L 可配），最接近真实电磁环境
      - iid    : 每个位置独立均匀 [-1,1]（误差完全不相关）
      - grid   : 网格常数场（同一格内误差相同，模拟'电磁环境分区'）
    """

    def __init__(self, rng: np.random.Generator, mode: str = "smooth", corr_len: float = 1500.0):
        self.rng = rng
        self.mode = mode
        self.corr_len = corr_len
        if mode == "smooth":
            self.K = 8
            amp = rng.normal(0.0, 1.0, self.K)
            self.amp = amp / np.abs(amp).sum()
            # 波数：使主要尺度约等于 corr_len
            k = 2 * math.pi / max(corr_len, 1.0)
            self.kx = k * rng.uniform(0.5, 1.6, self.K)
            self.ky = k * rng.uniform(0.5, 1.6, self.K)
            self.ph = rng.uniform(0, 2 * math.pi, self.K)
        elif mode == "iid":
            self._cache: dict[tuple[int, int], float] = {}
        elif mode == "grid":
            self.cell = max(corr_len, 1.0)
            self._cache = {}
        else:
            raise ValueError(mode)

    def __call__(self, x: float, y: float) -> float:
        if self.mode == "smooth":
            s = float(np.sum(self.amp * np.sin(self.kx * x + self.ky * y + self.ph)))
            return float(np.clip(s, -1.0, 1.0))
        key = (int(round(x)), int(round(y))) if self.mode == "iid" else (
            int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))
        if key not in self._cache:
            self._cache[key] = float(self.rng.uniform(-1.0, 1.0))
        return self._cache[key]


# ---------------- 案例数据 ----------------
@dataclass
class Case:
    """一局测试的隐藏真值。"""
    N: int
    channels: list[int]          # 每个干扰源的频道
    pos: np.ndarray              # (N,2)
    radius: np.ndarray           # (N,) 有效接收半径
    is_dir: np.ndarray           # (N,) 是否定向
    phi: np.ndarray              # (N,) 定向方向（弧度），全向源忽略
    seed: int = 0

    def channel_of(self, ch: int) -> int | None:
        for i, c in enumerate(self.channels):
            if c == ch:
                return i
        return None

    @property
    def n_dir(self) -> int:
        return int(self.is_dir.sum())


def make_case(rng: np.random.Generator, n_min: int = 10, n_max: int = 16,
              dir_prob: float = 0.0, seed: int = 0) -> Case:
    """随机生成一局案例。位置：圆盘内均匀；频道：从 1..20 无重复抽取；
    有效接收半径：U[1000,1500]；定向源：以 dir_prob 概率为定向，定向方向 U[0,2pi)。"""
    N = int(rng.integers(n_min, n_max + 1))
    channels = sorted(rng.choice(np.arange(1, 21), size=N, replace=False).tolist())
    # 圆盘内均匀
    r = R_ARENA * np.sqrt(rng.uniform(0, 1, N))
    th = rng.uniform(0, 2 * math.pi, N)
    pos = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    radius = rng.uniform(R_RECV_MIN, R_RECV_MAX, N)
    is_dir = rng.uniform(0, 1, N) < dir_prob
    phi = rng.uniform(0, 2 * math.pi, N)
    return Case(N, channels, pos, radius, is_dir, phi, seed=seed)


# ---------------- 模拟器 ----------------
class Simulator:
    """实现 /enter /measure /clear /exit 四条指令、虚拟计时规则、幂等与错误码。"""

    def __init__(self, case: Case, robot_id: str = "2026123456",
                 err: ErrorField | None = None, seed: int = 0,
                 real_limit: float = REAL_LIMIT, virtual_limit: float = VIRTUAL_LIMIT):
        self.case = case
        self.robot_id = robot_id
        self.err = err
        self.rng = np.random.default_rng(seed + 991)
        self.real_limit = real_limit
        self.virtual_limit = virtual_limit

        self.entered = False
        self.exited = False
        self.pos = np.array([0.0, 0.0])
        self.chan = 1                     # 测向机当前频道
        self.t = 0.0                      # 虚拟时刻
        self.cleared: set[int] = set()     # 已清除的源下标
        self._req: dict[str, tuple[str, dict]] = {}   # request_id -> (payload hash, response)
        self.n_measure = 0
        self.n_clear = 0
        self.n_switch = 0
        self.move_m = 0.0

    # ---- 工具 ----
    def _h(self, payload: dict) -> str:
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _resp(self, accepted: bool, **kw) -> dict:
        d = {"accepted": bool(accepted), "real_timestamp_ms": 1760000000000, "virtual_time_s": self.t}
        if accepted:
            d.update(kw)
        return d

    def _bad(self, msg="") -> tuple[int, dict]:
        return 400, {"accepted": False, "real_timestamp_ms": 1760000000000,
                     "virtual_time_s": 0.0, "err_msg": msg}

    def _check_common(self, path: str, payload: dict, allowed: set[str]):
        """返回 (status, resp) 或 None 表示通过。"""
        if not isinstance(payload, dict):
            return self._bad("not a json object")
        extra = set(payload) - allowed
        if extra:
            # 未声明字段 -> HTTP 200 且 accepted=false
            return 200, self._resp(False, err_msg=f"unknown fields {sorted(extra)}")
        for k in ("arena_id", "robot_id", "request_id"):
            if k not in payload:
                return self._bad(f"missing {k}")
        if payload["arena_id"] != "default":
            return 200, self._resp(False, err_msg="arena_id mismatch")
        if payload["robot_id"] != self.robot_id:
            return 200, self._resp(False, err_msg="robot_id mismatch")
        rid = payload["request_id"]
        if not isinstance(rid, str) or not (1 <= len(rid.encode()) <= 128):
            return self._bad("bad request_id")
        return None

    def _idem(self, path: str, payload: dict):
        """幂等：同 id 同内容 -> 返回首次完整响应；同 id 改内容 -> 409。"""
        rid = payload["request_id"]
        h = self._h(payload)
        if rid in self._req:
            h0, r0 = self._req[rid]
            if h0 == h:
                return 200, r0
            return 409, {"accepted": False, "real_timestamp_ms": 1760000000000,
                         "virtual_time_s": 0.0, "err_msg": "request_id reused"}
        return None

    def _move_time(self, p: np.ndarray) -> float:
        d = float(np.linalg.norm(p - self.pos))
        self.move_m += d
        return d / V_ROBOT

    # ---- 指令 ----
    def handle(self, path: str, payload: dict) -> tuple[int, dict]:
        ALLOWED = {"arena_id", "robot_id", "request_id"}
        if path == "/enter":
            r = self._check_common(path, payload, ALLOWED)
            if r:
                return r
            r = self._idem(path, payload)
            if r:
                return r
            self.entered = True
            resp = self._resp(True, remaining_real_duration_s=self.real_limit)
            self._req[payload["request_id"]] = (self._h(payload), resp)
            return 200, resp

        if path == "/exit":
            r = self._check_common(path, payload, ALLOWED)
            if r:
                return r
            r = self._idem(path, payload)
            if r:
                return r
            self.exited = True
            resp = self._resp(True, exit_reason="user_exit")
            self._req[payload["request_id"]] = (self._h(payload), resp)
            return 200, resp

        if path in ("/measure", "/clear"):
            allowed = ALLOWED | {"position", "channel"}
            r = self._check_common(path, payload, allowed)
            if r:
                return r
            pos = payload.get("position")
            if not isinstance(pos, dict) or set(pos) - {"x", "y"} or "x" not in pos or "y" not in pos:
                return self._bad("bad position")
            for k in ("x", "y"):
                v = pos[k]
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) \
                        or abs(v) > MAX_COORD:
                    return self._bad(f"bad {k}")
            ch = payload.get("channel")
            if not isinstance(ch, (int, float)) or isinstance(ch, bool) or int(ch) != ch \
                    or not (1 <= int(ch) <= 20):
                return self._bad("bad channel")
            ch = int(ch)
            if not self.entered or self.exited:
                return 200, self._resp(False, err_msg="not entered")
            r = self._idem(path, payload)
            if r:
                return r

            p = np.array([float(pos["x"]), float(pos["y"])])
            mv = self._move_time(p)
            self.pos = p
            idx = self.case.channel_of(ch)

            if path == "/measure":
                sw = 0.0
                if ch != self.chan:
                    sw = T_SWITCH
                    self.n_switch += 1
                self.t += mv + sw + T_MEASURE
                self.chan = ch
                self.n_measure += 1
                body = self._measure_body(p, ch, idx)
                resp = self._resp(True, **body)
            else:
                self.t += mv + T_CLEAR_FAIL
                self.n_clear += 1
                ok = (idx is not None and idx not in self.cleared
                      and float(np.linalg.norm(self.case.pos[idx] - p)) <= R_CLEAR)
                if ok:
                    self.cleared.add(idx)
                    self.t += (T_CLEAR_OK - T_CLEAR_FAIL)
                    resp = self._resp(True, clear_result="success")
                else:
                    resp = self._resp(True, clear_result="no_target_in_range")
            self._req[payload["request_id"]] = (self._h(payload), resp)
            return 200, resp

        return 404, {"accepted": False, "real_timestamp_ms": 1760000000000,
                     "virtual_time_s": 0.0, "err_msg": "unknown path"}

    def _measure_body(self, p: np.ndarray, ch: int, idx: int | None) -> dict:
        if idx is None or idx in self.cleared:
            return {"measure_result": "no_signal"}
        G = self.case.pos[idx]
        v = G - p
        d = float(np.linalg.norm(v))
        if d > self.case.radius[idx]:
            return {"measure_result": "no_signal"}
        if self.case.is_dir[idx]:
            cov = float(np.dot(v, np.array([math.cos(self.case.phi[idx]),
                                            math.sin(self.case.phi[idx])])))
            if cov < 0:                      # 定向源背后 -> 无辐射
                return {"measure_result": "no_signal"}
        if d <= R_NEAR:
            return {"measure_result": "near"}
        true_b = math.degrees(math.atan2(v[1], v[0]))
        e = self.err(p[0], p[1]) if self.err is not None else 0.0
        svd = round(norm_deg(true_b + e), 2) % 360.0
        return {"measure_result": "direction", "svd_deg": svd}

    # ---- 统计 ----
    def stats(self) -> dict:
        return dict(N=self.case.N, n_cleared=len(self.cleared), total_time=self.t,
                    move_m=round(self.move_m, 1), n_measure=self.n_measure,
                    n_clear=self.n_clear, n_switch=self.n_switch)


# ---------------- 客户端（传输层可替换） ----------------
class LocalTransport:
    def __init__(self, sim: Simulator):
        self.sim = sim

    def post(self, path: str, payload: dict):
        return self.sim.handle(path, payload)


class HttpTransport:
    """对真模拟器（默认 http://127.0.0.1:2026）发送请求，语义与 LocalTransport 一致。"""

    def __init__(self, base_url: str = "http://127.0.0.1:2026", timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, payload: dict):
        import urllib.error
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8"))
            except Exception:
                return e.code, {"accepted": False, "virtual_time_s": 0.0,
                                "real_timestamp_ms": 0, "err_msg": "http error"}
        except Exception as ex:                     # 连接失败：接口未开放/已结束
            return 0, {"accepted": False, "virtual_time_s": 0.0,
                       "real_timestamp_ms": 0, "err_msg": f"conn fail: {ex}"}


class RobotClient:
    """机器狗客户端：新动作用新 request_id；仅重试时复用原 id 与内容；串行。"""

    def __init__(self, transport, robot_id: str, max_retry: int = 3, verbose: bool = False):
        self.tp = transport
        self.robot_id = robot_id
        self.max_retry = max_retry
        self.verbose = verbose
        self.seq = 0
        self.log: list[dict] = []
        self.virtual_time = 0.0
        self.n_req = 0

    def _call(self, path: str, extra: dict | None = None):
        self.seq += 1
        payload = {"arena_id": "default", "robot_id": self.robot_id,
                   "request_id": f"{path.strip('/')}-{self.seq}"}
        if extra:
            payload.update(extra)
        for attempt in range(self.max_retry):
            self.n_req += 1
            status, resp = self.tp.post(path, payload)      # 重试复用同一 payload
            if status != 0:
                break
        else:
            raise RuntimeError(f"{path} 连接失败")
        if not resp.get("accepted"):
            raise RuntimeError(f"{path} 被拒绝: http={status} resp={resp}")
        if status >= 400:
            raise RuntimeError(f"{path} http={status} resp={resp}")
        self.virtual_time = float(resp.get("virtual_time_s", self.virtual_time))
        self.log.append({"path": path, "req": payload, "status": status, "resp": resp})
        return resp

    def enter(self):
        return self._call("/enter")

    def measure(self, x: float, y: float, channel: int):
        return self._call("/measure", {"position": {"x": float(x), "y": float(y)}, "channel": int(channel)})

    def clear(self, x: float, y: float, channel: int):
        return self._call("/clear", {"position": {"x": float(x), "y": float(y)}, "channel": int(channel)})

    def exit(self):
        return self._call("/exit")
