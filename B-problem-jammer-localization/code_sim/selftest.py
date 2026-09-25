"""自检：用附件1 表2 的官方例子核对虚拟计时是否逐秒吻合。"""

import numpy as np

from jammer_sim import Case, ErrorField, LocalTransport, RobotClient, Simulator

# 官方例子的场景：与干扰源无关，只看计时
case = Case(N=3, channels=[1, 2, 3],
            pos=np.array([[900.0, 900.0], [1000.0, 1000.0], [800.0, 800.0]]),
            radius=np.array([1500.0, 1500.0, 1500.0]),
            is_dir=np.array([False, False, False]),
            phi=np.zeros(3))
sim = Simulator(case, robot_id="team", err=None)
cli = RobotClient(LocalTransport(sim), "team")

cli.enter()
seq = [("measure", 300, 400, 1, 105), ("measure", 300, 400, 2, 111),
       ("clear", 300, 0, 3, 194), ("measure", 300, 0, 2, 199)]
ok = True
for kind, x, y, ch, want in seq:
    r = cli.measure(x, y, ch) if kind == "measure" else cli.clear(x, y, ch)
    got = r["virtual_time_s"]
    flag = "OK " if abs(got - want) < 1e-9 else "FAIL"
    if flag == "FAIL":
        ok = False
    print(f"{flag} {kind:8s} pos=({x},{y}) ch={ch}  期望 t={want:6.1f}  实得 t={got:6.1f}")
r = cli.exit()
print(f"    exit    期望 t=199.0  实得 t={r['virtual_time_s']:6.1f}  reason={r['exit_reason']}")
print("\n计时规则自检：", "全部通过" if ok else "存在不一致")
print(sim.stats())

# 协议自检：未知字段 / 频道越界 / 坐标越界 / 幂等
print("\n--- 协议自检 ---")
s2 = Simulator(case, robot_id="team", err=None)
base = {"arena_id": "default", "robot_id": "team", "request_id": "x1"}
print("未 enter 就 measure ->", s2.handle("/measure", {**base, "position": {"x": 0, "y": 0}, "channel": 1}))
print("未知字段 ->", s2.handle("/enter", {**base, "foo": 1}))
print("坐标越界 ->", s2.handle("/enter", base)[0], s2.handle(
    "/measure", {**base, "position": {"x": 3e6, "y": 0}, "channel": 1}))
print("频道越界 ->", s2.handle("/measure", {**base, "position": {"x": 0, "y": 0}, "channel": 21}))
s2.handle("/enter", base)
r1 = s2.handle("/measure", {**base, "position": {"x": 10, "y": 0}, "channel": 1})
r2 = s2.handle("/measure", {**base, "position": {"x": 10, "y": 0}, "channel": 1})
print("幂等（同 id 同内容，虚拟时刻不应推进）->", r1[1]["virtual_time_s"], r2[1]["virtual_time_s"])
r3 = s2.handle("/measure", {**base, "position": {"x": 20, "y": 0}, "channel": 1})
print("同 id 改内容 ->", r3[0], r3[1].get("err_msg"))
print("路径错误 ->", s2.handle("/foo", base)[0])
