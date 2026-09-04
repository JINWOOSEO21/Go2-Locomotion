"""같은 관절 목표를 주고 평지에 세웠을 때 **base 높이**가 IsaacLab 과 MuJoCo 에서 같은가.

링크 기하(관절 오프셋)는 2 mm 이내로 일치함을 확인했다. 그런데 실제 지지 높이는
링크 길이만으로 정해지지 않는다 — 발의 충돌 반지름, 접촉 침투(penetration),
액추에이터가 중력에 지는 정도가 모두 들어간다. 그 합이 다르면 같은 관절 궤적이
다른 보폭·다른 지면 여유를 만든다.

정책이 쓰는 것과 같은 액추에이터(ParkourDCMotor, kp=40/kd=1)로 기본 자세를 명령하고
가만히 두어 정착시킨 뒤, base z 와 관절 추종오차, 발 위치를 기록한다. MuJoCo 에서
같은 절차를 돌린 값과 견주면 된다.

Isaac 종료 시 stdout 이 잘리므로 결과는 파일로만 쓰고 [RESULT] 마커로 알린다.
"""
import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--settle", type=float, default=4.0, help="정착 시간 [s]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from parkour_tasks.default_cfg import ParkourDefaultSceneCfg  # noqa: E402

out = {}
ok = False
try:
    scene_cfg = ParkourDefaultSceneCfg(num_envs=1, env_spacing=1.0)
    robot_cfg = scene_cfg.robot.replace(prim_path="/World/Robot")
    dt = 0.005
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0"))
    # 지형과 같은 마찰의 평지 (학습 terrain 은 static/dynamic 1.0)
    ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average", restitution_combine_mode="average",
            static_friction=1.0, dynamic_friction=1.0,
        )
    )
    ground.func("/World/ground", ground)
    robot = Articulation(robot_cfg)
    sim.reset()

    names = list(robot.data.joint_names)
    q_def = robot.data.default_joint_pos.clone()

    # 스폰: 기본 자세 그대로 살짝 띄워 놓고 떨어뜨린다
    root = robot.data.default_root_state.clone()
    root[:, 2] = 0.45
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(q_def, torch.zeros_like(q_def))
    robot.reset()

    n = int(args.settle / dt)
    for _ in range(n):
        robot.set_joint_position_target(q_def)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    body_names = list(robot.data.body_names)
    bp = robot.data.body_pos_w[0].cpu().numpy()
    rp = robot.data.root_pos_w[0].cpu().numpy()
    out["joint_names"] = names
    out["q_target"] = q_def[0].cpu().numpy().tolist()
    out["q_actual"] = robot.data.joint_pos[0].cpu().numpy().tolist()
    out["base_z"] = float(rp[2])
    out["root_pos_w"] = rp.tolist()
    out["applied_torque"] = robot.data.applied_torque[0].cpu().numpy().tolist()
    out["body_names"] = body_names
    # 발 원점의 월드 z 와 base 기준 z
    feet = {}
    for i, bn in enumerate(body_names):
        if bn.endswith("_foot"):
            feet[bn] = {"world_z": float(bp[i, 2]),
                        "below_base": float(bp[i, 2] - rp[2])}
    out["feet"] = feet
    out["joint_vel_max"] = float(robot.data.joint_vel[0].abs().max())
    ok = True
except Exception:  # noqa: BLE001
    import traceback

    traceback.print_exc()

with open(args.out, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"[RESULT] {'OK' if ok else 'FAILED'}", flush=True)
simulation_app.close()
