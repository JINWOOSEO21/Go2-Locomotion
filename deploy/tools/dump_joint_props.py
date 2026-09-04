"""IsaacLab 이 쓰는 Go2 의 **관절 물성**과 **접촉 물성**을 실측해 JSON 으로 떨군다.

MJCF(menagerie) 는 관절마다 armature=0.01, damping=0.1, frictionloss=0.2 를 들고 있다.
이 값들은 PD 게인(kp=40, kd=1.0) 위에 **더해지는** 항이라, 학습 쪽에 없으면 같은
목표각을 줘도 다리가 다르게 움직인다. 폐루프에서만 발산하는 현상의 후보다.

접촉 쪽(마찰/반발)도 같이 뜬다. 발 마찰은 이미 맞췄지만 반발계수는 아직 대조한 적이 없다.

Isaac 종료 시 stdout 이 잘리므로 결과는 파일로만 쓰고 [RESULT] 마커로 알린다.
"""
import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from parkour_tasks.default_cfg import ParkourDefaultSceneCfg  # noqa: E402

out = {}
ok = False


def tol(x):
    return x[0].cpu().numpy().tolist()


try:
    scene_cfg = ParkourDefaultSceneCfg(num_envs=1, env_spacing=1.0)
    robot_cfg = scene_cfg.robot.replace(prim_path="/World/Robot")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(robot_cfg)
    sim.reset()

    v = robot.root_physx_view
    out["joint_names"] = list(robot.data.joint_names)

    # 실제로 물리 엔진에 들어가 있는 값 (cfg 가 아니라 뷰에서 읽는다)
    for key, fn in (
        ("armature", "get_dof_armatures"),
        ("joint_friction", "get_dof_friction_coefficients"),
        ("max_velocity", "get_dof_max_velocities"),
        ("max_force", "get_dof_max_forces"),
        ("stiffness_drive", "get_dof_stiffnesses"),
        ("damping_drive", "get_dof_dampings"),
        ("limits", "get_dof_limits"),
    ):
        try:
            out[key] = tol(getattr(v, fn)())
        except Exception as e:  # noqa: BLE001
            out[key] = f"<불가: {e}>"

    # ArticulationData 쪽 사본 (액추에이터가 덮어쓴 뒤의 값)
    for key in ("joint_stiffness", "joint_damping", "joint_armature", "joint_friction",
                "joint_pos_limits", "joint_vel_limits", "joint_effort_limits"):
        try:
            out[f"data.{key}"] = tol(getattr(robot.data, key))
        except Exception as e:  # noqa: BLE001
            out[f"data.{key}"] = f"<불가: {e}>"

    # 액추에이터 그룹이 실제로 들고 있는 값
    acts = {}
    for name, a in robot.actuators.items():
        d = {"class": type(a).__name__, "joint_names": list(a.joint_names)}
        for attr in ("stiffness", "damping", "armature", "friction", "effort_limit",
                     "velocity_limit", "saturation_effort", "effort_limit_sim",
                     "velocity_limit_sim"):
            val = getattr(a, attr, None)
            if val is None:
                continue
            try:
                d[attr] = val[0].cpu().numpy().tolist()
            except Exception:  # noqa: BLE001
                d[attr] = float(val)
        acts[name] = d
    out["actuators"] = acts

    # 접촉 물성 — 로봇 강체별 (static, dynamic, restitution)
    try:
        mats = v.get_material_properties()[0].cpu().numpy()
        out["body_names"] = list(robot.data.body_names)
        out["material_static_dynamic_restitution"] = mats.tolist()
    except Exception as e:  # noqa: BLE001
        out["material_static_dynamic_restitution"] = f"<불가: {e}>"

    # spawn cfg 쪽 물성 (contact offset 등은 여기에만 있다)
    for key in ("rigid_props", "collision_props", "articulation_props"):
        p = getattr(robot_cfg.spawn, key, None) if robot_cfg.spawn is not None else None
        out[key] = None if p is None else {
            k: getattr(p, k) for k in dir(p)
            if not k.startswith("_") and not callable(getattr(p, k))
        }
    ok = True
except Exception:  # noqa: BLE001
    import traceback

    traceback.print_exc()

with open(args.out, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"[RESULT] {'OK' if ok else 'FAILED'}", flush=True)
simulation_app.close()
