"""IsaacLab 이 파싱한 Go2 관절/바디 순서를 실측해 JSON 으로 떨군다.

지형을 만들지 않고 Articulation 하나만 띄운다 (VRAM 절약).
Isaac 종료 시 stdout 이 잘리는 함정이 있어 결과는 파일로만 쓴다.
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

from locomotion_tasks.default_cfg import LocomotionDefaultSceneCfg  # noqa: E402

out = {}
try:
    # 파쿠르 씬과 같은 robot cfg (actuator override 는 __post_init__ 에서 걸린다)
    scene_cfg = LocomotionDefaultSceneCfg(num_envs=1, env_spacing=1.0)
    robot_cfg = scene_cfg.robot.replace(prim_path="/World/Robot")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(robot_cfg)
    sim.reset()

    d = robot.data
    out["joint_names"] = list(d.joint_names)
    out["body_names"] = list(d.body_names)
    out["default_joint_pos"] = d.default_joint_pos[0].cpu().tolist()
    out["default_joint_stiffness"] = d.default_joint_stiffness[0].cpu().tolist()
    out["default_joint_damping"] = d.default_joint_damping[0].cpu().tolist()
    # actuator 그룹별 실제 해석된 한계값
    acts = {}
    for name, act in robot.actuators.items():
        acts[name] = {
            "joint_names": list(act.joint_names),
            "effort_limit": act.effort_limit[0].cpu().tolist(),
            "velocity_limit": act.velocity_limit[0].cpu().tolist(),
            "stiffness": act.stiffness[0].cpu().tolist(),
            "damping": act.damping[0].cpu().tolist(),
        }
        sat = getattr(act, "_saturation_effort", None)
        if sat is not None and hasattr(sat, "cpu"):
            acts[name]["saturation_effort"] = sat[0].cpu().tolist()
    out["actuators"] = acts
    out["ok"] = True
except Exception as e:  # noqa: BLE001
    import traceback

    out["ok"] = False
    out["error"] = f"{type(e).__name__}: {e}"
    out["traceback"] = traceback.format_exc()

with open(args.out, "w") as f:
    json.dump(out, f, indent=2)

simulation_app.close()
