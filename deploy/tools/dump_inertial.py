"""IsaacLab 이 쓰는 Go2 의 질량/관성/COM 을 실측해 JSON 으로 떨군다.

deploy.yaml 의 caveats 에 "질량/관성/COM 이 IsaacLab USD 와 menagerie MJCF 사이에서
일치하는지 미확인" 으로 남아 있던 항목이다. 관성이 다르면 같은 토크에도 몸이 다르게
돌기 때문에, 관측·액추에이터가 같은데도 자세가 발산하는 현상의 유력한 후보다.

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
try:
    scene_cfg = ParkourDefaultSceneCfg(num_envs=1, env_spacing=1.0)
    robot_cfg = scene_cfg.robot.replace(prim_path="/World/Robot")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(robot_cfg)
    sim.reset()

    names = list(robot.data.body_names)
    masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
    inertias = robot.root_physx_view.get_inertias()[0].cpu().numpy()   # (B, 9)
    coms = robot.root_physx_view.get_coms()[0].cpu().numpy()           # (B, 7) pos+quat

    out["body_names"] = names
    out["mass"] = masses.tolist()
    out["total_mass"] = float(masses.sum())
    out["inertia"] = inertias.tolist()
    out["com"] = coms.tolist()
    ok = True
except Exception:  # noqa: BLE001
    import traceback

    traceback.print_exc()

with open(args.out, "w") as f:
    json.dump(out, f, indent=2)
print(f"[RESULT] {'OK' if ok else 'FAILED'}", flush=True)
simulation_app.close()
