"""학습(IsaacLab PLAY) 자신의 램프 통과 성공률 — sim2sim 판정의 기준선.

MuJoCo 가 20초에 8회 중 3회 통과한다. 그런데 **학습이 몇 번 통과하는지** 를
재 본 적이 없다. 기준선을 1회만 돌려 놓고 "학습은 항상 된다" 고 가정했다.
그 가정이 틀리면 sim2sim 격차 자체가 없는 셈이 된다.

같은 정책·같은 지형(PLAY, difficulty 0.7)을 num_envs 개 병렬로 돌리고,
env 마다 "고원(30 cm 이상)에 올라갔는가" 와 전진 거리를 센다.
판정 기준은 MuJoCo 쪽 traverse_check 와 같게 맞춘다.

Isaac 종료 시 stdout 이 잘리므로 결과는 파일로만 쓰고 [RESULT] 마커로 알린다.
"""
import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts", "rsl_rl"))
# 바깥 parkour_tasks/ 가 네임스페이스 패키지로 진짜 패키지를 가리는 것을 막는다
# (dump_golden_trace.py 의 같은 주석 참조).
sys.path.insert(0, os.path.join(_REPO, "parkour_tasks"))

import cli_args  # noqa: E402  (scripts/rsl_rl/cli_args.py)
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1000, help="정책 스텝 (0.02 s/스텝)")

parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=1)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import parkour_tasks  # noqa: F401, E402

out = {}
ok = False
try:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    n = args_cli.num_envs
    start = robot.data.root_pos_w[:, :3].clone()
    max_h = torch.full((n,), -10.0, device=env.unwrapped.device)
    max_x = torch.zeros(n, device=env.unwrapped.device)
    resets = torch.zeros(n, device=env.unwrapped.device)
    origins = env.unwrapped.scene.env_origins.clone()

    with torch.inference_mode():
        for _i in range(args_cli.steps):
            if _i % 50 == 0:
                print(f"[progress] {_i}/{args_cli.steps}", flush=True)
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            p = robot.data.root_pos_w[:, :3]
            # 로봇 밑 지형 높이 ~ base z - 기립 높이(0.32). 등반 판정에 충분하다.
            h = p[:, 2] - 0.32
            max_h = torch.maximum(max_h, h)
            max_x = torch.maximum(max_x, p[:, 0] - start[:, 0])
            resets += dones.float()
            if dones.any():
                start[dones.bool()] = robot.data.root_pos_w[dones.bool(), :3]

    out["num_envs"] = n
    out["steps"] = args_cli.steps
    out["max_terrain_h"] = max_h.cpu().numpy().tolist()
    out["max_dx"] = max_x.cpu().numpy().tolist()
    out["resets"] = resets.cpu().numpy().tolist()
    out["env_origins"] = origins.cpu().numpy().tolist()
    ok = True
except Exception:  # noqa: BLE001
    import traceback
    traceback.print_exc()

with open(args_cli.out, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"[RESULT] {'OK' if ok else 'FAILED'}", flush=True)
simulation_app.close()
