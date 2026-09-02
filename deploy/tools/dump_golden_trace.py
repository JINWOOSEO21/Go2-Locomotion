"""IsaacLab EM student 를 N 스텝 굴려 '골든 트레이스' 를 npz 로 남긴다.

무엇에 쓰나
-----------
1) **정책 검증(Phase 0)**: 여기 기록된 prop/scan/hist 를 policy.onnx 에 넣어
   기록된 actions 가 재현되는지 본다. 재현되면 ONNX 추출·차원·활성함수·슬라이스
   정렬이 전부 맞다는 뜻이다. `deploy/tools/check_onnx_against_trace.py` 가 한다.
2) **물리 검증(Phase 2 이후)**: 같은 액션을 MuJoCo 에 먹였을 때 q/dq/토크/자세가
   얼마나 갈라지는지 비교하는 기준선.

play.py 를 고치지 않고 별도 스크립트로 둔 이유는, 이 저장소는 산출물(onnx/yaml/트레이스)
만 뱉는 자리로 두기로 했기 때문이다 (docs/mujoco_sim2sim_plan.md §6).

기록 시점이 중요하다
--------------------
정책 입력은 **env.step 을 부르기 전에** 떠야 한다. step 이 obs 를 갈아끼우므로
나중에 뜨면 "그 액션을 만든 입력" 이 아니라 "그 다음 스텝의 입력" 이 기록된다.
그래서 아래 루프는 policy 호출 직후, step 직전에 기록한다.

실행 (Isaac Sim 필요)
---------------------
    python deploy/tools/dump_golden_trace.py --headless --num_envs 1 --steps 100 \
        --task Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-Play-v0 \
        --checkpoint logs/.../trained_v1.3~30K/model_29998.pt \
        --out logs/.../trained_v1.3~30K/exported/golden_trace.npz
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "scripts", "rsl_rl"))
sys.path.insert(0, _REPO)

import cli_args  # noqa: E402  (scripts/rsl_rl/cli_args.py)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--out", required=True, help="출력 npz 경로")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=1, help="지형 노이즈/리셋/DR 재현용")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""이하 Isaac Sim 기동 후에만 import 가능."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import parkour_tasks  # noqa: F401, E402
from scripts.rsl_rl.checkpoint_utils import get_checkpoint_path_with_fallback  # noqa: E402
from scripts.rsl_rl.modules import OnPolicyRunnerWithExtractor  # noqa: E402
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: E402


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.seed is not None:
        env_cfg.seed = agent_cfg.seed

    # student 씬은 녹화용 추격 카메라(TiledCamera)를 상시로 들고 있다. 그대로 두면
    # "A camera was spawned without the --enable_cameras flag" 로 죽고, 켜면 env 마다
    # 렌더가 돌아 VRAM 을 먹는다. train.py 가 학습 경로에서 떼어내는 것과 같은 처리다.
    if getattr(env_cfg.scene, "record_camera", None) is not None:
        env_cfg.scene.record_camera = None
        print("[INFO] record_camera 제거 (--enable_cameras 불필요, VRAM 절약)")

    if agent_cfg.algorithm.class_name != "EMDistillation":
        raise SystemExit(
            f"이 스크립트는 EM student 전용이다. --task 의 algorithm={agent_cfg.algorithm.class_name}"
        )

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path_with_fallback(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    print(f"[INFO] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    estimator = runner.get_estimator_inference_policy(device=env.device)
    policy = runner.get_inference_depth_policy(device=env.unwrapped.device)

    paras = agent_cfg.to_dict()["estimator"]
    P, S, PE = paras["num_prop"], paras["num_scan"], paras["num_priv_explicit"]
    H = paras["num_hist"] * P
    print(f"[INFO] dims: prop={P} scan={S} priv_explicit={PE} hist={H}")

    robot = env.unwrapped.scene["robot"]
    contacts = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids = robot.find_bodies(".*_foot")[0]

    rec: dict[str, list] = {k: [] for k in (
        "prop", "scan", "hist", "actions", "obs",
        "joint_pos", "joint_vel", "applied_torque",
        "root_pos_w", "root_quat_w", "root_lin_vel_b", "root_ang_vel_b",
        "foot_contact_forces", "episode_length",
    )}

    def add(name, t):
        rec[name].append(t.detach().cpu().numpy().copy())

    obs, extras = env.get_observations()
    try:
        for step in range(args_cli.steps):
            em_scan = extras["observations"]["em_scan"].to(env.device)
            with torch.inference_mode():
                # play.py:616-631 의 EM 분기와 글자 그대로 같은 순서다.
                obs_em = obs.clone()
                obs_em[:, P : P + S] = em_scan
                obs_em[:, P + S : P + S + PE] = estimator.inference(obs_em[:, :P])
                actions = policy(obs_em, hist_encoding=True)

            # --- step 전에 기록한다 (이 액션을 만든 입력을 남겨야 한다) ---
            add("prop", obs_em[:, :P])
            add("scan", obs_em[:, P : P + S])
            add("hist", obs_em[:, -H:])
            add("actions", actions)
            add("obs", obs_em)
            add("joint_pos", robot.data.joint_pos)
            add("joint_vel", robot.data.joint_vel)
            add("applied_torque", robot.data.applied_torque)
            add("root_pos_w", robot.data.root_pos_w)
            add("root_quat_w", robot.data.root_quat_w)
            add("root_lin_vel_b", robot.data.root_lin_vel_b)
            add("root_ang_vel_b", robot.data.root_ang_vel_b)
            add("foot_contact_forces", contacts.data.net_forces_w[:, foot_ids])
            add("episode_length", env.unwrapped.episode_length_buf)

            obs, _, _, extras = env.step(actions)
            if (step + 1) % 20 == 0:
                print(f"[INFO] {step + 1}/{args_cli.steps} steps")
    finally:
        env.close()

    out = {k: np.stack(v, axis=0) for k, v in rec.items()}   # (T, num_envs, ...)
    out["joint_names"] = np.array(robot.data.joint_names)
    out["foot_body_names"] = np.array([robot.data.body_names[i] for i in foot_ids])
    out["meta_dims"] = np.array([P, S, PE, H])
    out["checkpoint"] = np.array(resume_path)
    out["task"] = np.array(args_cli.task)

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    np.savez_compressed(args_cli.out, **out)
    print(f"\n[INFO] 저장: {args_cli.out}")
    for k in ("prop", "scan", "hist", "actions"):
        print(f"  {k:10s} {out[k].shape}")


if __name__ == "__main__":
    # 주의: `try: main() finally: simulation_app.close()` 로 쓰면 안 된다.
    # SimulationApp.close() 가 프로세스를 강제 종료해서 예외도 **종료코드도** 통째로
    # 삼켜 버린다 — 실패가 "출력 없이 exit 0" 으로 보인다. 실제로 이 스크립트를
    # 처음 돌렸을 때 그 증상으로 두 번 헤맸다.
    # 그래서 (1) 예외를 먼저 찍고 flush 하고, (2) 아래 [RESULT] 마커를 남긴다.
    # 종료코드는 믿을 수 없으니 호출 측은 이 마커나 산출물 파일로 판정할 것.
    import traceback

    failed = False
    try:
        main()
    except BaseException:
        traceback.print_exc()
        failed = True
    finally:
        print(f"[RESULT] {'FAILED' if failed else 'OK'}")
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            simulation_app.close()
        except Exception:
            pass
    sys.exit(1 if failed else 0)
