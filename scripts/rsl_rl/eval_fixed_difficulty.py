# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""고정 난이도에서 teacher 체크포인트들을 순차 평가한다.

학습 로그의 reward / terrain_level 은 커리큘럼이 난이도를 정책 실력에 맞춰 계속
올려 버리기 때문에 학습 곡선으로 쓸 수 없다. 잘할수록 어려운 지형으로 가서 다시
비슷한 reward 를 받는 구조라, 지표가 원리적으로 포화한다. 실측으로도
teacher_reward_added run 의 terrain_levels 는 iter 600 부터 6.30 +- 0.03 으로
완전히 평평했다 (평균 레벨의 이론적 천장은 num_rows=10 에서 6.0 이다. 최고
레벨을 깨면 0~9 중 무작위 레벨로 재배치되기 때문).

그래서 여기서는 난이도를 한 점에 고정한 지형에서 체크포인트별로 같은 조건의
rollout 을 돌리고, 에피소드 단위로 reward 를 기록한다.

난이도 고정 방법:
    difficulty_range = (d, d) 로 두면 _generate_curriculum_terrains 의
        difficulty = lower + (upper - lower) * row/(num_rows-1)
    가 모든 row 에서 d 가 된다. 커리큘럼이 env 를 위아래로 옮겨도 난이도가 안
    바뀌므로 커리큘럼을 따로 끌 필요가 없다. 대신 타일 내부 랜덤(계단 수,
    디딤판 깊이, 코스 폭, plateau 길이)은 타일마다 다르므로 num_rows x num_cols
    개의 서로 다른 '같은 난이도' 지형이 생긴다.

reward 기록 규약:
    ParkourRewardManager 는 최종 reward 를 0 에서 클리핑한다
    (parkour_reward_manager.py 의 torch.clip(min=0.)). 여기서 저장하는
    rew_total_raw_per_s 는 그 클리핑 '이전' 의 가중합이다. 페널티가 온전히
    반영된 값이라 동작 품질 비교에는 이쪽이 맞다.

    항별 값은 가중치를 곱하기 '이전' 의 리워드 함수 원 출력을 저장한다.
    RewardManager._step_reward[:, i] 가 func * weight 이므로 weight 로 나눠서
    얻는다. 리워드 함수를 다시 호출하지 않으므로 중복 연산도, 부작용(예:
    reward_feet_edge 가 feet_at_edge 를 갱신하는 것)도 없다.
    가중치는 meta.json 에 저장되므로 raw * weight 로 언제든 복원할 수 있다.

실행 구조:
    지형은 체크포인트와 무관하므로 시뮬레이터를 한 번만 띄우고 그 안에서
    체크포인트 weight 만 갈아끼운다. 체크포인트마다 시뮬을 새로 띄우면 기동 +
    지형 생성 비용을 그 횟수만큼 다시 낸다.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate scandots-input checkpoints on a fixed-difficulty terrain.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Extreme-Parkour-Scandots-Unitree-Go2-Eval-v0",
    help="Eval 용 task id.",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="model_*.pt 가 들어 있는 폴더.",
)
parser.add_argument(
    "--iterations",
    type=str,
    default=None,
    help="쉼표로 구분한 iteration 목록. 생략하면 --iteration_step 배수 + 마지막 체크포인트.",
)
parser.add_argument("--iteration_step", type=int, default=1000, help="자동 선택 시 iteration 간격.")
parser.add_argument("--difficulty", type=float, default=0.7, help="고정할 난이도 (0~1).")
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="지형 생성과 초기 상태의 난수 seed. 체크포인트 비교가 성립하려면 고정해야 한다.",
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=2500, help="체크포인트당 rollout step 수.")
parser.add_argument("--num_rows", type=int, default=10)
parser.add_argument("--num_cols", type=int, default=10)
parser.add_argument("--preset", type=str, default="trapezoid_only", help="TERRAIN_PRESETS 의 키.")
parser.add_argument("--output_dir", type=str, required=True, help="결과 csv/json 을 쓸 폴더.")
parser.add_argument(
    "--calibrate",
    action="store_true",
    help="첫 체크포인트 하나만 돌리고 소요 시간을 보고한 뒤 종료한다.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import json
import os
import re
import subprocess
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from isaaclab_tasks.utils import parse_env_cfg
from tqdm import tqdm

from parkour_isaaclab.terrains.extreme_parkour.config.parkour import apply_terrain_preset
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlOnPolicyRunnerCfg,
)
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper


def select_checkpoints(checkpoint_dir: str, iterations: str | None, step: int) -> list[tuple[int, str]]:
    """평가할 (iteration, 경로) 목록을 만든다."""
    available = {}
    for name in os.listdir(checkpoint_dir):
        match = re.fullmatch(r"model_(\d+)\.pt", name)
        if match:
            available[int(match.group(1))] = os.path.join(checkpoint_dir, name)
    if not available:
        raise FileNotFoundError(f"{checkpoint_dir} 에 model_*.pt 가 없다.")

    if iterations is not None:
        wanted = [int(x) for x in iterations.split(",") if x.strip()]
        missing = [it for it in wanted if it not in available]
        if missing:
            raise FileNotFoundError(f"없는 체크포인트: {missing}")
    else:
        # step 배수 + 마지막 체크포인트. 학습이 15000 에서 끊기면 마지막은 14999 처럼
        # 배수가 아닌 번호로 저장되므로 따로 챙긴다.
        wanted = sorted(it for it in available if it % step == 0)
        last = max(available)
        if last not in wanted:
            wanted.append(last)
    return [(it, available[it]) for it in sorted(wanted)]


def resolve_difficulty_params(sub_cfg, difficulty: float) -> dict:
    """난이도 수식으로 적힌 지형 파라미터를 이 난이도에서의 실제 값으로 푼다.

    지형 cfg 는 난이도 의존 파라미터를 문자열 수식으로 들고 있고
    (예: trapezoid ramp 의 slope_angle = '10 + 27*difficulty'), 지형 생성 함수가
    eval(cfg.<field>, {"difficulty": difficulty}) 로 푼다. 여기서도 같은 방식으로
    풀어서 meta.json 에 남긴다. 그래야 나중에 그래프만 보고도 '난이도 0.7 이
    경사 몇 도였나' 를 알 수 있다.

    값을 하드코딩하지 않고 cfg 에서 푸는 이유는, 지형 수식이 바뀌면 그래프의
    설명도 자동으로 따라가야 하기 때문이다.
    """
    resolved = {}
    for field, value in vars(sub_cfg).items():
        if isinstance(value, str) and "difficulty" in value:
            try:
                resolved[field] = eval(value, {"__builtins__": {}}, {"difficulty": difficulty})
            except Exception:
                # 난이도 수식이 아닌 문자열이 우연히 'difficulty' 를 포함할 수 있다.
                continue
    return resolved


class EpisodeRecorder:
    """에피소드 단위로 reward 항과 종료 상태를 모은다.

    _reset_idx 를 감싸서 리셋 '직전' 에 종료 상태를 읽는다. env.step() 이 반환된
    뒤에는 이미 리셋이 끝나 cur_goal_idx 가 0 으로 밀려 있고 로봇도 출발선에 서
    있어서, 그 시점에 읽으면 종료 원인을 알 수 없다.
    """

    def __init__(self, env, weights: np.ndarray, num_goals: int):
        self.env_u = env.unwrapped
        self.rm = self.env_u.reward_manager
        self.pe = self.env_u.parkour_manager.get_term("base_parkour")
        self.robot = self.env_u.scene["robot"]
        self.terrain = self.env_u.scene.terrain
        self.num_goals = num_goals
        self.device = self.env_u.device
        self.num_envs = self.env_u.num_envs
        self.weights = weights
        self.num_terms = len(weights)

        self.acc = torch.zeros(self.num_envs, self.num_terms, device=self.device)
        self.acc_steps = torch.zeros(self.num_envs, device=self.device)

        self.capture_on = False
        self.accumulated = False
        self.iteration = -1
        self.seed = -1
        self.rows: list[dict] = []

        self._orig_reset_idx = self.env_u._reset_idx
        self.env_u._reset_idx = self._patched_reset_idx

    def close(self):
        self.env_u._reset_idx = self._orig_reset_idx

    def accumulate(self):
        """이번 step 의 항별 reward 를 누적한다. _step_reward 는 func * weight (초당)."""
        self.acc += self.rm._step_reward
        self.acc_steps += 1
        self.accumulated = True

    def reset_all(self):
        self.acc.zero_()
        self.acc_steps.zero_()

    def _patched_reset_idx(self, env_ids):
        if self.capture_on:
            # step() 안에서 reward_manager.compute 는 이미 끝났고 reset 은 아직이다.
            # 이 시점에 이번 step 분을 먼저 누적해야 종료 step 이 평균에 들어간다.
            if not self.accumulated:
                self.accumulate()
            self._record(env_ids)
            self.acc[env_ids] = 0.0
            self.acc_steps[env_ids] = 0.0
        self._orig_reset_idx(env_ids)

    @torch.no_grad()
    def _record(self, env_ids):
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device)
        ids = ids.reshape(-1)
        if ids.numel() == 0:
            return
        steps = self.acc_steps[ids]
        keep = steps > 0
        if not bool(keep.any()):
            return
        ids = ids[keep]
        steps = steps[keep]

        # 항별 초당 평균 (가중치 적용 상태) -> 가중치로 나눠 원값 복원
        per_s = (self.acc[ids] / steps.unsqueeze(1)).cpu().numpy().astype(np.float64)
        total_raw = per_s.sum(axis=1)
        raw = np.divide(
            per_s, self.weights[None, :], out=np.full_like(per_s, np.nan), where=self.weights[None, :] != 0.0
        )

        root = self.robot.data.root_state_w[ids]
        roll, pitch, _ = euler_xyz_from_quat(root[:, 3:7])
        roll = wrap_to_pi(roll).abs().cpu().numpy()
        pitch = wrap_to_pi(pitch).abs().cpu().numpy()
        height = root[:, 2].cpu().numpy()

        goal_idx = self.pe.cur_goal_idx[ids].cpu().numpy()
        dist = self.pe.dis_to_start_pos[ids].cpu().numpy()
        levels = self.terrain.terrain_levels[ids].cpu().numpy()
        ep_len = self.env_u.episode_length_buf[ids].cpu().numpy()
        ids_np = ids.cpu().numpy()
        names = np.asarray(self.pe.env_per_terrain_name)[ids_np].reshape(len(ids_np), -1)[:, 0]

        # terminate_episode 와 같은 판정. 우선순위는 goal > roll > pitch > height > timeout.
        reason = np.full(len(ids_np), "timeout", dtype=object)
        reason[height < -0.25] = "height"
        reason[pitch > 1.5] = "fall_pitch"
        reason[roll > 1.5] = "fall_roll"
        success = goal_idx >= self.num_goals
        reason[success] = "goal"

        steps_np = steps.cpu().numpy()
        dt = self.env_u.step_dt
        for k in range(len(ids_np)):
            row = {
                "iteration": self.iteration,
                "seed": self.seed,
                "env_id": int(ids_np[k]),
                "terrain_name": str(names[k]),
                "terrain_level": int(levels[k]),
                "steps": int(steps_np[k]),
                "duration_s": float(steps_np[k] * dt),
                "episode_length_buf": int(ep_len[k]),
                "success": int(bool(success[k])),
                "term_reason": str(reason[k]),
                "goal_idx_final": int(goal_idx[k]),
                "dist_from_start": float(dist[k]),
                "rew_total_raw_per_s": float(total_raw[k]),
            }
            for i, name in enumerate(TERM_NAMES):
                row[f"raw_{name}"] = float(raw[k, i])
            self.rows.append(row)


TERM_NAMES: list[str] = []


def main():
    global TERM_NAMES

    os.makedirs(args_cli.output_dir, exist_ok=True)
    seed = args_cli.seed if args_cli.seed is not None else 0

    checkpoints = select_checkpoints(args_cli.checkpoint_dir, args_cli.iterations, args_cli.iteration_step)
    if args_cli.calibrate:
        checkpoints = checkpoints[:1]
    print(f"[INFO] 평가할 체크포인트 {len(checkpoints)} 개: {[it for it, _ in checkpoints]}")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # -- 난이도 고정. difficulty_range 를 한 점으로 좁히면 모든 row 가 같은 난이도가 된다.
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None:
        raise RuntimeError("terrain_generator 가 없다. generator 기반 지형이어야 한다.")
    generator.curriculum = True
    generator.random_difficulty = False
    generator.difficulty_range = (args_cli.difficulty, args_cli.difficulty)
    generator.num_rows = args_cli.num_rows
    generator.num_cols = args_cli.num_cols
    generator.seed = seed
    apply_terrain_preset(generator, args_cli.preset, active_overrides={"noise_range": (0.02, 0.02)})
    env_cfg.seed = seed

    active_terrains = [k for k, v in generator.sub_terrains.items() if v.proportion > 0]
    terrain_params = {
        name: resolve_difficulty_params(generator.sub_terrains[name], args_cli.difficulty) for name in active_terrains
    }
    terrain_params = {k: v for k, v in terrain_params.items() if v}
    print(
        f"[INFO] 난이도 {args_cli.difficulty} 고정, 지형 {active_terrains}, "
        f"{args_cli.num_rows}x{args_cli.num_cols} 타일, seed {seed}"
    )
    for name, params in terrain_params.items():
        print(f"[INFO]   {name}: {params}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env_u = env.unwrapped

    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    estimator_paras = agent_cfg.to_dict()["estimator"]
    num_prop = estimator_paras["num_prop"]
    num_scan = estimator_paras["num_scan"]
    num_priv_explicit = estimator_paras["num_priv_explicit"]

    rm = env_u.reward_manager
    TERM_NAMES = list(rm.active_terms)
    weights = np.array([rm.get_term_cfg(n).weight for n in TERM_NAMES], dtype=np.float64)
    num_goals = env_u.scene.terrain.cfg.terrain_generator.num_goals

    recorder = EpisodeRecorder(env, weights, num_goals)
    recorder.seed = seed

    fieldnames = [
        "iteration",
        "seed",
        "env_id",
        "terrain_name",
        "terrain_level",
        "steps",
        "duration_s",
        "episode_length_buf",
        "success",
        "term_reason",
        "goal_idx_final",
        "dist_from_start",
        "rew_total_raw_per_s",
    ] + [f"raw_{n}" for n in TERM_NAMES]

    episodes_path = os.path.join(args_cli.output_dir, "episodes.csv")
    with open(episodes_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)), text=True
        ).strip()
    except Exception:
        commit = "unknown"

    meta = {
        "task": args_cli.task,
        "checkpoint_dir": os.path.abspath(args_cli.checkpoint_dir),
        "iterations": [it for it, _ in checkpoints],
        "difficulty": args_cli.difficulty,
        "terrain_preset": args_cli.preset,
        "active_terrains": active_terrains,
        "terrain_params": terrain_params,
        "num_rows": args_cli.num_rows,
        "num_cols": args_cli.num_cols,
        "num_envs": args_cli.num_envs,
        "steps_per_checkpoint": args_cli.steps,
        "seed": seed,
        "num_goals": num_goals,
        "episode_length_s": env_cfg.episode_length_s,
        "max_episode_length": int(env_u.max_episode_length),
        "step_dt": env_u.step_dt,
        "reward_weights": {n: float(w) for n, w in zip(TERM_NAMES, weights)},
        "reward_total_is_pre_clip": True,
        "git_commit": commit,
    }
    with open(os.path.join(args_cli.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # simulation_app.close() 가 파이썬 stdout 버퍼를 흘리지 않고 프로세스를 끊는 일이
    # 있어서, 리다이렉트된 로그에서는 마지막 요약 print 가 통째로 사라진다.
    # 진행 상황을 로그로 봐야 하므로 여기서부터는 전부 flush 해서 찍는다.
    def log(msg: str):
        print(msg, flush=True)

    durations = []
    for iteration, path in checkpoints:
        log(f"[INFO] === iteration {iteration} : {path}")
        runner.load(path, load_optimizer=False)
        policy = runner.get_inference_policy(device=env_u.device)
        estimator = runner.get_estimator_inference_policy(device=env_u.device)

        # 체크포인트를 갈아끼울 때는 진행 중이던 에피소드를 버린다. 이전 정책이
        # 절반 만든 상태를 새 정책 점수에 섞으면 안 된다.
        recorder.capture_on = False
        obs, _ = env.reset()
        recorder.reset_all()
        recorder.iteration = iteration
        recorder.rows.clear()
        recorder.capture_on = True

        start = time.time()
        # mininterval 을 크게 잡는다. 기본값(0.1s)이면 진행 막대가 로그 파일을
        # 수십 MB 로 불려서 나중에 요약을 찾기가 어려워진다.
        for _ in tqdm(range(args_cli.steps), desc=f"iter {iteration}", leave=False, mininterval=30.0):
            recorder.accumulated = False
            with torch.inference_mode():
                obs[:, num_prop + num_scan : num_prop + num_scan + num_priv_explicit] = estimator.inference(
                    obs[:, :num_prop]
                )
                actions = policy(obs, hist_encoding=True)
            obs, _, _, _ = env.step(actions)
            if not recorder.accumulated:
                recorder.accumulate()
        elapsed = time.time() - start
        durations.append(elapsed)

        with open(episodes_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(recorder.rows)

        n_ep = len(recorder.rows)
        if n_ep:
            rew = np.array([r["rew_total_raw_per_s"] for r in recorder.rows])
            ok = np.array([r["success"] for r in recorder.rows], dtype=bool)
            msg = (
                f"[INFO] iter {iteration}: {n_ep} 에피소드, {elapsed / 60:.1f} 분, "
                f"완주율 {ok.mean() * 100:.1f}%, reward(초당, clip 전) 전체 {rew.mean():.4f}"
            )
            if ok.any():
                msg += f", 완주만 {rew[ok].mean():.4f}"
            log(msg)
        else:
            log(f"[WARN] iter {iteration}: 완료된 에피소드가 없다 ({elapsed / 60:.1f} 분)")

    recorder.close()
    env.close()

    mean_dur = float(np.mean(durations)) if durations else 0.0
    log("\n[SUMMARY] ------------------------------------------------")
    log(f"체크포인트당 rollout 시간: 평균 {mean_dur / 60:.2f} 분 ({args_cli.steps} step)")
    if args_cli.calibrate:
        total = select_checkpoints(args_cli.checkpoint_dir, args_cli.iterations, args_cli.iteration_step)
        log(f"전체 {len(total)} 개 예상 rollout 시간: {mean_dur * len(total) / 3600:.2f} 시간 (기동 시간 별도)")
    log(f"결과: {episodes_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
