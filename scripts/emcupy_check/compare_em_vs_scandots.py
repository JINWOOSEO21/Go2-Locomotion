"""Phase 2/3 검증: 시뮬레이션 안에서 GT scandots vs EM 샘플 비교 (Isaac 필요, headless).

EM student 환경을 띄우고 zero action(제자리 서기)으로 N step 진행하면서, 매 EM tick 에
obs 의 GT scandots 구간(obs[53:185])과 em_scan 관측을 비교한다. 결과는 stdout 이
아니라 --out 파일에 쓴다 (headless 검증 관례).

판정 (계획서 §5-Q7): 유효 셀 RMSE < 0.05 m, invalid 비율 < 20 % (1 s 워밍업 후).
--num_envs 192 --steps 300 으로 돌리면 Phase 3 벤치(step 시간·VRAM)도 겸한다.

실행 (env_isaaclab 환경):
  python scripts/emcupy_check/compare_em_vs_scandots.py --num_envs 4 --steps 250 --headless
"""
import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--out", default="/tmp/emcupy_compare.txt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import parkour_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

NUM_PROP, NUM_SCAN = 53, 132


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if getattr(env_cfg.scene, "record_camera", None) is not None:
        env_cfg.scene.record_camera = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs_dict, _ = env.reset()

    actions = torch.zeros(args_cli.num_envs, 12, device=env.unwrapped.device)
    warmup_steps = 50  # 1 s
    rmse_sum, rmse_cnt = 0.0, 0
    valid_sum, ub_sum, unknown_sum, frac_cnt = 0.0, 0.0, 0.0, 0
    step_times = []

    for step in range(args_cli.steps):
        t0 = time.time()
        obs_dict, _, _, _, _ = env.step(actions)
        torch.cuda.synchronize()
        step_times.append(time.time() - t0)

        if env.unwrapped.common_step_counter % 5 == 0 and step >= warmup_steps:
            policy_obs = obs_dict["policy"]
            gt = policy_obs[:, NUM_PROP:NUM_PROP + NUM_SCAN]
            em = obs_dict["em_scan"]
            term = env.unwrapped.em_scan_term
            observed = term.valid_frac > 1e-6
            has_ub = (~observed) & (term.ub_frac > 1e-6)
            unknown = (~observed) & (term.ub_frac <= 1e-6)
            if observed.any():
                rmse_sum += float(torch.sqrt(((em - gt)[observed] ** 2).mean()))
                rmse_cnt += 1
            valid_sum += float(observed.float().mean())
            ub_sum += float(has_ub.float().mean())
            unknown_sum += float(unknown.float().mean())
            frac_cnt += 1

    mean_step_ms = sum(step_times[warmup_steps:]) / max(len(step_times[warmup_steps:]), 1) * 1000
    rmse = rmse_sum / max(rmse_cnt, 1)
    valid = valid_sum / max(frac_cnt, 1)
    ub = ub_sum / max(frac_cnt, 1)
    unknown = unknown_sum / max(frac_cnt, 1)
    vram = torch.cuda.max_memory_allocated() / 2**20

    log(f"task={args_cli.task} num_envs={args_cli.num_envs} steps={args_cli.steps}")
    log(f"[cover] 직접관측 {valid*100:.1f}% | upper_bound 채움 {ub*100:.1f}% | 완전미지(0 채움) {unknown*100:.1f}%")
    log(f"[gate ] RMSE(valid) {rmse:.4f} m (< 0.05 목표) | unknown {unknown*100:.1f}% (< 20% 목표)")
    log("       (zero-action 정지 테스트: 몸 아래/다리 그림자 셀은 물리적으로 관측 불가 —")
    log("        cascade 가 ub/0 으로 채우며, 보행 시에는 지나온 지형이 맵에 남아 줄어든다)")
    log(f"[perf ] mean {mean_step_ms:.1f} ms/step (50Hz 실시간 기준 20 ms)")
    log(f"[vram ] torch max allocated {vram:.0f} MiB (cupy pool 별도 — nvidia-smi 확인)")
    ok = rmse < 0.05 and unknown < 0.20
    log(f"[{'PASS' if ok else 'FAIL'}] compare_em_vs_scandots")

    with open(args_cli.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    env.close()
    simulation_app.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
