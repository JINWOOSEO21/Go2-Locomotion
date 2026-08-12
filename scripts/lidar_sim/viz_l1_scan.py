"""정지 로봇에서 L1 스캔 누적 시각화.

yaw(저속 모터) 1/3/11 회전 시점까지 누적된:
  (a) raw range map — 센서(돔) 프레임 (azimuth, elevation) 산점, 색 = range
  (b) 로봇 주변 top-down elevation map (5 cm 격자, 색 = 높이, 회색 = 미관측)

로봇은 사다리꼴 계단(trapezoid_stairs) 위에 정지(액션 0 = 기본 자세 유지).
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import parkour_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_apply, quat_conjugate  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from parkour_tasks.default_cfg import GO2_LIDAR_CFG  # noqa: E402

TASK = "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0"
ENV_VIS = 1  # trapezoid_only + one_col_per_terrain: env0=ramp, env1=stairs


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=2)
    env_cfg.scene.record_camera = None
    env_cfg.scene.lidar = copy.deepcopy(GO2_LIDAR_CFG)
    env_cfg.scene.lidar.update_period = env_cfg.sim.dt * env_cfg.decimation
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    lidar = env.unwrapped.scene["lidar"]
    yaw_T = env_cfg.scene.lidar.yaw_period_s
    windows_rev = [1, 3, 11]
    windows_t = [k * yaw_T for k in windows_rev]

    zero = torch.zeros(2, 12, device=env.unwrapped.device)
    offset_quat = torch.tensor(list(env_cfg.scene.lidar.offset.rot), device=env.unwrapped.device)

    # 정착 (계단 위에 서기)
    for _ in range(50):
        env.step(zero)

    # 센서는 0.1 s 프레임 단위로만 읽히므로, "yaw k 회전" 시점(k/11 s)에서 정확히
    # 자르려면 포인트별 방출 시각이 필요하다. valid_frame_times 가 레이 순서와
    # 같은 순서로 방출 시각을 돌려주므로 프레임마다 같이 모아 두고 나중에 자른다.
    # (예전 버전은 0.1 s 프레임 경계에서 통째로 찍어 "1 rev"가 실제로는 1.1 회전,
    #  float 누적 오차로 11 rev 가 1.1 s 에 찍히는 문제가 있었다.)
    acc = {"az": [], "el": [], "rng": [], "hits_w": [], "t_pt": []}
    dt5 = env.unwrapped.step_dt * 5  # 0.1 s (프레임 하나)
    n_frames = int(np.ceil(windows_t[-1] / dt5)) + 1
    base0 = robot.data.root_pos_w[ENV_VIS].clone()
    for _ in range(n_frames):
        for _ in range(5):
            env.step(zero)
        # 이번 프레임(직전 0.1 s)의 레이
        hits = lidar.data.ray_hits_w[ENV_VIS]                  # (R,3)
        origins = lidar._ray_starts_w[ENV_VIS]
        dirs_b = lidar.ray_directions[ENV_VIS]                 # base 프레임 (마운트 회전 포함)
        rng = torch.norm(hits - origins, dim=-1)
        # 돔(센서) 프레임 각도: 마운트 회전 역적용
        d_s = quat_apply(quat_conjugate(offset_quat).expand(dirs_b.shape[0], 4), dirs_b)
        az = torch.rad2deg(torch.atan2(d_s[:, 1], d_s[:, 0]))
        el = torch.rad2deg(torch.asin(d_s[:, 2].clamp(-1, 1)))
        t_pt, _ = lidar.valid_frame_times(
            lidar._timestamp[ENV_VIS].view(1, 1),
            lidar._l1_phase_pitch_deg[ENV_VIS].view(1, 1),
            lidar.num_rays, 1.0 / env_cfg.scene.lidar.pitch_period_s,
            env_cfg.scene.lidar.samples_per_second)
        acc["az"].append(az.cpu().numpy())
        acc["el"].append(el.cpu().numpy())
        acc["rng"].append(rng.cpu().numpy())
        acc["hits_w"].append(hits.cpu().numpy())
        acc["t_pt"].append(t_pt[0].cpu().numpy())

    all_pts = {key: np.concatenate(v) for key, v in acc.items()}
    t0 = all_pts["t_pt"].min()
    snaps = {}
    for k, T in zip(windows_rev, windows_t):
        m = all_pts["t_pt"] < t0 + T
        snaps[k] = {key: v[m] for key, v in all_pts.items() if key != "t_pt"}
        snaps[k]["t"] = T

    base = base0.cpu().numpy()
    os.makedirs(args_cli.out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    for col, k in enumerate(windows_rev):
        s = snaps[k]
        finite = np.isfinite(s["rng"])
        # (a) raw range map (dome 프레임)
        ax = axes[0, col]
        ax.scatter(s["az"][~finite], s["el"][~finite], s=1.5, c="lightgray", label="no-return")
        sc = ax.scatter(s["az"][finite], s["el"][finite], s=1.5, c=s["rng"][finite],
                        cmap="turbo", vmin=0, vmax=3.0)
        ax.set_title(f"raw range map — yaw {k} rev ({s['t']:.3f}s, {finite.size} pts)")
        ax.set_xlabel("azimuth [deg]")
        ax.set_ylabel("elevation [deg]")
        ax.set_xlim(-180, 180)
        ax.set_ylim(90, 0)  # x축 대칭: 아래 = 90°(수직 하방), 위 = 0°(수평)
        plt.colorbar(sc, ax=ax, label="range [m]")
        # (b) top-down elevation map (5 cm 격자, 로봇 중심 ±2.5 m)
        ax = axes[1, col]
        half, res = 2.5, 0.05
        n = int(2 * half / res)
        h = np.full((n, n), np.nan)
        pts = s["hits_w"][np.isfinite(s["hits_w"]).all(axis=1)]
        ix = ((pts[:, 0] - base[0] + half) / res).astype(int)
        iy = ((pts[:, 1] - base[1] + half) / res).astype(int)
        m = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        for x, y, z in zip(ix[m], iy[m], pts[m, 2]):
            if np.isnan(h[y, x]) or z > h[y, x]:
                h[y, x] = z
        im = ax.imshow(h, origin="lower", extent=[-half, half, -half, half],
                       cmap="turbo", vmin=base[2] - 0.5, vmax=base[2] + 0.3)
        im.cmap.set_bad("lightgray")
        ax.plot(0, 0, "wo", markersize=8, markeredgecolor="k")
        ax.arrow(0, 0, 0.4, 0, width=0.03, color="w", edgecolor="k")
        cov = 100 * np.isfinite(h).mean()
        ax.set_title(f"elevation map — yaw {k} rev (observed {cov:.0f}%)")
        ax.set_xlabel("world x - base [m]")
        ax.set_ylabel("world y - base [m]")
        plt.colorbar(im, ax=ax, label="z [m]")
    fig.suptitle(
        f"L1 scan accumulation (static robot, trapezoid_stairs) — yaw_period={yaw_T:.4f}s, "
        f"pitch_period={env_cfg.scene.lidar.pitch_period_s:.5f}s, 21600 pts/s", fontsize=13)
    fig.tight_layout()
    out = os.path.join(args_cli.out_dir, "l1_scan_accumulation.png")
    fig.savefig(out, dpi=110)
    print(f"[viz] saved {out}")
    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)
