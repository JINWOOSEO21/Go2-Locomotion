"""IsaacLab 파쿠르 지형을 MuJoCo 로 옮길 수 있는 형태로 추출한다.

무엇을 뽑나
-----------
- `terrain.obj`      : IsaacLab 이 실제로 물리/레이캐스팅에 쓰는 삼각망 그대로.
- `terrain_meta.npz` : MuJoCo `<hfield>` 용 높이 격자 + 목표점/원점/이름 등 메타.

왜 hfield 인가
--------------
MuJoCo 의 `type="mesh"` geom 은 **충돌 시 볼록껍질로 근사**된다. 계단·단차가 있는
지형을 mesh 로 넣으면 통째로 하나의 볼록 덩어리가 되어 물리가 완전히 달라진다.
비볼록 지형의 표준 표현은 `<hfield>` 이므로 높이 격자를 함께 뽑는다.
OBJ 는 시각 확인·검증용으로만 남긴다.

왜 꼭짓점 스냅이 아니라 레이캐스팅인가
--------------------------------------
파쿠르 지형은 높이맵(height_field_raw)에서 출발하지만, **최종 삼각망은 그 높이맵의
조밀 격자가 아니다.** 실측으로 확인한 것:
  - `parkour_field_to_mesh` 가 `convert_height_field_to_mesh(..., slope_threshold)` 를
    쓰는데, 이게 단차 옆면을 수직으로 만들려고 꼭짓점을 **반칸(0.05m)** 위치로 옮긴다.
    그래서 꼭짓점의 57% 만 0.1 격자에 있고 43% 는 반칸에 있다.
  - `cfg.use_simplified` 가 켜져 있으면 quadric decimation 으로 면을 35% 줄인다.
    평지에는 내부 꼭짓점이 아예 남지 않는다 (지형 영역 고유 좌표 12,516개 <
    조밀 격자 19,521개).
꼭짓점을 격자에 되꽂는 방식으로 처음 짰다가 채움률 19.6% 에 검증 오차 53mm 로
실패했다. 레이캐스팅은 이런 가정이 필요 없고, **물리와 height_scanner 가 실제로
보는 바로 그 삼각망**(= 간략화 이후)을 재현한다. 격자 간격 기본값은
horizontal_scale/2 = 0.05m 로, slope_threshold 가 만든 반칸 수직면을 담을 수 있다.

지형 재현을 위한 시드 고정
------------------------------------
TerrainGeneratorCfg.seed 기본값이 None 이라 지형 난수는 **전역 numpy 시드**를 따른다
(noise_range 가 셀마다 난수를 쓴다). 그래서 생성기만 따로 만들지 않고
학습/재생과 같은 방식으로 env 를 만들어 그 안의 생성기를 꺼낸다.
같은 --seed 를 주면 같은 지형이 나온다.

검증
----
구운 높이 격자를 IsaacLab 자신의 height_scanner 레이 적중점과 대조한다. 같은
지형이라면 두 값이 일치해야 한다. 불일치가 크면 격자화가 틀린 것이다.

실행 (Isaac Sim 필요)
---------------------
    python deploy/tools/export_terrain.py --headless --num_envs 1 --seed 1 \
        --task Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Play-v0 \
        --out-dir logs/.../exported
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "scripts", "rsl_rl"))
sys.path.insert(0, _REPO)

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-Extreme-Parkour-Lidar-Unitree-Go2-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--margin", type=float, default=2.0, help="지형 bbox 바깥으로 더 굽는 여유 [m]")
parser.add_argument(
    "--res",
    type=float,
    default=None,
    help="hfield 격자 간격 [m]. 기본 horizontal_scale/2 — slope_threshold 가 만드는 "
    "반칸(0.05) 수직면을 표현하려면 절반 간격이 필요하다.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import numpy as np  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import parkour_tasks  # noqa: F401, E402


def bake_heightfield(mesh, res: float, xlo, xhi, ylo, yhi):
    """삼각망을 위에서 아래로 레이캐스팅해 높이 격자를 만든다.

    반환: (H[ny, nx], x0, y0, 적중률) — H[j, i] 는 (x0 + i*res, y0 + j*res) 의 높이.

    처음에는 "꼭짓점이 높이맵 격자점 위에 있으니 되꽂으면 된다" 로 짰는데 틀렸다.
    실측해 보니 ① slope_threshold 처리가 꼭짓점을 반칸(0.05) 위치로 옮기고,
    ② `use_simplified` 가 quadric decimation 으로 면을 35% 줄여서 평지에는 내부
    꼭짓점이 아예 없다. 즉 삼각망은 원본 높이맵의 조밀 격자가 아니다.
    레이캐스팅은 이런 가정이 필요 없고, **물리와 height_scanner 가 실제로 보는
    바로 그 삼각망**을 그대로 재현한다.
    """
    nx = int(np.floor((xhi - xlo) / res)) + 1
    ny = int(np.floor((ylo * 0 + (yhi - ylo)) / res)) + 1
    x0, y0 = xlo, ylo
    gx = x0 + np.arange(nx) * res
    gy = y0 + np.arange(ny) * res
    GX, GY = np.meshgrid(gx, gy)  # (ny, nx)
    n = GX.size
    z_top = float(np.asarray(mesh.vertices)[:, 2].max()) + 1.0
    origins = np.stack([GX.ravel(), GY.ravel(), np.full(n, z_top)], axis=1)
    dirs = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))

    loc, idx_ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    flat = np.full(n, -np.inf)
    # 한 레이에 적중이 여러 개 와도 가장 높은 면을 취한다 (위에서 본 표면).
    np.maximum.at(flat, idx_ray, loc[:, 2])
    hit = np.isfinite(flat)
    flat[~hit] = np.nan
    return flat.reshape(ny, nx), x0, y0, float(hit.mean())


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = agent_cfg.seed
    if getattr(env_cfg.scene, "record_camera", None) is not None:
        env_cfg.scene.record_camera = None
        print("[INFO] record_camera 제거")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        terrain = env.unwrapped.scene.terrain
        gen = terrain.terrain_generator_class
        tcfg = terrain.cfg.terrain_generator
        hs, vs = float(tcfg.horizontal_scale), float(tcfg.vertical_scale)
        mesh = gen.terrain_mesh
        print(f"[INFO] mesh: vertices={len(mesh.vertices)} faces={len(mesh.faces)}")
        print(
            f"[INFO] horizontal_scale={hs} vertical_scale={vs} "
            f"num_rows={tcfg.num_rows} num_cols={tcfg.num_cols} size={tcfg.size}"
        )

        os.makedirs(args_cli.out_dir, exist_ok=True)
        obj_path = os.path.join(args_cli.out_dir, "terrain.obj")
        mesh.export(obj_path)
        print(f"[INFO] OBJ 저장: {obj_path}")

        # 테두리(border_width=20m)는 평지라 통째로 구울 필요가 없다. 지형 타일
        # 영역 + margin 만 굽고, 바깥은 MuJoCo 쪽에서 평면으로 깐다.
        origins = np.asarray(gen.terrain_origins)  # (rows, cols, 3)
        half_x = tcfg.num_rows * tcfg.size[0] / 2.0
        half_y = tcfg.num_cols * tcfg.size[1] / 2.0
        m = args_cli.margin
        res = args_cli.res if args_cli.res else hs / 2.0
        H, x0, y0, covered = bake_heightfield(mesh, res, -half_x - m, half_x + m, -half_y - m, half_y + m)
        print(f"[INFO] hfield: {H.shape} (ny,nx) res={res}m, 레이 적중률 {covered * 100:.1f}%")
        base_fill = float(np.nanmin(H))
        H = np.where(np.isfinite(H), H, base_fill)

        # --- 검증: IsaacLab 자신의 height_scanner 적중점과 대조 ---
        scanner = env.unwrapped.scene.sensors["height_scanner"]
        env.reset()
        hits = scanner.data.ray_hits_w[0].detach().cpu().numpy()  # (132, 3)
        ok = np.isfinite(hits).all(axis=-1)
        hx, hy, hz = hits[ok, 0], hits[ok, 1], hits[ok, 2]
        fi = np.clip((hx - x0) / res, 0, H.shape[1] - 1.001)
        fj = np.clip((hy - y0) / res, 0, H.shape[0] - 1.001)
        i0, j0 = fi.astype(int), fj.astype(int)
        tx, ty = fi - i0, fj - j0
        baked = (
            H[j0, i0] * (1 - tx) * (1 - ty)
            + H[j0, i0 + 1] * tx * (1 - ty)
            + H[j0 + 1, i0] * (1 - tx) * ty
            + H[j0 + 1, i0 + 1] * tx * ty
        )
        d = np.abs(baked - hz)
        print(f"\n[검증] height_scanner 적중점 {ok.sum()}개 vs 구운 격자 (쌍선형 보간)")
        print(f"  max|diff| = {d.max() * 1000:.2f} mm,  mean = {d.mean() * 1000:.2f} mm")
        verdict = "PASS" if d.max() < 0.01 else "FAIL"
        print(f"  => {verdict} (기준 10mm — 격자 간격 {res * 1000:.0f}mm 의 보간 오차 범위)")

        robot_spawn = np.asarray(env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu())
        np.savez_compressed(
            os.path.join(args_cli.out_dir, "terrain_meta.npz"),
            hfield=H.astype(np.float32),
            x0=x0,
            y0=y0,
            res=res,
            hs=hs,
            vs=vs,
            terrain_origins=origins,
            env_origins=np.asarray(terrain.env_origins.detach().cpu()),
            goals=np.asarray(gen.goals),
            terrain_names=np.asarray(gen.terrain_names),
            terrain_type=np.asarray(gen.terrain_type),
            num_rows=tcfg.num_rows,
            num_cols=tcfg.num_cols,
            tile_size=np.asarray(tcfg.size),
            difficulty_range=np.asarray(tcfg.difficulty_range),
            border_width=tcfg.border_width,
            robot_spawn_pos_w=robot_spawn,
            scan_check_max_mm=d.max() * 1000.0,
            task=np.array(args_cli.task),
            seed=agent_cfg.seed,
        )
        print(f"\n[INFO] meta 저장: {os.path.join(args_cli.out_dir, 'terrain_meta.npz')}")
        print(f"  hfield {H.shape}  x0={x0:.3f} y0={y0:.3f} res={res}")
        print(f"  z 범위 [{H.min():.3f}, {H.max():.3f}]")
        print(f"  로봇 스폰 (world) = {robot_spawn}")
        print(f"  goals shape = {np.asarray(gen.goals).shape}")
        print(f"  terrain_names = {np.asarray(gen.terrain_names).ravel().tolist()}")
    finally:
        env.close()


if __name__ == "__main__":
    # 시뮬레이터 종료 전에 예외를 출력해 오류 원인을 남긴다.
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
