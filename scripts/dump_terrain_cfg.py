# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""지형 설정의 '최종 확정값'을 찍어보는 도구.

지형 파라미터는 4단계에 걸쳐 덮어써진다.

    terrains/extreme_parkour/config/parkour.py   (EXTREME_PARKOUR_TERRAINS_CFG)
      -> ParkourStudentSceneCfg.__post_init__
        -> UnitreeGo2StudentParkourEnvCfg_EVAL.__post_init__
          -> UnitreeGo2StudentParkourEnvCfg_PLAY.__post_init__

그래서 어떤 값이 실제로 쓰이는지 알려면 네 파일을 다 따라가야 한다.
이 스크립트는 parse_env_cfg 로 전부 해석한 뒤의 결과만 보여준다.
시뮬레이터를 띄우지 않으므로(=지형을 생성하지 않으므로) 몇 초면 끝난다.

사용 예:
  python scripts/dump_terrain_cfg.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0
  PARKOUR_DIFFICULTY=0.9 python scripts/dump_terrain_cfg.py --task ... --difficulty 0.9
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump the fully-resolved terrain config.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--num_envs", type=int, default=4, help="Used only to resolve the cfg.")
parser.add_argument(
    "--difficulty",
    type=float,
    default=None,
    help="이 값으로 문자열 수식(예: '0.1 + 0.7*difficulty')을 평가해서 실제 치수까지 보여준다.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np

from isaaclab_tasks.utils import parse_env_cfg

import parkour_tasks  # noqa: F401


def p(*a):
    """simulation_app.close() 가 버퍼를 비우기 전에 프로세스를 끝내는 경우가 있어
    출력이 통째로 사라진다. 매 줄 flush 한다."""
    print(*a, flush=True)


def resolve(value, difficulty):
    """문자열 수식이면 difficulty 를 넣어 계산하고, 아니면 그대로 돌려준다."""
    if isinstance(value, str) and difficulty is not None:
        try:
            return eval(value, {"difficulty": difficulty})  # noqa: S307 - 원본 코드와 동일한 방식
        except Exception:
            return value
    return value


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=args_cli.num_envs)
    gen = env_cfg.scene.terrain.terrain_generator
    d = args_cli.difficulty

    p("=" * 78)
    p(f"TASK: {args_cli.task}")
    p("=" * 78)
    p("[격자]")
    p(f"  num_rows x num_cols : {gen.num_rows} x {gen.num_cols}   (행=난이도 단계, 열=지형 종류)")
    p(f"  타일 크기            : {gen.size} m")
    p(f"  difficulty_range     : {gen.difficulty_range}")
    p(f"  random_difficulty    : {gen.random_difficulty}")
    p(f"  horizontal/vertical  : {gen.horizontal_scale} / {gen.vertical_scale} m")
    p(f"  num_goals            : {gen.num_goals}")
    p(f"  max_init_terrain_lvl : {env_cfg.scene.terrain.max_init_terrain_level}")

    # 열 -> 지형 종류 매핑 (ParkourTerrainGenerator._generate_curriculum_terrains 와 같은 식)
    names = list(gen.sub_terrains.keys())
    props = np.array([c.proportion for c in gen.sub_terrains.values()], dtype=float)
    p("\n[열 배정]")
    if props.sum() <= 0:
        p("  proportion 합이 0 이라 배정 불가")
    else:
        norm = props / props.sum()
        for i in range(gen.num_cols):
            idx = int(np.min(np.where(i / gen.num_cols + 0.001 < np.cumsum(norm))[0]))
            p(f"  col {i} -> {names[idx]}")
        n = args_cli.num_envs
        types = np.floor(np.arange(n) / (n / gen.num_cols)).astype(int)
        p(f"  num_envs={n} -> terrain_types = {types.tolist()}")

    p("\n[서브지형별 최종값]" + (f"  (difficulty={d} 로 수식 평가)" if d is not None else ""))
    for key, cfg in gen.sub_terrains.items():
        mark = "  <-- 비활성(proportion 0)" if cfg.proportion == 0 else ""
        p(f"\n  {key}  proportion={cfg.proportion}{mark}")
        p(f"    function       : {cfg.function.__name__}")
        p(f"    size           : {cfg.size}")
        for f in ("platform_len", "x_range", "y_range", "half_valid_width", "noise_range",
                  "gap_size", "gap_depth", "hurdle_height_range", "step_height",
                  "stone_len", "stone_width", "incline_height", "last_incline_height", "pit_depth"):
            if not hasattr(cfg, f):
                continue
            raw = getattr(cfg, f)
            out = resolve(raw, d)
            extra = f"   -> {out}" if (isinstance(raw, str) and out != raw) else ""
            p(f"    {f:<19}: {raw}{extra}")

    p("\n" + "=" * 78)
    p("주의: 이 값들은 상속 체인(config/parkour.py -> SceneCfg -> EVAL -> PLAY)이")
    p("      전부 적용된 뒤의 최종값이다. 개별 파일의 값과 다를 수 있다.")
    p("=" * 78)


if __name__ == "__main__":
    main()
    simulation_app.close()
