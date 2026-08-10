# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""env 마다 1개씩 mp4 를 쓰는 녹화 헬퍼.

씬의 ``record_camera`` (TiledCamera, env 당 1대) 를 직접 읽는다. TiledCamera 는 모든
env 를 한 번의 렌더 패스로 처리하므로 env 별 카메라를 따로 두는 것보다 싸다.
``data.output["rgb"]`` 가 (num_envs, H, W, 3) 로 나온다.

play.py 의 ``--multicam`` 이 쓴다.
"""

from __future__ import annotations

import numpy as np
import os
import torch

import imageio.v2 as imageio

from scripts.rsl_rl.video_overlay import depth_to_panel


def resolve_terrain_names(env) -> list[str]:
    """env 별로 배정된 서브지형 이름을 돌려준다.

    TerrainImporter 가 env 를 (level=row, type=col) 로 배정하고
    ParkourTerrainGenerator 가 terrain_names[row, col] 에 이름을 채워둔다.
    그 배열을 우선 쓰고, 접근이 안 되면 generator 와 동일한 규칙으로
    컬럼→서브지형 매핑을 직접 계산한다.
    """
    terrain = env.unwrapped.scene.terrain
    types = terrain.terrain_types.cpu().numpy()
    levels = terrain.terrain_levels.cpu().numpy()

    # 1순위: 생성기가 기록해둔 이름 배열
    try:
        names_arr = terrain.terrain_generator_class.terrain_names
        return [str(np.asarray(names_arr[levels[i], types[i]]).reshape(-1)[0]) for i in range(len(types))]
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] terrain_names 조회 실패, 설정에서 재계산한다: {exc!r}")

    # 2순위: proportion 으로 컬럼 매핑을 재현 (ParkourTerrainGenerator 와 같은 식)
    gen_cfg = env.unwrapped.scene.cfg.terrain.terrain_generator
    keys = list(gen_cfg.sub_terrains.keys())
    props = np.array([c.proportion for c in gen_cfg.sub_terrains.values()], dtype=float)
    props = props / props.sum()
    num_cols = gen_cfg.num_cols
    col_names = [
        keys[int(np.min(np.where(i / num_cols + 0.001 < np.cumsum(props))[0]))] for i in range(num_cols)
    ]
    return [col_names[t] for t in types]


class PerEnvVideoRecorder:
    """env 당 mp4 1개. 매 스텝 카메라를 로봇 위로 옮기고 프레임을 받아 쓴다.

    카메라 자세는 고정하고 위치만 로봇을 따라간다. 오프셋이 월드 기준 상수이므로
    eye->target 방향도 항상 같고, 결과적으로 자세는 고정된 채 평행이동만 하게 된다.
    (로봇 base 에 부착하면 몸체의 pitch/roll 을 물려받아 화면이 같이 기운다.)

    기본 오프셋은 원본 VIEWER(eye=(-0., 2.6, 1.6), origin_type='asset_root') 와 같은
    좌측 측면 시점이다. +Y 가 좌측, +Z 가 위.
    """

    def __init__(self, env, out_dir: str, fps: int = 50, cam_offset=(0.0, 2.6, 1.6)):
        self.env = env
        self.camera = env.unwrapped.scene["record_camera"]
        self.robot = env.unwrapped.scene["robot"]
        self.cam_offset = torch.tensor(cam_offset, device=env.unwrapped.device)

        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        # 지형 이름은 리셋 후에야 확정되므로 첫 관측 뒤에 만들어야 한다.
        self.terrain_names = resolve_terrain_names(env)
        self.paths = []
        self.writers = []
        for i, name in enumerate(self.terrain_names):
            path = os.path.join(self.out_dir, f"env{i}_{name}.mp4")
            # quality 8 은 imageio-ffmpeg 기준 고화질(기본값 5보다 높음).
            self.writers.append(imageio.get_writer(path, fps=fps, quality=8, macro_block_size=1))
            self.paths.append(path)
            print(f"[INFO] env {i} -> {name} -> {path}", flush=True)
        self.written = 0

    def track_camera(self):
        """로봇 위치 + 고정 오프셋에서 로봇을 바라보게 카메라를 옮긴다.

        스텝 중에 렌더가 일어나므로 env.step() '전에' 불러야 한다.
        """
        target = self.robot.data.root_pos_w
        self.camera.set_world_poses_from_view(eyes=target + self.cam_offset, targets=target)

    def capture(self, depth=None) -> bool:
        """현재 카메라 출력에서 env 별로 한 프레임씩 쓴다. 쓸 게 없으면 False.

        Args:
            depth: (num_envs, ...) 모양의 depth 텐서/배열. 주면 각 프레임 오른쪽에
                depth 패널을 붙인다 (play.py --with_depth). None 이면 RGB 만 쓴다.
        """
        rgb = self.camera.data.output["rgb"]
        if rgb is None:
            return False
        frames = rgb[..., :3].detach().cpu().numpy()
        if frames.dtype != np.uint8:
            # float 로 나오는 경우 0..1 로 보고 변환한다.
            frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        for i, w in enumerate(self.writers):
            frame = frames[i]
            if depth is not None:
                frame = np.hstack([frame, depth_to_panel(depth[i], frame.shape[0])])
            w.append_data(frame)
        self.written += 1
        return True

    def close(self):
        """mp4 는 moov atom 을 close() 시점에 쓴다. 중간에 죽으면 파일이 통째로
        재생 불가가 되므로 어떤 경로로 빠져나가든 반드시 불러야 한다."""
        for w in self.writers:
            try:
                w.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] writer close 실패: {exc!r}", flush=True)
        print(f"[INFO] wrote {self.written} frames to {self.out_dir}", flush=True)
        for p in self.paths:
            print(f"[INFO]   {p}", flush=True)
