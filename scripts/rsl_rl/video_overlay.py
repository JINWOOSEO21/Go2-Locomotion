# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""녹화 프레임에 붙이는 오버레이 유틸.

play.py --multicam 과 demo.py 가 --with_depth 로 똑같은 depth 패널을 그리도록
구현을 여기 한 곳에만 둔다. 양쪽에 복사해 두면 컬러맵이나 정규화 규칙이
갈라져서, 같은 정책을 찍은 영상인데 depth 가 달라 보이게 된다.

--with_scandots 의 height-scan 패널도 같은 이유로 여기 둔다.
"""

from __future__ import annotations

import cv2
import numpy as np


def depth_to_panel(depth: np.ndarray, height: int) -> np.ndarray:
    """정책이 먹는 depth 한 장을 영상 오른쪽에 붙일 RGB 패널로 만든다.

    observations.image_features 가 돌려주는 값은
        depth_m / clipping_range - 0.5
    로 정규화돼 있어 대략 [-0.5, 0.5] 범위다(가까울수록 작다). 여기에 0.5 를 더해
    0..1 로 되돌린 뒤 8bit 로 만든다.

    확대는 INTER_NEAREST 로 한다. 원본이 87x58 밖에 안 되는데 부드럽게 보간하면
    정책이 실제로 보는 해상도보다 정보가 많아 보인다. 픽셀을 그대로 키워야
    "이 정도로 성긴 입력을 보고 있다"가 눈에 들어온다.

    TURBO 컬러맵을 씌운다. 회색조는 지형의 원근 차이가 잘 안 읽힌다.
    """
    d8 = np.clip((depth + 0.5) * 255.0, 0, 255).astype(np.uint8)
    src_h, src_w = d8.shape
    panel_w = int(round(height * src_w / src_h))
    big = cv2.resize(d8, (panel_w, height), interpolation=cv2.INTER_NEAREST)
    # applyColorMap 은 BGR 로 돌려주므로 채널을 뒤집어 RGB 로 맞춘다.
    return cv2.applyColorMap(big, cv2.COLORMAP_TURBO)[:, :, ::-1]


# scandots 값은 observations._get_heights() 의
#     clip(base_z - hit_z - 0.3, -1, 1)
# 이다. 평지에서 GO2 의 base 가 0.3~0.4m 쯤이므로 0 근처에서 놀고, 앞에 턱이 있으면
# hit_z 가 올라가 음수로, 구덩이면 양수로 간다. [-1,1] 전체를 다 쓰면 대비가 죽어서
# 실제로 움직이는 구간만 창으로 잘라 쓴다. 프레임마다 자동 정규화하면 같은 색이
# 프레임마다 다른 높이를 뜻하게 되므로 창은 상수로 고정한다.
SCANDOTS_VALUE_RANGE = (-0.5, 0.5)


def scandots_to_panel(
    scandots: np.ndarray,
    height: int,
    grid_shape: tuple[int, int],
    robot_cell: tuple[float, float] | None = None,
    value_range: tuple[float, float] = SCANDOTS_VALUE_RANGE,
) -> np.ndarray:
    """teacher 정책이 먹는 height scan 132 개를 탑뷰 패널로 만든다.

    Args:
        scandots: obs[num_prop : num_prop+num_scan] 한 env 분. 길이 n_y*n_x.
        height: 붙일 프레임 높이(px). 패널은 격자 비율을 지켜 이 높이에 맞춘다.
        grid_shape: (n_y, n_x). GridPatternCfg 의 size/resolution 에서 나온다.
        robot_cell: 로봇 base 가 놓인 격자 좌표 (y_idx, x_idx). 주면 표시한다.
        value_range: 색으로 펼칠 값의 창 (lo, hi).

    격자 순서는 GridPatternCfg(ordering="xy") 기준이라 평탄화 인덱스가
    ``y_idx * n_x + x_idx`` 다. 따라서 reshape 는 (n_y, n_x) 로 해야 한다.

    센서가 ray_alignment="yaw" 라 격자는 로봇 헤딩을 따라 돈다. 즉 이 패널은
    항상 "로봇 기준" 탑뷰다. 위쪽이 로봇 앞(+x), 왼쪽이 로봇의 왼쪽(+y)이 되게
    전치 후 두 축을 뒤집는다.

    depth 패널과 같은 이유로 INTER_NEAREST 로 키운다. 12x11 짜리 성긴 격자라는
    사실이 화면에 그대로 남아야 한다.
    """
    n_y, n_x = grid_shape
    grid = np.asarray(scandots, dtype=np.float32).reshape(n_y, n_x)
    # [y_idx, x_idx] -> 전치하면 [x_idx, y_idx]. 행(x)을 뒤집어 앞이 위로,
    # 열(y)을 뒤집어 로봇의 왼쪽이 화면 왼쪽으로 오게 한다.
    img = grid.T[::-1, ::-1]

    lo, hi = value_range
    u8 = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    panel_w = max(1, int(round(height * n_y / n_x)))
    big = cv2.resize(u8, (panel_w, height), interpolation=cv2.INTER_NEAREST)
    # applyColorMap 은 BGR 이라 뒤집어 RGB 로 맞춘다. 뒤집힌 뷰는 음수 stride 라
    # 그대로는 cv2 그리기 함수에 못 넘기므로 연속 배열로 복사한다.
    panel = np.ascontiguousarray(cv2.applyColorMap(big, cv2.COLORMAP_TURBO)[:, :, ::-1])

    if robot_cell is not None:
        # 격자 좌표 -> 화면 좌표. 위의 전치/뒤집기와 같은 변환이다.
        y_idx, x_idx = robot_cell
        cell_h = height / n_x
        cell_w = panel_w / n_y
        cy = int(round(((n_x - 1 - x_idx) + 0.5) * cell_h))
        cx = int(round(((n_y - 1 - y_idx) + 0.5) * cell_w))
        radius = max(2, int(round(min(cell_h, cell_w) * 0.18)))
        # 검은 테두리를 깔아야 TURBO 어느 색 위에서도 흰 점이 보인다.
        cv2.circle(panel, (cx, cy), radius + 1, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        # 위쪽(=로봇 앞)을 가리키는 짧은 선. 격자가 헤딩을 따라 도므로 항상 위다.
        tip = (cx, int(round(cy - cell_h * 0.9)))
        cv2.line(panel, (cx, cy), tip, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.line(panel, (cx, cy), tip, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return panel
