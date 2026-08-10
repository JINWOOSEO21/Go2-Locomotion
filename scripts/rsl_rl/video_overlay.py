# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""녹화 프레임에 붙이는 오버레이 유틸.

play.py --multicam 과 demo.py 가 --with_depth 로 똑같은 depth 패널을 그리도록
구현을 여기 한 곳에만 둔다. 양쪽에 복사해 두면 컬러맵이나 정규화 규칙이
갈라져서, 같은 정책을 찍은 영상인데 depth 가 달라 보이게 된다.
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
