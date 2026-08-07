
from __future__ import annotations

import numpy as np
import random
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING
from ..utils import parkour_field_to_mesh
if TYPE_CHECKING:
    from . import extreme_parkour_terrains_cfg

"""
Reference from https://arxiv.org/pdf/2309.14341
"""

def padding_height_field_raw(
    height_field_raw:np.ndarray, 
    cfg:extreme_parkour_terrains_cfg.ExtremeParkourRoughTerrainCfg
    )->np.ndarray:
    pad_width = int(cfg.pad_width // cfg.horizontal_scale)
    pad_height = int(cfg.pad_height // cfg.vertical_scale)
    height_field_raw[:, :pad_width] = pad_height
    height_field_raw[:, -pad_width:] = pad_height
    height_field_raw[:pad_width, :] = pad_height
    height_field_raw[-pad_width:, :] = pad_height
    height_field_raw = np.rint(height_field_raw).astype(np.int16)
    return height_field_raw

def random_uniform_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourRoughTerrainCfg,
    height_field_raw: np.ndarray,
    ):
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    max_height = (cfg.noise_range[1] - cfg.noise_range[0]) * difficulty + cfg.noise_range[0]
    height_min = int(-cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(max_height / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    z_upsampled = np.rint(z_upsampled).astype(np.int16)
    height_field_raw += z_upsampled 
    return height_field_raw 

@parkour_field_to_mesh
def parkour_gap_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourGapTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))
        mid_y = length_pixels // 2  # length is actually y width
        gap_size = eval(cfg.gap_size,{"difficulty":difficulty})
        gap_size = round(gap_size / cfg.horizontal_scale)

        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale) + gap_size
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) + gap_size

        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height

        gap_depth = -round(np.random.uniform(cfg.gap_depth[0], cfg.gap_depth[1]) / cfg.vertical_scale)
        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height
        goals[0] = [platform_len - 1, mid_y]
        dis_x = platform_len
        last_dis_x = dis_x
        for i in range(num_goals - 2):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            dis_x += rand_x
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            if not cfg.apply_flat:
                height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

            height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
            height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
            
            last_dis_x = dis_x
            goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)

        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale

@parkour_field_to_mesh
def parkour_hurdle_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourHurdleTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        stone_len = eval(cfg.stone_len, {"difficulty": difficulty})
        stone_len = round(stone_len / cfg.horizontal_scale)

        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))

        mid_y = length_pixels // 2  # length is actually y width
        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) 
        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
        hurdle_height_range = eval(cfg.hurdle_height_range, {"difficulty": difficulty})
        hurdle_height_max = round(hurdle_height_range[1] / cfg.vertical_scale)
        hurdle_height_min = round(hurdle_height_range[0] / cfg.vertical_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height
        dis_x = platform_len
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height

        goals[0] = [platform_len - 1, mid_y]

        for i in range(num_goals-2):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            dis_x += rand_x
            if not cfg.apply_flat:
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, :mid_y+rand_y-half_valid_width] = 0
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, mid_y+rand_y+half_valid_width:] = 0
            goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)

        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


@parkour_field_to_mesh
def parkour_step_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourStepTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        step_height = eval(cfg.step_height,{'difficulty':difficulty} )
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))

        mid_y = length_pixels // 2  # length is actually y width
        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) 
        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        step_height = round(step_height / cfg.vertical_scale)

        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height

        dis_x = platform_len
        last_dis_x = dis_x
        stair_height = 0
        goals = np.zeros((num_goals, 2))
        goals[0] = [platform_len - round(1 / cfg.horizontal_scale), mid_y]
        goal_heights = np.ones((num_goals)) * platform_height

        num_stones = num_goals - 2
        for i in range(num_stones):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            if i < num_stones // 2:
                stair_height += step_height
            elif i > num_stones // 2:
                stair_height -= step_height
            height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
            dis_x += rand_x
            height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
            height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
            
            last_dis_x = dis_x
            goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
            goal_heights[i+1] = stair_height
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
        # import ipdb; ipdb.set_trace()
        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights 

@parkour_field_to_mesh
def parkour_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))
        height_field_raw[:] = -round(np.random.uniform(cfg.pit_depth[0], cfg.pit_depth[1]) / cfg.vertical_scale)
        mid_y = length_pixels // 2  # length is actually y width
        stone_len = eval(cfg.stone_len, {"difficulty": difficulty})
        stone_len = np.random.uniform(*stone_len)
        stone_len = 2 * round(stone_len / 2.0, 1)
        stone_len = round(stone_len / cfg.horizontal_scale)
        x_range = eval(cfg.x_range, {"difficulty": difficulty})
        y_range = eval(cfg.y_range, {"difficulty": difficulty})
        dis_x_min = stone_len + round(x_range[0] / cfg.horizontal_scale)
        dis_x_max = stone_len + round(x_range[1] / cfg.horizontal_scale)
        dis_y_min = round(y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(y_range[1] / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height
        
        stone_width = round(cfg.stone_width / cfg.horizontal_scale)
        last_stone_len = round(cfg.last_stone_len / cfg.horizontal_scale)

        incline_height = eval(cfg.incline_height, {"difficulty": difficulty})
        last_incline_height = eval(cfg.last_incline_height, {"difficulty": difficulty, "incline_height":incline_height})
        last_incline_height = round(last_incline_height / cfg.vertical_scale)
        incline_height = round(incline_height / cfg.vertical_scale)

        dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height
        goals[0] = [platform_len -  stone_len // 2, mid_y]
        left_right_flag = np.random.randint(0, 2)
        dis_z = 0
        num_stones = num_goals - 2
        for i in range(num_stones):
            dis_x += np.random.randint(dis_x_min, dis_x_max)
            pos_neg = round(2*(left_right_flag - 0.5))
            dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
            if i == num_stones - 1:
                dis_x += last_stone_len // 4
                heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
                height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
            else:
                heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
            
            goals[i+1] = [dis_x, dis_y]
            goal_heights[i+1] = np.mean(heights.astype(int))

            left_right_flag = 1 - left_right_flag
        final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
        final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // cfg.horizontal_scale)
        height_field_raw[final_platform_start:, :] = platform_height
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale




@parkour_field_to_mesh
def parkour_demo_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourDemoTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    goals = np.zeros((num_goals, 2))
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    mid_y = length_pixels // 2  # length is actually y width

    height_field_raw = np.zeros((width_pixels, length_pixels))
    goal_heights = np.ones((num_goals)) * round(cfg.platform_height / cfg.vertical_scale)
    platform_length = round(2 / cfg.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / cfg.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / cfg.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height

    platform_length += round(np.random.uniform(1.5, 2.5) / cfg.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / cfg.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    goal_heights[1] = first_step_height

    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    goal_heights[2] = second_step_height

    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / cfg.horizontal_scale)
    
    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    goal_heights[3] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    goal_heights[4] = forth_step_height
    
    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / cfg.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / cfg.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / cfg.horizontal_scale)
    slope_width = round(1.0 / cfg.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / cfg.vertical_scale)

    goals[5] = [platform_length+slope_depth/2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    goal_heights[5] = np.mean(heights.astype(int) + platform_height)
    
    platform_length += slope_depth + gap_size
    goals[6] = [platform_length+slope_depth/2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    goal_heights[6] = np.mean(heights.astype(int) + platform_height)
    
    platform_length += slope_depth + gap_size + round(0.4 / cfg.horizontal_scale)
    goals[-1] = [platform_length, left_y]

    height_field_raw = padding_height_field_raw(height_field_raw,cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


@parkour_field_to_mesh
def parkour_pyramid_stairs_terrain(
    difficulty: float,
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourPyramidStairsTerrainCfg,
    num_goals: int,
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IsaacLab 의 :func:`isaaclab.terrains.height_field.hf_terrains.pyramid_stairs_terrain`
    (= ``HfPyramidStairsTerrainCfg`` / ``HfInvertedPyramidStairsTerrainCfg`` 가 쓰는 함수) 을
    parkour 지형 규약에 맞춘 것.

    ``cfg.inverted`` 가 True 면 안쪽으로 파 내려가는 구덩이(내려갔다 올라오기),
    False 면 위로 쌓아 올린 피라미드(직각 계단을 올라갔다 내려오기)가 된다.
    원본과 마찬가지로 계단 면은 slope_threshold 에 의해 수직으로 세워진다.

    원본과 달라진 점은 네 가지다.

    1. 시그니처. 원본은 ``(difficulty, cfg)`` 를 받아 높이맵만 돌려주지만, parkour
       생성기는 ``cfg.function(difficulty, cfg, num_goals)`` 를 부르고
       ``(mesh, origin, goals, goal_heights, x_edge_mask)`` 를 기대한다.
       그래서 ``@height_field_to_mesh`` 대신 ``@parkour_field_to_mesh`` 를 쓴다.
    2. 시작 플랫폼. 원본 피라미드는 타일 전체를 덮는다. 그런데 로봇은 타일 시작점에서
       ``reset_root_state`` 의 offset(1.0m) 만큼 들어온 지점에 스폰되므로, 원본을 그대로
       쓰면 이미 꼭짓점(구덩이 바닥 / 정상 평지) 한가운데서 시작해 계단을 볼 일이 없다.
       여기서는 다른 parkour 지형과 똑같이 앞쪽 ``platform_len`` 을 평지로 남기고,
       그 뒤 ``run_up_len`` 만큼 더 띄운 다음 ``pyramid_len`` 구간에만 피라미드를 만든다.
    3. x/y 계단 폭 분리. 원본은 x, y 모두 ``step_width`` 씩 줄여 정사각 피라미드를 만든다.
       parkour 타일은 4m 폭 × 24m 길이의 복도라 같은 폭으로 줄이면 y 가 먼저 바닥나
       "짧은 계단 + 아주 긴 평평한 바닥" 이 되어버린다. ``step_depth`` 로 x 방향
       감소폭을 따로 주면, x/y 가 동시에 닫혀 바닥이 ``apex_width`` 정사각형인
       진짜 피라미드가 되고 디딤판(tread)도 로봇이 밟기 좋은 깊이가 된다.
       ``step_depth=None`` 이면 원본과 동일하게 ``step_width`` 를 쓴다.
    4. goal / 거칠기. 코스를 따라 8개 goal 을 깔고 각 goal 의 높이를 높이맵에서 읽는다.
       다른 지형과 마찬가지로 ``apply_roughness`` 노이즈와 테두리 패딩도 적용한다.

    계단 한 칸의 높이 계산과 링을 안쪽으로 줄여가는 루프 자체는 원본 그대로다.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))
    mid_y = length_pixels // 2  # length is actually y width

    # -- 계단 파라미터 (원본과 동일). inverted 면 부호를 뒤집어 구덩이가 된다.
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height *= -1
    step_height = round(step_height / cfg.vertical_scale)
    step_width = round(cfg.step_width / cfg.horizontal_scale)
    step_depth = round((cfg.step_depth if cfg.step_depth is not None else cfg.step_width) / cfg.horizontal_scale)
    apex_width = round(cfg.apex_width / cfg.horizontal_scale)
    if step_width < 1 or step_depth < 1:
        # 0 이면 아래 루프가 영원히 돌아간다. horizontal_scale 보다 작은 계단은 만들 수 없다.
        raise ValueError(
            f"step_width({cfg.step_width}) / step_depth({cfg.step_depth}) 는 "
            f"horizontal_scale({cfg.horizontal_scale}) 이상이어야 한다."
        )

    # -- 시작 플랫폼 (로봇 스폰 지점)
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height

    # -- 피라미드가 차지할 x 구간
    run_up = round(cfg.run_up_len / cfg.horizontal_scale)
    pyr_start = platform_len + run_up
    pyr_stop = min(pyr_start + round(cfg.pyramid_len / cfg.horizontal_scale), width_pixels)

    # 원본 루프. x 는 step_depth, y 는 step_width 씩 안쪽으로 줄이며 한 칸씩 파 내려간다.
    start_x, stop_x = pyr_start, pyr_stop
    start_y, stop_y = 0, length_pixels
    current_step_height = 0
    while (stop_x - start_x) > apex_width and (stop_y - start_y) > apex_width:
        # increment position
        start_x += step_depth
        stop_x -= step_depth
        start_y += step_width
        stop_y -= step_width
        # increment height
        current_step_height += step_height
        # add the step
        height_field_raw[start_x:stop_x, start_y:stop_y] = current_step_height

    # -- goals: 시작 플랫폼 끝 -> 피라미드 진입/바닥/탈출 -> 빠져나온 평지
    goals = np.zeros((num_goals, 2))
    goal_heights = np.zeros((num_goals))
    goals[0] = [platform_len - 1, mid_y]
    goal_heights[0] = platform_height
    # 계단 수는 난이도/크기에 따라 달라지므로 중간 goal 은 등간격으로 깐다.
    # 그래야 계단이 몇 칸이든 내려가기-바닥-올라오기가 고르게 커버된다.
    mid_goals_x = np.linspace(pyr_start + step_depth * 0.5, pyr_stop - step_depth * 0.5, num_goals - 2)
    for i, goal_x in enumerate(mid_goals_x):
        goals[i + 1] = [goal_x, mid_y]
        goal_heights[i + 1] = height_field_raw[int(round(goal_x)), mid_y]
    final_dis_x = pyr_stop + run_up
    if final_dis_x > width_pixels:
        final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    goal_heights[-1] = platform_height

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


def _lay_goals_over_trapezoid(
    cfg,
    height_field_raw: np.ndarray,
    num_goals: int,
    course_start: int,
    course_end: int,
    width_pixels: int,
    length_pixels: int,
    y_lo: int,
    y_hi: int,
    ):
    """사다리꼴(오르막-평지-내리막) 코스용 goal 배치.

    중간 goal 은 코스 구간 [course_start, course_end] 에 등간격으로 깔고
    (pyramid_stairs 와 같은 방식 — 경사/계단이 몇 픽셀이든 오르막-평지-내리막을
    고르게 커버한다), y 는 ``cfg.y_range`` 로 흔들어 코스 폭 안에서 랜덤화한다.
    goal 높이는 높이맵에서 그대로 읽는다.

    goal_heights 는 vertical_scale 을 곱하지 않은 원시 단위로 돌려준다.
    소비처(parkour_event._debug_vis_callback)가 vertical_scale 을 곱하기 때문이다.
    (step 지형과 같은 규약. gap/hurdle 처럼 미터로 돌려주면 int16 캐스팅에서 0 이
    되어 goal 마커가 바닥 높이에 찍힌다.)
    """
    mid_y = length_pixels // 2
    dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
    dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)

    goals = np.zeros((num_goals, 2))
    goal_heights = np.ones((num_goals)) * platform_height
    goals[0] = [platform_len - 1, mid_y]

    mid_goals_x = np.linspace(course_start, course_end, num_goals - 2)
    for i, goal_x in enumerate(mid_goals_x):
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        goal_y = int(np.clip(mid_y + rand_y, y_lo + 1, y_hi - 2))
        goals[i + 1] = [goal_x, goal_y]
        goal_heights[i + 1] = height_field_raw[int(round(goal_x)), goal_y]

    final_dis_x = course_end + np.random.randint(
        round(0.8 / cfg.horizontal_scale), round(1.5 / cfg.horizontal_scale)
    )
    final_dis_x = min(final_dis_x, width_pixels - round(0.5 / cfg.horizontal_scale))
    goals[-1] = [final_dis_x, mid_y]
    goal_heights[-1] = height_field_raw[int(final_dis_x), mid_y]
    return goals, goal_heights


def _course_span(cfg, width_pixels: int, length_pixels: int):
    """사다리꼴 지형 공통: 시작 플랫폼과 코스 폭(y 구간)을 계산한다.

    ``course_width`` (기본 5.0m) 가 타일 폭보다 넓으면 타일 전체 폭을 쓴다.
    (현재 타일은 4m 폭이라 사실상 전폭 코스가 된다. 폭 5m 를 그대로 쓰려면
    generator 의 size[1] 을 5.0 이상으로 키워야 한다.)
    """
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    mid_y = length_pixels // 2
    half_w = round(min(cfg.course_width, cfg.size[1]) / cfg.horizontal_scale) // 2
    y_lo = max(0, mid_y - half_w)
    y_hi = min(length_pixels, mid_y + half_w)
    return platform_len, mid_y, y_lo, y_hi


@parkour_field_to_mesh
def parkour_trapezoid_ramp_terrain(
    difficulty: float,
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourTrapezoidRampTerrainCfg,
    num_goals: int,
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """옆에서 보면 사다리꼴인 경사로 지형: 오르막 - 평지 - 내리막.

    - 경사 각도는 ``slope_angle`` (도 단위 difficulty 수식) 로 정하고,
      오르막과 내리막의 기울기는 같다. 기본 '10 + 27*difficulty' 는 10행 커리큘럼
      (difficulty = row/9) 에서 정확히 10, 13, ..., 37도가 된다.
    - 평지(꼭대기) 길이와 높이는 ``plateau_len_range`` / ``plateau_height_range``
      에서 랜덤으로 뽑는다. 각도가 스펙이므로, 완만한 각도에서 뽑힌 높이가
      타일 길이를 넘치게 하면 각도를 유지한 채 높이를 낮춰서 맞춘다.
    - 경사면은 slope_threshold(기본 1.5) 기준을 넘지 않아 (37도에서 픽셀당
      0.075m < 0.15m) 직각화되지 않고 매끈한 램프로 렌더된다. roughness 노이즈가
      경사면 위에 그대로 얹혀 발 디딤 랜덤화가 된다.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))

    platform_len, mid_y, y_lo, y_hi = _course_span(cfg, width_pixels, length_pixels)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height

    slope_deg = eval(cfg.slope_angle, {"difficulty": difficulty})
    slope = np.tan(np.deg2rad(slope_deg))

    plateau_len = round(np.random.uniform(*cfg.plateau_len_range) / cfg.horizontal_scale)
    plateau_height_m = np.random.uniform(*cfg.plateau_height_range)

    # 각도를 유지한 채 타일에 들어가도록 높이를 clamp 한다.
    end_margin = round(cfg.end_margin / cfg.horizontal_scale)
    avail = width_pixels - platform_len - plateau_len - end_margin
    run_len = round(plateau_height_m / slope / cfg.horizontal_scale)
    run_len = int(np.clip(run_len, 2, max(avail // 2, 2)))
    plateau_height = round(run_len * cfg.horizontal_scale * slope / cfg.vertical_scale)

    up_start = platform_len
    up_end = up_start + run_len
    down_start = up_end + plateau_len
    down_end = down_start + run_len

    # 오르막 / 평지 / 내리막 (내리막은 같은 기울기를 뒤집은 것).
    # linspace(0, 높이, run_len) 는 픽셀 간 기울기가 h/(run_len-1) 이라 램프가 짧을수록
    # (37도, 높이 0.4m 면 5픽셀) 실제 각도가 스펙보다 가팔라진다. 픽셀당 상승량을
    # 정확히 tan(angle)*horizontal_scale 로 깔아 어떤 길이에서도 각도를 보존한다.
    rise_per_px = cfg.horizontal_scale * slope / cfg.vertical_scale
    ramp = np.rint(np.arange(1, run_len + 1) * rise_per_px)[:, None]
    height_field_raw[up_start:up_end, y_lo:y_hi] = ramp
    height_field_raw[up_end:down_start, y_lo:y_hi] = plateau_height
    height_field_raw[down_start:down_end, y_lo:y_hi] = plateau_height - ramp

    goals, goal_heights = _lay_goals_over_trapezoid(
        cfg, height_field_raw, num_goals,
        course_start=up_start + round(0.5 / cfg.horizontal_scale),
        course_end=down_end,
        width_pixels=width_pixels, length_pixels=length_pixels, y_lo=y_lo, y_hi=y_hi,
    )

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights


@parkour_field_to_mesh
def parkour_trapezoid_stairs_terrain(
    difficulty: float,
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourTrapezoidStairsTerrainCfg,
    num_goals: int,
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """사다리꼴 계단 지형: 계단 오르막 - 평지 - 계단 내리막.

    - 단차(riser) 는 ``step_height`` (difficulty 수식) 로 정한다. 기본
      '0.05 + 0.18*difficulty' 는 10행 커리큘럼에서 정확히 5, 7, ..., 23cm 가 된다.
    - 단의 개수는 ``num_steps`` (기본 5개) 로 고정이고, 평지 높이는 자연히
      num_steps * step_height 가 된다.
    - 디딤판(tread) 깊이는 step 지형과 같은 방식으로 계단마다 ``x_range`` 에서
      따로 뽑는다 (오르막/내리막 각각 독립 샘플). step 지형의 x_range (0.3, 1.5)
      를 그대로 쓰면 왕복 10칸이 타일을 넘칠 수 있어 기본값을 (0.3, 0.8) 로 줄였다.
    - 내리막도 같은 단차의 계단이다 (스펙에는 오르막만 명시돼 있지만, 사다리꼴
      구조를 유지하려면 내려오는 쪽도 필요하다. 내리막을 경사로로 바꾸려면
      이 함수에서 down 루프만 램프로 갈아끼우면 된다).
    - slope_threshold 기본 0.3: 가장 낮은 단차 5cm 도 기준(0.03m)을 넘어
      계단 면이 수직으로 선다. 기본값 1.5(기준 0.15m)면 23cm 를 빼고는 전부
      경사로로 뭉개진다 (pyramid_stairs 와 같은 이유).
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))

    platform_len, mid_y, y_lo, y_hi = _course_span(cfg, width_pixels, length_pixels)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height

    step_height = round(eval(cfg.step_height, {"difficulty": difficulty}) / cfg.vertical_scale)
    dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale)
    plateau_len = round(np.random.uniform(*cfg.plateau_len_range) / cfg.horizontal_scale)

    # 디딤판 깊이를 미리 다 뽑고, 넘치면 비율로 줄여 타일에 맞춘다.
    end_margin = round(cfg.end_margin / cfg.horizontal_scale)
    treads = np.random.randint(dis_x_min, dis_x_max, size=2 * cfg.num_steps)
    avail = width_pixels - platform_len - plateau_len - end_margin
    if treads.sum() > avail:
        treads = np.maximum((treads * (avail / treads.sum())).astype(int), 2)

    plateau_height = cfg.num_steps * step_height
    dis_x = platform_len
    # 오르막 계단: i번째 단의 윗면 높이는 (i+1)*step_height
    for i in range(cfg.num_steps):
        depth = treads[i]
        height_field_raw[dis_x:dis_x + depth, y_lo:y_hi] = (i + 1) * step_height
        dis_x += depth
    up_end = dis_x
    # 평지 (꼭대기)
    height_field_raw[up_end:up_end + plateau_len, y_lo:y_hi] = plateau_height
    dis_x = up_end + plateau_len
    # 내리막 계단: 마지막 단에서 바닥(0)에 닿는다
    for i in range(cfg.num_steps):
        depth = treads[cfg.num_steps + i]
        height_field_raw[dis_x:dis_x + depth, y_lo:y_hi] = plateau_height - (i + 1) * step_height
        dis_x += depth
    down_end = dis_x

    goals, goal_heights = _lay_goals_over_trapezoid(
        cfg, height_field_raw, num_goals,
        course_start=platform_len + max(treads[0] // 2, 1),
        course_end=down_end,
        width_pixels=width_pixels, length_pixels=length_pixels, y_lo=y_lo, y_hi=y_hi,
    )

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights


def _lay_goals_along_corridor(
    cfg,
    height_field_raw: np.ndarray,
    num_goals: int,
    width_pixels: int,
    length_pixels: int,
    clear_half: int = 0,
    ):
    """지물이 균질하게 흩어진 지형용 goal 배치.

    gap/hurdle 처럼 ``x_range`` / ``y_range`` 에서 간격을 뽑아 코스를 따라 goal 을 깐다.
    ``clear_half`` 가 0 보다 크면 각 goal 둘레 (2*clear_half+1) 픽셀을 평지로 밀어
    로봇이 디딜 자리를 만든다 (goal 이 기둥 위나 구덩이 속에 찍히는 것을 막는다).

    height_field_raw 를 제자리에서 수정하고 (goals, goal_heights) 를 돌려준다.
    """
    mid_y = length_pixels // 2
    dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale)
    dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
    dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)

    goals = np.zeros((num_goals, 2))
    goal_heights = np.zeros((num_goals))
    goals[0] = [platform_len - 1, mid_y]
    goal_heights[0] = platform_height

    dis_x = platform_len
    for i in range(num_goals - 1):
        if i == num_goals - 2:
            # 마지막 goal 은 코스를 완전히 빠져나온 지점
            dis_x += np.random.randint(dis_x_min, dis_x_max)
            goal_x = min(dis_x, width_pixels - 1)
            goal_y = mid_y
        else:
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            dis_x += rand_x
            goal_x = min(dis_x - rand_x // 2, width_pixels - 1)
            goal_y = int(np.clip(mid_y + rand_y, 0, length_pixels - 1))
        if clear_half > 0:
            x0, x1 = max(0, goal_x - clear_half), min(width_pixels, goal_x + clear_half + 1)
            y0, y1 = max(0, goal_y - clear_half), min(length_pixels, goal_y + clear_half + 1)
            height_field_raw[x0:x1, y0:y1] = platform_height
        goals[i + 1] = [goal_x, goal_y]
        goal_heights[i + 1] = height_field_raw[int(goal_x), int(goal_y)]

    return goals, goal_heights


@parkour_field_to_mesh
def parkour_discrete_obstacles_terrain(
    difficulty: float,
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourDiscreteObstaclesTerrainCfg,
    num_goals: int,
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IsaacLab 의 :func:`isaaclab.terrains.height_field.hf_terrains.discrete_obstacles_terrain`
    (= ``HfDiscreteObstaclesTerrainCfg`` 가 쓰는 함수) 을 parkour 지형 규약에 맞춘 것.

    장애물 하나를 뽑는 절차 — 높이를 ``choice`` 면 ``[-h, -h/2, h/2, h]`` 에서 고르고,
    폭/길이를 4픽셀 간격 격자에서 고르고, 시작 위치도 4픽셀 격자에서 고른 뒤 지형 밖으로
    나가면 안쪽으로 당기는 — 은 원본 그대로다. 달라진 점은 세 가지다.

    1. 시그니처/데코레이터. 다른 이식 지형과 같은 이유로 ``@parkour_field_to_mesh`` 를 쓰고
       ``(difficulty, cfg, num_goals)`` 를 받아 goal 까지 돌려준다.
    2. 평지로 비우는 위치. 원본은 '타일 한가운데' 정사각형만 비운다. parkour 타일에서는
       로봇이 타일 시작점 1.0m 지점에 스폰하므로 한가운데를 비워봐야 소용이 없다.
       여기서는 앞쪽 ``platform_len`` 전체를 시작 플랫폼으로 비운다.
    3. goal 자리 확보. 원본에는 goal 개념이 없어 장애물이 어디 놓이든 상관없지만, parkour
       에서는 goal 이 기둥 꼭대기나 구덩이 바닥에 찍히면 로봇이 도달할 수 없어 태스크가
       깨진다. goal 둘레 ``goal_clear_width`` 만큼을 평지로 밀어 디딜 자리를 만든다.
       (장애물 사이를 헤집고 지나가는 성격은 그대로 남는다.)
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # -- 원본과 동일한 파라미터 해석
    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )
    obs_height = int(obs_height / cfg.vertical_scale)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)

    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    if len(obs_width_range) == 0 or len(obs_length_range) == 0:
        raise ValueError(
            f"obstacle_width_range({cfg.obstacle_width_range}) 가 horizontal_scale"
            f"({cfg.horizontal_scale}) 기준 4픽셀 격자를 하나도 못 만든다. 범위를 넓혀라."
        )
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)

    height_field_raw = np.zeros((width_pixels, length_pixels))
    for _ in range(cfg.num_obstacles):
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        else:
            raise ValueError(
                f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'."
            )
        width = int(np.random.choice(obs_width_range))
        length = int(np.random.choice(obs_length_range))
        x_start = int(np.random.choice(obs_x_range))
        y_start = int(np.random.choice(obs_y_range))
        if x_start + width > width_pixels:
            x_start = width_pixels - width
        if y_start + length > length_pixels:
            y_start = length_pixels - length
        height_field_raw[x_start : x_start + width, y_start : y_start + length] = height

    # -- 시작 플랫폼 (원본의 '가운데 비우기' 대신)
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height

    clear_half = max(0, round(cfg.goal_clear_width / cfg.horizontal_scale) // 2)
    goals, goal_heights = _lay_goals_along_corridor(
        cfg, height_field_raw, num_goals, width_pixels, length_pixels, clear_half=clear_half
    )

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


@parkour_field_to_mesh
def parkour_random_grid_terrain(
    difficulty: float,
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourRandomGridTerrainCfg,
    num_goals: int,
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IsaacLab 의 :func:`isaaclab.terrains.trimesh.mesh_terrains.random_grid_terrain`
    (= ``MeshRandomGridTerrainCfg`` 가 쓰는 함수) 을 parkour 지형 규약에 맞춘 것.

    격자 칸마다 윗면 높이를 ``uniform(-grid_height, +grid_height)`` 로 뽑아 평평한 사각
    타일을 깔고, 칸 경계는 수직으로 떨어지는 — 그 성격은 원본 그대로다.
    달라진 점은 네 가지다.

    1. 표현 방식. 원본은 trimesh 지형이라 칸마다 상자를 만들어 붙이고
       ``(meshes, origin)`` 만 돌려준다. parkour 생성기는 높이맵 기반
       ``@parkour_field_to_mesh`` 를 거쳐 mesh 와 함께 ``x_edge_mask`` (발이 모서리를
       밟았는지 판정하는 마스크) 를 만들어야 하는데, 이건 높이맵이 있어야 계산된다.
       그래서 같은 '칸별 랜덤 높이' 를 높이맵으로 구현했다. 칸 경계가 한 픽셀에서
       수직으로 꺾이므로 slope_threshold 를 거치면 원본과 같은 직각 단차가 된다.
    2. 정사각 제한. 원본은 ``if cfg.size[0] != cfg.size[1]: raise ValueError`` 로 정사각
       타일만 받는다. parkour 타일은 24m x 4m 복도라 그대로는 아예 생성이 거부된다.
       높이맵 방식에는 그런 제약이 없어 그냥 직사각형 타일을 채운다.
    3. 시작 플랫폼. 원본은 타일 한가운데에 평지 플랫폼을 놓지만, 여기서는 다른 parkour
       지형과 같이 앞쪽 ``platform_len`` 을 평지로 남긴다 (로봇 스폰 지점).
    4. goal. 코스를 따라 8개를 깔고 각 goal 높이를 높이맵에서 읽는다. 칸 자체가 평평해서
       goal 이 칸 위에 찍히면 그대로 디딜 수 있으므로 따로 비우지 않는다.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # -- 원본과 동일: 난이도로 칸 높이 진폭을 정한다
    grid_height = cfg.grid_height_range[0] + difficulty * (cfg.grid_height_range[1] - cfg.grid_height_range[0])
    grid_height = grid_height / cfg.vertical_scale
    grid_width = max(1, round(cfg.grid_width / cfg.horizontal_scale))

    num_cells_x = int(np.ceil(width_pixels / grid_width))
    num_cells_y = int(np.ceil(length_pixels / grid_width))
    # 칸마다 하나의 높이를 뽑아 픽셀로 펼친다 (원본의 h_noise.uniform_(-grid_height, grid_height))
    cell_heights = np.random.uniform(-grid_height, grid_height, size=(num_cells_x, num_cells_y))
    height_field_raw = np.repeat(np.repeat(cell_heights, grid_width, axis=0), grid_width, axis=1)
    height_field_raw = np.rint(height_field_raw[:width_pixels, :length_pixels])

    # -- 시작 플랫폼
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height

    goals, goal_heights = _lay_goals_along_corridor(
        cfg, height_field_raw, num_goals, width_pixels, length_pixels, clear_half=0
    )

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale
