from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.extreme_parkour import * 

EXTREME_PARKOUR_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.08, ## original scale is 0.05, But Computing issue in IsaacLab see this issue in https://github.com/isaac-sim/IsaacLab/issues/2187
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum= True,
    sub_terrains={
        "parkour_gap": ExtremeParkourGapTerrainCfg(
                        proportion=0.2,
                        apply_roughness=True,
                        x_range = (0.8, 1.5),
                        half_valid_width = (0.6, 1.2),
                        gap_size = '0.1 + 0.7*difficulty'
                        ),
        "parkour_hurdle": ExtremeParkourHurdleTerrainCfg(
                        proportion=0.2,
                        apply_roughness=True,
                        x_range = (1.2, 2.2),
                        half_valid_width = (0.4,0.8),
                        hurdle_height_range= '0.1+0.1*difficulty, 0.15+0.25*difficulty'
                        ),
        "parkour_flat": ExtremeParkourHurdleTerrainCfg(
                        proportion=0.2,
                        apply_roughness=True,
                        apply_flat=True,
                        x_range = (1.2, 2.2),
                        half_valid_width = (0.4,0.8),
                        hurdle_height_range= '0.1+0.1*difficulty, 0.15+0.15*difficulty'
                        ),
        "parkour_step": ExtremeParkourStepTerrainCfg(
                        proportion=0.2,
                        apply_roughness=True,
                        x_range = (0.3,1.5),
                        half_valid_width = (0.5, 1),
                        step_height = '0.1 + 0.35*difficulty'
                        ),
        "parkour": ExtremeParkourTerrainCfg(
                        proportion=0.2,
                        apply_roughness=True,
                        x_range  = '-0.1, 0.1+0.3*difficulty',
                        y_range  = '0.2, 0.3+0.1*difficulty',
                        stone_len  = '0.9 - 0.3*difficulty, 1 - 0.2*difficulty',
                        incline_height = '0.25*difficulty',
                        last_incline_height = 'incline_height + 0.1 - 0.1*difficulty'
                        ),
        "parkour_demo": ExtremeParkourDemoTerrainCfg(
                        proportion=0.0,
                        apply_roughness=True,
                        ),
        # IsaacLab HfInvertedPyramidStairsTerrainCfg 를 parkour 규약에 맞춘 지형.
        # proportion=0.0 이라 teacher/student '학습' 분포는 그대로 두고,
        # PLAY/EVAL 설정에서만 비중을 켜서 쓴다. 학습에도 넣고 싶으면 여기를 올리면 된다.
        #
        # 4.0m 폭 타일에서 step_width=0.3 이면 y 는 4칸 만에 apex_width(1.6m) 에 닿는다.
        # step_depth 를 (pyramid_len - apex_width) / (2*4) = (8.0-1.6)/8 = 0.8 로 맞추면
        # x 도 같은 4칸에서 닫혀 바닥이 1.6m 정사각형인 피라미드가 된다.
        "parkour_pyramid_stairs": ExtremeParkourInvertedPyramidStairsTerrainCfg(
                        proportion=0.0,
                        apply_roughness=True,
                        step_height_range=(0.05, 0.20),
                        step_width=0.3,
                        step_depth=0.8,
                        apex_width=1.6,
                        pyramid_len=8.0,
                        run_up_len=1.5,
                        ),
        # 같은 함수의 inverted=False 판. 구덩이로 내려가는 대신 직각 계단을 올라갔다
        # 내려온다. 올라가는 쪽이 더 힘드므로 계단 한 칸을 조금 낮게 잡았다
        # (난이도 1.0 에서 4칸 x 0.16 = 0.64m 상승).
        #
        # slope_threshold 를 반드시 낮춰야 '직각' 계단이 된다.
        # convert_height_field_to_mesh 는 한 픽셀 사이 높이차가
        #   slope_threshold * horizontal_scale / vertical_scale
        # 를 넘을 때만 그 면을 수직으로 세운다. 기본값 1.5 + horizontal_scale 0.1 이면
        # 0.15m 를 넘는 단차만 직각이 되고, 그보다 낮은 계단은 폭 0.1m 짜리 경사로로
        # 렌더된다(실측: 난이도 0.7 에서 x_edge_mask 가 13픽셀밖에 안 잡혔다).
        # 0.3 이면 기준이 0.03m 라 어느 난이도에서든 계단 면이 수직이 된다.
        # roughness 노이즈는 ±0.02m 라 이 기준에 걸리지 않는다.
        "parkour_pyramid_stairs_up": ExtremeParkourPyramidStairsTerrainCfg(
                        proportion=0.0,
                        apply_roughness=True,
                        step_height_range=(0.05, 0.16),
                        step_width=0.3,
                        step_depth=0.8,
                        apex_width=1.6,
                        pyramid_len=8.0,
                        run_up_len=1.5,
                        slope_threshold=0.3,
                        ),
        # IsaacLab HfDiscreteObstaclesTerrainCfg 를 parkour 규약에 맞춘 지형.
        # x_range 는 goal 간격이다. 7구간 * 최대 2.5m + 플랫폼 2.5m = 20m 로 24m 타일에 들어간다.
        "parkour_discrete_obstacles": ExtremeParkourDiscreteObstaclesTerrainCfg(
                        proportion=0.0,
                        apply_roughness=True,
                        obstacle_height_mode="choice",
                        obstacle_width_range=(0.4, 1.2),
                        obstacle_height_range=(0.05, 0.20),
                        num_obstacles=40,
                        goal_clear_width=0.6,
                        x_range=(1.5, 2.5),
                        ),
        # IsaacLab MeshRandomGridTerrainCfg 를 parkour 규약에 맞춘 지형.
        # apply_roughness 는 끈다. 칸 윗면이 평평한 것이 이 지형의 성격인데 노이즈를
        # 덧씌우면 그 평평함이 사라진다.
        # grid_height_range 는 원본 프리셋과 같은 (0.02, 0.10) 을 쓴다.
        #
        # slope_threshold 는 위 계단과 같은 이유로 낮춘다. 원본은 상자를 쌓아 만들어
        # 칸 옆면이 항상 수직인데, 기본값 1.5 로 두면 칸 경계 대부분이 0.15m 를 못 넘어
        # 경사로가 되어버린다(실측: 난이도 0.7 에서 x_edge_mask 가 0픽셀이었다).
        "parkour_random_grid": ExtremeParkourRandomGridTerrainCfg(
                        proportion=0.0,
                        apply_roughness=False,
                        grid_width=0.5,
                        grid_height_range=(0.02, 0.10),
                        x_range=(1.5, 2.5),
                        slope_threshold=0.3,
                        ),

    },
)