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
        # IsaacLab HfRandomUniformTerrainCfg 를 parkour 규약에 맞춘 지형.
        # 위 피라미드와 같은 이유로 proportion=0.0 (학습 분포 유지, PLAY/EVAL 에서만 켠다).
        #
        # apply_roughness 는 끈다. 다른 지형은 '지물 + 잔잔한 노이즈' 라서 roughness 를
        # 덧씌우지만, 여기서는 균등 노이즈 자체가 지형이라 두 번 얹을 이유가 없다.
        # x_range 는 goal 간격이다. 7구간 * 최대 2.5m + 플랫폼 2.5m = 20m 로 24m 타일에 들어간다.
        "parkour_random_uniform": ExtremeParkourRandomUniformTerrainCfg(
                        proportion=0.0,
                        apply_roughness=False,
                        # IsaacLab ROUGH_TERRAINS_CFG 의 HfRandomUniformTerrainCfg 는
                        # noise_range=(0.02, 0.10) 즉 peak-to-peak 8cm 다. 여기서는 시작
                        # 플랫폼(높이 0)과 이어지도록 0 을 중심으로 대칭을 주되 진폭은
                        # 비슷하게 맞춘다 (난이도 1.0 에서 p2p 10cm).
                        # 참고로 다른 지형에 공통으로 덧씌우는 roughness 는 최대 6cm 다.
                        uniform_noise_range=(-0.05, 0.05),
                        uniform_noise_step=0.02,
                        uniform_downsampled_scale=0.2,
                        x_range=(1.5, 2.5),
                        ),

    },
)