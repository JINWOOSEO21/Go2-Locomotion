from locomotion_isaaclab.terrains.locomotion_terrains.locomotion_terrains_cfg import *
from locomotion_isaaclab.terrains.locomotion_terrain_generator_cfg import LocomotionTerrainGeneratorCfg

TEST_TERRAINS_CFG = LocomotionTerrainGeneratorCfg(
    size=(22.0, 12.0),
    border_width=5.0,
    num_rows=2,
    num_cols=2,
    # horizontal_scale=0.05,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    num_goals=8,
    curriculum=True,
    sub_terrains={
        "gap": LocomotionGapTerrainCfg(
            proportion=0.2,
            apply_roughness=True,
            x_range=(0.8, 1.5),
            y_range=(-0.1, 0.1),
            half_valid_width=(0.6, 1.2),
        )
        # "hurdle": LocomotionHurdleTerrainCfg(
        #                 proportion=0.2,
        #                 apply_roughness=True,
        #                 x_range = (1.2, 2.2),
        #                 y_range = (-0.1, 0.1),
        #                 half_valid_width = (0.4,0.8),
        #                 hurdle_height_range= '0.1+0.1*difficulty, 0.15+0.25*difficulty'
        #                 ),
        # "flat": LocomotionHurdleTerrainCfg(
        #                 proportion=0.2,
        #                 apply_roughness=True,
        #                 apply_flat=True,
        #                 x_range = (1.2, 2.2),
        #                 y_range = (-0.1, 0.1),
        #                 half_valid_width = (0.4,0.8),
        #                 hurdle_height_range= '0.1+0.1*difficulty, 0.15+0.15*difficulty'
        #                 ),
        # "step": LocomotionStepTerrainCfg(
        #                 proportion=0.2,
        #                 apply_roughness=True,
        #                 x_range = (0.3,1.5),
        #                 y_range = (-0.1, 0.1),
        #                 half_valid_width = (0.5, 1)
        #                 ),
        # "obstacle_course": LocomotionObstacleCourseTerrainCfg(
        #                 proportion=0.2,
        #                 apply_roughness=True,
        #                 y_range = (-0.1, 0.1),
        #                 ),
        # "demo": LocomotionDemoTerrainCfg(
        #         y_range = (-0.1, 0.1),
        #         proportion=0.0,
        #         apply_roughness=True,
        #         ),
    },
)
