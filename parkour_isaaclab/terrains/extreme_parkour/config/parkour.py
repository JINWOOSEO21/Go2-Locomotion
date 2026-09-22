from parkour_isaaclab.terrains.extreme_parkour import *
from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg

EXTREME_PARKOUR_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.08,  ## original scale is 0.05, But Computing issue in IsaacLab see this issue in https://github.com/isaac-sim/IsaacLab/issues/2187
    vertical_scale=0.005,
    # 높이맵을 메쉬로 바꿀 때 '한 픽셀 사이 단차를 수직 벽으로 세울지' 를 정하는
    # 기울기(rise/run) 기준. horizontal_scale 0.08 에서는 단차 0.12m 초과분만
    # 수직이 되고, 그보다 낮은 단차는 가로 0.08m 짜리 비탈로 렌더된다.
    # x_edge_mask(발이 모서리를 밟았는지 판정) 도 '수직으로 세운 픽셀' 목록이라
    # 이 값이 곧 edge 로 인정되는 최소 단차다.
    #
    # 주의: 이 값은 지형 전체에 공통으로 적용된다. IsaacLab TerrainGenerator.__init__
    # 이 모든 sub_terrain 의 slope_threshold 를 여기 값으로 덮어쓰기 때문에
    # (isaaclab/terrains/terrain_generator.py 의 sub_cfg.slope_threshold 대입),
    # sub_terrains 쪽에 따로 적어도 효과가 없다.
    # 낮추면 낮은 계단이 직각이 되지만 roughness 노이즈(픽셀당 최대 6cm)까지
    # 기준을 넘겨 램프/평지가 통째로 미세계단 + edge 로 뭉개진다. 실측: 0.3 에서
    # trapezoid_ramp 의 edge 픽셀이 9 -> 2846 으로 폭증했다.
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "parkour_gap": ExtremeParkourGapTerrainCfg(
            proportion=0.2,
            apply_roughness=True,
            x_range=(0.8, 1.5),
            half_valid_width=(0.6, 1.2),
            gap_size="0.1 + 0.7*difficulty",
        ),
        "parkour_hurdle": ExtremeParkourHurdleTerrainCfg(
            proportion=0.2,
            apply_roughness=True,
            x_range=(1.2, 2.2),
            half_valid_width=(0.4, 0.8),
            hurdle_height_range="0.1+0.1*difficulty, 0.15+0.25*difficulty",
        ),
        "parkour_flat": ExtremeParkourHurdleTerrainCfg(
            proportion=0.2,
            apply_roughness=True,
            apply_flat=True,
            x_range=(1.2, 2.2),
            half_valid_width=(0.4, 0.8),
            hurdle_height_range="0.1+0.1*difficulty, 0.15+0.15*difficulty",
        ),
        "parkour_step": ExtremeParkourStepTerrainCfg(
            proportion=0.2,
            apply_roughness=True,
            x_range=(0.3, 1.5),
            half_valid_width=(0.5, 1),
            step_height="0.1 + 0.35*difficulty",
        ),
        "parkour": ExtremeParkourTerrainCfg(
            proportion=0.2,
            apply_roughness=True,
            x_range="-0.1, 0.1+0.3*difficulty",
            y_range="0.2, 0.3+0.1*difficulty",
            stone_len="0.9 - 0.3*difficulty, 1 - 0.2*difficulty",
            incline_height="0.25*difficulty",
            last_incline_height="incline_height + 0.1 - 0.1*difficulty",
        ),
        "parkour_demo": ExtremeParkourDemoTerrainCfg(
            proportion=0.0,
            apply_roughness=True,
        ),
        # IsaacLab HfInvertedPyramidStairsTerrainCfg 를 parkour 규약에 맞춘 지형.
        # 여기의 proportion 은 기본값일 뿐이고, 실제 분포는 각 env cfg 가
        # apply_terrain_preset() 으로 덮어쓴다. 학습에 넣으려면 TERRAIN_PRESETS 의
        # *_train 프리셋에 가중치를 추가하면 된다.
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
        # 같은 함수의 inverted=False 판. 구덩이로 내려가는 대신 계단을 올라갔다
        # 내려온다. 올라가는 쪽이 더 힘드므로 계단 한 칸을 조금 낮게 잡았다
        # (난이도 1.0 에서 4칸 x 0.16 = 0.64m 상승).
        #
        # 단차가 generator 의 slope_threshold 기준(0.08 스케일에서 0.12m)보다 낮은
        # 난이도 구간에서는 계단 면이 수직으로 서지 않고 x_edge_mask 도 안 생긴다.
        "parkour_pyramid_stairs_up": ExtremeParkourPyramidStairsTerrainCfg(
            proportion=0.0,
            apply_roughness=True,
            step_height_range=(0.05, 0.16),
            step_width=0.3,
            step_depth=0.8,
            apex_width=1.6,
            pyramid_len=8.0,
            run_up_len=1.5,
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
        # 원본은 상자를 쌓아 만들어 칸 옆면이 항상 수직이지만, 여기서는 높이맵을
        # 거치므로 칸 경계 높이차가 generator 의 slope_threshold 기준(0.08 스케일에서
        # 0.12m)을 넘지 못해 수직 벽도 x_edge_mask 도 생기지 않는다.
        "parkour_random_grid": ExtremeParkourRandomGridTerrainCfg(
            proportion=0.0,
            apply_roughness=False,
            grid_width=0.5,
            grid_height_range=(0.02, 0.10),
            x_range=(1.5, 2.5),
        ),
        # 사다리꼴 경사로: 오르막(10~37도) - 평지 - 내리막. 오르막/내리막 기울기 동일.
        # 10행 커리큘럼에서 각도가 정확히 10,13,...,37도가 되도록 slope_angle 을 잡았다.
        # 평지 길이/높이는 랜덤 (완만한 각도에서 넘치면 각도 유지, 높이만 clamp).
        "parkour_trapezoid_ramp": ExtremeParkourTrapezoidRampTerrainCfg(
            proportion=0.0,
            apply_roughness=True,
            slope_angle="10 + 27*difficulty",
            course_width_range=(2.0, 4.0),
            plateau_len_range=(1.5, 3.0),
            plateau_height_range=(0.4, 0.8),
        ),
        # 사다리꼴 계단: 계단 오르막 - 평지 - 계단 내리막 (단 수는 타일마다 3~7개 랜덤).
        # 단차는 10행 커리큘럼에서 정확히 5,7,...,23cm. 디딤판 깊이는 step 지형처럼
        # 계단마다 x_range 에서 랜덤.
        #
        # generator 의 slope_threshold(0.08 스케일에서 0.12m) 때문에 단차가 그보다
        # 낮은 row 0~3 은 계단 면이 수직으로 서지 않고 x_edge_mask 도 비어 있다.
        # reward_feet_edge 가 terrain_levels > 3 에서만 켜지므로 실제 리워드에는
        # 영향이 없다 (실측 edge 픽셀: row3=89, row4=232, row6=543).
        "parkour_trapezoid_stairs": ExtremeParkourTrapezoidStairsTerrainCfg(
            proportion=0.0,
            apply_roughness=True,
            step_height="0.05 + 0.18*difficulty",
            num_steps_range=(3, 7),
            x_range=(0.3, 0.8),
            course_width_range=(2.0, 4.0),
            plateau_len_range=(1.5, 3.0),
        ),
    },
)

# ---------------------------------------------------------------------------
# 지형 분포 프리셋
#
# 위 sub_terrains 는 '지형의 기하(모양)' 정의이고, '어떤 지형을 얼마나 섞을지'는
# 전부 여기서 관리한다. env cfg 마다 if/elif 로 proportion 을 재설정하다가
# 한 곳을 빠뜨려 분포가 조용히 틀어지는 사고를 막기 위한 단일 소스다.
#
# 새 지형을 추가하는 절차:
#   1. sub_terrains 에 기하 정의 1개 추가 (proportion 은 0.0 으로)
#   2. 그 지형을 쓰고 싶은 프리셋에 가중치 1줄 추가 (필요하면 그룹 튜플에도)
# 프리셋에 안 적힌 지형은 apply_terrain_preset() 이 자동으로 0 으로 꺼 준다.
#
# 가중치는 ParkourTerrainGenerator 가 합으로 정규화하므로 절대값은 의미 없고
# 비율만 중요하다. 균등 분포 프리셋은 전부 1.0 으로 적는다.
# ---------------------------------------------------------------------------

# 원조 extreme-parkour 4종 (goal 추종 장애물 코스)
CORE_TERRAINS = ("parkour_gap", "parkour_hurdle", "parkour_step", "parkour")
# IsaacLab 지형을 parkour 규약으로 이식한 4종
EXTRA_TERRAINS = (
    "parkour_pyramid_stairs",
    "parkour_pyramid_stairs_up",
    "parkour_discrete_obstacles",
    "parkour_random_grid",
)
# 사다리꼴(오르막-평지-내리막) 2종
TRAPEZOID_TERRAINS = ("parkour_trapezoid_ramp", "parkour_trapezoid_stairs")

TERRAIN_PRESETS: dict[str, dict[str, float]] = {
    # 사다리꼴 2종 + 워밍업용 평지. teacher/student 학습이 현재 이걸 쓴다.
    # flat 은 초반 학습이 막히지 않도록 섞는 완충재다 (기존 student_train 에서
    # demo 0.15 + flat 0.05 가 하던 역할). 지형 종류 배정은 env 생성 시 한 번
    # 정해지고 학습 내내 바뀌지 않으므로(커리큘럼은 난이도 행만 움직인다),
    # 이 비율은 끝까지 그대로 유지된다.
    # 원래 분포로 되돌리려면 각 env cfg 의 apply_terrain_preset 인자를
    # "teacher_train" / "student_train" 으로 바꾸면 된다.
    "trapezoid_train": {**{k: 0.45 for k in TRAPEZOID_TERRAINS}, "parkour_flat": 0.1},
    # PLAY/EVAL 용: 사다리꼴 2종만 균등 (flat 은 학습 보조라 뺀다).
    "trapezoid_only": {k: 1.0 for k in TRAPEZOID_TERRAINS},
    # PLAY/EVAL 용: trapezoid_train 이 실제로 학습에 쓰는 3종을 '균등하게' 본다.
    # trapezoid_train 을 그대로 PLAY 에 쓰면 안 된다. 커리큘럼 모드의 컬럼→지형
    # 매핑은 cumsum(proportion) 을 num_cols 로 잘라 정하는데(_generate_curriculum_terrains),
    # 0.45/0.45/0.1 이면 flat 구간(0.9~1.0)에 걸리는 컬럼이 거의 안 나와
    # flat 이 화면에서 통째로 사라진다. 균등 + one_col_per_terrain=True 면
    # 컬럼 하나가 지형 하나에 1:1 로 떨어진다.
    "trapezoid_train_all": {k: 1.0 for k in (*TRAPEZOID_TERRAINS, "parkour_flat")},
    # teacher 학습: 원조 4종 + flat. (기존 체크포인트와 분포를 맞춰야 하므로
    # EXTRA_TERRAINS 는 뺀다. 포함해 재학습하려면 여기에 추가할 것.)
    "teacher_train": {**{k: 0.2 for k in CORE_TERRAINS}, "parkour_flat": 0.2},
    # student 학습: flat 비중을 줄이고 demo 를 섞는다.
    "student_train": {
        **{k: 0.2 for k in CORE_TERRAINS},
        "parkour_flat": 0.05,
        "parkour_demo": 0.15,
    },
    # teacher EVAL: 원조 4종 균등.
    "eval_core": {k: 1.0 for k in CORE_TERRAINS},
    # student EVAL/PLAY: 실전 장애물 8종 균등 (flat/demo 제외).
    "all_obstacles": {k: 1.0 for k in (*CORE_TERRAINS, *EXTRA_TERRAINS)},
    # teacher PLAY: 8종 + demo 균등.
    "all_with_demo": {k: 1.0 for k in (*CORE_TERRAINS, *EXTRA_TERRAINS, "parkour_demo")},
}


def apply_terrain_preset(
    generator_cfg: ParkourTerrainGeneratorCfg,
    preset: "str | dict[str, float]",
    *,
    one_col_per_terrain: bool = False,
    active_overrides: "dict | None" = None,
) -> None:
    """sub_terrains 의 proportion 을 프리셋으로 일괄 재설정한다.

    Args:
        generator_cfg: 대상 terrain generator cfg.
        preset: TERRAIN_PRESETS 의 키, 또는 {지형이름: 가중치} 딕셔너리.
            여기에 없는 지형은 proportion=0 으로 꺼진다.
        one_col_per_terrain: True 면 num_cols 를 활성 지형 수로 맞춘다.
            커리큘럼 모드의 컬럼→지형 매핑은 '활성 지형 수 == num_cols' 일 때만
            1:1 로 떨어지므로 (PLAY 처럼 로봇 1마리당 1종을 원할 때) 이걸 켠다.
        active_overrides: 활성(가중치>0) 지형에만 일괄 적용할 속성.
            예: {"noise_range": (0.02, 0.02)}
    """
    weights = TERRAIN_PRESETS[preset] if isinstance(preset, str) else preset
    unknown = set(weights) - set(generator_cfg.sub_terrains)
    if unknown:
        raise KeyError(
            f"sub_terrains 에 정의되지 않은 지형: {sorted(unknown)} (사용 가능: {sorted(generator_cfg.sub_terrains)})"
        )
    for key, sub_terrain in generator_cfg.sub_terrains.items():
        sub_terrain.proportion = weights.get(key, 0.0)
        if sub_terrain.proportion > 0 and active_overrides:
            for attr, value in active_overrides.items():
                setattr(sub_terrain, attr, value)
    if one_col_per_terrain:
        generator_cfg.num_cols = sum(w > 0 for w in weights.values())
