from isaaclab.utils import configclass
from ..parkour_terrain_generator_cfg import ParkourSubTerrainBaseCfg
from . import extreme_parkour_terrians

@configclass
class ExtremeParkourRoughTerrainCfg(ParkourSubTerrainBaseCfg):
    apply_roughness: bool = True 
    apply_flat: bool = False 
    downsampled_scale: float | None = 0.075
    noise_range: tuple[float,float] = (0.02, 0.06)
    noise_step: float = 0.005
    x_range: tuple[float, float] = (0.8, 1.5)
    y_range: tuple[float, float] = (-0.4, 0.4)
    half_valid_width: tuple[float, float] = (0.6, 1.2)
    pad_width: float = 0.1 
    pad_height: float = 0.0

@configclass
class ExtremeParkourGapTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_gap_terrain
    gap_size: str = '0.1 + 0.7*difficulty'
    gap_depth: tuple[float, float] = (0.2, 1) 

@configclass
class ExtremeParkourHurdleTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_hurdle_terrain
    stone_len: str = '0.1 + 0.3 * difficulty'
    hurdle_height_range: str = '0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty'

@configclass
class ExtremeParkourStepTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_step_terrain
    step_height: str = '0.1 + 0.35*difficulty'

@configclass
class ExtremeParkourTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_terrain
    pit_depth: tuple[float, float] = (0.2, 1)
    stone_width: float = 1.0
    last_stone_len: float =1.6
    x_range: str = '-0.1, 0.1+0.3*difficulty'
    y_range: str = '0.2, 0.3+0.1*difficulty'
    stone_len: str = '0.9 - 0.3*difficulty, 1 - 0.2*difficulty'
    incline_height: str = '0.25*difficulty'
    last_incline_height: str = 'incline_height + 0.1 - 0.1*difficulty'

@configclass
class ExtremeParkourDemoTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_demo_terrain

@configclass
class ExtremeParkourPyramidStairsTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """IsaacLab ``HfPyramidStairsTerrainCfg`` 를 parkour 지형 규약에 맞춘 설정.

    기본값은 원본과 같이 ``inverted=False`` 라, 직각 계단을 올라갔다가 내려오는
    위로 솟은 피라미드가 된다. 구덩이형은 아래
    :class:`ExtremeParkourInvertedPyramidStairsTerrainCfg` 를 쓴다.

    step_height_range / step_width / platform_width(-> apex_width) / inverted 는 원본과
    같은 의미이고, 나머지는 4m 폭 복도형 타일에 맞추려고 추가한 값이다.
    자세한 내용은 :func:`extreme_parkour_terrians.parkour_pyramid_stairs_terrain` 참고.
    """

    function = extreme_parkour_terrians.parkour_pyramid_stairs_terrain

    step_height_range: tuple[float, float] = (0.05, 0.20)
    """계단 한 칸 높이의 (최소, 최대). 실제 높이는 difficulty 로 선형 보간한다 (원본과 동일)."""

    step_width: float = 0.3
    """y 방향으로 한 칸씩 좁아지는 폭 (m). 원본의 ``step_width``."""

    step_depth: float | None = 0.8
    """x 방향으로 한 칸씩 좁아지는 폭 (m). None 이면 ``step_width`` 를 써서 원본과 같아진다.

    복도형 타일에서는 y 가 먼저 바닥나므로, x 를 더 크게 줄여야 x/y 가 같이 닫히면서
    로봇이 밟을 만한 깊이의 디딤판이 나온다.
    """

    apex_width: float = 1.6
    """피라미드 꼭짓점(정상 평지 / 구덩이 바닥) 정사각형의 한 변 (m). 원본의 ``platform_width``.

    시작 플랫폼 길이인 ``platform_len`` 과 헷갈리지 않도록 이름을 바꿨다.
    """

    inverted: bool = False
    """False 면 위로 쌓아 올린 피라미드(올라갔다 내려오기), True 면 파 내려간 구덩이."""

    pyramid_len: float = 8.0
    """피라미드가 차지할 x 구간 길이 (m)."""

    run_up_len: float = 1.5
    """시작 플랫폼 끝 ~ 피라미드 시작까지의 평지 길이 (m). 마지막 goal 여유로도 쓰인다."""

@configclass
class ExtremeParkourInvertedPyramidStairsTerrainCfg(ExtremeParkourPyramidStairsTerrainCfg):
    """IsaacLab ``HfInvertedPyramidStairsTerrainCfg`` 에 대응.

    원본과 마찬가지로 ``inverted=True`` 만 다른 서브클래스다.
    """

    inverted: bool = True

@configclass
class ExtremeParkourDiscreteObstaclesTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """IsaacLab ``HfDiscreteObstaclesTerrainCfg`` 를 parkour 지형 규약에 맞춘 설정.

    obstacle_* 와 num_obstacles 는 원본과 같은 의미다.
    원본의 ``platform_width`` (타일 한가운데를 비우는 정사각형) 는 parkour 에서는 쓸모가
    없어 빼고, 대신 앞쪽 ``platform_len`` 을 시작 플랫폼으로 비운다.
    자세한 내용은 :func:`extreme_parkour_terrians.parkour_discrete_obstacles_terrain` 참고.
    """

    function = extreme_parkour_terrians.parkour_discrete_obstacles_terrain

    obstacle_height_mode: str = "choice"
    """"choice" 면 [-h, -h/2, h/2, h] 에서 고르고 "fixed" 면 h 로 고정 (원본과 동일)."""

    obstacle_width_range: tuple[float, float] = (0.4, 1.2)
    """장애물 한 변의 (최소, 최대) (m). 원본과 동일하게 4픽셀 격자로 양자화된다."""

    obstacle_height_range: tuple[float, float] = (0.05, 0.20)
    """장애물 높이의 (최소, 최대) (m). difficulty 로 선형 보간한다 (원본과 동일)."""

    num_obstacles: int = 40
    """타일에 뿌릴 장애물 개수."""

    goal_clear_width: float = 0.6
    """각 goal 둘레를 평지로 미는 정사각형의 한 변 (m).

    원본에는 goal 이 없어 필요 없던 값이다. parkour 에서는 goal 이 기둥 꼭대기나 구덩이
    바닥에 찍히면 로봇이 도달할 수 없어 태스크가 깨지므로 디딜 자리를 확보한다.
    """

@configclass
class ExtremeParkourRandomGridTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """IsaacLab ``MeshRandomGridTerrainCfg`` 를 parkour 지형 규약에 맞춘 설정.

    grid_width / grid_height_range 는 원본과 같은 의미다. 원본은 trimesh 지형이지만
    여기서는 같은 '칸별 랜덤 높이' 를 높이맵으로 구현했다 (x_edge_mask 를 만들려면
    높이맵이 필요하다). 원본의 정사각 타일 제한도 자연히 사라진다.
    자세한 내용은 :func:`extreme_parkour_terrians.parkour_random_grid_terrain` 참고.
    """

    function = extreme_parkour_terrians.parkour_random_grid_terrain

    grid_width: float = 0.5
    """격자 칸 한 변의 길이 (m). 원본의 ``grid_width``."""

    grid_height_range: tuple[float, float] = (0.02, 0.10)
    """칸 높이 진폭의 (최소, 최대) (m). difficulty 로 보간한 값을 h 라 할 때
    각 칸의 윗면은 ``uniform(-h, +h)`` 다 (원본과 동일)."""

@configclass
class ExtremeParkourTrapezoidRampTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """옆에서 보면 사다리꼴인 경사로 지형: 오르막 램프 - 평지 - 내리막 램프.

    오르막/내리막 기울기는 같고 ``slope_angle`` 로 정한다. 평지 길이/높이는
    랜덤. 자세한 내용은 :func:`extreme_parkour_terrians.parkour_trapezoid_ramp_terrain` 참고.
    """

    function = extreme_parkour_terrians.parkour_trapezoid_ramp_terrain

    slope_angle: str = '10 + 27*difficulty'
    """경사 각도 (도). 10행 커리큘럼(difficulty = row/9)에서 10, 13, ..., 37도가 된다."""

    course_width: float = 5.0
    """구조물 폭 (m). 타일 폭(size[1])보다 크면 타일 전체 폭을 쓴다.
    현재 타일이 4m 폭이라 사실상 전폭이다. 5m 를 그대로 쓰려면 generator 의
    size 를 (x, 5.0) 이상으로 키워야 한다 (모든 지형에 영향)."""

    plateau_len_range: tuple[float, float] = (1.5, 3.0)
    """꼭대기 평지 길이의 랜덤 범위 (m)."""

    plateau_height_range: tuple[float, float] = (0.4, 0.8)
    """꼭대기 평지 높이의 랜덤 범위 (m). 완만한 각도에서 타일을 넘치면
    각도를 유지한 채 높이를 낮춰 맞춘다 (10도, 높이 0.8m 일 때 램프 한쪽이
    4.5m 라 16m 타일에 딱 들어가는 상한이다)."""

    end_margin: float = 1.5
    """내리막이 끝난 뒤 남겨둘 평지 길이 (m). 마지막 goal 이 이 안에 찍힌다."""

@configclass
class ExtremeParkourTrapezoidStairsTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """사다리꼴 계단 지형: 계단 오르막 - 평지 - 계단 내리막.

    단차는 ``step_height``, 단 수는 ``num_steps`` 로 고정, 디딤판 깊이는 step 지형처럼
    계단마다 ``x_range`` 에서 랜덤. 자세한 내용은
    :func:`extreme_parkour_terrians.parkour_trapezoid_stairs_terrain` 참고.
    """

    function = extreme_parkour_terrians.parkour_trapezoid_stairs_terrain

    step_height: str = '0.05 + 0.18*difficulty'
    """단차 높이 (m). 10행 커리큘럼(difficulty = row/9)에서 5, 7, ..., 23cm 가 된다."""

    num_steps_range: tuple[int, int] = (3, 7)
    """오르막 계단 단 수의 랜덤 범위 (양 끝 포함, 내리막도 같은 수).
    타일마다 하나를 뽑으며, 평지 높이 = 뽑힌 단 수 * step_height 가 된다.
    (최대 7단 x 23cm = 1.61m. 단 수가 많고 디딤판이 깊게 뽑혀 타일을 넘치면
    디딤판 깊이를 비율로 줄여 맞춘다.)"""

    x_range: tuple[float, float] = (0.3, 0.8)
    """디딤판(tread) 깊이의 랜덤 범위 (m). step 지형의 x_range(0.3, 1.5) 방식을
    따르되, 왕복 10칸이 16m 타일을 넘치지 않도록 상한을 0.8 로 줄였다."""

    course_width: float = 5.0
    """구조물 폭 (m). ramp 쪽과 같은 의미."""

    plateau_len_range: tuple[float, float] = (1.5, 3.0)
    """꼭대기 평지 길이의 랜덤 범위 (m)."""

    end_margin: float = 1.5
    """내리막이 끝난 뒤 남겨둘 평지 길이 (m)."""

    slope_threshold: float = 0.3
    """계단 면을 수직으로 세우는 기준. 0.3 이면 기준 단차가 0.03m 라 가장 낮은
    5cm 단도 직각이 된다. 기본값 1.5(기준 0.15m)를 쓰면 낮은 계단이 경사로로
    뭉개진다 (pyramid_stairs 와 같은 이유)."""
