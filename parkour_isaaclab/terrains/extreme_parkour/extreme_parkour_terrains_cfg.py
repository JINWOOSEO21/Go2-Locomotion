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
