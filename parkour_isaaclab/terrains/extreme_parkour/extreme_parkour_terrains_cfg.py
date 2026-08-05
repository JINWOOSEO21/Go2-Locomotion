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
class ExtremeParkourInvertedPyramidStairsTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """IsaacLab ``HfInvertedPyramidStairsTerrainCfg`` 를 parkour 지형 규약에 맞춘 설정.

    step_height_range / step_width / inverted 는 원본과 같은 의미이고,
    나머지는 4m 폭 복도형 타일에 맞추려고 추가한 값이다.
    자세한 내용은 :func:`extreme_parkour_terrians.parkour_inverted_pyramid_stairs_terrain` 참고.
    """

    function = extreme_parkour_terrians.parkour_inverted_pyramid_stairs_terrain

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
    """구덩이 바닥(피라미드 꼭짓점) 정사각형의 한 변 (m). 원본의 ``platform_width``.

    시작 플랫폼 길이인 ``platform_len`` 과 헷갈리지 않도록 이름을 바꿨다.
    """

    inverted: bool = True
    """True 면 안쪽으로 파 내려가는 구덩이, False 면 위로 쌓아 올린 피라미드."""

    pyramid_len: float = 8.0
    """피라미드가 차지할 x 구간 길이 (m)."""

    run_up_len: float = 1.5
    """시작 플랫폼 끝 ~ 피라미드 시작까지의 평지 길이 (m). 마지막 goal 여유로도 쓰인다."""
