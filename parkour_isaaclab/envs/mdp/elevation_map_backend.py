"""elevation_mapping_cupy 래퍼 — per-env 인스턴스, ROS 없이 코어만 사용.

계획서 docs/emcupy_student_plan.md §2.4. 설계 요점:
- 코어 라이브러리는 순수 python+cupy 라 그대로 쓰되, 확인된 함정들을 여기서 흡수한다:
  1) import 부작용으로 걸리는 프로세스 전역 managed-memory allocator 를 일반
     device pool 로 되돌린다 (IsaacLab/torch 와 VRAM 경합 방지).
  2) input_pointcloud 가 t 를 in-place 로 바꾸므로 복사본을 넘긴다.
  3) move()/move_to() 의 grid shift 부호가 상반 — move_to 만 쓴다.
  4) traversability CNN 은 우리 파이프라인에 불필요 — zero-stub 으로 교체한다.
     (normal map 은 visibility cleanup 의 cos 문턱이 쓰므로 살려 둔다.)
- 이 모듈은 isaaclab 을 import 하지 않는다. cupy/em_cupy 는 지연 import 라
  샘플링 순수 함수(sample_scan_heights)는 CPU 단위 테스트에서 그대로 쓸 수 있다.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# (cp, ElevationMap, Parameter, repo_root) — _import_emcupy() 가 채운다.
_EM = None


def _find_emcupy_root() -> Path:
    """elevation_mapping_cupy 클론 위치를 찾는다.

    main checkout 의 repo root 에 클론되어 있고(untracked), 워크트리에서 돌 때는
    거기 없으므로 상위(.claude/worktrees/<name> -> repo root)도 훑는다.
    EMCUPY_ROOT 환경변수가 있으면 그것이 이긴다.
    """
    candidates = []
    if os.environ.get("EMCUPY_ROOT"):
        candidates.append(Path(os.environ["EMCUPY_ROOT"]))
    here = Path(__file__).resolve()
    # parents[3] == <repo root>/parkour_isaaclab/envs/mdp 기준 repo root.
    for up in (3, 4, 5, 6):
        if up < len(here.parents):
            candidates.append(here.parents[up] / "elevation_mapping_cupy")
    for root in candidates:
        if (root / "elevation_mapping_cupy" / "script" / "elevation_mapping_cupy" / "elevation_mapping.py").exists():
            return root
    raise ImportError(
        "elevation_mapping_cupy 클론을 찾지 못했다. repo root 에 클론하거나 "
        "EMCUPY_ROOT 환경변수로 위치를 지정할 것. 찾아본 곳: "
        + ", ".join(str(c) for c in candidates)
    )


def _import_emcupy():
    global _EM
    if _EM is not None:
        return _EM
    import sys

    root = _find_emcupy_root()
    script_dir = str(root / "elevation_mapping_cupy" / "script")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import cupy as cp
    from elevation_mapping_cupy import ElevationMap, Parameter  # noqa: E402

    # 함정 1: elevation_mapping.py 가 import 시점에 cupy 전역 allocator 를
    # managed(unified) memory pool 로 바꿔 버린다. torch 와 VRAM 을 두 풀이
    # 나눠 잡는 것은 어쩔 수 없지만, managed pool 은 페이지 폴트 비용까지
    # 얹으므로 일반 device pool 로 되돌린다.
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    _EM = (cp, ElevationMap, Parameter, root)
    return _EM


def sample_scan_heights(
    maps: torch.Tensor,
    centers: torch.Tensor,
    points_xy: torch.Tensor,
    base_z: torch.Tensor,
    resolution: float,
    cell_n: int,
    height_offset: float = 0.3,
    clip_range: float = 1.0,
):
    """elevation map 을 scandot 위치에서 샘플해 teacher 와 동일한 높이 obs 를 만든다.

    순수 torch 함수 (cupy/isaaclab 무관 — CPU 단위 테스트 대상).

    Args:
        maps: (N, 7, H, W) — em_cupy elevation_map 레이어 스택.
              [0]=elevation(맵 center z 상대값), [2]=is_valid, [5]=upper_bound, [6]=is_upper_bound
        centers: (N, 3) — 각 맵의 center (odom frame).
        points_xy: (N, P, 2) — 샘플 위치 (odom frame).
        base_z: (N,) — 로봇 base 높이 (odom frame).
        resolution, cell_n: 맵 그리드 사양.

    Returns:
        h_obs: (N, P) — clip(base_z − h_abs − height_offset, ±clip_range).
               invalid 셀 cascade: valid → h, 아니면 upper_bound, 아니면 0.
        valid_frac: (N, P) — 셀 4이웃의 유효 가중치 합 (진단용, 0이면 채움값 사용).
    """
    n, p = points_xy.shape[0], points_xy.shape[1]
    # 커널 규약: i = floor((x−c)/res + 0.5·cell_n), 셀 i 의 중심은 연속좌표 i+0.5.
    # bilinear 는 셀 중심 격자 위에서 한다: u 정수값 == 셀 중심.
    u = (points_xy[..., 0] - centers[:, None, 0]) / resolution + 0.5 * cell_n - 0.5
    v = (points_xy[..., 1] - centers[:, None, 1]) / resolution + 0.5 * cell_n - 0.5
    u0 = torch.floor(u).long()
    v0 = torch.floor(v).long()
    fu = (u - u0.to(u.dtype)).unsqueeze(-1)  # (N,P,1)
    fv = (v - v0.to(v.dtype)).unsqueeze(-1)

    # 4 이웃 (dx, dy) 와 bilinear 가중치
    w = torch.cat(
        [(1 - fu) * (1 - fv), (1 - fu) * fv, fu * (1 - fv), fu * fv], dim=-1
    )  # (N,P,4)
    ux = torch.stack([u0, u0, u0 + 1, u0 + 1], dim=-1)  # (N,P,4)
    vy = torch.stack([v0, v0 + 1, v0, v0 + 1], dim=-1)

    # 1셀 보더 링(커널의 is_inside 규약)과 맵 밖은 invalid 취급
    inside = (ux >= 1) & (ux <= cell_n - 2) & (vy >= 1) & (vy <= cell_n - 2)
    ux = ux.clamp(0, cell_n - 1)
    vy = vy.clamp(0, cell_n - 1)

    flat = (ux * cell_n + vy).view(n, -1)  # (N, P*4)

    def gather(layer: int) -> torch.Tensor:
        return maps[:, layer].reshape(n, -1).gather(1, flat).view(n, p, 4)

    h = gather(0)
    valid = (gather(2) > 0.5) & inside
    ub = gather(5)
    is_ub = (gather(6) > 0.5) & inside

    w_valid = w * valid.to(w.dtype)
    w_ub = w * is_ub.to(w.dtype)
    sum_valid = w_valid.sum(-1)
    sum_ub = w_ub.sum(-1)

    eps = 1e-9
    h_valid = (w_valid * h).sum(-1) / sum_valid.clamp_min(eps)
    h_ubval = (w_ub * ub).sum(-1) / sum_ub.clamp_min(eps)

    h_abs_rel = torch.where(sum_valid > eps, h_valid, h_ubval)  # 맵 center z 상대값
    h_abs = h_abs_rel + centers[:, None, 2]
    h_obs = torch.clip(base_z[:, None] - h_abs - height_offset, -clip_range, clip_range)
    # 관측도 상한도 없는 셀: teacher 식 기준 0 (= base 기준 평지 가정)
    h_obs = torch.where((sum_valid > eps) | (sum_ub > eps), h_obs, torch.zeros_like(h_obs))
    return h_obs, sum_valid


class ElevationMapBackend:
    """per-env ElevationMap 인스턴스 묶음.

    성능 노트 (계획서 §2.4/리스크 1): stock 커널은 단일 맵 설계라 여기서는
    python 루프로 env 별 인스턴스를 직렬 갱신한다. Phase 3 벤치마크에서
    이 루프가 병목으로 판정되면 커널 3개(add_points/error_counting/average_map)에
    env 배치 차원을 더한 batched 포트로 교체한다 (인터페이스는 이 클래스 유지).
    """

    # elevation_map 레이어 인덱스 (em_cupy 규약)
    L_ELEV, L_VAR, L_VALID, L_UB, L_IS_UB = 0, 1, 2, 5, 6

    def __init__(
        self,
        num_envs: int,
        device: str,
        resolution: float = 0.1,
        map_length: float = 3.2,
        sensor_noise_factor: float = 0.05,
        min_valid_distance: float = 0.10,
        max_ray_length: float = 3.0,
        enable_visibility_cleanup: bool = True,
    ):
        cp, ElevationMap, Parameter, root = _import_emcupy()
        self.cp = cp
        self.device = device
        self.num_envs = num_envs
        self.resolution = float(resolution)

        cfg_dir = root / "elevation_mapping_cupy" / "config" / "core"
        param = Parameter(
            use_chainer=False,
            weight_file=str(cfg_dir / "weights.dat"),
            plugin_config_file=str(cfg_dir / "plugin_config.yaml"),
        )
        param.resolution = float(resolution)
        param.map_length = float(map_length)
        param.sensor_noise_factor = float(sensor_noise_factor)
        # 캡슐 self-filter(t0=0.12m)가 근거리를 이미 걸렀다 — 내장 필터는 그보다 안쪽만.
        param.min_valid_distance = float(min_valid_distance)
        # GT(+백색잡음) odometry 라 드리프트 보정은 끈다. error_counting 커널은
        # 그래도 돌지만(update_map_with_kernel 고정 순서) 보정 분기는 스킵된다.
        param.enable_drift_compensation = False
        # upper_bound(=Q3 cascade 2단계)를 만들려면 visibility cleanup 이 필요하다.
        # 비용은 점당 max_ray_length/resolution 회 반복 — 기본 yaml(10m/0.04m) 금지.
        param.enable_visibility_cleanup = bool(enable_visibility_cleanup)
        param.max_ray_length = float(max_ray_length)
        # 맵(3.2m)이 clear range(4m 기본)보다 작아 사실상 전맵 소거가 된다 — 끈다.
        param.enable_overlap_clearance = False
        # semantic/이미지 경로 전부 봉인
        param.subscriber_cfg = {}
        param.additional_layers = []
        param.fusion_algorithms = []
        param.update()  # 필수: cell_n 등 파생값 계산
        self.param = param
        self.cell_n = int(param.cell_n)

        self.maps = [ElevationMap(param) for _ in range(num_envs)]
        # traversability CNN 우회: update_map_with_kernel 이 매 갱신 호출하지만
        # 우리는 그 레이어를 안 쓴다. dilation/normal 은 visibility cleanup 이
        # 쓰므로 그대로 두고 CNN forward 만 zero-stub 으로 바꾼다.
        self._trav_zero = cp.zeros((1, 1, self.cell_n - 6, self.cell_n - 6), dtype=cp.float32)
        for m in self.maps:
            m.traversability_filter = self._trav_stub

    def _trav_stub(self, _x):
        return self._trav_zero

    def clear(self, env_ids) -> None:
        """env reset(teleport) 시 해당 맵 초기화. 다음 move_to 가 다시 중심을 맞춘다."""
        for i in env_ids:
            self.maps[int(i)].clear()

    @torch.no_grad()
    def update(
        self,
        points_sensor: list[torch.Tensor],
        R_sensor: torch.Tensor,
        t_sensor: torch.Tensor,
        base_pos: torch.Tensor,
        R_base: torch.Tensor,
    ) -> None:
        """한 tick 의 관측을 전 env 맵에 반영한다.

        Args:
            points_sensor: env 별 (Mi, 3) 센서 프레임 점군 (self-hit 제거 후).
            R_sensor/t_sensor: (N,3,3)/(N,3) — odom frame 기준 센서 pose (노이즈 포함).
            base_pos/R_base: (N,3)/(N,3,3) — odom frame 기준 base pose (노이즈 포함).
        """
        cp = self.cp
        for i, m in enumerate(self.maps):
            # move_to 를 input 보다 먼저: 커널이 "현재 center" 기준으로 셀을 계산한다.
            m.move_to(cp.asarray(base_pos[i]), cp.asarray(R_base[i]))
            pts = points_sensor[i]
            if pts.shape[0] > 0:
                m.input_pointcloud(
                    cp.asarray(pts.contiguous()),
                    ["x", "y", "z"],
                    cp.asarray(R_sensor[i].contiguous()),
                    # 함정 2: t 는 커널 진입 전에 in-place 로 center 를 빼므로 복사본.
                    cp.asarray(t_sensor[i].contiguous()).copy(),
                    0.0,
                    0.0,
                )
            m.update_variance()
            m.update_time()

    def layers(self) -> torch.Tensor:
        """(N, 7, H, W) torch 뷰 스택. as_tensor 는 zero-copy, stack 이 1회 복사."""
        return torch.stack([torch.as_tensor(m.elevation_map, device=self.device) for m in self.maps])

    def centers(self) -> torch.Tensor:
        return torch.stack([torch.as_tensor(m.center, device=self.device) for m in self.maps])

    def sample(self, points_xy: torch.Tensor, base_z: torch.Tensor):
        """scandot 위치 샘플 → (h_obs, valid_frac). 규약은 sample_scan_heights 참조."""
        return sample_scan_heights(
            self.layers(), self.centers(), points_xy, base_z, self.resolution, self.cell_n
        )
