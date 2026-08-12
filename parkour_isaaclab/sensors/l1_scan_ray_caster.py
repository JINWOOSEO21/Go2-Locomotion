"""실제 Unitree L1 스캔 운동학을 재현하는 MultiMeshRayCaster ([A] 고충실도 sim 전용).

레이 방향을 임의 지터가 아니라 **L1 의 실제 이중 모터 운동학**으로 생성한다.
근거: unilidar_sdk 의 MavLink 파싱 문서
(HowToParsePointCloudAndIMUDataFromMavLinkMessages.md) 가 공개한 포인트 기하 —

    A = (−cosθ·sinψ + sinθ·sinα·cosψ)·r + b_axis_dist
    B = cosα·cosψ·r
    x = cosβ·A − sinβ·B
    y = sinβ·A + cosβ·B
    z = (sinθ·sinψ + cosθ·sinα·cosψ)·r + z_bias

α = pitch(반사경, 고속 180 Hz), β = yaw(코어, 저속 11 Hz), θ/ψ = 고정 캘리브레이션
틸트(기기 내부 광학 상수 — 로봇 자세와 무관).

스캔 모델: 샘플 클럭 43,200 samples/s 에서 α(t) = ((360°·180·t + φ) mod 360) − 180
이 **α ∈ (−90°, 90°) 인 반주기(1/360 s)에 있을 때만 유효** — 나머지 반주기는 FOV
밖(하우징)이라 버려진다. 회전당 240 샘플 중 유효 120점(= SDK 패킷 크기), 유효율
21,600 pts/s(공식 스펙). θ=ψ=45° 기본값에서 유효 반주기는 elevation =
asin((1+sinα)/2) ∈ (0°, 90°) 반구로 단조 매핑된다. 방향 랜덤성 없음(결정론) —
env 별 초기 모터 위상만 랜덤이다.

주의: 모터 주기와 θ/ψ 의 **정밀 값은 기기 런타임 aux 패킷**
(sys/com_rotation_period, theta/ksi_angle)이 준다. 실기 확보 시 교체할 것 [실측 반영].

self-filter 프리미티브 테이블(GO2_SELF_FILTER_CAPSULES)도 여기 둔다 — 센서와 함께
"로봇 기하"라는 한 가지 지식을 이루기 때문이다. 소비자는 [A] 파이프라인(scripts/lidar_sim).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.sensors import MultiMeshRayCaster, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

# Go2 self-filter 프리미티브: (body_a, offset_a, body_b, offset_b, radius[m]).
# 두 endpoint 는 각 body 프레임의 offset 을 월드로 변환한 점이고, 그 선분 + radius 가
# 캡슐이다. body_a == body_b, offset 0 이면 구(sphere)가 된다.
# 반경은 collision 지오메트리 + 패딩 (스윙 다리 프레임 내 이동, extrinsic 오차 대비).
GO2_SELF_FILTER_CAPSULES: list[tuple[str, tuple[float, float, float], str, tuple[float, float, float], float]] = [
    # 몸통: base 프레임 x축 캡슐 (몸 폭 0.19, 높이 0.12 를 반경 0.11 로 덮는다)
    ("base", (0.22, 0.0, 0.02), "base", (-0.18, 0.0, 0.02), 0.11),
    # 머리: Head_upper - Head_lower 를 잇는 캡슐
    ("Head_upper", (0.0, 0.0, 0.0), "Head_lower", (0.0, 0.0, 0.0), 0.06),
    # 다리 (링크 원점이 관절 위치라 링크 쌍 = 뼈대 선분)
    *[(f"{leg}_hip", (0.0, 0.0, 0.0), f"{leg}_hip", (0.0, 0.0, 0.0), 0.07) for leg in ("FL", "FR", "RL", "RR")],
    *[(f"{leg}_thigh", (0.0, 0.0, 0.0), f"{leg}_calf", (0.0, 0.0, 0.0), 0.055) for leg in ("FL", "FR", "RL", "RR")],
    *[(f"{leg}_calf", (0.0, 0.0, 0.0), f"{leg}_foot", (0.0, 0.0, 0.0), 0.05) for leg in ("FL", "FR", "RL", "RR")],
]


def ray_capsule_penetrates(origins, dirs, t_hit, seg_a, seg_b, radius, t0=0.12):
    """레이(o + t d, t ∈ [t0, t_hit])가 캡슐(선분 seg_a-seg_b + radius)을 관통하는지.

    self-filter 판정은 hit 점-몸체 "거리"가 아니라 **레이 경로의 관통**으로 한다
    (설계서 §5.2 — shadow/mixed point 는 거리로 못 잡는다). 센서 마운트가 몸통
    캡슐 내부에 있으므로 t0(기본 0.12 m — 마운트 캡슐 최대 탈출 거리 0.09 m 초과,
    L1 최소 측정거리 0.05 m 초과) 이후 구간만 검사한다.

    origins/dirs: (N,R,3), t_hit: (N,R) (miss 는 0 으로 넣을 것),
    seg_a/seg_b: (N,3), radius: (N,1) 또는 스칼라. 반환 (N,R) bool.
    선분-선분 최근접 거리 (Ericson, Real-Time Collision Detection).
    """
    seg_len = (t_hit - t0).clamp_min(0.0)
    p1 = origins + dirs * t0
    d1 = dirs * seg_len.unsqueeze(-1)
    p2 = seg_a.unsqueeze(1)
    d2 = (seg_b - seg_a).unsqueeze(1)
    r = p1 - p2
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    denom = (a * e - b * b).clamp_min(1e-9)
    s = ((b * f - c * e) / denom).clamp(0.0, 1.0)
    t = ((b * s + f) / e.clamp_min(1e-9)).clamp(0.0, 1.0)
    s = ((b * t - c) / a.clamp_min(1e-9)).clamp(0.0, 1.0)
    closest1 = p1 + d1 * s.unsqueeze(-1)
    closest2 = p2 + d2 * t.unsqueeze(-1)
    return torch.norm(closest1 - closest2, dim=-1) < radius


class L1ScanRayCaster(MultiMeshRayCaster):
    """매 갱신마다 레이 방향을 L1 이중 모터 운동학으로 계산하는 MultiMeshRayCaster.

    스캔 모델 (사용자 검증 + SDK 패킷 구조와 정합):
      - 샘플 클럭 43,200 samples/s, 수직(반사경) 180 Hz, 수평(코어) 11 Hz
      - 수직각 α(t) = ((360°·180·t + φ) mod 360°) − 180°, **α ∈ (−90°, 90°) 인
        반주기(1/360 s)만 유효** — 나머지 반주기는 FOV 밖(하우징)이라 버린다
      - 따라서 회전당 240 샘플 중 120점이 유효 (= SDK 패킷 크기), 유효율 21,600 pts/s
      - 43,200 = 240 x 180 으로 샘플 클럭이 수직 모터와 정확히 동기이므로 유효
        샘플의 α 격자는 매 회전 동일: α_i = −90° + 1.5°·(i+0.5), i=0..119.
        무효 샘플은 시각이 폐형식으로 알려져 있어 **캐스트하지 않는다** (비용 0).
      - 프레임 = 마지막 완결된 18개 유효 창 (0.1 s) = 2,160 rays

    `pattern_cfg` 는 레이 개수(2,160) 할당용이다. env 별 초기 모터 위상만 랜덤
    (리셋 시 고정)이고 이후는 결정론적 — 방향 랜덤성은 없다.

    주의: 정확히 11/180 Hz 는 1 초마다 정수 회전이라 패턴이 1 s 주기로 반복된다.
    실기의 비반복성은 주기 미세 오차에서 오므로 aux 패킷 실측값으로 교체할 것.
    """

    cfg: L1ScanRayCasterCfg

    @staticmethod
    def directions_from_angles(
        alpha: torch.Tensor,
        beta: torch.Tensor,
        theta_rad: float,
        ksi_rad: float,
    ) -> torch.Tensor:
        """(α, β) [rad] → 센서 프레임 단위 방향 (..., 3).

        unilidar_sdk 문서의 xyz 식에서 r 의 계수만 취한 것 (원점 오프셋 b_axis_dist,
        z_bias 는 방향이 아니라 시작점 보정이라 제외 — 마운트 offset 에 흡수).
        순수 텐서 함수 — CPU 단위 테스트 대상.
        """
        sin_t, cos_t = math.sin(theta_rad), math.cos(theta_rad)
        sin_k, cos_k = math.sin(ksi_rad), math.cos(ksi_rad)
        sin_a, cos_a = torch.sin(alpha), torch.cos(alpha)
        sin_b, cos_b = torch.sin(beta), torch.cos(beta)
        A = -cos_t * sin_k + sin_t * sin_a * cos_k
        B = cos_a * cos_k
        x = cos_b * A - sin_b * B
        y = sin_b * A + cos_b * B
        z = sin_t * sin_k + cos_t * sin_a * cos_k
        dirs = torch.stack([x, y, z], dim=-1)
        return dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    @staticmethod
    def valid_frame_times(
        t_end: torch.Tensor,
        phase_pitch_deg: torch.Tensor,
        num_rays: int,
        pitch_hz: float,
        samples_per_second: float,
    ):
        """프레임(끝시각 t_end 이전의 마지막 완결 유효 창들)의 방출 시각과 α.

        t_end/phase_pitch_deg: (n_env, 1). 반환: t (n_env, R), alpha_rad (R,).
        유효 창: α 가 −90° 를 지나는 순간부터 1/360 s (= 회전당 120 샘플).
        순수 텐서 함수 — CPU 단위 테스트 대상.
        """
        samples_per_rev = samples_per_second / pitch_hz          # 240
        valid_per_rev = int(round(samples_per_rev / 2))          # 120
        n_revs = num_rays // valid_per_rev                       # 18
        rev_T = 1.0 / pitch_hz
        # α(t) = ((360·pitch_hz·t + φ) mod 360) − 180 이 −90 이 되는 최초 시각:
        # (360·pitch_hz·t + φ) mod 360 == 90
        t_cross0 = torch.remainder(90.0 - phase_pitch_deg, 360.0) / (360.0 * pitch_hz)  # (n,1)
        # t_end 이전에 창이 완결(시작+1/360s ≤ t_end)된 마지막 회전 번호
        n_last = torch.floor((t_end - t_cross0 - 0.5 * rev_T) * pitch_hz)
        revs = n_last - torch.arange(n_revs - 1, -1, -1, device=t_end.device).view(1, -1)  # (n, n_revs)
        i = torch.arange(valid_per_rev, device=t_end.device, dtype=t_end.dtype)            # (120,)
        t = (t_cross0.unsqueeze(-1) + revs.unsqueeze(-1) * rev_T
             + (i + 0.5).view(1, 1, -1) / samples_per_second)                              # (n, n_revs, 120)
        alpha_deg = -90.0 + (i + 0.5) * (360.0 / samples_per_rev)                          # (120,)
        alpha = torch.deg2rad(alpha_deg).repeat(n_revs)                                    # (R,)
        return t.reshape(t.shape[0], -1), alpha

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        samples_per_rev = self.cfg.samples_per_second * self.cfg.pitch_period_s
        if abs(samples_per_rev - round(samples_per_rev)) > 1e-6 or self.num_rays % int(round(samples_per_rev / 2)):
            raise ValueError(
                f"num_rays({self.num_rays}) 는 회전당 유효 샘플 수({samples_per_rev / 2:.1f})의 배수여야 한다.")
        n_env = self._view.count
        # env 별 초기 모터 위상 (고정 랜덤 — 이후 결정론적)
        self._l1_phase_pitch_deg = torch.rand(n_env, 1, device=self._device) * 360.0
        self._l1_phase_yaw_deg = torch.rand(n_env, 1, device=self._device) * 360.0
        self._l1_offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device, dtype=torch.float32)

    def _update_ray_infos(self, env_ids: Sequence[int]):
        n_env = len(env_ids)
        t_end = self._timestamp[env_ids].view(n_env, 1)
        t, alpha = self.valid_frame_times(
            t_end, self._l1_phase_pitch_deg[env_ids], self.num_rays,
            1.0 / self.cfg.pitch_period_s, self.cfg.samples_per_second)
        beta = torch.deg2rad(360.0 * t / self.cfg.yaw_period_s + self._l1_phase_yaw_deg[env_ids])
        dirs = self.directions_from_angles(
            alpha.unsqueeze(0).expand_as(t), beta,
            math.radians(self.cfg.theta_angle_deg), math.radians(self.cfg.ksi_angle_deg))
        # 마운트 회전 적용 (base 클래스 _initialize_rays_impl 이 초기 패턴에 하던 것과 동일)
        dirs = quat_apply(self._l1_offset_quat.expand(n_env, self.num_rays, 4), dirs)
        self.ray_directions[env_ids] = dirs
        super()._update_ray_infos(env_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        if hasattr(self, "_l1_phase_pitch_deg"):
            n = self._l1_phase_pitch_deg[env_ids].shape[0]
            self._l1_phase_pitch_deg[env_ids] = torch.rand(n, 1, device=self._device) * 360.0
            self._l1_phase_yaw_deg[env_ids] = torch.rand(n, 1, device=self._device) * 360.0


@configclass
class L1ScanRayCasterCfg(MultiMeshRayCasterCfg):
    """Configuration for the L1 scan-kinematics ray-cast sensor."""

    class_type: type = L1ScanRayCaster

    rays_per_frame: int = 2160
    """한 프레임(0.1 s)의 유효 포인트 수 = rays_per_frame / (samples_per_second/2) 초
    분량. 기본 2,160 = 21,600 pts/s x 0.1 s. 회전당 유효 샘플 수(120)의 배수여야 한다."""

    samples_per_second: float = 43200.0
    """원시 샘플 클럭. 유효 반주기만 취하면 21,600 pts/s (공식 스펙)가 된다."""

    theta_angle_deg: float = 45.0
    ksi_angle_deg: float = 45.0
    """고정 캘리브레이션 틸트(기기 내부 광학 상수 — 로봇 자세와 무관). 45/45 는
    유효 반주기 α∈(−90°,90°)가 elevation ∈ (0°, 90°) 반구로 단조 매핑되는 유도값.
    실제 값은 기기 aux 패킷(theta/ksi_angle)로 교체 [실측 반영]."""

    pitch_period_s: float = 1.0 / 180.0
    """수직(반사경, 고속) 모터 주기 = 180 Hz. 회전당 240 샘플, 유효 120점
    (= SDK 패킷 크기). 실제 값은 aux 패킷 rotation_period 로 교체."""

    yaw_period_s: float = 1.0 / 11.0
    """수평(코어 360°, 저속) 모터 주기 = 11 Hz. 정확히 11/180 Hz 면 패턴이 1 s
    주기로 반복되므로, 실기 비반복성은 aux 패킷 실측 주기로 교체 시 반영된다."""

    def __post_init__(self):
        # pattern_cfg 는 base 클래스(RayCaster)의 레이 버퍼 "개수" 할당에만 쓰이고
        # 방향은 매 갱신 L1 운동학이 덮어쓴다. 사용자가 만질 필요가 없도록
        # rays_per_frame 로부터 자동 생성한다 (구 channels=12 류 파라미터 노출 제거).
        self.pattern_cfg = LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0.0, float(self.rays_per_frame)),
            horizontal_res=1.0,
        )
