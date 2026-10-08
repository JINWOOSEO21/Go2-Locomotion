# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_apply, wrap_to_pi

from locomotion_isaaclab.envs.mdp.goals import GoalEvent

if TYPE_CHECKING:
    from isaaclab.managers import ObservationTermCfg

    from locomotion_isaaclab.envs import LocomotionManagerBasedRLEnv


class LocomotionObservations(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: LocomotionManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors["contact_forces"]
        self.ray_sensor: RayCaster = env.scene.sensors["height_scanner"]
        self.goal_event: GoalEvent = env.goal_manager.get_term(cfg.params["goal_name"])
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.history_length = cfg.params["history_length"]
        self._obs_history_buffer = torch.zeros(
            self.num_envs, self.history_length, 3 + 2 + 3 + 4 + 36 + 5, device=self.device
        )
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device)
        self.measured_heights = torch.zeros(self.num_envs, 132, device=self.device)
        self.env = env
        self.body_id = self.asset.find_bodies("base")[0]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._obs_history_buffer[env_ids, :, :] = 0.0

    def __call__(
        self,
        env: LocomotionManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        goal_name: str,
        history_length: int,
    ) -> torch.Tensor:

        # 지형 타입(flat/non-flat) 플래그를 상수로 고정: 항상 non-flat=1, flat=0.
        # 원본 프로젝트는 이 플래그로 flat 고속 모드 전환을 학습시켰지만,
        # 이 프로젝트의 목표는 험지 robust 보행이고 실기에는 "지금 flat 이다"를
        # 알려줄 오라클이 없다 — flat 감지 후 모드 전환을 실세계에서 재현할 수
        # 없으므로 정책이 이 신호에 의존하지 않게 한다. 지형 종류 정보는 보상
        # 셰이핑(rewards.py 의 flat 분기)에만 남는다 — 그쪽은 sim 전용이다.
        env_idx_tensor = torch.ones(self.num_envs, 1, device=self.device)
        invert_env_idx_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        roll, pitch, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)
        imu_obs = torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=1).to(self.device)
        if env.common_step_counter % 5 == 0:
            self.delta_yaw = self.goal_event.target_yaw - wrap_to_pi(yaw)
            self.delta_next_yaw = self.goal_event.next_target_yaw - wrap_to_pi(yaw)
            self.measured_heights = self._get_heights()
        commands = env.command_manager.get_command("base_velocity")
        obs_buf = torch.cat(
            (
                self.asset.data.root_ang_vel_b * 0.25,  # [1,3] 0~2
                imu_obs,  # [1,2] 3~4
                0 * self.delta_yaw[:, None],  # [1,1] 5
                1.5 * self.delta_yaw[:, None],  # [1,1] 6: scaled heading error (radians)
                1.5 * self.delta_next_yaw[:, None],  # [1,1] 7: scaled next-heading error
                0 * commands[:, 0:2],  # [1,2] 8
                commands[:, 0:1],  # [1,1] 9
                env_idx_tensor,
                invert_env_idx_tensor,
                self.asset.data.joint_pos - self.asset.data.default_joint_pos,
                self.asset.data.joint_vel * 0.05,
                env.action_manager.get_term("joint_pos").action_history_buf[:, -1],
                self._get_contact_fill(),
            ),
            dim=-1,
        )
        priv_explicit = self._get_priv_explicit()
        priv_latent = self._get_priv_latent()
        observations = torch.cat(
            [
                obs_buf,  # 53
                self.measured_heights,  # 132
                priv_explicit,  # 9
                priv_latent,  # 29
                self._obs_history_buffer.view(self.num_envs, -1),
            ],
            dim=-1,
        )
        obs_buf[:, 6:8] = 0
        self._obs_history_buffer = torch.where(
            (env.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.history_length, dim=1),
            torch.cat([self._obs_history_buffer[:, 1:], obs_buf.unsqueeze(1)], dim=1),
        )
        return observations

    def _get_contact_fill(
        self,
    ):
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids]  # (N, 4, 3)
        contact = torch.norm(contact_forces, dim=-1) > 2.0
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[
            :, -1, self.sensor_cfg.body_ids
        ]  # N, 4, 3
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.0
        contact_filt = torch.logical_or(contact, last_contacts)
        return (contact_filt.float() - 0.5).to(self.device)

    def _get_priv_explicit(
        self,
    ):
        base_lin_vel = self.asset.data.root_lin_vel_b
        return torch.cat((base_lin_vel * 2.0, 0 * base_lin_vel, 0 * base_lin_vel), dim=-1).to(self.device)

    def _get_priv_latent(
        self,
    ):
        body_mass = self.asset.root_physx_view.get_masses()[:, self.body_id].to(self.device)
        body_com = self.asset.data.com_pos_b[:, self.body_id, :].to(self.device).squeeze(1)
        mass_params_tensor = torch.cat([body_mass, body_com], dim=-1).to(self.device)
        friction_coeffs_tensor = self.asset.root_physx_view.get_material_properties()[:, 0, 0]
        joint_stiffness = self.asset.data.joint_stiffness.to(self.device)
        default_joint_stiffness = self.asset.data.default_joint_stiffness.to(self.device)
        joint_damping = self.asset.data.joint_damping.to(self.device)
        default_joint_damping = self.asset.data.default_joint_damping.to(self.device)
        return torch.cat(
            (
                mass_params_tensor,
                friction_coeffs_tensor.unsqueeze(1).to(self.device),
                (joint_stiffness / default_joint_stiffness) - 1,
                (joint_damping / default_joint_damping) - 1,
            ),
            dim=-1,
        ).to(self.device)

    def _get_heights(self):
        return torch.clip(
            self.ray_sensor.data.pos_w[:, 2].unsqueeze(1) - self.ray_sensor.data.ray_hits_w[..., 2] - 0.3, -1, 1
        ).to(self.device)


class elevation_map_scan(ManagerTermBase):
    """L1 LiDAR → self-hit 필터 → elevation_mapping_cupy → scandots 격자 샘플.

    LiDAR 정책용 관측으로, GT scandots(obs[53:185])와 같은 위치·같은
    정규화(clip(base_z − h − 0.3, ±1))의 132-vector 를 낸다. EM tick(= 센서 자연
    프레임 0.1s, 10Hz)에서만 갱신되고 사이 step 은 최신 값을 그대로 돌려준다.
    """

    def __init__(self, cfg: ObservationTermCfg, env: LocomotionManagerBasedRLEnv):
        super().__init__(cfg, env)
        from locomotion_isaaclab.envs.mdp.elevation_map_backend import (
            BatchedElevationMapBackend,
            ElevationMapBackend,
        )
        from locomotion_isaaclab.sensors import GO2_SELF_FILTER_CAPSULES

        self.env = env
        self.lidar = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.update_interval = int(cfg.params.get("update_interval", 5))
        # 노이즈 — 전부 EM 입력에만 걸리고 policy obs(GT)는 무관.
        # L1 측정: 거리(range) 백색잡음 + 빔 지향(az/el) 백색잡음.
        self.range_std = float(cfg.params.get("range_std", 0.02))
        self.ray_dir_std_rad = float(np.deg2rad(cfg.params.get("ray_dir_std_deg", 0.2)))
        # odometry 위치(xyz): 변화량 기반 drift 모델. tick 마다 (body frame 축별로)
        #   Δ_meas = Δ_true·(1+b+bias) + n,  b~N(0, odom_scale_var), n~N(0, walk_std²)
        # b 는 tick 마다 재샘플되는 백색 계수(√거리 random walk), bias 는 episode 당
        # 1회 uniform[-max, max] 로 뽑혀 유지되는 계통적 scale 오차(캘리브레이션류,
        # 이동거리에 선형으로 누적) — yaw 의 gyro bias 와 대칭 구조다.
        # b·bias 는 body frame 에서 걸린다(보폭/슬립 오차는 로봇 기준). 등방성 n 은
        # 회전 불변이라 frame 구분이 없다.
        # odom_scale_var 는 분산이다. curriculum scale s 는 표준편차에 선형으로
        # 적용되므로 실제 분산은 odom_scale_var * s² 가 된다.
        self.odom_scale_std = float(cfg.params.get("odom_scale_var", 0.02)) ** 0.5
        self.odom_pos_walk_std = float(cfg.params.get("odom_pos_walk_std", 0.005))
        self.odom_scale_bias_max = float(cfg.params.get("odom_scale_bias_max", 0.03))
        # yaw: gyro bias drift 모델 — Δ_meas = Δ_true + bias·dt + n.
        # bias 는 reset 마다 [범위] deg/s 크기·랜덤 부호로 재샘플되는 상수,
        # n~N(0, walk_std²). 오차가 시간에 선형(bias) + √t(walk) 로 커진다.
        self.odom_yaw_bias_range_dps = tuple(cfg.params.get("odom_yaw_bias_range_dps", (0.01, 0.05)))
        self.odom_yaw_walk_std_rad = float(np.deg2rad(cfg.params.get("odom_yaw_walk_std_deg", 0.003)))
        # roll/pitch 는 중력(IMU) 관측으로 드리프트하지 않으므로 백색잡음만.
        self.odom_rp_std_rad = float(np.deg2rad(cfg.params.get("odom_rp_std_deg", 0.5)))
        # EM tick 주기 [s] — yaw bias(deg/s)를 tick 당 증분으로 바꿀 때 쓴다.
        self._tick_dt = float(env.step_dt) * self.update_interval

        # scandot 격자는 teacher 의 height_scanner 에서 그대로 가져온다 —
        # ray_starts 는 (base frame) 패턴점 + cfg offset(+0.375m) 이라 순서/위치가
        # obs 의 scan 구간과 1:1 로 일치한다.
        scanner: RayCaster = env.scene.sensors["height_scanner"]
        self._scan_offsets_xy = scanner.ray_starts[0][:, :2].clone().to(self.device)
        self.num_points = self._scan_offsets_xy.shape[0]

        # "batched"(기본): 커널 배치판 — 192 env 기준 tick 당 수 ms.
        # "loop": em_cupy 인스턴스 직렬 루프 — 회귀 비교/디버깅용 (250ms/tick).
        backend_cls = (
            BatchedElevationMapBackend if cfg.params.get("em_backend", "batched") == "batched" else ElevationMapBackend
        )
        self.backend = backend_cls(
            num_envs=self.num_envs,
            device=self.device,
            resolution=float(cfg.params.get("em_resolution", 0.1)),
            map_length=float(cfg.params.get("em_map_length", 3.2)),
        )

        # self-filter 캡슐: body 이름 → 인덱스 해석 (센서와 함께 사는 로봇 기하 테이블)
        self._capsules = []
        for name_a, off_a, name_b, off_b, radius in GO2_SELF_FILTER_CAPSULES:
            ia = self.asset.find_bodies(name_a)[0][0]
            ib = self.asset.find_bodies(name_b)[0][0]
            self._capsules.append(
                (
                    ia,
                    torch.tensor(off_a, device=self.device),
                    ib,
                    torch.tensor(off_b, device=self.device),
                    float(radius),
                )
            )
        # 마운트 회전 (GO2_LIDAR_CFG.offset.rot) — base→센서 프레임 회전
        offset_quat = torch.tensor(list(self.lidar.cfg.offset.rot), device=self.device)
        self._R_offset = matrix_from_quat(offset_quat.unsqueeze(0)).squeeze(0)

        self.h_obs = torch.zeros(self.num_envs, self.num_points, device=self.device)
        self.valid_frac = torch.zeros(self.num_envs, self.num_points, device=self.device)
        self.ub_frac = torch.zeros(self.num_envs, self.num_points, device=self.device)
        # odometry drift 상태 — EM tick 간 유지, env reset 시 0 에서 다시 시작
        # (실기에서 odometry 는 에피소드 시작점 기준으로 초기화된다).
        self._odom_pos_err = torch.zeros(self.num_envs, 3, device=self.device)
        self._odom_yaw_err = torch.zeros(self.num_envs, device=self.device)
        self._odom_yaw_bias = torch.zeros(self.num_envs, device=self.device)
        self._odom_pos_scale_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._odom_needs_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._resample_yaw_bias(torch.arange(self.num_envs, device=self.device))
        self._resample_pos_scale_bias(torch.arange(self.num_envs, device=self.device))
        # 외부에서 valid_frac 등 elevation-map 상태에 접근할 수 있게 노출
        env.em_scan_term = self

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        # teleport 후 옛 지형이 남으면 안 된다. 관측도 "아직 아무것도 못 봄"(=0)으로.
        self.backend.clear([int(i) for i in env_ids])
        self.h_obs[env_ids] = 0.0
        self.valid_frac[env_ids] = 0.0
        self.ub_frac[env_ids] = 0.0
        self._odom_pos_err[env_ids] = 0.0
        self._odom_yaw_err[env_ids] = 0.0
        self._resample_yaw_bias(env_ids)
        self._resample_pos_scale_bias(env_ids)
        # teleport 직후 pose 로 prev 를 다시 잡아야 하므로 다음 tick 에서 재초기화
        self._odom_needs_init[env_ids] = True

    def _resample_yaw_bias(self, env_ids) -> None:
        # gyro bias: 크기 uniform[lo, hi] deg/s, 부호 랜덤 — tick 당 증분(rad)으로 저장
        lo = float(np.deg2rad(self.odom_yaw_bias_range_dps[0])) * self._tick_dt
        hi = float(np.deg2rad(self.odom_yaw_bias_range_dps[1])) * self._tick_dt
        n = len(env_ids)
        mag = torch.empty(n, device=self.device).uniform_(lo, hi)
        sign = torch.where(
            torch.rand(n, device=self.device) < 0.5,
            torch.tensor(-1.0, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        self._odom_yaw_bias[env_ids] = mag * sign

    def _resample_pos_scale_bias(self, env_ids) -> None:
        # 위치 scale 계통 오차: episode 당 1회, 축별 독립 uniform[-max, max].
        # yaw bias 와 달리 0 근처도 뽑힌다 — "거의 정확한 odometry" 도 DR 분포에 포함.
        n = len(env_ids)
        self._odom_pos_scale_bias[env_ids] = (
            torch.rand(n, 3, device=self.device) * 2.0 - 1.0
        ) * self.odom_scale_bias_max

    def __call__(
        self,
        env: LocomotionManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        # 아래 파라미터들은 __init__ 에서 cfg.params 로 소비된다. ObservationManager 의
        # 시그니처 검사를 통과하기 위해 여기 명시한다 (**kwargs 는 'kwargs' 라는
        # 필수 파라미터로 오해석된다).
        em_resolution: float = 0.1,
        em_map_length: float = 3.2,
        em_backend: str = "batched",
        update_interval: int = 5,
        range_std: float = 0.02,
        ray_dir_std_deg: float = 0.2,
        odom_scale_var: float = 0.02,
        odom_pos_walk_std: float = 0.005,
        odom_scale_bias_max: float = 0.03,
        odom_yaw_bias_range_dps: tuple = (0.01, 0.05),
        odom_yaw_walk_std_deg: float = 0.003,
        odom_rp_std_deg: float = 0.5,
    ) -> torch.Tensor:
        # 센서 프레임(0.1s)과 같은 위상: 5 control step마다 갱신.
        if env.common_step_counter % self.update_interval == 0:
            self._update()
        return self.h_obs

    @torch.no_grad()
    def _update(self):
        # Runner 가 iteration 마다 env.lidar_noise_scale 을 갱신한다. 직접 값이 아직
        # 없으면 env cfg 의 초기값을 쓰며, 둘 다 없으면 이전 동작과 같은 최대(1.0).
        noise_scale = float(
            getattr(self.env, "lidar_noise_scale", getattr(self.env.cfg, "lidar_noise_scale", 1.0))
        )
        lidar = self.lidar
        # .data 접근이 센서 lazy 갱신을 트리거 — 직전 0.1s 프레임(2160 ray, yaw 1.1회전)
        hits = lidar.data.ray_hits_w  # (N,R,3), miss=inf
        origins = lidar._ray_starts_w  # (N,R,3) — 전 ray 공통 마운트 원점
        dirs = lidar._ray_directions_w  # (N,R,3) 단위벡터

        t_hit = (hits - origins).norm(dim=-1)
        finite = torch.isfinite(t_hit)
        t_hit = torch.where(finite, t_hit, torch.zeros_like(t_hit))

        # self-hit 필터: 부풀린 캡슐(로봇 기하 테이블)에 대한 ray 경로 관통 판정.
        # 로봇 collision mesh 가 캐스트 타깃이라 body 에 맞은 ray 는 t_hit 이 몸에서
        # 끝나 있고, 그 경로는 캡슐을 관통하므로 여기서 걸린다.
        from locomotion_isaaclab.sensors import ray_capsule_penetrates

        body_pos = self.asset.data.body_pos_w
        body_quat = self.asset.data.body_quat_w
        self_hit = torch.zeros_like(finite)
        for ia, off_a, ib, off_b, radius in self._capsules:
            pa = body_pos[:, ia]
            pb = body_pos[:, ib]
            if bool((off_a != 0).any()):
                pa = pa + quat_apply(body_quat[:, ia], off_a.expand(self.num_envs, 3))
            if bool((off_b != 0).any()):
                pb = pb + quat_apply(body_quat[:, ib], off_b.expand(self.num_envs, 3))
            self_hit |= ray_capsule_penetrates(origins, dirs, t_hit, pa, pb, radius)
        valid = finite & ~self_hit

        # L1 거리 노이즈 (스펙 ±2cm): range 백색잡음.
        if noise_scale > 0.0 and self.range_std > 0:
            t_hit = (t_hit + torch.randn_like(t_hit) * self.range_std * noise_scale).clamp_min(0.0)

        # 센서 pose (GT) — 점군을 센서 프레임으로 되돌리는 데 쓴다. 실기에서 점군은
        # 애초에 센서 프레임으로 들어오므로 이 변환에는 odometry 오차가 없다.
        t_s = origins[:, 0, :]
        R_base = matrix_from_quat(self.asset.data.root_quat_w)
        R_s = torch.bmm(R_base, self._R_offset.expand(self.num_envs, 3, 3))
        # 전 ray 가 공통 원점이라 p_sensor = (R_sᵀ d_w) · t_hit
        dirs_sensor = torch.einsum("nij,nri->nrj", R_s, dirs)

        # 빔 지향 노이즈: 실기 ray 방향은 공칭값과 어긋난다. 노이즈 방향으로
        # 재캐스팅할 수는 없으므로 "실제 빔이 어긋난 방향으로 나갔는데 공칭 거리로
        # 기록됐다"의 역, 즉 어긋난 방향에 측정 거리를 놓는 것으로 근사한다.
        if noise_scale > 0.0 and self.ray_dir_std_rad > 0:
            az = torch.atan2(dirs_sensor[..., 1], dirs_sensor[..., 0])
            el = torch.asin(dirs_sensor[..., 2].clamp(-1.0, 1.0))
            az = az + torch.randn_like(az) * self.ray_dir_std_rad * noise_scale
            el = (el + torch.randn_like(el) * self.ray_dir_std_rad * noise_scale).clamp(-1.5707, 1.5707)
            cos_el = torch.cos(el)
            dirs_sensor = torch.stack([cos_el * torch.cos(az), cos_el * torch.sin(az), torch.sin(el)], dim=-1)
        points_sensor = dirs_sensor * t_hit.unsqueeze(-1)

        # odometry drift — EM 이 아는 "odom frame 상의 pose" 에만 건다.
        # 위치: Δ_meas = Δ_true·(1+b+bias) + n → 오차 증분 R·((Rᵀ·Δ_true)∘(b+bias)) + n
        #      (b 는 tick 백색, bias 는 reset 마다 재샘플되는 episode 상수 scale 오차,
        #       둘 다 body frame 축별 적용).
        # yaw: Δ_meas = Δ_true + bias·dt + n → 오차 증분 bias·dt + n 누적
        #      (bias 는 reset 마다 재샘플되는 gyro 상수 bias).
        base_pos = self.asset.data.root_pos_w
        _, _, yaw_now = euler_xyz_from_quat(self.asset.data.root_quat_w)
        yaw_now = wrap_to_pi(yaw_now)
        if self._odom_needs_init.any():
            ids = self._odom_needs_init
            self._prev_base_pos[ids] = base_pos[ids]
            self._odom_needs_init[:] = False
        d_pos_true = base_pos - self._prev_base_pos
        self._prev_base_pos = base_pos.clone()
        if noise_scale > 0.0 and (
            self.odom_scale_std > 0 or self.odom_pos_walk_std > 0 or self.odom_scale_bias_max > 0
        ):
            b_pos = torch.randn(self.num_envs, 3, device=self.device) * self.odom_scale_std * noise_scale
            n_pos = torch.randn(self.num_envs, 3, device=self.device) * self.odom_pos_walk_std * noise_scale
            # scale 오차(b, bias)는 body frame 축별로 건다 — 보폭/슬립 오차는 로봇
            # 기준(진행/횡/수직)으로 생기므로 Δ 를 body 로 돌려 곱하고 world 로
            # 되돌려 누적한다. 등방성 walk n 은 회전 불변(R·N(0,σ²I)=N(0,σ²I))이라
            # frame 을 가릴 필요가 없다 — 축별 σ 를 도입하면 그때 body 로 옮길 것.
            d_pos_body = torch.bmm(R_base.transpose(1, 2), d_pos_true.unsqueeze(-1)).squeeze(-1)
            # episode bias 자체는 항상 최대 범위로 보관하고 적용할 때만 scale 한다.
            # 따라서 cfg 초기값이 0 이어도 curriculum 이 증가하면 즉시 활성화된다.
            err_body = d_pos_body * (b_pos + self._odom_pos_scale_bias * noise_scale)
            self._odom_pos_err += torch.bmm(R_base, err_body.unsqueeze(-1)).squeeze(-1) + n_pos
        if noise_scale > 0.0:
            n_yaw = torch.randn(self.num_envs, device=self.device) * self.odom_yaw_walk_std_rad * noise_scale
            # 누적된 과거 drift 에 scale 을 다시 곱하지 않고 이번 tick 증분만 조절한다.
            self._odom_yaw_err += self._odom_yaw_bias * noise_scale + n_yaw

        # 회전 오차 = (roll/pitch 백색, small-angle) ∘ Rz(누적 yaw drift)
        zeros = torch.zeros(self.num_envs, device=self.device)
        ones = torch.ones(self.num_envs, device=self.device)
        cz, sz = torch.cos(self._odom_yaw_err), torch.sin(self._odom_yaw_err)
        R_z = torch.stack(
            [
                torch.stack([cz, -sz, zeros], dim=-1),
                torch.stack([sz, cz, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=1,
        )
        if noise_scale > 0.0:
            d_rp = torch.randn(self.num_envs, 2, device=self.device) * self.odom_rp_std_rad * noise_scale
        else:
            d_rp = torch.zeros(self.num_envs, 2, device=self.device)
        skew_rp = torch.stack(
            [
                torch.stack([zeros, zeros, d_rp[:, 1]], dim=-1),
                torch.stack([zeros, zeros, -d_rp[:, 0]], dim=-1),
                torch.stack([-d_rp[:, 1], d_rp[:, 0], zeros], dim=-1),
            ],
            dim=1,
        )
        R_err = torch.bmm(torch.eye(3, device=self.device).expand(self.num_envs, 3, 3) + skew_rp, R_z)
        R_s_n = torch.bmm(R_err, R_s)
        t_s_n = t_s + self._odom_pos_err
        base_pos_n = base_pos + self._odom_pos_err
        R_base_n = torch.bmm(R_err, R_base)

        pts_list = [points_sensor[i][valid[i]] for i in range(self.num_envs)]
        self.backend.update(pts_list, R_s_n, t_s_n, base_pos_n, R_base_n)

        # scandot 위치(ray_alignment="yaw" 와 동일: yaw 만 따라 회전) 에서 샘플.
        # 실기에서 query pose 도 같은 odometry 에서 나오므로 map 과 같은 odom frame
        # 을 쓴다 — GT pose 로 샘플하면 누적 drift 전체가 query 오차로 들어가
        # 실제보다 훨씬 비관적인(틀린) 관측이 된다. odom frame 을 쓰면 map−query
        # 간 상대 오차(= 셀 관측 시점 이후의 drift)만 남는다.
        yaw_n = yaw_now + self._odom_yaw_err
        cy, sy = torch.cos(yaw_n), torch.sin(yaw_n)
        ox, oy = self._scan_offsets_xy[:, 0], self._scan_offsets_xy[:, 1]
        px = base_pos_n[:, 0:1] + cy[:, None] * ox[None, :] - sy[:, None] * oy[None, :]
        py = base_pos_n[:, 1:2] + sy[:, None] * ox[None, :] + cy[:, None] * oy[None, :]
        points_xy = torch.stack([px, py], dim=-1)
        self.h_obs, self.valid_frac, self.ub_frac = self.backend.sample(points_xy, base_pos_n[:, 2])
