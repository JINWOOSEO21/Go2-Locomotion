# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""
from __future__ import annotations
import torchvision
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi, matrix_from_quat, quat_apply
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from collections.abc import Sequence
import numpy as np
import cv2
import warnings
if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


class ExtremeParkourObservations(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors['contact_forces']
        self.ray_sensor: RayCaster = env.scene.sensors['height_scanner']
        self.parkour_event: ParkourEvent =  env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.history_length = cfg.params['history_length']
        self._obs_history_buffer = torch.zeros(self.num_envs, self.history_length, 3 + 2 + 3 + 4 + 36 + 5, device=self.device)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device)
        self.measured_heights = torch.zeros(self.num_envs, 132, device=self.device)
        self.env = env
        self.body_id = self.asset.find_bodies('base')[0]
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._obs_history_buffer[env_ids, :, :] = 0. 

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        history_length: int,
        ) -> torch.Tensor:
        
        # 지형 타입(flat/non-flat) 플래그를 상수로 고정: 항상 non-flat=1, flat=0.
        # 원조 Extreme Parkour 는 이 플래그로 flat 고속 모드 전환을 학습시켰지만,
        # 이 프로젝트의 목표는 험지 robust 보행이고 실기에는 "지금 flat 이다"를
        # 알려줄 오라클이 없다 — flat 감지 후 모드 전환을 실세계에서 재현할 수
        # 없으므로 정책이 이 신호에 의존하지 않게 한다. 지형 종류 정보는 보상
        # 셰이핑(rewards.py 의 parkour_flat 분기)에만 남는다 — 그쪽은 sim 전용이다.
        env_idx_tensor = torch.ones(self.num_envs, 1, device=self.device)
        invert_env_idx_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        roll, pitch, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)
        imu_obs = torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=1).to(self.device)
        if env.common_step_counter % 5 == 0:
            self.delta_yaw = self.parkour_event.target_yaw - wrap_to_pi(yaw)
            self.delta_next_yaw = self.parkour_event.next_target_yaw - wrap_to_pi(yaw)
            self.measured_heights = self._get_heights()
        commands = env.command_manager.get_command('base_velocity')
        obs_buf = torch.cat((
                            self.asset.data.root_ang_vel_b * 0.25,   #[1,3] 0~2
                            imu_obs,    #[1,2] 3~4
                            0*self.delta_yaw[:, None],   #[1,1] 5
                            self.delta_yaw[:, None], #[1,1] 6
                            self.delta_next_yaw[:, None], #[1,1] 7 
                            0*commands[:, 0:2], #[1,2] 8 
                            commands[:, 0:1],  #[1,1] 9
                            env_idx_tensor,
                            invert_env_idx_tensor,
                            self.asset.data.joint_pos - self.asset.data.default_joint_pos,
                            self.asset.data.joint_vel * 0.05 ,
                            env.action_manager.get_term('joint_pos').action_history_buf[:, -1],
                            self._get_contact_fill(),
                            ),dim=-1)
        priv_explicit = self._get_priv_explicit()
        priv_latent = self._get_priv_latent()
        observations = torch.cat([obs_buf, #53
                                  self.measured_heights, #132
                                  priv_explicit, # 9
                                  priv_latent, # 29
                                  self._obs_history_buffer.view(self.num_envs, -1)
                                  ],dim=-1)
        obs_buf[:, 6:8] = 0
        self._obs_history_buffer = torch.where(
            (env.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.history_length, dim=1),
            torch.cat([
                self._obs_history_buffer[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        return observations 

    def _get_contact_fill(
        self,
        ):
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        contact = torch.norm(contact_forces, dim=-1) > 2.
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        return (contact_filt.float()-0.5).to(self.device)
    
    def _get_priv_explicit(
        self,
        ):
        base_lin_vel = self.asset.data.root_lin_vel_b 
        return torch.cat((base_lin_vel * 2.0,
                        0 * base_lin_vel,
                        0 * base_lin_vel), dim=-1).to(self.device)
    
    def _get_priv_latent(
        self,
        ):
        body_mass = self.asset.root_physx_view.get_masses()[:,self.body_id].to(self.device)
        body_com = self.asset.data.com_pos_b[:,self.body_id,:].to(self.device).squeeze(1)
        mass_params_tensor = torch.cat([body_mass, body_com],dim=-1).to(self.device)
        friction_coeffs_tensor = self.asset.root_physx_view.get_material_properties()[:, 0, 0]
        joint_stiffness = self.asset.data.joint_stiffness.to(self.device)
        default_joint_stiffness = self.asset.data.default_joint_stiffness.to(self.device)
        joint_damping = self.asset.data.joint_damping.to(self.device)
        default_joint_damping = self.asset.data.default_joint_damping.to(self.device)
        return torch.cat((
            mass_params_tensor,
            friction_coeffs_tensor.unsqueeze(1).to(self.device),
            (joint_stiffness/ default_joint_stiffness) - 1, 
            (joint_damping/ default_joint_damping) - 1
        ), dim=-1).to(self.device)
    
    def _get_heights(self):
        return torch.clip(self.ray_sensor.data.pos_w[:, 2].unsqueeze(1) - self.ray_sensor.data.ray_hits_w[..., 2] - 0.3, -1, 1).to(self.device)

class elevation_map_scan(ManagerTermBase):
    """L1 LiDAR → self-hit 필터 → elevation_mapping_cupy → scandots 격자 샘플.

    계획서 docs/emcupy_student_plan.md §2.1 의 파이프라인. depth camera 파이프라인을
    대체하는 student 전용 관측으로, teacher scandots(obs[53:185])와 같은 위치·같은
    정규화(clip(base_z − h − 0.3, ±1))의 132-vector 를 낸다. EM tick(= 센서 자연
    프레임 0.1s, 10Hz)에서만 갱신되고 사이 step 은 최신 값을 그대로 돌려준다.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        from parkour_isaaclab.sensors import GO2_SELF_FILTER_CAPSULES
        from parkour_isaaclab.envs.mdp.elevation_map_backend import (
            BatchedElevationMapBackend,
            ElevationMapBackend,
        )

        self.env = env
        self.lidar = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.update_interval = int(cfg.params.get("update_interval", 5))
        # 노이즈 (계획서 §5 잔여 가정): EM 입력에만 걸리고 policy obs(GT)는 무관.
        self.odom_pos_std = float(cfg.params.get("odom_pos_std", 0.01))
        self.odom_rot_std_rad = float(np.deg2rad(cfg.params.get("odom_rot_std_deg", 0.5)))
        self.range_std = float(cfg.params.get("range_std", 0.02))

        # scandot 격자는 teacher 의 height_scanner 에서 그대로 가져온다 —
        # ray_starts 는 (base frame) 패턴점 + cfg offset(+0.375m) 이라 순서/위치가
        # obs 의 scan 구간과 1:1 로 일치한다.
        scanner: RayCaster = env.scene.sensors["height_scanner"]
        self._scan_offsets_xy = scanner.ray_starts[0][:, :2].clone().to(self.device)
        self.num_points = self._scan_offsets_xy.shape[0]

        # "batched"(기본): 커널 배치판 — 192 env 기준 tick 당 수 ms.
        # "loop": em_cupy 인스턴스 직렬 루프 — 회귀 비교/디버깅용 (250ms/tick).
        backend_cls = (
            BatchedElevationMapBackend
            if cfg.params.get("em_backend", "batched") == "batched"
            else ElevationMapBackend
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
        # 검증 스크립트(scripts/emcupy_check)가 valid_frac 등에 접근할 수 있게 노출
        env.em_scan_term = self

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        # teleport 후 옛 지형이 남으면 안 된다. 관측도 "아직 아무것도 못 봄"(=0)으로.
        self.backend.clear([int(i) for i in env_ids])
        self.h_obs[env_ids] = 0.0
        self.valid_frac[env_ids] = 0.0
        self.ub_frac[env_ids] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        # 아래 파라미터들은 __init__ 에서 cfg.params 로 소비된다. ObservationManager 의
        # 시그니처 검사를 통과하기 위해 여기 명시한다 (**kwargs 는 'kwargs' 라는
        # 필수 파라미터로 오해석된다).
        em_resolution: float = 0.1,
        em_map_length: float = 3.2,
        em_backend: str = "batched",
        update_interval: int = 5,
        odom_pos_std: float = 0.01,
        odom_rot_std_deg: float = 0.5,
        range_std: float = 0.02,
    ) -> torch.Tensor:
        # 센서 프레임(0.1s)과 같은 위상: 기존 depth/scandots 의 %5 게이트와 일치.
        if env.common_step_counter % self.update_interval == 0:
            self._update()
        return self.h_obs

    @torch.no_grad()
    def _update(self):
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
        from parkour_isaaclab.sensors import ray_capsule_penetrates

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

        # L1 거리 노이즈 (스펙 ±2cm). 방향은 결정론이라 range 에만 건다.
        if self.range_std > 0:
            t_hit = (t_hit + torch.randn_like(t_hit) * self.range_std).clamp_min(0.0)
        points_w = origins + dirs * t_hit.unsqueeze(-1)

        # 센서 pose (GT) — 점군을 센서 프레임으로 되돌리는 데 쓴다. 실기에서 점군은
        # 애초에 센서 프레임으로 들어오므로 이 변환에는 odometry 오차가 없다.
        t_s = origins[:, 0, :]
        R_base = matrix_from_quat(self.asset.data.root_quat_w)
        R_s = torch.bmm(R_base, self._R_offset.expand(self.num_envs, 3, 3))
        p_rel = points_w - t_s.unsqueeze(1)
        points_sensor = torch.einsum("nij,nri->nrj", R_s, p_rel)  # R_sᵀ (p−t)

        # odometry(6D pose) 백색잡음 — EM 이 아는 "odom frame 상의 pose" 에만 건다.
        base_pos = self.asset.data.root_pos_w
        if self.odom_pos_std > 0 or self.odom_rot_std_rad > 0:
            d_pos = torch.randn(self.num_envs, 3, device=self.device) * self.odom_pos_std
            d_rpy = torch.randn(self.num_envs, 3, device=self.device) * self.odom_rot_std_rad
            # small-angle 회전: R_err ≈ I + [δθ]×
            zeros = torch.zeros(self.num_envs, device=self.device)
            skew = torch.stack(
                [
                    torch.stack([zeros, -d_rpy[:, 2], d_rpy[:, 1]], dim=-1),
                    torch.stack([d_rpy[:, 2], zeros, -d_rpy[:, 0]], dim=-1),
                    torch.stack([-d_rpy[:, 1], d_rpy[:, 0], zeros], dim=-1),
                ],
                dim=1,
            )
            R_err = torch.eye(3, device=self.device).expand(self.num_envs, 3, 3) + skew
            R_s_n = torch.bmm(R_err, R_s)
            t_s_n = t_s + d_pos
            base_pos_n = base_pos + d_pos
            R_base_n = torch.bmm(R_err, R_base)
        else:
            R_s_n, t_s_n, base_pos_n, R_base_n = R_s, t_s, base_pos, R_base

        pts_list = [points_sensor[i][valid[i]] for i in range(self.num_envs)]
        self.backend.update(pts_list, R_s_n, t_s_n, base_pos_n, R_base_n)

        # scandot 위치(ray_alignment="yaw" 와 동일: yaw 만 따라 회전) 에서 샘플
        _, _, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        ox, oy = self._scan_offsets_xy[:, 0], self._scan_offsets_xy[:, 1]
        px = base_pos[:, 0:1] + cy[:, None] * ox[None, :] - sy[:, None] * oy[None, :]
        py = base_pos[:, 1:2] + sy[:, None] * ox[None, :] + cy[:, None] * oy[None, :]
        points_xy = torch.stack([px, py], dim=-1)
        self.h_obs, self.valid_frac, self.ub_frac = self.backend.sample(points_xy, base_pos[:, 2])


class image_features(ManagerTermBase):
    
    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.camera_sensor: RayCasterCamera = env.scene[cfg.params["sensor_cfg"].name]
        self.clipping_range = self.camera_sensor.cfg.max_distance
        resized = cfg.params["resize"]
        self.buffer_len = cfg.params['buffer_len']
        self.debug_vis = cfg.params['debug_vis']
        # cv2.imshow 가 가능한 환경인지. 첫 실패 시 False 로 내려간다.
        self._can_show = True
        self.resize_transform = torchvision.transforms.Resize(
                                    (resized[0], resized[1]), 
                                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC).to(env.device)
        self.depth_buffer = torch.zeros(self.num_envs,  
                                        self.buffer_len, 
                                        resized[0], 
                                        resized[1]).to(self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(0, self.num_envs)
        depth_images = self.camera_sensor.data.output["distance_to_camera"].squeeze(-1)[env_ids]
        for depth_image, env_id in zip(depth_images, env_ids):
            processed_image = self._process_depth_image(depth_image)
            self.depth_buffer[env_id] = torch.stack([processed_image]* 2, dim=0)

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        sensor_cfg: SceneEntityCfg,
        resize: tuple(int,int), 
        buffer_len: int,
        debug_vis:bool
        ):
        if env.common_step_counter % 5 == 0:
            depth_images = self.camera_sensor.data.output["distance_to_camera"].squeeze(-1)
            for env_id, depth_image in enumerate(depth_images):
                processed_image = self._process_depth_image(depth_image)
                self.depth_buffer[env_id] = torch.cat([self.depth_buffer[env_id, 1:], 
                                                    processed_image.to(self.device).unsqueeze(0)], dim=0)
        if self.debug_vis and self._can_show:
            depth_images_np = self.depth_buffer[:, -2].detach().cpu().numpy()
            depth_images_norm = []
            for img in depth_images_np:
                depth_images_norm.append(img)
            rows = []
            ncols = 4
            for i in range(0, len(depth_images_norm), ncols):
                chunk = list(depth_images_norm[i:i+ncols])
                # 마지막 행이 ncols 개를 못 채우면 행마다 폭이 달라져 vstack 이 실패한다.
                # (num_envs=5 -> 4 + 1 이면 348 vs 87 로 어긋난다.)
                # 빈 칸을 0 으로 채워 폭을 맞춘다.
                if len(chunk) < ncols:
                    chunk += [np.zeros_like(chunk[0])] * (ncols - len(chunk))
                rows.append(np.hstack(chunk))

            grid_img = np.vstack(rows)
            try:
                cv2.imshow("depth_images_grid", grid_img)
                cv2.waitKey(1)
            except cv2.error as e:
                # 헤드리스(SSH, --headless)이거나 OpenCV 가 GUI 지원 없이 빌드된 경우.
                # 관측 계산 자체는 문제없으므로 디버그 창만 끄고 계속 진행한다.
                self._can_show = False
                warnings.warn(f"depth debug window disabled (no display / OpenCV built without GUI): {e}")
        return self.depth_buffer[:, -2].to(env.device)

    def _process_depth_image(self, depth_image):
        depth_image = self._crop_depth_image(depth_image)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self._normalize_depth_image(depth_image)
        return depth_image

    def _crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def _normalize_depth_image(self, depth_image):
        depth_image = depth_image  # make similiar to scandot
        depth_image = (depth_image) / (self.clipping_range)  - 0.5
        return depth_image
