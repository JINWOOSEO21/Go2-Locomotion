from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi, quat_apply
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from collections.abc import Sequence

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

import cv2
import numpy as np 

class reward_feet_edge(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.parkour_event: ParkourEvent =  env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.body_id = self.contact_sensor.find_bodies('base')[0]
        generator_cfg = env.scene.terrain.cfg.terrain_generator
        self.horizontal_scale = generator_cfg.horizontal_scale
        self.tile_size_x, self.tile_size_y = generator_cfg.size
        self.num_rows, self.num_cols = generator_cfg.num_rows, generator_cfg.num_cols
        self.rows_offset = (self.tile_size_x * self.num_rows/2)
        self.cols_offset = (self.tile_size_y * self.num_cols/2)
        # 타일 하나가 x_edge_maskes 에서 차지하는 픽셀 수. 높이맵은 칸이 아니라
        # 격자점을 저장하므로 (칸 수 + 1) 이고, 따라서 타일의 물리 길이를
        # horizontal_scale 로 나눈 값보다 정확히 1 크다. 아래에서 이어 붙인
        # 이미지도 이 폭으로 타일을 나열하므로, 월드 좌표를 그냥
        # horizontal_scale 로 나눈 값을 인덱스로 쓰면 행/열마다 1픽셀씩 밀린다
        # (16m x 4m 타일 / hs 0.08 기준 최대 x 0.72m, y 3.12m).
        self.tile_width_pixels = int(self.tile_size_x / self.horizontal_scale) + 1
        self.tile_length_pixels = int(self.tile_size_y / self.horizontal_scale) + 1
        total_x_edge_maskes = torch.from_numpy(self.parkour_event.terrain.terrain_generator_class.x_edge_maskes).to(device = self.device)
        self.x_edge_masks_tensor = total_x_edge_maskes.permute(0, 2, 1, 3).reshape(
            env.scene.terrain.terrain_generator_class.total_width_pixels, env.scene.terrain.terrain_generator_class.total_length_pixels
        )

    def _to_mask_index(
        self,
        pos: torch.Tensor,
        tile_size: float,
        num_tiles: int,
        tile_pixels: int,
        ) -> torch.Tensor:
        """지형 좌하단 기준 좌표(m) -> 이어 붙인 edge mask 이미지의 픽셀 인덱스.

        타일 인덱스를 먼저 구하고 타일 내부 픽셀을 더한다. 물리 간격(tile_size /
        horizontal_scale 픽셀)이 아니라 실제 저장 폭(tile_pixels)으로 타일을
        건너뛰어야 마스크가 어긋나지 않는다.
        """
        tile = torch.div(pos, tile_size, rounding_mode="floor").long()
        tile = torch.clip(tile, 0, num_tiles - 1)
        local = ((pos - tile * tile_size) / self.horizontal_scale).round().long()
        local = torch.clip(local, 0, tile_pixels - 1)
        return tile * tile_pixels + local

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        ) -> torch.Tensor:
        feet_pos_x = self._to_mask_index(
            self.asset.data.body_state_w[:, self.asset_cfg.body_ids, 0] + self.rows_offset,
            self.tile_size_x, self.num_rows, self.tile_width_pixels,
        )
        feet_pos_y = self._to_mask_index(
            self.asset.data.body_state_w[:, self.asset_cfg.body_ids, 1] + self.cols_offset,
            self.tile_size_y, self.num_cols, self.tile_length_pixels,
        )
        feet_at_edge = self.x_edge_masks_tensor[feet_pos_x, feet_pos_y]
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        contact = torch.norm(contact_forces, dim=-1) > 2.
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        self.feet_at_edge = contact_filt & feet_at_edge
        rew = (self.parkour_event.terrain.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        ## This is for debugging to matching index and x_edge_mask
        # origin = self.x_edge_masks_tensor.detach().cpu().numpy().astype(np.uint8) * 255
        # cv2.imshow('origin',origin)
        # origin[feet_pos_x.detach().cpu().numpy(), feet_pos_y.detach().cpu().numpy()] -= 100
        # cv2.imshow('feet_edge',origin)
        # cv2.waitKey(1)
        return rew

def reward_torques(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)

def reward_dof_error(    
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

def reward_hip_pos(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] \
                                    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)

def reward_ang_vel_xy(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:,:2]), dim=1)

class reward_action_rate(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_actions = torch.zeros(env.num_envs, 2,  asset.num_joints, dtype= torch.float ,device=self.device)
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_actions[env_ids, 0,:] = 0.
        self.previous_actions[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_actions[:, 0, :] = self.previous_actions[:, 1, :]
        self.previous_actions[:, 1, :] = env.action_manager.get_term('joint_pos').raw_actions
        return torch.norm(self.previous_actions[:, 1, :] - self.previous_actions[:,0,:], dim=1)
    
class reward_dof_acc(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_joint_vel = torch.zeros(env.num_envs, 2,  asset.num_joints, dtype= torch.float ,device=self.device)
        self.dt = env.cfg.decimation * env.cfg.sim.dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_joint_vel[env_ids, 0,:] = 0.
        self.previous_joint_vel[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        self.previous_joint_vel[:, 0, :] = self.previous_joint_vel[:, 1, :]
        self.previous_joint_vel[:, 1, :] = asset.data.joint_vel
        return torch.sum(torch.square((self.previous_joint_vel[:, 1, :] - self.previous_joint_vel[:,0,:]) / self.dt), dim=1)
        
def reward_lin_vel_z(
    env: ParkourManagerBasedRLEnv,        
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.square(asset.data.root_lin_vel_b[:, 2])
    rew[(terrain_names !='parkour_flat')[:,-1]] *= 0.5
    return rew

def reward_orientation(
    env: ParkourManagerBasedRLEnv,   
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    rew[(terrain_names !='parkour_flat')[:,-1]] = 0.
    return rew

def reward_feet_stumble(
    env: ParkourManagerBasedRLEnv,        
    sensor_cfg: SceneEntityCfg ,
    ) -> torch.Tensor: 
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
    rew = torch.any(torch.norm(net_contact_forces[:, :, :2], dim=2) >\
            4 *torch.abs(net_contact_forces[:, :, 2]), dim=1)
    return rew.float()

def reward_feet_slip(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    ) -> torch.Tensor:
    """Penalize horizontal foot speed while the foot is in contact."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    contacts = torch.norm(contact_forces, dim=-1) >= threshold
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.norm(foot_vel_xy, dim=-1) * contacts, dim=1)

def reward_tracking_goal_vel(
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    target_pos_rel = parkour_event.target_pos_rel
    target_vel = target_pos_rel / (torch.norm(target_pos_rel, dim=-1, keepdim=True) + 1e-5)
    cur_vel = asset.data.root_vel_w[:, :2]
    proj_vel = torch.sum(target_vel * cur_vel, dim=-1)
    command_vel = env.command_manager.get_command('base_velocity')[:, 0]
    rew_move = torch.minimum(proj_vel, command_vel) / (command_vel + 1e-5)
    return rew_move

def reward_tracking_yaw(     
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    yaw = torch.atan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
                    1 - 2*(q[:,2]**2 + q[:,3]**2))
    # Compare headings using the shortest signed angle across the +/-pi boundary.
    yaw_error = wrap_to_pi(parkour_event.target_yaw - yaw)
    return torch.exp(-torch.abs(yaw_error))

class reward_delta_torques(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_torque = torch.zeros(env.num_envs, 2,  self.asset.num_joints, dtype= torch.float ,device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_torque[env_ids, 0,:] = 0.
        self.previous_torque[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_torque[:, 0, :] = self.previous_torque[:, 1, :]
        self.previous_torque[:, 1, :] = self.asset.data.applied_torque
        return torch.sum(torch.square((self.previous_torque[:, 1, :] - self.previous_torque[:,0,:])), dim=1)

def reward_collision(
    env: ParkourManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg ,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
    return torch.sum(1.*(torch.norm(net_contact_forces, dim=-1) > 0.1), dim=1)
