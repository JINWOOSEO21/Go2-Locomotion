from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import wrap_to_pi

from parkour_isaaclab.managers import ParkourTerm
from parkour_isaaclab.terrains import ParkourTerrainGenerator, ParkourTerrainGeneratorCfg, ParkourTerrainImporter

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

    from .parkour_events_cfg import ParkourEventsCfg


class ParkourEvent(ParkourTerm):
    cfg: ParkourEventsCfg

    def __init__(self, cfg: ParkourEventsCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.episode_length_s = env.cfg.episode_length_s
        self.reach_goal_delay = cfg.reach_goal_delay
        self.num_future_goal_obs = cfg.num_future_goal_obs
        self.next_goal_threshold = cfg.next_goal_threshold
        self.simulation_time = env.step_dt
        self.arrow_num = cfg.arrow_num
        self.env = env
        self.debug_vis = cfg.debug_vis

        self.robot: Articulation = env.scene[cfg.asset_name]
        # -- metrics
        self.metrics["far_from_current_goal"] = torch.zeros(self.num_envs, device="cpu")
        self.metrics["how_far_from_start_point"] = torch.zeros(self.num_envs, device="cpu")
        self.metrics["terrain_levels"] = torch.zeros(self.num_envs, device="cpu")
        self.metrics["current_goal_idx"] = torch.zeros(self.num_envs, device="cpu")
        self.dis_to_start_pos = torch.zeros(self.num_envs, device=self.device)
        self.terrain: ParkourTerrainImporter = self.env.scene.terrain
        terrain_generator: ParkourTerrainGenerator = self.terrain.terrain_generator_class
        parkour_terrain_cfg: ParkourTerrainGeneratorCfg = self.terrain.cfg.terrain_generator
        self.num_goals = parkour_terrain_cfg.num_goals
        self.env_class = torch.zeros(self.num_envs, device=self.device)
        self.env_origins = self.terrain.env_origins
        self.terrain_type = terrain_generator.terrain_type
        self.terrain_class = torch.from_numpy(self.terrain_type).to(self.device).to(torch.float)
        self.env_class[:] = self.terrain_class[self.terrain.terrain_levels, self.terrain.terrain_types]

        terrain_goals = terrain_generator.goals
        self.terrain_goals = torch.from_numpy(terrain_goals).to(self.device).to(torch.float)
        self.env_goals = torch.zeros(
            self.num_envs,
            self.terrain_goals.shape[2] + self.num_future_goal_obs,
            3,
            device=self.device,
            requires_grad=False,
        )
        self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        temp = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float).to(device=self.device)

        if self.debug_vis:
            self.total_heights = torch.from_numpy(terrain_generator.goal_heights).to(device=self.device)
            self.future_goal_idx = torch.ones(self.num_goals, device=self.device, dtype=torch.bool).repeat(
                self.num_envs, 1
            )
            self.future_goal_idx[:, 0] = False
            self.env_per_heights = self.total_heights[self.terrain.terrain_levels, self.terrain.terrain_types]

        self.total_terrain_names = terrain_generator.terrain_names
        numpy_terrain_levels = self.terrain.terrain_levels.detach().cpu().numpy()  ## string type can't convert to torch
        numpy_terrain_types = self.terrain.terrain_types.detach().cpu().numpy()
        self.env_per_terrain_name = self.total_terrain_names[numpy_terrain_levels, numpy_terrain_types]
        self._reset_offset = self.env.event_manager.get_term_cfg("reset_root_state").params["offset"]

        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.target_pos_rel = self.cur_goals[:, :2] - robot_root_pos_w
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_root_pos_w
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    def __call__(self):
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    def restore_terrain_levels(self, levels: torch.Tensor):
        """지형 커리큘럼 단계를 통째로 덮어쓰고, 거기서 파생되는 캐시를 전부 다시 만든다.

        체크포인트에서 학습을 재개할 때 쓴다. terrain_levels 는 모델 파라미터와 달리
        어디에도 저장되지 않아서, 그냥 재개하면 _compute_env_origins_curriculum 이
        randint(0, max_init_terrain_level+1) 로 다시 뽑아 커리큘럼이 바닥으로 돌아간다.

        _resample_command 가 리셋된 env 에 대해 하는 재계산과 같은 일을 전 env 에
        대해 한다. 다만 여기서는 move_up/move_down 판정을 하지 않고 주어진 값을
        그대로 쓴다. cur_goal_idx / reach_goal_timer / dis_to_start_pos 는 건드리지
        않는다. 호출자가 곧바로 env 를 리셋해 그 값들을 0 으로 만드는 것을 전제한다.

        env_origins 는 terrain.env_origins 와 같은 텐서라 여기서 제자리 수정하면
        씬 쪽에도 그대로 반영된다.
        """
        levels = levels.to(device=self.terrain.terrain_levels.device, dtype=self.terrain.terrain_levels.dtype)
        # 저장 당시보다 지형이 얕아졌을 수 있다. 그대로 색인하면 out-of-bounds 다.
        self.terrain.terrain_levels[:] = levels.clamp_(0, self.terrain.max_terrain_level - 1)

        rows, cols = self.terrain.terrain_levels, self.terrain.terrain_types
        self.env_origins[:] = self.terrain.terrain_origins[rows, cols]
        self.env_class[:] = self.terrain_class[rows, cols]

        temp = self.terrain_goals[rows, cols]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        numpy_terrain_levels = rows.detach().cpu().numpy()
        numpy_terrain_types = cols.detach().cpu().numpy()
        self.env_per_terrain_name = self.total_terrain_names[numpy_terrain_levels, numpy_terrain_types]

        if self.debug_vis:
            self.env_per_heights = self.total_heights[rows, cols]

    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(
            1, (self.cur_goal_idx[:, None, None] + future).expand(-1, -1, self.env_goals.shape[-1])
        ).squeeze(1)

    def __str__(self) -> str:
        msg = "ParkourCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    def _update_command(self):
        """Re-target the current goal position to the current root state."""
        next_flag = self.reach_goal_timer > self.reach_goal_delay / self.simulation_time
        if self.debug_vis:
            tmp_mask = torch.nonzero(self.cur_goal_idx > 0).squeeze(-1)
            if tmp_mask.numel() > 0:
                self.future_goal_idx[tmp_mask, self.cur_goal_idx[tmp_mask]] = False
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0
        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.reached_goal_ids = torch.norm(robot_root_pos_w - self.cur_goals[:, :2], dim=1) < self.next_goal_threshold
        reached_goal_idx = self.reached_goal_ids.nonzero(as_tuple=False).squeeze(-1)
        if reached_goal_idx.numel() > 0:
            self.reach_goal_timer[reached_goal_idx] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - robot_root_pos_w
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_root_pos_w
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        start_pos = self._spawn_pos_xy()

        self.dis_to_start_pos = torch.norm(start_pos - self.robot.data.root_pos_w[:, :2], dim=1)

    def _spawn_pos_xy(self, env_ids=slice(None)) -> torch.Tensor:
        """로봇이 리셋될 때 놓이는 위치(타일 앞쪽 시작 플랫폼 위)의 xy.

        :func:`parkour_isaaclab.envs.mdp.events.reset_root_state` 와 반드시 같은 식이어야
        한다. 거기서는 env_origins(타일 '중앙') 기준으로
            positions = env_origins - (size[0] * 0.5 - offset, 0, 0)
        에 로봇을 놓는다. 여기서 다른 식을 쓰면 dis_to_start_pos 가 실제 이동 거리와
        어긋나고, 그 값으로 판정하는 지형 커리큘럼(move_up / move_down)이 통째로 틀어진다.

        원래는 size[1](y 폭) + offset 을 빼고 있었다. 이는 16x4 타일 + offset 3.0
        조합에서만 4.0 + 3.0 = 0.5*16 - 1.0 = 7.0 으로 우연히 맞던 식이다.
        offset 이 1.0 으로 바뀌고 타일이 24m 로 길어지면서 6m 어긋나 있었다
        (스폰 직후 dis_to_start_pos 가 0 이 아니라 6 으로 읽히고, norm 이라
        정확히 6m 전진한 로봇은 0 으로 읽혀 오히려 레벨이 강등됐다).
        """
        back_from_center = self.terrain.cfg.terrain_generator.size[0] * 0.5 - self._reset_offset
        back_from_center = torch.tensor((back_from_center, 0.0), device=self.device)
        return self.env_origins[env_ids, :2] - back_from_center

    def _resample_command(self, env_ids: Sequence[int]):
        ## we are use reset_root_state events for initalize robot position in a subterrain
        ## original robot root init position is (0,0) in the subterrain axis, so we subtracted off from current robot position

        # 판정에는 '이번 에피소드에서 얼마나 갔나' 가 필요하다. 그 값은 _update_command 가
        # 매 스텝 갱신해 둔 self.dis_to_start_pos 에 이미 들어 있으므로 그대로 쓴다.
        #
        # 여기서 로봇의 '현재' 위치로 다시 계산하면 안 된다. ParkourManagerBasedRLEnv._reset_idx
        # 는 event_manager.apply(mode="reset") 으로 로봇을 스폰 지점에 되돌린 '뒤'
        # command_manager.reset() -> 이 함수를 부른다. 즉 이 시점의 로봇은 이미 출발선에
        # 서 있어서 재계산하면 항상 ~0 이 나오고, move_up 은 영원히 False,
        # move_down 은 항상 True 가 된다.
        # (실측: 판정에 쓰인 거리가 0.03m 로 고정되고 terrain_level 이 3->2->1->0 으로
        #  단조 감소해 모든 env 가 최저 레벨로 주저앉았다.)
        dis_to_start_pos = self.dis_to_start_pos[env_ids]
        threshold = self.env.command_manager.get_command("base_velocity")[env_ids, 0] * self.episode_length_s
        move_up = dis_to_start_pos > 0.8 * threshold
        move_down = dis_to_start_pos < 0.4 * threshold
        # 리셋된 env 는 다시 출발선에 서므로 누적값을 0 으로 되돌린다.
        # (다음 스텝의 _update_command 가 어차피 덮어쓰지만, 그 사이에 metrics 로
        #  읽히면 이전 에피소드 값이 섞인다.)
        self.dis_to_start_pos[env_ids] = 0.0

        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # # Robots that solve the last level are sent to a random one
        self.terrain.terrain_levels[env_ids] = torch.where(
            self.terrain.terrain_levels[env_ids] >= self.terrain.max_terrain_level,
            torch.randint_like(self.terrain.terrain_levels[env_ids], self.terrain.max_terrain_level),
            torch.clip(self.terrain.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain.terrain_origins[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]
        ]
        self.env_class[env_ids] = self.terrain_class[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]
        ]

        temp = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.target_pos_rel = self.cur_goals[:, :2] - robot_root_pos_w
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_root_pos_w
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        numpy_terrain_levels = self.terrain.terrain_levels.detach().cpu().numpy()
        numpy_terrain_types = self.terrain.terrain_types.detach().cpu().numpy()
        self.env_per_terrain_name = self.total_terrain_names[numpy_terrain_levels, numpy_terrain_types]

        self.reach_goal_timer[env_ids] = 0
        self.cur_goal_idx[env_ids] = 0

        if self.debug_vis:
            self.future_goal_idx[env_ids, 0] = False
            self.future_goal_idx[env_ids, 1:] = True
            self.env_per_heights = self.total_heights[self.terrain.terrain_levels, self.terrain.terrain_types]

    def _update_metrics(self):
        # logs data
        self.metrics["terrain_levels"] = (self.terrain.terrain_levels.float()).to(device="cpu")
        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.metrics["far_from_current_goal"] = (
            torch.norm(self.cur_goals[:, :2] - robot_root_pos_w, dim=-1) - self.next_goal_threshold
        ).to(device="cpu")
        self.metrics["current_goal_idx"] = self.cur_goal_idx.to(device="cpu", dtype=float)
        self.metrics["how_far_from_start_point"] = self.dis_to_start_pos.to(device="cpu")

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "current_goal_pose_visualizer"):
                self.current_goal_pose_visualizer = VisualizationMarkers(self.cfg.current_goal_pose_visualizer_cfg)
            # set their visibility to true
            self.current_goal_pose_visualizer.set_visibility(True)
            if not hasattr(self, "future_goal_poses_visualizer"):
                self.future_goal_poses_visualizer = VisualizationMarkers(self.cfg.future_goal_poses_visualizer_cfg)
            self.future_goal_poses_visualizer.set_visibility(True)

            if not hasattr(self, "current_arrow_visualizer"):
                self.current_arrow_visualizer = VisualizationMarkers(self.cfg.current_arrow_visualizer_cfg)
            # set their visibility to true
            self.current_arrow_visualizer.set_visibility(True)
            if not hasattr(self, "future_arrow_visualizer"):
                self.future_arrow_visualizer = VisualizationMarkers(self.cfg.future_arrow_visualizer_cfg)
            self.future_arrow_visualizer.set_visibility(True)

        else:
            if hasattr(self, "current_goal_pose_visualizer"):
                self.current_goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "future_goal_poses_visualizer"):
                self.future_goal_poses_visualizer.set_visibility(False)

            if hasattr(self, "current_arrow_visualizer"):
                self.current_arrow_visualizer.set_visibility(False)
            if hasattr(self, "future_arrow_visualizer"):
                self.future_arrow_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        env_per_goals = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types]
        env_per_xy_goals = env_per_goals[:, :, :2].reshape(self.num_envs, -1, 2)  ## (env_num, 8, 2 )
        env_per_xy_goals = env_per_xy_goals + self.env_origins[:, :2].unsqueeze(1)
        goal_height = self.env_per_heights.unsqueeze(-1) * self.terrain.cfg.terrain_generator.vertical_scale
        env_per_goal_pos = torch.concat([env_per_xy_goals, goal_height], dim=-1)
        env_per_current_goal_pos = env_per_goal_pos[~self.future_goal_idx, :]
        env_per_future_goal_pos = env_per_goal_pos[self.future_goal_idx, :].reshape(-1, 3)
        self.current_goal_pose_visualizer.visualize(
            translations=env_per_current_goal_pos,
        )
        if len(env_per_future_goal_pos) > 0:
            self.future_goal_poses_visualizer.visualize(
                translations=env_per_future_goal_pos,
            )
        current_arrow_list = []
        future_arrow_list = []
        for i in range(self.arrow_num):
            norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            current_pose_arrow = self.robot.data.root_pos_w[:, :2] + 0.1 * (i + 3) * target_vec_norm[:, :2]
            current_arrow_list.append(
                torch.concat(
                    [
                        current_pose_arrow[:, 0][:, None],
                        current_pose_arrow[:, 1][:, None],
                        self.robot.data.root_pos_w[:, 2][:, None],
                    ],
                    dim=1,
                )
            )
            if len(env_per_future_goal_pos) > 0:
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                future_pose_arrow = self.robot.data.root_pos_w[:, :2] + 0.2 * (i + 3) * target_vec_norm[:, :2]
                future_arrow_list.append(
                    torch.concat(
                        [
                            future_pose_arrow[:, 0][:, None],
                            future_pose_arrow[:, 1][:, None],
                            self.robot.data.root_pos_w[:, 2][:, None],
                        ],
                        dim=1,
                    )
                )
            else:
                future_arrow_list.append(
                    torch.concat(
                        [
                            current_pose_arrow[:, 0][:, None],
                            current_pose_arrow[:, 1][:, None],
                            self.robot.data.root_pos_w[:, 2][:, None],
                        ],
                        dim=1,
                    )
                )

        current_arrow_positions = torch.cat(current_arrow_list, dim=0)
        future_arrow_positions = torch.cat(future_arrow_list, dim=0)
        self.current_arrow_visualizer.visualize(
            translations=current_arrow_positions,
        )

        self.future_arrow_visualizer.visualize(
            translations=future_arrow_positions,
        )

    @property
    def command(self):
        """Null command.

        Raises:
            RuntimeError: No command is generated. Always raises this error.
        """
        raise RuntimeError("NullCommandTerm does not generate any commands.")
