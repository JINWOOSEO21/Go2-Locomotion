from isaaclab.envs.mdp.events import apply_external_force_torque, randomize_rigid_body_mass, reset_joints_by_scale
from isaaclab.envs.mdp.rewards import undesired_contacts
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from parkour_isaaclab.envs.mdp import events, observations, parkour_commands, parkours, rewards, terminations
from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = parkour_commands.ParkourCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
        heading_control_stiffness=0.8,
        ranges=parkour_commands.ParkourCommandCfg.Ranges(lin_vel_x=(0.3, 0.8), heading=(-1.6, 1.6)),
        clips=parkour_commands.ParkourCommandCfg.Clips(lin_vel_clip=0.2, ang_vel_clip=0.4),
    )


@configclass
class ParkourEventsCfg:
    """Command specifications for the MDP."""

    base_parkour = parkours.ParkourEventsCfg(
        asset_name="robot",
    )


@configclass
class TeacherObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "parkour_name": "base_parkour",
                "history_length": 10,
            },
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class LidarObservationsCfg(TeacherObservationsCfg):
    """LiDAR 관측 구성.

    ``policy``는 Scandots 경로와 같은 753차원 레이아웃을 유지한다. 별도의
    ``em_scan`` 그룹이 LiDAR와 odometry로 만든 elevation map의 132개 샘플을
    제공하고, RSL-RL wrapper가 policy의 GT scandots 구간만 이 값으로 교체한다.
    """

    @configclass
    class EMScanPolicyCfg(ObsGroup):
        em_scan = ObsTerm(
            func=observations.elevation_map_scan,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("lidar"),
                # 계획서 §2.4 확정값 + §5 잔여 가정 (노이즈는 EM 입력에만 걸린다)
                "em_resolution": 0.1,
                "em_map_length": 3.2,
                "em_backend": "batched",  # "loop" = em_cupy 인스턴스 직렬(회귀 비교용)
                "update_interval": 5,  # 센서 자연 프레임 0.1s = 10Hz (Q1)
                "range_std": 0.02,  # L1 거리 노이즈 σ [m] (스펙 ±2cm)
                "ray_dir_std_deg": 0.2,  # L1 빔 지향(az/el) 백색잡음 σ [deg]
                # odometry 위치 drift: tick 마다 Δ_meas = Δ_true·(1+b+bias) + n 으로 오차 누적
                "odom_scale_var": 0.02,  # b ~ N(0, 0.02) — tick 백색 scale 오차 분산
                "odom_pos_walk_std": 0.005,  # n ~ N(0, 0.005²) [m/tick] 위치 random walk
                "odom_scale_bias_max": 0.03,  # bias ~ U(-0.03, 0.03) — episode 당 1회, 축별 독립
                # yaw drift: Δ_meas = Δ_true + bias·dt + n (bias 는 reset 마다 재샘플)
                "odom_yaw_bias_range_dps": (0.01, 0.05),  # gyro bias 크기 [deg/s], 부호 랜덤
                "odom_yaw_walk_std_deg": 0.003,  # yaw random walk σ [deg/tick]
                "odom_rp_std_deg": 0.5,  # roll/pitch 백색잡음 σ [deg] (IMU 관측, 비누적)
            },
        )

    em_scan: EMScanPolicyCfg = EMScanPolicyCfg()


# 이전에 저장된 Hydra 설정과 외부 import를 위한 호환 alias. 새 task 등록과
# 실행 경로에서는 LidarObservationsCfg를 사용한다.
EMStudentObservationsCfg = LidarObservationsCfg


@configclass
class TeacherRewardsCfg:
    """Reward terms for the MDP.
    ['base',
    'FL_hip',
    'FL_thigh',
    'FL_calf',
    'FL_foot',
    'FR_hip',
    'FR_thigh',
    'FR_calf',
    'FR_foot',
    'Head_upper',
    'Head_lower',
    'RL_hip',
    'RL_thigh',
    'RL_calf',
    'RL_foot',
    'RR_hip',
    'RR_thigh',
    'RR_calf',
    'RR_foot']
    """

    # Available Body strings:
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_calf", ".*_thigh"]),
        },
    )
    reward_feet_edge = RewTerm(
        func=rewards.reward_feet_edge,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=".*_foot"),
            "parkour_name": "base_parkour",
        },
    )
    reward_torques = RewTerm(
        func=rewards.reward_torques,
        weight=-0.00001,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_dof_error = RewTerm(
        func=rewards.reward_dof_error,
        weight=-0.04,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_hip_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
        },
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_feet_slip = RewTerm(
        func=rewards.reward_feet_slip,
        weight=-0.04,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "threshold": 5.0,
        },
    )
    reward_delta_torques = RewTerm(
        func=rewards.reward_delta_torques,
        weight=-1.0e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=terminations.time_out,
        time_out=True,
    )
    goal_reached = DoneTerm(
        func=terminations.goal_reached,
        time_out=False,
    )
    fallen = DoneTerm(
        func=terminations.fallen,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class EventCfg:
    ### Modified origin events, plz see relative issue https://github.com/isaac-sim/IsaacLab/issues/1955
    """Configuration for events."""

    reset_root_state = EventTerm(
        func=events.reset_root_state,
        # offset = 타일 시작점에서 spawn 까지의 x 거리. 시작 플랫폼은 0 ~ platform_len(2.5m)
        # 이므로 그 안에 들어와야 한다. 1.0 은 원본(16m 지형)의 실제 spawn 위치와 동일하다.
        params={"offset": 1.0},
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=reset_joints_by_scale,
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
        mode="reset",
    )
    physics_material = EventTerm(  # Okay
        func=events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_range": (0.25, 2.0),
            "friction_intervals": ((0.25, 0.5), (0.5, 1.0), (1.0, 2.0)),
            "num_buckets": 96,
        },
    )

    ## we don't use this event, If you use this, you will get a bad result
    # randomize_actuator_gains = EventTerm(
    #     func= events.randomize_actuator_gains,
    #     params={
    #         "asset_cfg" :SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.975, 1.025),
    #         "damping_distribution_params": (0.975, 1.025),
    #         "operation": "scale",
    #         },
    #     mode="startup",
    # )
    randomize_rigid_body_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=events.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    push_by_setting_velocity = EventTerm(  # Okay
        func=events.push_by_setting_velocity,
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        interval_range_s=(8.0, 8.0),
        is_global_time=True,
        mode="interval",
    )
    push_angular_velocity = EventTerm(
        func=events.push_by_setting_velocity,
        params={"velocity_range": {"roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}},
        interval_range_s=(7.0, 7.0),
        is_global_time=True,
        mode="interval",
    )
    base_external_force_torque = EventTerm(  # Okay
        func=apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )


@configclass
class ActionsCfg:
    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=8,
        use_delay=True,
        clip={".*": (-4.8, 4.8)},
    )
