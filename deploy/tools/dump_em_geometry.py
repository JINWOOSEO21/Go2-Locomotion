"""EM 사이드카가 필요로 하는 로봇/센서 기하를 IsaacLab 에서 실측해 npz 로 떨군다.

배포측(MuJoCo/실기)은 IsaacLab 을 import 하지 않는다. 그래서 학습 때 쓰인 기하
상수를 **손으로 옮겨 적지 않고** 여기서 뽑아 계약 파일에 못박는다. 뽑는 것:

1. scandots 격자 offset (132, 2)
   teacher height_scanner 의 ray_starts[:, :2] 와 같은 값. RayCaster 는
   `pattern_cfg.func(...)` 결과에 `offset.pos` 를 더할 뿐이므로(ray_caster.py
   L218-224), **teacher cfg 객체 자체**에서 같은 함수를 같은 인자로 호출한다.
   지형 생성(수십 초 + VRAM)을 피하려고 센서 인스턴스를 만들지는 않는다.

2. self-filter 캡슐 끝점의 로컬 오프셋
   MJCF 에는 base/{leg}_hip/{leg}_thigh/{leg}_calf 바디만 있고 Head_upper,
   Head_lower, {leg}_foot 은 없다(고정 링크라 부모에 합쳐졌다). 이들은 부모
   프레임에 대해 고정이므로 그 오프셋을 실측해 상수로 굳힌다.

3. FK 교차검증용 기준값
   임의 관절각 K 개에 대한 전 캡슐 바디의 base 프레임 위치. 사이드카의
   MuJoCo FK 가 이 값을 재현하는지 대조한다(게이트: max 오차 < 5 mm).

Isaac 종료 시 stdout 이 잘리는 함정이 있어 결과는 파일로만 쓰고, 성공/실패는
[RESULT] 마커로 알린다.
"""

import argparse

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_poses", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

from locomotion_isaaclab.sensors import GO2_SELF_FILTER_CAPSULES  # noqa: E402
from locomotion_tasks.default_cfg import LocomotionDefaultSceneCfg  # noqa: E402

ok = False
try:
    # --- 1. scandots 격자 -----------------------------------------------------
    # teacher cfg 를 그대로 읽는다 (값을 재입력하지 않는다).
    from locomotion_tasks.locomotion_task.config.go2.locomotion_teacher_cfg import (
        LocomotionTeacherSceneCfg,
    )

    # configclass 는 dataclass 라 클래스 속성이 아니라 인스턴스에서 읽어야 한다.
    # (씬 cfg 인스턴스화는 객체 조립일 뿐, 지형 생성은 일어나지 않는다.)
    scanner_cfg = LocomotionTeacherSceneCfg(num_envs=1, env_spacing=1.0).height_scanner
    pat = scanner_cfg.pattern_cfg
    starts, _dirs = pat.func(pat, "cpu")
    starts = starts + torch.tensor(list(scanner_cfg.offset.pos))
    scan_offsets_xy = starts[:, :2].numpy().astype(np.float64)

    # --- 2/3. 로봇 기하 --------------------------------------------------------
    scene_cfg = LocomotionDefaultSceneCfg(num_envs=1, env_spacing=1.0)
    robot_cfg = scene_cfg.robot.replace(prim_path="/World/Robot")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(robot_cfg)
    sim.reset()

    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)

    # 캡슐이 참조하는 바디 이름 (중복 제거, 등장 순서 유지)
    cap_bodies: list[str] = []
    for na, _oa, nb, _ob, _r in GO2_SELF_FILTER_CAPSULES:
        for n in (na, nb):
            if n not in cap_bodies:
                cap_bodies.append(n)
    cap_body_ids = [robot.find_bodies(n)[0][0] for n in cap_bodies]

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    q_lo = robot.data.soft_joint_pos_limits[0, :, 0].cpu()
    q_hi = robot.data.soft_joint_pos_limits[0, :, 1].cpu()

    q_used = []
    pos_base = []  # (K, n_cap, 3) — base 프레임에서 본 캡슐 바디 원점
    quat_w = []  # (K, n_cap, 4)
    base_quat = []

    for k in range(args.num_poses):
        if k == 0:
            q = robot.data.default_joint_pos[0].cpu().clone()
        else:
            u = torch.rand(len(joint_names), generator=gen)
            q = q_lo + u * (q_hi - q_lo)
        qd = torch.zeros_like(q)
        robot.write_joint_state_to_sim(q.to(sim.device).unsqueeze(0), qd.to(sim.device).unsqueeze(0))
        # 관절각을 강제로 넣은 뒤 바디 pose 가 갱신되려면 물리 한 스텝이 필요하다.
        # 5 ms 동안 중력으로 조금 흐르므로, 명령값이 아니라 **읽어온 실제값**을
        # 기준값으로 저장한다 — 그래야 대조가 정확해진다.
        sim.step(render=False)
        robot.update(sim.get_physics_dt())

        qa = robot.data.joint_pos[0].cpu().numpy()
        bp = robot.data.body_pos_w[0].cpu()
        bq = robot.data.body_quat_w[0].cpu()
        ib = robot.find_bodies("base")[0][0]
        R_b = matrix_from_quat(bq[ib].unsqueeze(0))[0]
        rel = (bp[cap_body_ids] - bp[ib]) @ R_b  # (R_bᵀ x)ᵀ == xᵀ R_b

        q_used.append(qa)
        pos_base.append(rel.numpy())
        quat_w.append(bq[cap_body_ids].numpy())
        base_quat.append(bq[ib].numpy())

    np.savez(
        args.out,
        scan_offsets_xy=scan_offsets_xy,
        scan_resolution=np.float64(pat.resolution),
        scan_size=np.asarray(pat.size, dtype=np.float64),
        scan_offset_pos=np.asarray(scanner_cfg.offset.pos, dtype=np.float64),
        joint_names=np.array(joint_names),
        body_names=np.array(body_names),
        capsule_bodies=np.array(cap_bodies),
        capsule_a=np.array([c[0] for c in GO2_SELF_FILTER_CAPSULES]),
        capsule_off_a=np.array([c[1] for c in GO2_SELF_FILTER_CAPSULES], dtype=np.float64),
        capsule_b=np.array([c[2] for c in GO2_SELF_FILTER_CAPSULES]),
        capsule_off_b=np.array([c[3] for c in GO2_SELF_FILTER_CAPSULES], dtype=np.float64),
        capsule_radius=np.array([c[4] for c in GO2_SELF_FILTER_CAPSULES], dtype=np.float64),
        q_test=np.stack(q_used),
        cap_pos_in_base=np.stack(pos_base),
        cap_quat_w=np.stack(quat_w),
        base_quat_w=np.stack(base_quat),
    )
    ok = True
except Exception:  # noqa: BLE001
    import traceback

    traceback.print_exc()

# 함정: simulation_app.close() 가 예외/종료코드를 삼킨다. 닫기 전에 결과를 못박는다.
print(f"[RESULT] {'OK' if ok else 'FAILED'}", flush=True)
simulation_app.close()
