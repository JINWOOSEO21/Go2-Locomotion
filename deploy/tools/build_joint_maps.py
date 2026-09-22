"""관절 순서 3종(IsaacLab / MJCF / Unitree SDK)의 인덱스 매핑을 계산·검증한다.

입력:
  --isaaclab-json : deploy/tools/dump_isaaclab_joint_order.py 가 만든 JSON
  --mjcf          : Go2 MJCF (예: mujoco_menagerie/unitree_go2/go2.xml)
출력:
  매핑/기본자세/바디 차이를 담은 JSON. deploy/contract/deploy.yaml 의 근거다.

실행 환경이 둘로 갈린다 — IsaacLab 덤프는 env_isaaclab, 이 스크립트는 mujoco 가
있는 env_sim2real 에서 돌린다.

  python deploy/tools/build_joint_maps.py \
      --isaaclab-json /tmp/isaaclab_joint_order.json \
      --mjcf ../mujoco_menagerie/unitree_go2/go2.xml \
      --out /tmp/maps.json
"""

import argparse
import json

import mujoco

# unitree_rl_lab source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py 의
# UNITREE_GO2_CFG.joint_sdk_names. LowCmd.motor_cmd / LowState.motor_state 인덱스 규약.
SDK_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaaclab-json", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    il_dump = json.load(open(args.isaaclab_json))
    if not il_dump.get("ok"):
        raise SystemExit(f"IsaacLab 덤프가 실패한 상태다: {il_dump.get('error')}")
    il = il_dump["joint_names"]

    m = mujoco.MjModel.from_xml_path(args.mjcf)
    name = lambda obj, i: mujoco.mj_id2name(m, obj, i)  # noqa: E731
    mj = [name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    mj_act = [name(mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0]) for a in range(m.nu)]
    # 이게 깨지면 qpos 인덱스와 ctrl 인덱스를 따로 관리해야 한다.
    assert mj == mj_act, f"MJCF 힌지 순서 != actuator 순서\n  hinge={mj}\n  act={mj_act}"

    sdk = SDK_JOINT_NAMES
    assert sorted(il) == sorted(mj) == sorted(sdk), "세 순서의 관절 집합이 다르다"

    il_to_sdk = [sdk.index(n) for n in il]
    il_to_mj = [mj.index(n) for n in il]
    sdk_to_il = [il.index(n) for n in sdk]
    mj_to_il = [il.index(n) for n in mj]

    # 왕복 일관성
    for i, n in enumerate(il):
        assert sdk[il_to_sdk[i]] == n and mj[il_to_mj[i]] == n
        assert sdk_to_il[il_to_sdk[i]] == i and mj_to_il[il_to_mj[i]] == i

    q = il_dump["default_joint_pos"]
    res = {
        "isaaclab": il,
        "mjcf": mj,
        "sdk": sdk,
        "il_to_sdk": il_to_sdk,
        "il_to_mj": il_to_mj,
        "sdk_to_il": sdk_to_il,
        "mj_to_il": mj_to_il,
        "default_joint_pos_il": [round(v, 6) for v in q],
        "default_joint_pos_mj": [round(q[mj_to_il[i]], 6) for i in range(12)],
        "default_joint_pos_sdk": [round(q[sdk_to_il[i]], 6) for i in range(12)],
        # MJCF 의 기본 자세 키프레임은 IsaacLab 기본 자세와 다를 수 있다 (실제로 다르다).
        "mj_key_home_qpos_joints": ([round(float(v), 6) for v in m.key_qpos[0][7:]] if m.nkey else None),
        "il_body_names": il_dump["body_names"],
        "mj_body_names": [name(mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)],
        "actuators": il_dump["actuators"],
    }
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("isaaclab", "mjcf", "sdk", "il_to_sdk", "il_to_mj")}, indent=2))
    print(f"\nOK -> {args.out}")


if __name__ == "__main__":
    main()
