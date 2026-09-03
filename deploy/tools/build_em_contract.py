"""dump_em_geometry.py 의 실측 덤프에서 EM 사이드카용 순기구학 상수를 **역산**한다.

왜 역산인가
-----------
사이드카는 lowstate 의 관절각으로부터 self-filter 캡슐 끝점의 월드 좌표를 알아야
한다. 즉 Go2 순기구학이 필요하다. 후보는 셋이었다:

  (a) MuJoCo 를 사이드카에 링크해 mj_kinematics 로 푼다
      → env_isaaclab 에 mujoco 를 새로 깔아야 하고, 젯슨에도 얹어야 한다.
  (b) MJCF 의 body pos / joint axis 를 사람이 옮겨 적는다
      → 옮겨 적는 순간이 곧 버그가 들어오는 순간이다.
  (c) IsaacLab 실측 덤프에서 상수를 **풀어낸다**  ← 이것

(c) 는 아무 값도 옮겨 적지 않는다. 학습을 실제로 돌린 IsaacLab 의 바디 pose 가
유일한 출처이고, 산출된 상수가 그 pose 를 재현하는지 같은 자리에서 검증한다.

기구학 모델
-----------
다리마다 base → {leg}_hip → {leg}_thigh → {leg}_calf 의 회전 3관절 사슬이다.
부모 프레임에서 본 자식의 pose 는

    p_child = p0                       (관절각과 무관 — 축이 자식 원점을 지난다)
    R_child = R0 · Rot(axis, θ)

이고, 두 자세 i, j 에 대해

    R(θ_i)ᵀ R(θ_j) = Rot(axis, θ_j − θ_i)

이므로 축은 상대 회전에서 바로 나온다. R0 는 그 축으로 θ 를 되돌려 얻는다.
Head_upper / Head_lower(base 고정)와 {leg}_foot(calf 고정)은 관절과 무관한
고정 오프셋이라 평균으로 굳힌다.

산출물은 contract/em_geometry.npz — 사이드카가 런타임에 읽는 유일한 기하 파일이다.
"""
from __future__ import annotations

import argparse

import numpy as np

LEGS = ("FL", "FR", "RL", "RR")


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) → 3x3. IsaacLab 규약."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_axis_angle(a: np.ndarray, th: float) -> np.ndarray:
    """Rodrigues."""
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """회전행렬 → 축·각 벡터 (크기가 각도)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2.0 * np.sin(th)) * th


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="dump_em_geometry.py 산출 npz")
    ap.add_argument("--out", required=True, help="contract/em_geometry.npz")
    ap.add_argument("--tol_mm", type=float, default=5.0, help="FK 재현 게이트 [mm]")
    a = ap.parse_args()

    d = np.load(a.dump)
    body_names = [str(s) for s in d["capsule_bodies"]]
    bi = {n: i for i, n in enumerate(body_names)}
    joint_names = [str(s) for s in d["joint_names"]]
    ji = {n: i for i, n in enumerate(joint_names)}

    q_test = d["q_test"].astype(np.float64)  # (K, 12) IsaacLab 관절 순서
    pos_b = d["cap_pos_in_base"].astype(np.float64)  # (K, B, 3) base 프레임 위치
    quat_w = d["cap_quat_w"].astype(np.float64)  # (K, B, 4) 월드 회전
    base_q = d["base_quat_w"].astype(np.float64)  # (K, 4)
    K = q_test.shape[0]

    # 월드 회전을 base 프레임 회전으로 (R_base ᵀ R_body)
    R_base = np.stack([quat_to_mat(base_q[k]) for k in range(K)])
    R_in_base = np.zeros((K, len(body_names), 3, 3))
    for k in range(K):
        for b in range(len(body_names)):
            R_in_base[k, b] = R_base[k].T @ quat_to_mat(quat_w[k, b])

    report: list[str] = []

    def solve_link(child: str, parent: str, joint: str):
        """부모 프레임에서 본 자식의 (p0, R0, axis) 를 K 자세에서 역산한다."""
        ic, ip_ = bi[child], bi[parent]
        th = q_test[:, ji[joint]]
        # 위치: 부모 프레임 기준 자식 원점 — 관절각과 무관해야 한다.
        p_rel = np.einsum("kij,kj->ki", R_in_base[:, ip_].transpose(0, 2, 1),
                          pos_b[:, ic] - pos_b[:, ip_])
        p0 = p_rel.mean(0)
        p_spread = np.abs(p_rel - p0).max()

        # 회전: R_rel(θ) = R0 · Rot(axis, θ)
        R_rel = np.einsum("kij,kjl->kil", R_in_base[:, ip_].transpose(0, 2, 1), R_in_base[:, ic])
        # 자세 쌍마다 상대 회전의 축을 구해 평균낸다 (부호는 Δθ 로 맞춘다).
        axes = []
        for i in range(K):
            for j in range(i + 1, K):
                dth = th[j] - th[i]
                if abs(dth) < 1e-3:
                    continue
                w = log_so3(R_rel[i].T @ R_rel[j])
                nw = np.linalg.norm(w)
                if nw < 1e-6:
                    continue
                axes.append(w / nw * np.sign(dth))
        axis = np.mean(axes, axis=0)
        axis /= np.linalg.norm(axis)
        # R0 = R_rel(θ)·Rot(axis, θ)ᵀ — 자세마다 같아야 한다.
        R0s = np.stack([R_rel[k] @ rot_axis_angle(axis, th[k]).T for k in range(K)])
        R0 = R0s[0]
        R0_spread = max(np.abs(log_so3(R0.T @ R0s[k])).max() for k in range(K))
        report.append(
            f"  {parent:10s} -> {child:10s} ({joint:15s}) "
            f"axis=[{axis[0]:+.4f} {axis[1]:+.4f} {axis[2]:+.4f}] "
            f"p0=[{p0[0]:+.4f} {p0[1]:+.4f} {p0[2]:+.4f}] "
            f"p흔들림={p_spread*1e3:.3f}mm R0흔들림={np.rad2deg(R0_spread):.4f}deg"
        )
        return p0, R0, axis

    print("[1] 관절 사슬 역산 (부모→자식)")
    chain = {}
    for leg in LEGS:
        chain[f"{leg}_hip"] = ("base", f"{leg}_hip_joint", *solve_link(f"{leg}_hip", "base", f"{leg}_hip_joint"))
        chain[f"{leg}_thigh"] = (
            f"{leg}_hip", f"{leg}_thigh_joint",
            *solve_link(f"{leg}_thigh", f"{leg}_hip", f"{leg}_thigh_joint"),
        )
        chain[f"{leg}_calf"] = (
            f"{leg}_thigh", f"{leg}_calf_joint",
            *solve_link(f"{leg}_calf", f"{leg}_thigh", f"{leg}_calf_joint"),
        )
    print("\n".join(report))

    print("\n[2] 고정 링크 (관절 없음)")
    fixed = {}
    for child, parent in (("Head_upper", "base"), ("Head_lower", "base"),
                          *[(f"{leg}_foot", f"{leg}_calf") for leg in LEGS]):
        ic, ip_ = bi[child], bi[parent]
        p_rel = np.einsum("kij,kj->ki", R_in_base[:, ip_].transpose(0, 2, 1),
                          pos_b[:, ic] - pos_b[:, ip_])
        R_rel = np.einsum("kij,kjl->kil", R_in_base[:, ip_].transpose(0, 2, 1), R_in_base[:, ic])
        p0 = p_rel.mean(0)
        R0 = R_rel[0]
        spread = np.abs(p_rel - p0).max()
        r_spread = max(np.abs(log_so3(R0.T @ R_rel[k])).max() for k in range(K))
        fixed[child] = (parent, p0, R0)
        print(f"  {parent:10s} -> {child:10s} p0=[{p0[0]:+.4f} {p0[1]:+.4f} {p0[2]:+.4f}] "
              f"흔들림={spread*1e3:.3f}mm / {np.rad2deg(r_spread):.4f}deg")

    # --- 3. 게이트: 역산한 모델로 FK 를 돌려 덤프의 base 프레임 위치를 재현하는가 ---
    # 사이드카가 실제로 쓸 코드와 같은 순서로 계산한다.
    order: list[tuple[str, str, str | None]] = []  # (child, parent, joint|None)
    for leg in LEGS:
        order += [(f"{leg}_hip", "base", f"{leg}_hip_joint"),
                  (f"{leg}_thigh", f"{leg}_hip", f"{leg}_thigh_joint"),
                  (f"{leg}_calf", f"{leg}_thigh", f"{leg}_calf_joint")]
    order += [("Head_upper", "base", None), ("Head_lower", "base", None)]
    order += [(f"{leg}_foot", f"{leg}_calf", None) for leg in LEGS]

    p0_arr, R0_arr, ax_arr, par_idx, jnt_idx = [], [], [], [], []
    name_to_row = {"base": -1}
    for row, (child, parent, joint) in enumerate(order):
        name_to_row[child] = row
    for child, parent, joint in order:
        if joint is None:
            p0, R0 = fixed[child][1], fixed[child][2]
            axis, jj = np.array([0.0, 0.0, 1.0]), -1
        else:
            _, _, p0, R0, axis = chain[child]
            jj = ji[joint]
        p0_arr.append(p0)
        R0_arr.append(R0)
        ax_arr.append(axis)
        par_idx.append(name_to_row[parent])
        jnt_idx.append(jj)

    p0_arr = np.stack(p0_arr)
    R0_arr = np.stack(R0_arr)
    ax_arr = np.stack(ax_arr)
    par_idx = np.array(par_idx, dtype=np.int64)
    jnt_idx = np.array(jnt_idx, dtype=np.int64)
    fk_names = [c for c, _, _ in order]

    def fk(q: np.ndarray):
        """base 프레임 기준 (pos, R) — 사이드카 kinematics.py 와 동일한 알고리즘."""
        n = len(order)
        P = np.zeros((n, 3))
        R = np.zeros((n, 3, 3))
        for i in range(n):
            pp = np.zeros(3) if par_idx[i] < 0 else P[par_idx[i]]
            pR = np.eye(3) if par_idx[i] < 0 else R[par_idx[i]]
            P[i] = pp + pR @ p0_arr[i]
            Ri = R0_arr[i]
            if jnt_idx[i] >= 0:
                Ri = Ri @ rot_axis_angle(ax_arr[i], q[jnt_idx[i]])
            R[i] = pR @ Ri
        return P, R

    print("\n[3] 게이트 — 역산 모델의 FK vs IsaacLab 실측 (base 프레임 위치)")
    worst = 0.0
    for k in range(K):
        P, _ = fk(q_test[k])
        err = np.linalg.norm(P - pos_b[k, [bi[n] for n in fk_names]], axis=1)
        worst = max(worst, err.max())
        print(f"  자세 {k}: max {err.max()*1e3:8.4f} mm   mean {err.mean()*1e3:8.4f} mm")
    ok = worst * 1e3 < a.tol_mm
    print(f"\n  최대 오차 {worst*1e3:.4f} mm  (허용 {a.tol_mm} mm) → {'PASS' if ok else 'FAIL'}")

    np.savez(
        a.out,
        # scandots 격자 (policy obs 순서 그대로)
        scan_offsets_xy=d["scan_offsets_xy"],
        scan_resolution=d["scan_resolution"],
        # 순기구학 (base 프레임 기준 사슬)
        fk_names=np.array(fk_names),
        fk_parent=par_idx,
        fk_joint=jnt_idx,
        fk_p0=p0_arr,
        fk_R0=R0_arr,
        fk_axis=ax_arr,
        joint_names=d["joint_names"],
        # self-filter 캡슐
        capsule_a=d["capsule_a"],
        capsule_off_a=d["capsule_off_a"],
        capsule_b=d["capsule_b"],
        capsule_off_b=d["capsule_off_b"],
        capsule_radius=d["capsule_radius"],
        # 회귀용 원본 (사이드카 단위테스트가 다시 대조한다)
        q_test=q_test,
        cap_pos_in_base_ref=np.stack([pos_b[k, [bi[n] for n in fk_names]] for k in range(K)]),
    )
    print(f"\n  → {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
