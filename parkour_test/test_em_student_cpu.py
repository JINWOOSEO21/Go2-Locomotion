"""EM student 파이프라인 CPU 전용 단위 테스트 (Isaac Sim / GPU / cupy 불필요).

VRAM 이 바쁠 때도 돌릴 수 있다. 대상:
1. L1 스캔 운동학 (l1_scan_ray_caster.py 의 순수 텐서 함수 2개 — AST 추출):
   유효 반주기 / 프레임 구조(18창 x 120점 = 0.1s) / 반구 방향 / 결정론.
2. self-hit 필터 ray_capsule_penetrates (AST 추출): 관통/비관통/근거리 t0 스킵.
3. elevation_map_backend.sample_scan_heights (직접 import — cupy 는 지연 import 라
   모듈 로드는 CPU 에서 안전): teacher 식 정규화, bilinear, invalid cascade
   (valid → upper_bound → 0), 보더 링 처리.

실행: python parkour_test/test_em_student_cpu.py
"""
import ast
import importlib.util
import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ---------------------------------------------------------------- AST 추출 유틸
def extract_from_sensor_file(names):
    """l1_scan_ray_caster.py 에서 순수 함수/스태틱메서드를 추출해 실행 가능하게.

    (파일 자체는 isaaclab import 가 있어 SimulationApp 없이 못 불린다.)
    """
    path = os.path.join(REPO, "parkour_isaaclab/sensors/l1_scan_ray_caster.py")
    tree = ast.parse(open(path).read())
    ns = {"torch": torch, "math": math}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
        if isinstance(node, ast.ClassDef) and node.name == "L1ScanRayCaster":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in names:
                    item.decorator_list = []
                    exec(compile(ast.Module(body=[item], type_ignores=[]), path, "exec"), ns)
    return [ns[n] for n in names]


# ---------------------------------------------------------------- 1. L1 운동학
def test_l1_pattern():
    dirs_fn, times_fn = extract_from_sensor_file(["directions_from_angles", "valid_frame_times"])
    R, sps, pitch_hz, yaw_T = 2160, 43200.0, 180.0, 1.0 / 11.0
    phase_p = torch.tensor([[123.4]], dtype=torch.float64)
    phase_y_deg = 77.0
    t_end = torch.tensor([[5.0]], dtype=torch.float64)
    t, alpha = times_fn(t_end, phase_p, R, pitch_hz, sps)

    alpha_from_t = torch.remainder(360.0 * pitch_hz * t + phase_p, 360.0) - 180.0
    in_fov = ((alpha_from_t > -90.0) & (alpha_from_t < 90.0)).all()
    consistent = torch.allclose(torch.deg2rad(alpha_from_t[0]), alpha, atol=1e-4)
    check("1a.l1_valid_halfcycle", bool(in_fov) and consistent)

    tt = t.view(18, 120)
    win_span = (tt[:, -1] - tt[:, 0]).max()
    win_gap = tt[1:, 0] - tt[:-1, 0]
    check(
        "1b.l1_frame_structure",
        abs(float(win_span) - (119 / sps)) < 1e-6
        and torch.allclose(win_gap, torch.full_like(win_gap, 1 / pitch_hz), atol=1e-6)
        and float(t.max()) <= 5.0,
        f"(창 길이 {float(win_span)*1000:.3f} ms, 창 간격 {float(win_gap[0])*1000:.3f} ms)",
    )

    beta = torch.deg2rad(360.0 * t / yaw_T + phase_y_deg)
    d = dirs_fn(alpha.unsqueeze(0).expand_as(t), beta, math.radians(45.0), math.radians(45.0))
    el = torch.rad2deg(torch.asin(d[..., 2].clamp(-1, 1)))
    norm_err = (d.norm(dim=-1) - 1).abs().max().item()
    check(
        "1c.l1_hemisphere",
        norm_err < 1e-5 and float(el.min()) >= -1e-3 and float(el.max()) <= 90.001,
        f"(|1-norm| {norm_err:.1e}, elevation {float(el.min()):.2f}..{float(el.max()):.2f} deg)",
    )

    t_r, _ = times_fn(t_end, phase_p, R, pitch_hz, sps)
    check("1d.l1_deterministic", torch.equal(t, t_r))


# ---------------------------------------------------------------- 2. self-hit 필터
def test_capsule_filter():
    (pen_fn,) = extract_from_sensor_file(["ray_capsule_penetrates"])
    # 캡슐: 원점 중심 x축 선분 (-0.2,0,0)-(0.2,0,0), r=0.11 (Go2 몸통과 동일 규격)
    seg_a = torch.tensor([[-0.2, 0.0, 0.0]])
    seg_b = torch.tensor([[0.2, 0.0, 0.0]])
    origins = torch.tensor([[0.28, 0.0, 0.10]]).expand(1, 3, 3).clone()  # 마운트 근사
    dirs = torch.tensor(
        [
            [
                [-1.0, 0.0, 0.0],   # 뒤로 수평 — 몸통 관통
                [1.0, 0.0, 0.0],    # 앞으로 — 멀어지므로 비관통
                [0.0, 0.0, 1.0],    # 상방 — 비관통
            ]
        ]
    )
    t_hit = torch.tensor([[2.0, 2.0, 2.0]])
    hit = pen_fn(origins, dirs, t_hit, seg_a, seg_b, 0.11)
    check("2a.capsule_hit_pattern", hit.tolist() == [[True, False, False]], f"({hit.tolist()})")

    # t0(0.12m) 스킵: 캡슐 축 위(내부)에서 상방으로 쏘면 t0 이후 구간은 반경(0.11) 밖
    # → 비관통. t0=0 으로 끄면 시작점이 내부라 관통 판정.
    o_in = torch.zeros(1, 1, 3)
    d_up = torch.tensor([[[0.0, 0.0, 1.0]]])
    t2 = torch.tensor([[2.0]])
    skip_on = pen_fn(o_in, d_up, t2, seg_a, seg_b, 0.11)
    skip_off = pen_fn(o_in, d_up, t2, seg_a, seg_b, 0.11, t0=0.0)
    check("2b.capsule_t0_skip", (not bool(skip_on.any())) and bool(skip_off.all()),
          f"(t0=0.12: {skip_on.tolist()}, t0=0: {skip_off.tolist()})")


# ---------------------------------------------------------------- 3. 샘플링
def load_backend_module():
    path = os.path.join(REPO, "parkour_isaaclab/envs/mdp/elevation_map_backend.py")
    spec = importlib.util.spec_from_file_location("em_backend_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sampling():
    mod = load_backend_module()
    fn = mod.sample_scan_heights
    res, cell_n = 0.1, 34
    N, P = 1, 1

    def make(center_z=0.0):
        maps = torch.zeros(N, 7, cell_n, cell_n)
        centers = torch.tensor([[0.0, 0.0, center_z]])
        return maps, centers

    # (a) 전셀 valid, 높이 0 (center 상대) → h_abs=center_z. base_z=0.3+cz → h_obs=0
    cz = 1.7
    maps, centers = make(cz)
    maps[:, 2] = 1.0
    h, vf = fn(maps, centers, torch.zeros(N, P, 2), torch.tensor([0.3 + cz]), res, cell_n)
    check("3a.flat_teacher_zero", abs(float(h)) < 1e-6 and float(vf) > 0.99, f"(h={float(h):.4f})")

    # (b) teacher 식: 셀 높이 +0.2 (지형이 20cm 위) → h_obs = base_z − (0.2+cz) − 0.3
    maps[:, 0] = 0.2
    h, _ = fn(maps, centers, torch.zeros(N, P, 2), torch.tensor([0.3 + cz]), res, cell_n)
    check("3b.height_formula", abs(float(h) + 0.2) < 1e-6, f"(h={float(h):.4f}, 기대 -0.2)")

    # (c) bilinear: 인접 두 셀 0 / 0.1, 그 중간점 → 0.05
    maps, centers = make(0.0)
    maps[:, 2] = 1.0
    i0 = cell_n // 2  # x=0.05 근방 셀 (연속좌표 u=i0+0.5 ↔ x=(i0+0.5-17)*0.1)
    maps[:, 0, i0, :] = 0.0
    maps[:, 0, i0 + 1, :] = 0.1
    x_mid = (i0 + 1.0 - 0.5 * cell_n) * res  # 두 셀 중심의 중간
    h, _ = fn(maps, centers, torch.tensor([[[x_mid, 0.0]]]), torch.tensor([0.3]), res, cell_n)
    check("3c.bilinear_mid", abs(float(h) + 0.05) < 1e-6, f"(h={float(h):.4f}, 기대 -0.05)")

    # (d) cascade 2단계: valid 없음 + upper_bound 만 → ub 사용
    maps, centers = make(0.0)
    maps[:, 5] = 0.15
    maps[:, 6] = 1.0
    h, vf = fn(maps, centers, torch.zeros(N, P, 2), torch.tensor([0.3]), res, cell_n)
    check("3d.cascade_upper_bound", abs(float(h) + 0.15) < 1e-6 and float(vf) < 1e-6, f"(h={float(h):.4f})")

    # (e) cascade 3단계: 아무것도 없음 → 0 (base 기준 평지 가정)
    maps, centers = make(0.0)
    h, _ = fn(maps, centers, torch.zeros(N, P, 2), torch.tensor([1.0]), res, cell_n)
    check("3e.cascade_fallback_zero", abs(float(h)) < 1e-9)

    # (f) 보더 링/맵 밖은 invalid → fallback 0
    maps, centers = make(0.0)
    maps[:, 2] = 1.0
    far = torch.tensor([[[10.0, 0.0]]])
    h, vf = fn(maps, centers, far, torch.tensor([0.3]), res, cell_n)
    check("3f.outside_map_fallback", abs(float(h)) < 1e-9 and float(vf) < 1e-6)

    # (g) clip 범위 ±1
    maps, centers = make(0.0)
    maps[:, 2] = 1.0
    maps[:, 0] = -5.0
    h, _ = fn(maps, centers, torch.zeros(N, P, 2), torch.tensor([0.3]), res, cell_n)
    check("3g.clip", abs(float(h) - 1.0) < 1e-6)


if __name__ == "__main__":
    test_l1_pattern()
    test_capsule_filter()
    test_sampling()
    print("=" * 60)
    print(f"PASS: {PASS}")
    print(f"FAIL: {FAIL}")
    sys.exit(0 if not FAIL else 1)
