"""Phase 3′ 회귀 검증: BatchedElevationMapBackend vs stock 인스턴스 루프.

같은 합성 입력(smoke_em_backend 의 지형/라이다 생성기 재사용)을 두 백엔드에
동시에 먹이고 비교한다. 주의: em_cupy 의 upper_bound 기록은 비원자적 write
경쟁(여러 ray 가 같은 셀에 plain write)이라 **stock 끼리도 실행마다 값이
다르다** (REG_FLOOR=1 로 실측: ub ~245mm, 파생 elevation ~25mm). 따라서
비트단위 비교는 무의미하고, 판정은 결정적/통계적 성질로 한다:

1. 커버리지 마스크(is_valid, is_upper_bound) 일치 < 0.5%
2. tick 0 의 elevation 완전 일치 (< 1mm) — 경쟁 피드백이 끼기 전의
   transform→insert→Kalman→average 경로가 동일함을 입증
3. 해석적 정답 대비 정확도: 두 백엔드 모두 RMSE(valid) < 0.05m 이고
   서로 5mm 이내 — 경쟁으로 갈라진 셀들이 품질에 영향 없음을 입증

실행: python scripts/emcupy_check/regression_batched.py  (GPU 필요, Isaac 불필요)
노이즈 플로어(stock vs stock) 측정: REG_FLOOR=1 로 실행.
"""
import argparse
import importlib.util
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_module(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=25)
    ap.add_argument("--rays", type=int, default=2160)
    ap.add_argument("--bench-envs", type=int, default=192)
    ap.add_argument("--out", default="/tmp/emcupy_regression.txt")
    args = ap.parse_args()

    be = load_module("parkour_isaaclab/envs/mdp/elevation_map_backend.py", "em_backend_standalone")
    sm = load_module("scripts/emcupy_check/smoke_em_backend.py", "em_smoke_standalone")

    device = "cuda"
    n, r = args.num_envs, args.rays
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    if os.environ.get("REG_FLOOR"):
        # 노이즈 플로어 측정: stock vs stock (같은 입력, 다른 인스턴스).
        # em_cupy 의 upper_bound 기록은 비원자적 write 경쟁이라 자체적으로
        # 비결정적이다 — 이 모드로 그 크기를 잰다.
        stock = be.ElevationMapBackend(num_envs=n, device=device)
        bat = be.ElevationMapBackend(num_envs=n, device=device)
    else:
        stock = be.ElevationMapBackend(num_envs=n, device=device)
        bat = be.BatchedElevationMapBackend(num_envs=n, device=device)
    offsets = sm.make_scandot_offsets(device)

    tick0_elev_err, max_layer_mismatch = 0.0, 0.0
    sq_s, cnt_s, sq_b, cnt_b = 0.0, 0, 0.0, 0  # GT 대비 RMSE 누적 (워밍업 이후)
    for k in range(args.ticks):
        bx = -0.5 + 0.02 * k
        base_z = 0.3 + float(sm.terrain_height(torch.tensor([bx])))
        base_pos = torch.tensor([[bx, 0.0, base_z]], device=device).expand(n, 3).contiguous()
        t_s = base_pos + torch.tensor([[0.28, 0.0, 0.10]], device=device)
        R_base = torch.eye(3, device=device).expand(n, 3, 3).contiguous()
        R_off = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))
        R_s = torch.matmul(R_base, R_off.expand(n, 3, 3))
        dirs_w = sm.hemisphere_dirs(r, device, seed=k).unsqueeze(0).expand(n, r, 3)
        origins = t_s.unsqueeze(1).expand(n, r, 3)
        t_hit = sm.cast_rays(origins, dirs_w)
        finite = torch.isfinite(t_hit) & (t_hit < 10.0) & (t_hit > 0.12)
        pts_w = origins + dirs_w * torch.where(finite, t_hit, torch.zeros_like(t_hit)).unsqueeze(-1)
        p_rel = pts_w - t_s.unsqueeze(1)
        p_s = torch.einsum("nij,nri->nrj", R_s, p_rel)
        pts_list = [p_s[i][finite[i]] for i in range(n)]

        stock.update([p.clone() for p in pts_list], R_s, t_s, base_pos, R_base)
        bat.update([p.clone() for p in pts_list], R_s, t_s, base_pos, R_base)

        Ls, Lb = stock.layers(), bat.layers()
        valid_both = (Ls[:, 2] > 0.5) & (Lb[:, 2] > 0.5)
        e_elev = float((Ls[:, 0] - Lb[:, 0])[valid_both].abs().max()) if valid_both.any() else 0.0
        ub_both = (Ls[:, 6] > 0.5) & (Lb[:, 6] > 0.5)
        e_ub = float((Ls[:, 5] - Lb[:, 5])[ub_both].abs().max()) if ub_both.any() else 0.0
        e_var = float((Ls[:, 1] - Lb[:, 1])[valid_both].abs().max()) if valid_both.any() else 0.0
        cs = stock.centers()
        cb = bat.centers if isinstance(getattr(bat, "centers", None), torch.Tensor) else bat.centers()
        e_c = float((cs - cb).abs().max())
        for layer in (2, 6):
            mism = float(((Ls[:, layer] > 0.5) != (Lb[:, layer] > 0.5)).float().mean())
            max_layer_mismatch = max(max_layer_mismatch, mism)

        q = base_pos[:, :2].unsqueeze(1) + offsets.unsqueeze(0)
        hs, vs, us = stock.sample(q, base_pos[:, 2])
        hb, vb, ub = bat.sample(q, base_pos[:, 2])
        e_samp = float((hs - hb).abs().max())
        if k == 0:
            tick0_elev_err = e_elev
        if k < 3:
            log(f"  tick {k}: elev {e_elev*1000:.2f}mm ub {e_ub*1000:.2f}mm var {e_var:.4f} "
                f"center {e_c*1000:.2f}mm sample {e_samp*1000:.2f}mm")
        # 해석적 정답 대비 (경쟁 노이즈에 불변인 품질 지표)
        if k >= 5:
            px = base_pos[:, 0:1] + offsets[None, :, 0]
            h_gt = torch.clip(base_pos[:, 2:3] - sm.terrain_height(px) - 0.3, -1, 1)
            ms_, mb_ = vs > 1e-6, vb > 1e-6
            sq_s += float(((hs - h_gt)[ms_] ** 2).sum()); cnt_s += int(ms_.sum())
            sq_b += float(((hb - h_gt)[mb_] ** 2).sum()); cnt_b += int(mb_.sum())

    rmse_s = (sq_s / max(cnt_s, 1)) ** 0.5
    rmse_b = (sq_b / max(cnt_b, 1)) ** 0.5
    log(f"[reg  ] {args.ticks} ticks x {n} env")
    log(f"[gate1] valid/is_ub 마스크 불일치 최대 {max_layer_mismatch*100:.3f}% (< 0.5%)")
    log(f"[gate2] tick0 elevation 오차 {tick0_elev_err*1000:.2f} mm (< 1mm — 결정적 경로 일치)")
    log(f"[gate3] GT RMSE stock {rmse_s*1000:.1f} mm vs batched {rmse_b*1000:.1f} mm "
        f"(둘 다 < 50mm, 차이 < 5mm)")
    ok = (max_layer_mismatch < 0.005 and tick0_elev_err < 0.001
          and rmse_s < 0.05 and rmse_b < 0.05 and abs(rmse_s - rmse_b) < 0.005)

    # 벤치: batched 단독 tick 시간
    nb = args.bench_envs
    bat2 = be.BatchedElevationMapBackend(num_envs=nb, device=device)
    base_pos = torch.tensor([[0.0, 0.0, 0.3]], device=device).expand(nb, 3).contiguous()
    t_s = base_pos + torch.tensor([[0.28, 0.0, 0.10]], device=device)
    R_base = torch.eye(3, device=device).expand(nb, 3, 3).contiguous()
    R_s = torch.matmul(R_base, torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device)).expand(nb, 3, 3))
    dirs_w = sm.hemisphere_dirs(args.rays, device, seed=0).unsqueeze(0).expand(nb, args.rays, 3)
    origins = t_s.unsqueeze(1).expand(nb, args.rays, 3)
    t_hit = sm.cast_rays(origins, dirs_w)
    finite = torch.isfinite(t_hit) & (t_hit < 10.0) & (t_hit > 0.12)
    pts_w = origins + dirs_w * torch.where(finite, t_hit, torch.zeros_like(t_hit)).unsqueeze(-1)
    p_s = torch.einsum("nij,nri->nrj", R_s, pts_w - t_s.unsqueeze(1))
    pts_list = [p_s[i][finite[i]] for i in range(nb)]
    offs = sm.make_scandot_offsets(device)
    for _ in range(3):  # 워밍업(컴파일)
        bat2.update(pts_list, R_s, t_s, base_pos, R_base)
    torch.cuda.synchronize()
    t0 = time.time()
    reps = 20
    for _ in range(reps):
        bat2.update(pts_list, R_s, t_s, base_pos, R_base)
        bat2.sample(base_pos[:, :2].unsqueeze(1) + offs.unsqueeze(0), base_pos[:, 2])
    torch.cuda.synchronize()
    ms = (time.time() - t0) / reps * 1000
    log(f"[bench] batched {nb} env: {ms:.2f} ms/tick (stock 루프 실측 250.8 ms 대비)")

    log(f"[{'PASS' if ok else 'FAIL'}] regression_batched")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
