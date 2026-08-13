"""Phase 0/3 검증: em_cupy 백엔드 단독 구동 + 벤치마크 (Isaac Sim 불필요, GPU 필요).

합성 지형(평지 + x>0.5 에 0.2m 단차)에 대해 하향 반구 라이다를 해석적으로 시뮬레이션
해서 ElevationMapBackend 에 먹이고, teacher scandots 격자(12x11, 0.15m)에서 샘플한
높이를 해석 정답과 비교한다. 로봇은 +x 로 전진(0.02 m/tick = 0.2 m/s @10Hz).

판정 (계획서 §5-Q7 제안 기본값):
- 유효 셀 RMSE < 0.05 m
- invalid 비율 < 20 % (워밍업 10 tick 이후)

벤치마크: --num-envs 192 로 돌리면 Phase 3 의 tick 당 EM 총 시간을 근사한다
(점 수/맵 크기가 실제와 같고, 커널 런치 오버헤드가 지배 항이므로 근사가 유효).

실행:
  python scripts/emcupy_check/smoke_em_backend.py                  # Phase 0 (2 env)
  python scripts/emcupy_check/smoke_em_backend.py --num-envs 192   # Phase 3 벤치
결과는 stdout + --out 파일(기본 /tmp/emcupy_smoke.txt)에 쓴다.
"""
import argparse
import math
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 패키지 경로(parkour_isaaclab.envs.__init__)가 isaaclab→pxr 를 끌고 오므로
# (SimulationApp 없이는 import 불가), 백엔드 모듈만 파일에서 직접 로드한다.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "em_backend_standalone", os.path.join(REPO, "parkour_isaaclab/envs/mdp/elevation_map_backend.py")
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ElevationMapBackend = _mod.ElevationMapBackend


def make_scandot_offsets(device):
    """teacher 격자: GridPatternCfg(res 0.15, size [1.65,1.5]) + offset x 0.375."""
    xs = torch.arange(-0.825, 0.825 + 1e-9, 0.15, device=device) + 0.375
    ys = torch.arange(-0.75, 0.75 + 1e-9, 0.15, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")  # ordering="xy" 와 동일
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (132,2)


def terrain_height(x):
    return torch.where(x > 0.5, torch.full_like(x, 0.2), torch.zeros_like(x))


def cast_rays(origins, dirs):
    """합성 지형(z=0 평지, x>0.5 에 z=0.2 단·x=0.5 수직벽) 해석 레이캐스트.

    origins/dirs: (N,R,3). 반환 t_hit (N,R), miss 는 inf.
    """
    inf = torch.full_like(dirs[..., 0], float("inf"))

    def plane_t(z_plane):
        dz = dirs[..., 2]
        t = (z_plane - origins[..., 2]) / torch.where(dz.abs() > 1e-9, dz, torch.full_like(dz, 1e-9))
        return torch.where((dz < 0) & (t > 0), t, inf)

    t0 = plane_t(0.0)
    x0 = origins[..., 0] + dirs[..., 0] * t0
    t0 = torch.where(x0 <= 0.5, t0, inf)

    t1 = plane_t(0.2)
    x1 = origins[..., 0] + dirs[..., 0] * t1
    t1 = torch.where(x1 > 0.5, t1, inf)

    dx = dirs[..., 0]
    tw = (0.5 - origins[..., 0]) / torch.where(dx.abs() > 1e-9, dx, torch.full_like(dx, 1e-9))
    zw = origins[..., 2] + dirs[..., 2] * tw
    tw = torch.where((tw > 0) & (zw >= 0.0) & (zw <= 0.2), tw, inf)

    return torch.minimum(torch.minimum(t0, t1), tw)


def hemisphere_dirs(n_rays, device, seed):
    """하향 반구 무작위 방향 (L1 의 1.1회전 프레임 근사 — 방향 분포만 흉내)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    az = torch.rand(n_rays, generator=g) * 2 * math.pi
    el = torch.rand(n_rays, generator=g) * (math.pi / 2)  # 0(수평)..90(수직 하방)
    d = torch.stack(
        [torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), -torch.sin(el)], dim=-1
    )
    return d.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=30)
    ap.add_argument("--rays", type=int, default=2160)
    ap.add_argument("--out", default="/tmp/emcupy_smoke.txt")
    args = ap.parse_args()

    device = "cuda"
    n, r = args.num_envs, args.rays
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    t_build = time.time()
    backend = ElevationMapBackend(num_envs=n, device=device)
    torch.cuda.synchronize()
    log(f"[build] {n} instances, cell_n={backend.cell_n}, {time.time()-t_build:.2f}s")
    log(f"[vram ] torch allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB")

    offsets = make_scandot_offsets(device)
    rmse_hist, invalid_hist, tick_ms = [], [], []

    for k in range(args.ticks):
        # 로봇 전진: base (bx, 0, 0.3 + H(bx)), 센서 base+(0.28,0,0.10), 하향 마운트
        bx = -0.5 + 0.02 * k
        base_z = 0.3 + float(terrain_height(torch.tensor([bx])))
        base_pos = torch.tensor([[bx, 0.0, base_z]], device=device).expand(n, 3).contiguous()
        # env 마다 y 를 벌려 서로 다른 (그러나 동일 지형의) 맵을 만들지 않게 y=0 공유
        t_s = base_pos + torch.tensor([[0.28, 0.0, 0.10]], device=device)
        R_base = torch.eye(3, device=device).expand(n, 3, 3).contiguous()
        # 하향 마운트(X축 180°): R_s = R_base @ diag(1,-1,-1)
        R_off = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))
        R_s = torch.matmul(R_base, R_off.expand(n, 3, 3))

        dirs_w = hemisphere_dirs(r, device, seed=k).unsqueeze(0).expand(n, r, 3)
        origins = t_s.unsqueeze(1).expand(n, r, 3)
        t_hit = cast_rays(origins, dirs_w)
        finite = torch.isfinite(t_hit) & (t_hit < 10.0) & (t_hit > 0.12)
        pts_w = origins + dirs_w * torch.where(finite, t_hit, torch.zeros_like(t_hit)).unsqueeze(-1)
        # 센서 프레임으로: p_s = R_sᵀ (p − t)
        p_rel = pts_w - t_s.unsqueeze(1)
        p_s = torch.einsum("nij,nri->nrj", R_s, p_rel)
        pts_list = [p_s[i][finite[i]] for i in range(n)]

        torch.cuda.synchronize()
        t0 = time.time()
        backend.update(pts_list, R_s, t_s, base_pos, R_base)
        h_obs, valid_frac, ub_frac = backend.sample(
            base_pos[:, :2].unsqueeze(1) + offsets.unsqueeze(0), base_pos[:, 2]
        )
        torch.cuda.synchronize()
        tick_ms.append((time.time() - t0) * 1000)

        # 해석 정답: h_gt = clip(base_z − H(x) − 0.3, ±1)
        px = base_pos[:, 0:1] + offsets[None, :, 0]
        h_gt = torch.clip(base_pos[:, 2:3] - terrain_height(px) - 0.3, -1, 1)
        observed = valid_frac > 1e-6
        unknown = (valid_frac <= 1e-6) & (ub_frac <= 1e-6)  # 관측도 상한도 없음
        if observed.any():
            rmse = float(torch.sqrt(((h_obs - h_gt)[observed] ** 2).mean()))
        else:
            rmse = float("nan")
        rmse_hist.append(rmse)
        invalid_hist.append(float(unknown.float().mean()))

    warm = 10
    rmse_ss = sum(rmse_hist[warm:]) / len(rmse_hist[warm:])
    inv_ss = sum(invalid_hist[warm:]) / len(invalid_hist[warm:])
    ms_ss = sum(tick_ms[warm:]) / len(tick_ms[warm:])
    log(f"[tick ] mean {ms_ss:.1f} ms/tick ({ms_ss/n:.2f} ms/env) after warmup "
        f"(first tick {tick_ms[0]:.0f} ms incl. kernel compile)")
    log(f"[gate ] RMSE(valid) {rmse_ss:.4f} m (< 0.05 목표) | unknown {inv_ss*100:.1f}% (< 20% 목표)")
    log(f"[vram ] torch allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB "
        f"(cupy pool 은 별도 — nvidia-smi 로 총량 확인)")
    ok = rmse_ss < 0.05 and inv_ss < 0.20
    log(f"[{'PASS' if ok else 'FAIL'}] smoke_em_backend (num_envs={n})")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
