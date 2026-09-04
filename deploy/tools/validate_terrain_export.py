"""내보낸 MuJoCo 지형이 IsaacLab 지형과 정말 같은가 — 기준선의 발 위치로 검증.

지금까지의 3-5 게이트는 "점군 → 월드" 와 "scandots" 를 **내가 내보낸 격자**와 견줬다.
둘 다 같은 격자를 쓰므로 격자 자체가 틀렸으면 통과한다. 이 검증은 다르다:
IsaacLab PLAY 기준 주행에서 **접지한 발의 월드 좌표**는 곧 IsaacLab 지형 표면이다.
그것을 내보낸 격자와 견주면 지형 자체가 검증된다.

같이 본다: 실제 계단 단차와 경사각. PLAY 설정은
  trapezoid_stairs  step_height = 0.05 + 0.18*difficulty → 0.7 에서 0.176 m
  slope             slope_angle = 10 + 27*difficulty     → 0.7 에서 28.9°
"""
import sys

import numpy as np

PARKOUR = "/home/seo-jinwoo/workspace/codes/unitree_rl_lab/deploy/parkour"
sys.path.insert(0, PARKOUR)
from em_sidecar.kinematics import Go2Kinematics, quat_to_mat  # noqa: E402
from em_sidecar.tests.test_live_terrain import META, Terrain  # noqa: E402

CONTRACT = f"{PARKOUR}/contract"
kin = Go2Kinematics(f"{CONTRACT}/em_geometry.npz")
FK = [str(s) for s in np.load(f"{CONTRACT}/em_geometry.npz")["fk_names"]]
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
IDX = [FK.index(f) for f in FEET]
FOOT_R = 0.0234          # IsaacLab 실측(dump_stand_height): 발 원점이 지면 위 23.4 mm

terr = Terrain(META)
SPAWN = terr.spawn
d = np.load("/home/seo-jinwoo/.claude/jobs/c76485cb/tmp/long_trace.npz")
q = d["joint_pos"][:, 0, :]
bp = d["root_pos_w"][:, 0, :]
bq = d["root_quat_w"][:, 0, :]
fc = np.linalg.norm(d["foot_contact_forces"][:, 0], axis=-1)
el = d["episode_length"][:, 0]
t = np.arange(len(q)) * 0.02
ep1 = el > 5

print("[1] IsaacLab 접지 발 z  vs  내보낸 격자 높이")
rows = []
for k in np.where(ep1)[0]:
    pb, _ = kin.link_poses_base(q[k])
    R = quat_to_mat(bq[k])
    fw = bp[k] + pb[IDX] @ R.T
    for j in range(4):
        if fc[k, j] > 20.0:                     # 확실히 하중이 실린 발만
            wx, wy = fw[j, 0] - SPAWN[0], fw[j, 1] - SPAWN[1]
            h = terr.height(np.array([wx]), np.array([wy]))[0]
            if np.isfinite(h):
                rows.append((t[k], wx, fw[j, 2] - FOOT_R, h))
R = np.array(rows)
err = R[:, 2] - R[:, 3]
print(f"  접지 표본 {len(R)}개 (스폰 기준 x {R[:,1].min():+.2f} ~ {R[:,1].max():+.2f} m)")
print(f"  발 접촉점 z − 격자 z :  평균 {err.mean()*1000:+7.2f} mm   "
      f"표준편차 {err.std()*1000:6.2f} mm   |최대| {np.abs(err).max()*1000:6.1f} mm")
print(f"  {'구간별':>10} " + "".join(f"{v:>10}" for v in ("x<0", "0~2", "2~5", "5~8", ">8")))
bins = [(-99, 0), (0, 2), (2, 5), (5, 8), (8, 99)]
line = ""
for lo, hi in bins:
    m = (R[:, 1] >= lo) & (R[:, 1] < hi)
    line += f"{err[m].mean()*1000:10.1f}" if m.sum() > 3 else f"{'-':>10}"
print(f"  {'평균오차[mm]':>10} " + line)
ok1 = np.abs(err).mean() < 0.02

print("\n[2] IsaacLab 이 실제로 지나간 지형 단면 (기준선 궤적을 따라)")
print("    발 접촉점 z 를 x 로 5 cm 격자에 모아 본 것 = IsaacLab 지형 실측")
xb = np.arange(np.floor(R[:, 1].min() * 20) / 20, R[:, 1].max(), 0.05)
prof_il, prof_ex = [], []
for x0 in xb:
    m = (R[:, 1] >= x0) & (R[:, 1] < x0 + 0.05)
    prof_il.append(np.median(R[m, 2]) if m.sum() else np.nan)
    prof_ex.append(np.median(R[m, 3]) if m.sum() else np.nan)
prof_il, prof_ex = np.array(prof_il), np.array(prof_ex)
print(f"    {'x[m]':>6} {'IsaacLab[cm]':>13} {'내보낸격자[cm]':>15}  프로파일")
for i in range(0, len(xb), 2):
    if np.isnan(prof_il[i]):
        continue
    bar = "#" * max(0, int(round(prof_il[i] * 50)))
    print(f"    {xb[i]:6.2f} {prof_il[i]*100:13.1f} {prof_ex[i]*100:15.1f}  {bar}")

print("\n[3] 단차/경사 실측 (IsaacLab 발 접촉점 프로파일에서)")
v = prof_il[~np.isnan(prof_il)]
xv = xb[~np.isnan(prof_il)]
dz = np.diff(v)
dx = np.diff(xv)
rise = dz > 0.02
if rise.any():
    print(f"    상승 구간 총 높이 {v.max()-v.min():.3f} m")
    seg = (v.max() - v.min())
    xs = xv[np.argmax(v > v.min() + 0.02)]
    xe = xv[np.argmax(v >= v.max() - 0.02)]
    if xe > xs:
        print(f"    상승 시작 x={xs:.2f} → 정상 x={xe:.2f} (길이 {xe-xs:.2f} m) "
              f"→ 평균 경사 {np.degrees(np.arctan2(seg, xe-xs)):.1f}°")
    print(f"    5cm 구간당 최대 상승 {dz.max()*100:.1f} cm "
          f"(계단이면 단차 ≈ 이 값의 누적)")
print(f"\n[RESULT] {'OK — 내보낸 지형이 IsaacLab 과 일치' if ok1 else 'MISMATCH — 지형이 다르다'}")
