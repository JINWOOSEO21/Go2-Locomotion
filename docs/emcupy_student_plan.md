# Student v1.3 계획서: L1 LiDAR + elevation_mapping_cupy 기반 terrain latent

작성일: 2026-08-12 (세션 v1.3). 대상 브랜치: master(`ad11a4f`) 기준 신규 작업.
개정: v1.2 (2026-08-12) — Q2 재개정: **scandots 그리드 현행 유지(12×11, 0.15m)**,
teacher 재학습 없음. Q5 지연 모델링 제외. 확정 사항은 §5 참조.

## 0. 목표

기존 student의 depth camera + ConvNet+GRU 파이프라인을 다음으로 대체한다.

```
L1 LiDAR (MultiMeshRayCaster, 11Hz yaw / 180Hz pitch, 21,600 pt/s)
   → self-hit 필터 (부풀린 로봇 캡슐, GT proprioception)
   → elevation_mapping_cupy (노이즈 부여된 GT odometry, per-env 인스턴스)
   → teacher scandots 그리드(12×11=132점, 0.15m, base 기준 +1.2/−0.45/±0.75m)에서 높이 샘플
   → teacher scan_encoder MLP (132→128→64→32 Tanh, teacher weight로 init 후 전체 fine-tune)
   → 32차원 terrain latent → DAgger distillation (기존 action-imitation L2 유지)
```

scandots 그리드와 teacher는 **현행 그대로 유지**한다(Q2 최종 결정) — 기존 teacher
ckpt(`model_16999.pt`)를 그대로 사용하며 재학습 없음. EM 내부 그리드 해상도(0.1m)는
scandots 간격(0.15m)과 별개의 값이다(§2.4).

정책은 50Hz로 동작하고, terrain latent는 센서 자연 프레임 주기(0.1s, 10Hz)마다 갱신되며
그 사이 4 step은 최신 값을 재사용한다(Q1 결정).

---

## 1. 사전 확인 결과

### 1.1 Teacher scandots / encoder (master, 현행 유지)

- 그리드: `GridPatternCfg(resolution=0.15, size=[1.65, 1.5])` + offset x=+0.375
  → x: base 기준 −0.45…+1.20m 12점, y: ±0.75m 11점, **12×11=132점**
  (`parkour_teacher_cfg.py:17-27`). 변경 없음.
- flatten 순서: `ordering="xy"` → shape (n_y=11, n_x=12), index = `y_idx*12 + x_idx`
  (`scripts/rsl_rl/multicam_recorder.py:82-110`의 `resolve_scandots_grid` 참조).
- 높이 식: `clip(pos_w_z − hit_z − 0.3, −1, 1)` (`observations.py:139-140`).
  **`data.pos_w`는 cfg offset(z=+20)을 포함하지 않음을 IsaacLab 소스에서 확인**
  (`ray_caster.py:155` — `self._offset`은 추적 prim 자체 오프셋, base면 identity; cfg offset은
  ray 시작점에만 적용). 즉 실질적으로 **`clip(base_z − hit_z − 0.3, −1, 1)`**.
- ray_alignment="yaw": 그리드는 로봇 yaw만 따라 회전 (roll/pitch 무시).
- 갱신: `common_step_counter % 5 == 0`(10Hz)에서만 재계산, 캐시 유지 (`observations.py:64-67`).
  노이즈 없음. → Q1 결정(10Hz)과 정확히 일치하는 주기.
- encoder: `Linear(132→128) ELU → Linear(128→64) ELU → Linear(64→32) Tanh`
  (`actor_critic_with_encoder.py:55-70`).
- checkpoint 키: `model_state_dict["actor.scan_encoder.{0,2,4}.{weight,bias}"]`
  (teacher `model_16999.pt`에서 shape 확인 완료 — 그대로 사용).
- **주입 지점**: `Actor.forward(obs, hist_encoding, scandots_latent=None)` — `None`이면
  obs[53:185]를 자체 scan_encoder에 통과 (`actor_critic_with_encoder.py:89-101`).
  distillation 시 `depth_actor = deepcopy(teacher actor)` 로드
  (`on_policy_runner_with_extractor.py:639-648`) → scan_encoder가 teacher weight로
  자동 초기화되고, 옵티마이저에 포함되어 전체 fine-tune됨(Q6 결정과 일치).

### 1.2 Student 현행 구조 (master)

- obs `policy` 그룹 753 = prop 53 | scan 132 | priv_explicit 9 | priv_latent 29 | history 530.
  teacher/student 동일 벡터. 변경 없음.
- depth encoder(ConvNet+GRU) forward는 `% 5 == 0`(10Hz)에서만, 사이 4 step은 같은 latent
  재사용 (`on_policy_runner_with_extractor.py:365-374`) — "최신 latent 재사용" 인프라 존재.
- 손실: `(teacher_action.detach() − student_action).norm(p=2, dim=1).mean()` 단독,
  Adam lr 1e-3, iteration당 120 env-step 수집 후 1회 업데이트 (`distillation_with_extractor.py:58-72`).
- 학습 기본값: num_envs 192, 50Hz(sim.dt 0.005 × decimation 4), max_iterations 5000.
  RTX 3060에서 192 env ≈ 11.1GB / 5k iter ≈ 11.4h (기존 실측).

### 1.3 L1 LiDAR 모델 (`lidar` 브랜치)

- **단일 파일 이식 가능**: `parkour_isaaclab/sensors/l1_scan_ray_caster.py`(240줄) +
  `parkour_isaaclab/sensors/__init__.py` + `default_cfg.py`의 `GO2_LIDAR_CFG` 블록.
  의존성은 `isaaclab.sensors`, `isaaclab.utils.math`뿐. master에 `parkour_isaaclab/sensors/`가
  없어 충돌 없이 드롭인 가능.
- 스캔 패턴: unilidar_sdk 듀얼모터 운동학. pitch 180Hz / yaw 11Hz / 43,200 sample/s 중
  α∈(−90°,90°)인 절반만 유효 → 21,600 pt/s. 매 업데이트 `_update_ray_infos`에서 방향 재계산.
- 프레임: **0.1s당 2160 ray** (18개 유효 윈도, 정확히 타일링). 0.1s에 yaw 1.1회전이므로
  **매 프레임이 이미 360° 전방위 커버** — 별도 sweep 누적 버퍼 불필요. (Q1: 이 프레임을
  그대로 사용.)
- 마운트: base 기준 (0.28, 0, 0.10), 돔 축 완전 하향(rot=(0,1,0,0)). max_distance 10m.
- 출력: `data.ray_hits_w` (N_env, 2160, 3) **world frame**, miss는 `inf`.
- 타깃: `/World/ground` + **자기 로봇 collision mesh**(per-env, 타 로봇 미포함).
  로봇 body에 ray가 막히는 물리적 self-occlusion 재현. `update_mesh_ids=False` 유지
  (upstream shape 버그 회피 — 켜지 말 것).
- **self-hit 필터 기구현**: `GO2_SELF_FILTER_CAPSULES`(14개 캡슐: base r=0.11, head r=0.06,
  hip×4 r=0.07, thigh-calf×4 r=0.055, calf-foot×4 r=0.05 — 실제보다 부풀린 값) +
  `ray_capsule_penetrates(..., t0=0.12)` (ray 경로 관통 판정, 마운트가 body 캡슐 내부인
  문제 회피). 캡슐 끝점은 GT link pose에서 계산. 실측 self-hit 비율 ~61%.
- 센서 자체 노이즈 없음. lidar 브랜치는 다운스트림에서 range σ=0.02m를 부여했음.
- 가져오지 않을 것: `SimLegOdometry`/KF, `LidarElevationMap`, `ExpandedScanEncoder`,
  [h|var|ub] 3채널 obs — 전부 이번 설계로 대체.

### 1.4 elevation_mapping_cupy (클론된 repo, ROS1 main 브랜치 `20a8a26`)

- **코어는 ROS 없이 동작**: `.../script/elevation_mapping_cupy/`의 `ElevationMap`/`Parameter`는
  순수 python+cupy (CI도 ROS 없이 pytest). `sys.path`에 `.../script` 추가 필요.
  torch↔cupy zero-copy 인터롭 기존 코드에 존재 (`traversability_filter.py:29,42`).
- 입력: `input_pointcloud(points(N,3) [sensor frame], ["x","y","z"], R(3,3), t(3)
  [map frame 기준 센서 pose], position_noise, orientation_noise)`.
  `move_to(base_pos, R_base)`로 맵 recenter. 그리드는 map frame 축정렬(회전 없음),
  높이는 **map center z 상대값**(절대값 = `elevation + center[2]`).
- 조회: `elevation_map[0]`(height) / `[1]`(variance) / `[2]`(is_valid) / `[5]`(upper_bound) /
  `[6]`(is_upper_bound) cupy 배열 직접 접근 + `torch.as_tensor` zero-copy가 유일한 실용 경로.
  `get_map_with_name_ref`는 GPU→CPU 복사라 학습 루프 사용 금지.
  임의 (x,y) 샘플 API 없음 — 자체 gather 구현(단순).
- **미관측 셀의 원래 처리(Q3 배경)**: 셀은 is_valid 플래그를 가질 뿐이며, 라이브러리는
  내보낼 때 invalid 셀을 NaN으로 표기하고 값 결정은 소비자에게 맡긴다. 구멍 메움용
  inpainting 플러그인(cv2 기반)이 있으나 CPU 경유라 학습 루프에 부적합. → §2.1의
  cascade 채움 규칙이 우리 쪽 대응.
- 인덱싱: `row=(x−cx)/res + cell_n/2`, `col=(y−cy)/res + cell_n/2` (row↔x, col↔y).
- 자체 self-filter 없음(실기 ANYmal도 외부 필터 노드 사용) — 우리 캡슐 필터가 표준 위치.
- **단일 맵 설계 — 배치 미지원.** 커널 소스는 파라미터 동일 시 캐시 재사용되어 인스턴스
  N개 생성 자체는 저렴. 메모리도 작은 맵(3.2m@0.1m, cell_n=34) 기준 인스턴스당 ~0.13MB.
  **진짜 병목은 커널 런치 오버헤드**: 업데이트 1회 = 인스턴스당 커널 ~6회, 192 env 직렬
  python 루프 → 라운드당 천 단위 런치.
- 확인된 함정:
  1. import 시 **프로세스 전역 cupy managed-memory allocator** 설정
     (`elevation_mapping.py:45-46`) — import 후 allocator 교체 필요.
  2. `input_pointcloud`가 `t`를 **in-place 변경** — 복사본 전달 필수.
  3. `move()`와 `move_to()`의 grid shift 부호가 상반 — **`move_to`만 사용**.
  4. `param.update()` 필수 호출(cell_n 계산), weight/plugin 경로는 절대경로로.
  5. visibility cleanup은 점당 최대 `max_ray_length/resolution`회 반복 — 기본 yaml값
     (10m/0.04m)이면 폭발. 우리 설정(3.0m/0.1m → ≤42회)으로 축소.
  6. GT 기반 odometry면 `enable_drift_compensation=False` → error_counting 커널 생략(~1/3 절약).
  7. traversability CNN이 업데이트 경로에 내장 — 불필요하므로 우회 패치.
  8. `min_valid_distance`(기본 0.3/0.5m)는 캡슐 필터 t0=0.12와 정합하게 0.10으로.

---

## 2. 설계

### 2.1 파이프라인 (매 EM tick = 5 policy step = 0.1s, 10Hz)

```
1. lidar.data.ray_hits_w 읽기 (N,2160,3 world, miss=inf) — 직전 0.1s 프레임(1.1회전)
2. self-hit 마스크: ray_capsule_penetrates × 14 캡슐 (GT link pose, 부풀린 반경)
3. 유효점 = finite ∧ ¬self_hit → sensor frame으로 변환: p_s = R_sᵀ(p_w − t_s)
4. odometry 노이즈 부여 (Q4): EM에 넘기는 6D pose에만 백색 잡음
   (기본값 제안: σ_xyz=0.01m, σ_rpy=0.5°; policy obs는 GT 유지)
   → t̃_s, R̃_s, base 포즈도 동일 잡음 계열로 t̃_b, R̃_b
5. per-env: em.move_to(t̃_b, R̃_b); em.input_pointcloud(p_s, R̃_s, t̃_s.copy(), 0, 0)
   (주기적으로 update_variance/update_time)
6. 132점 샘플: scandot 위치 = base_xy + yaw회전(그리드 오프셋) [ray_alignment="yaw" 동일]
   h_abs = bilinear(elevation) + center[2]
   invalid 셀 채움(Q3 cascade): is_valid → h, 아니면 is_upper_bound → upper_bound, 아니면 0
   h_obs = clip(base_z − h_abs − 0.3, −1, 1)  ← teacher 식과 동일
7. student obs 캐시 갱신: scan 슬라이스 [53:185]를 h_obs로 (다음 tick까지 유지)
[매 policy step, 50Hz]
8. student action = depth_actor(obs_em, hist_encoding=True, scandots_latent=None)
   → 자체 scan_encoder(=teacher init, 전체 fine-tune)가 latent 계산
   teacher action = policy.act_inference(원본 obs, GT scandots) → L2 imitation
```

지연 모델링(Q5): **하지 않음.** tick 종료 즉시 h_obs 사용.

### 2.2 L1 센서 이식

`git show lidar:...`로 아래를 v1.3 브랜치에 복사:

- `parkour_isaaclab/sensors/{__init__.py, l1_scan_ray_caster.py}` (그대로)
- `default_cfg.py`에 `GO2_LIDAR_CFG` 블록 (그대로: 마운트 (0.28,0,0.10) 하향, 2160 ray, 10m)
- `parkour_student_cfg.py`: `camera` → `lidar = GO2_LIDAR_CFG` 교체, `update_period` 설정
- 검증 도구: `scripts/lidar_sim/viz_l1_scan.py`, `test_l1_pattern()` (선택)

소비는 EM tick(`%5`)에서만 — 프레임(0.1s)과 소비 주기가 정확히 타일링되어 모든 점이
정확히 1회 사용된다.

### 2.3 Self-hit 필터

lidar 브랜치의 `GO2_SELF_FILTER_CAPSULES` + `ray_capsule_penetrates`를 그대로 사용.
캡슐 끝점은 GT link pose에서 매 tick 계산. 로봇 collision mesh는 raycast 타깃으로 유지
(필터="body에 맞은 점 제거", occlusion="body 뒤 지형이 안 보임" — 역할이 다르고 둘 다
실기와 일치).

### 2.4 elevation_mapping_cupy 통합

래퍼 모듈 `parkour_isaaclab/envs/mdp/elevation_map_backend.py`(신규)로 격리:

```python
Parameter (확정값):
  resolution=0.1, map_length=3.2          # EM 내부 그리드. scandot 최원점 √(1.2²+0.75²)=1.415m
                                          # + 여유; cell_n=34. scandots 간격 0.15m보다 조밀해
                                          # bilinear 샘플에 충분.
  sensor_noise_factor=0.05, mahalanobis_thresh=2.0
  min_valid_distance=0.10                 # 캡슐 필터 t0=0.12보다 작게
  max_height_range=1.0, ramped_* 기본값   # Q8: ceiling 필터 적용(ramp/stair라 무해)
  enable_drift_compensation=False
  enable_visibility_cleanup=True, max_ray_length=3.0   # Q3의 upper_bound 채움에 필요
  enable_overlap_clearance=False
  plugin 없음, subscriber_cfg={}, additional_layers=[], fusion=[]
```

- import 직후 cupy allocator 교체(함정 1), traversability CNN 우회(서브클래스 no-op).
- per-env 인스턴스 리스트, **env reset 시 해당 인스턴스 `clear()`** (teleport 대응).
- 좌표계: odom≡world(노이즈는 입력 pose에만). move_to가 base 추종 recenter.
- 성능 전략 2단계: stock 인스턴스 루프로 정확성 확보(Phase 3) → 병목이면 커널 3개
  (add_points/error_counting/average_map)에 env 배치 차원을 추가한 batched 포트(Phase 3′).

### 2.5 관측/학습 통합

- runner에서 obs 복제 후 scan 슬라이스 [53:185]만 EM 샘플로 교체해 student에 공급.
  teacher는 원본 obs(GT scandots) 사용 — 현행 구조와 동일한 이원화.
- `learn_vision` 기반의 단순화된 학습 루프 `learn_em`:
  - depth encoder/GRU/BPTT/hidden 워밍업 제거 (resume 스파이크 이슈 소멸).
  - 학습 파라미터 = `depth_actor.parameters()` (scan_encoder 포함, Q6: 전체 fine-tune).
  - 손실/옵티마이저/수집 길이(120)는 현행 유지.
  - h_obs 캐시는 tick에서만 갱신, 사이 step 재사용 (기존 `depth_latent` 패턴; 미초기화
    NameError 버그는 초기값 세팅으로 함께 수정).
- play/eval 동일 패턴 수정. 기존 depth 파이프라인은 신규 task 이름으로 분리·공존.

### 2.6 타이밍 (확정)

Q1 결정: **10Hz(5 step) 고정, 센서 자연 프레임(0.1s=1.1회전) 그대로 사용.** 모든 점이
정확히 1회 입력되고, 기존 `%5` 인프라 및 teacher scandots 갱신 주기와 일치. 정확 11Hz
절취 방식은 채택하지 않음.

---

## 3. 구현 단계

| Phase | 내용 | 검증 게이트 |
|---|---|---|
| 0 | em_cupy 단독 구동: sys.path, allocator 교체, 기존 pytest 통과, 우리 Parameter로 smoke | 합성 점군에서 기대 높이 재현 |
| 1 | L1 센서 이식(§2.2) + viz/test 재실행 | `test_l1_pattern` 통과, viz 확인 |
| 2 | 단일 env 통합: 필터→변환→odometry 노이즈→EM→132점 샘플. play.py에 GT vs EM 동시 시각화 | §5 게이트: 유효 셀 RMSE < 0.05m, invalid 비율 < 20% (보행 1s 후) — 학습 전 파이프라인 sanity check일 뿐, 학습/모델에는 불개입(§5-Q7) |
| 3 | 192 env 확장(stock 루프) + 벤치마크(시간·VRAM) | tick당 EM 총 시간 실측 → 3′ 진행 판단 |
| 3′ | (조건부) batched 커널 포트 | Phase 2와 출력 일치(회귀) |
| 4 | 학습 통합: runner/cfg/task 등록, 기존 teacher ckpt(`model_16999.pt`)로 짧은 학습(수백 iter) | distillation loss 하강, play 정성 평가 |
| 5 | 본학습 5k iter | 기존 depth student 대비 성공률 |

각 Phase 종료 시 커밋. 검증은 headless 원칙(env cfg 검증은 `env_isaaclab python` +
`SimulationApp(headless)` 선행, print는 파일로) 준수.

## 4. 리스크

1. **인스턴스 루프 성능** (최대 리스크): 192 env 직렬 → tick당 수십~수백 ms 가능성.
   Phase 3에서 조기 판정, 3′로 해소. 최악 시 num_envs 축소 병행.
2. **VRAM**: §7 분석 참조. 순효과는 중립~개선 예상이나 실측 필요.
3. **미관측 셀**: 채움 규칙(Q3 cascade)이 학습 품질 좌우. Phase 2에서 invalid 비율
   곡선 확인.
4. **teacher와의 분포 차이**: EM은 지연·부분 관측이라 초기 loss가 depth student보다
   높을 수 있음(정상).
5. em_cupy 수정(traversability 우회 등)은 vendored 패치로 격리.

## 5. 결정 사항 (사용자 확정) 및 잔여 가정

| # | 결정 |
|---|---|
| Q1 | 센서 자연 프레임 0.1s(10Hz, 1.1회전)만 사용 |
| Q2 | **scandots 그리드 현행 유지: 12×11=132점, 0.15m, +1.2/−0.45/±0.75m — teacher 변경·재학습 없음.** EM 내부 해상도는 별개로 0.1m |
| Q3 | invalid 셀 채움 cascade: valid→h, 아니면 upper_bound, 아니면 0 (visibility cleanup 활성으로 지원) |
| Q4 | odometry(6D pose) 노이즈 추가 — EM 입력 pose에만, policy obs는 GT 유지 |
| Q5 | EM 지연 모델링 안 함 |
| Q6 | teacher init 후 scan_encoder 포함 전체 fine-tune (현행 방식) |
| Q7 | RMSE/invalid 지표는 **학습·모델 설계에 불개입** — Phase 2에서 학습 착수 전 매핑 파이프라인의 정확성을 판정하는 sanity check 게이트(미달 시 버그 수정·EM 파라미터 조정의 트리거)로만 사용. 최종 성능 평가는 기존대로 성공률 |
| Q8 | ceiling 필터(max_height_range/ramped) 기본값으로 적용 — ramp/stair 지형이라 무해 |

잔여 가정 (기본값으로 진행, 조정 가능):
- odometry 노이즈 (2026-08-13 개정, 사용자 지시로 drift 모델 도입): EM tick마다
  변화량 기반 오차 누적 — `Δ_meas = Δ_true·(1+b) + n`, `b~N(0, 0.02)`,
  `n~N(0, 0.005²)`. 위치(xyz)와 yaw에 적용해 **시간에 따라 오차가 커진다**
  (0.5m/s 보행 20s 기준 위치 σ≈0.125m, yaw σ≈4.2°). roll/pitch는 IMU(중력)
  관측으로 드리프트하지 않으므로 종전 백색잡음 σ=0.5° 유지. scandots 샘플도
  odom frame pose(노이즈 포함)로 수행 — 실기에서 map과 query가 같은 odometry를
  공유하므로 상대 오차(셀 관측 이후 drift)만 남는 구조를 재현.
- lidar 노이즈: range **σ=0.02m** (L1 스펙 ±2cm) + 빔 지향(az/el) 백색잡음
  **σ=0.2°**(2026-08-13 추가; 어긋난 방향에 측정 거리를 놓는 근사). 끌 수 있게
  플래그화.
- Q7 게이트 수치(RMSE 0.05m / invalid 20%)는 제안 기본값 — Phase 2 실측 후 재조정 가능.

## 6. 이번에 만들지 않는 것 (혼동 방지)

- `SimLegOdometry`/KF odometry — 노이즈 부여 GT 사용 (lidar 브랜치와의 차이점 1)
- 자체 KF elevation map(`LidarElevationMap`) — em_cupy로 대체 (차이점 2)
- `ExpandedScanEncoder`(396입력) — teacher 원형 132입력 MLP 사용 (차이점 3)
- depth camera 파이프라인 수정 — 기존 task 유지, 신규 task로 병행

## 7. VRAM 분석 (RTX 3060 12GB, 현 student 192 env = 11.1GB)

제거되는 것 (기존 student 대비):
- RayCasterCamera 106×60 = 6,360 ray/env × 192 = 1.22M ray + 이미지/리사이즈 버퍼
- depth encoder: `Linear(62400→128)`(~32MB weight + Adam 상태 ×2) + **120-step BPTT
  그래프**(conv 활성값 × 192 env × 120 step — 현 student 메모리의 상당 지분)

추가되는 것:
- L1 raycaster 2160 ray/env × 192 = 415k ray (카메라의 1/3)
- cupy CUDA 컨텍스트 + 커널 캐시: ~0.3–0.5GB (프로세스당 1회)
- EM 인스턴스: 27그리드 × 34² × 4B ≈ 0.13MB × 192 ≈ **25MB** (무시 가능)

순효과: **중립~개선 예상** (BPTT 그래프 제거분이 cupy 컨텍스트 추가분보다 클 가능성).
Phase 3에서 실측하고, 부족 시 대응 순서:
1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (단편화 완화)
2. cupy를 소형 고정 풀로 제한 (맵 총량 25MB라 풀 상한 낮게 설정 가능)
3. student num_envs 192→128 (학습 시간 ~1.5배)
4. batched 포트(3′) 시 인스턴스별 scratch 그리드 통합으로 추가 절감
