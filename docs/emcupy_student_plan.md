# Student v1.3 계획서: L1 LiDAR + elevation_mapping_cupy 기반 terrain latent

작성일: 2026-08-12 (세션 v1.3). 대상 브랜치: master(`ad11a4f`) 기준 신규 작업.

## 0. 목표

기존 student의 depth camera + ConvNet+GRU 파이프라인을 다음으로 대체한다.

```
L1 LiDAR (MultiMeshRayCaster, 11Hz yaw / 180Hz pitch, 21,600 pt/s)
   → self-hit 필터 (부풀린 로봇 캡슐, GT proprioception)
   → elevation_mapping_cupy (GT odometry, per-env 인스턴스)
   → teacher scandots 그리드(12×11, base 기준 +1.2/−0.45/±0.75m)에서 높이 샘플
   → teacher scan_encoder MLP(132→128→64→32 Tanh, teacher weight로 init)
   → 32차원 terrain latent → DAgger distillation (기존 action-imitation L2 유지)
```

정책은 50Hz로 동작하고, terrain latent는 LiDAR 1회전 주기마다 갱신되며 그 사이에는 최신 값을 재사용한다.

---

## 1. 사전 확인 결과

### 1.1 Teacher scandots / encoder (master)

- 그리드: `GridPatternCfg(resolution=0.15, size=[1.65, 1.5])` + offset x=+0.375
  → x: base 기준 −0.45 … +1.20m 12점, y: ±0.75m 11점, **12×11=132점, 간격 0.15m**.
  요구사항과 정확히 일치. (`parkour_teacher_cfg.py:17-27`)
- flatten 순서: `ordering="xy"` → shape (n_y=11, n_x=12), index = `y_idx*12 + x_idx`.
  (`scripts/rsl_rl/multicam_recorder.py:82-110`의 `resolve_scandots_grid` 참조)
- 높이 식: `clip(pos_w_z − hit_z − 0.3, −1, 1)` (`parkour_isaaclab/envs/mdp/observations.py:139-140`).
  **`data.pos_w`는 cfg offset(z=+20)을 포함하지 않음을 IsaacLab 소스에서 확인**
  (`ray_caster.py:155` — `self._offset`은 추적 prim 자체 오프셋, base면 identity; cfg offset은
  ray 시작점에만 적용). 즉 실질적으로 **`clip(base_z − hit_z − 0.3, −1, 1)`**.
- ray_alignment="yaw": 그리드는 로봇 yaw만 따라 회전 (roll/pitch 무시).
- 갱신: `common_step_counter % 5 == 0`에서만 재계산(10Hz), 캐시 유지 (`observations.py:64-67`).
  노이즈 없음.
- encoder: `Linear(132→128) ELU → Linear(128→64) ELU → Linear(64→32) Tanh`
  (`actor_critic_with_encoder.py:55-70`, `scan_encoder_dims=[128,64,32]`).
- checkpoint 키: `model_state_dict["actor.scan_encoder.{0,2,4}.{weight,bias}"]`
  (teacher `model_16999.pt`에서 shape 확인 완료).
- **주입 지점**: `Actor.forward(obs, hist_encoding, scandots_latent=None)` — latent를 넘기면
  scan_encoder 우회, `None`이면 obs[53:185]를 자체 scan_encoder에 통과
  (`actor_critic_with_encoder.py:89-101`). 후자를 이용하면 별도 encoder 클래스 없이
  obs의 scandot 슬라이스만 교체하는 것으로 충분.
- distillation 시 `depth_actor = deepcopy(teacher actor)` 로드
  (`on_policy_runner_with_extractor.py:639-648`) → **scan_encoder가 teacher weight로
  자동 초기화됨**. "teacher MLP를 init으로 사용" 요구사항이 기존 코드 경로로 충족.

### 1.2 Student 현행 구조 (master)

- obs `policy` 그룹 753 = prop 53 | scan 132 | priv_explicit 9 | priv_latent 29 | history 530.
  teacher/student 동일 벡터. depth는 별도 그룹 `depth_camera`(58×87).
- depth encoder(ConvNet+GRU) forward는 `% 5 == 0`(10Hz)에서만, 사이 4 step은 같은 latent 재사용
  (`on_policy_runner_with_extractor.py:365-374`) — "최신 latent 재사용" 인프라가 이미 존재.
- 손실: `(teacher_action.detach() − student_action).norm(p=2, dim=1).mean()` 단독,
  Adam lr 1e-3, iteration당 120 env-step 수집 후 1회 업데이트, GRU BPTT
  (`distillation_with_extractor.py:58-72`).
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
  **매 프레임이 이미 360° 전방위 커버** — 별도 sweep 누적 버퍼 불필요.
  점별 방출시각은 `valid_frame_times()`로 복원 가능(임의 시각 경계로 자르기 가능).
- 마운트: base 기준 (0.28, 0, 0.10), 돔 축 완전 하향(rot=(0,1,0,0)). max_distance 10m.
- 출력: `data.ray_hits_w` (N_env, 2160, 3) **world frame**, miss는 `inf`.
- 타깃: `/World/ground` + **자기 로봇 collision mesh**(`{ENV_REGEX_NS}/Robot/.*/collisions`,
  per-env라 타 로봇 미포함). 로봇 body에 ray가 실제로 막히는 물리적 self-occlusion 재현.
  `update_mesh_ids=False`로 두어 upstream shape 버그 회피(켜지 말 것).
- **self-hit 필터가 이미 요구사항 그대로 구현되어 있음**: `GO2_SELF_FILTER_CAPSULES`
  (14개 캡슐: base r=0.11, head r=0.06, hip×4 r=0.07, thigh-calf×4 r=0.055,
  calf-foot×4 r=0.05 — 실제보다 부풀린 값) + `ray_capsule_penetrates(origins, dirs, t_hit,
  seg_a, seg_b, r, t0=0.12)` (ray 경로 관통 판정, t0=0.12m로 마운트가 body 캡슐 내부인 문제
  회피). 캡슐 끝점은 GT link pose(proprioception)에서 계산. 실측 self-hit 비율 ~61%.
- 센서 자체에는 노이즈 없음. lidar 브랜치는 다운스트림에서 range σ=0.02m,
  캡슐 반경 per-env 오차 σ=0.01m를 부여했음(참고값).
- 가져오지 않을 것: `SimLegOdometry`, KF 기반 `LidarElevationMap`, `ExpandedScanEncoder`,
  [h|var|ub] 3채널 obs — 이번 설계와 기능이 겹치지만 전부 대체 대상.

### 1.4 elevation_mapping_cupy (클론된 repo, ROS1 main 브랜치 `20a8a26`)

- **코어는 ROS 없이 동작**: `elevation_mapping_cupy/script/elevation_mapping_cupy/`의
  `ElevationMap`/`Parameter`는 순수 python+cupy (CI도 ROS 없이 pytest).
  `sys.path`에 `.../script` 추가 필요. torch↔cupy zero-copy 인터롭 기존 코드에 존재
  (`traversability_filter.py:29,42`).
- 입력: `input_pointcloud(points(N,3) [sensor frame], ["x","y","z"], R(3,3), t(3)
  [map frame 기준 센서 pose], position_noise, orientation_noise)`.
  `move_to(base_pos, R_base)`로 맵 recenter(셀 스냅, grid roll). 그리드는 map frame
  축정렬(회전 없음), 높이는 **map center z 상대값**(절대값 = `elevation + center[2]`).
- 조회: `elevation_map[0]`(height) / `[1]`(variance) / `[2]`(is_valid) / `[5]`(upper_bound)
  cupy 배열 직접 접근 + `torch.as_tensor` zero-copy가 유일한 실용 경로.
  `get_map_with_name_ref`는 GPU→CPU 복사라 학습 루프 사용 금지.
  **임의 (x,y) 높이 샘플 API 없음** — 자체 gather 구현 필요(단순).
- 인덱싱: `row=(x−cx)/res + cell_n/2`, `col=(y−cy)/res + cell_n/2` (row↔x, col↔y).
- 자체 self-filter 없음(실기 ANYmal도 외부 필터 노드 사용) — 우리 캡슐 필터가 표준 위치.
- **단일 맵 설계 — 배치 미지원.** 커널 소스는 파라미터 동일 시 캐시 재사용되어 인스턴스
  N개 생성 자체는 저렴. 메모리도 작은 맵(3.2m@0.1m) 기준 인스턴스당 ~0.1MB 수준으로 무시 가능.
  **진짜 병목은 커널 런치 오버헤드**: 업데이트 1회 = 인스턴스당 커널 ~6회 + torch CNN 1회,
  192 env 직렬 python 루프 → 라운드당 수천 런치.
- 확인된 함정:
  1. import 시 **프로세스 전역 cupy managed-memory allocator** 설정
     (`elevation_mapping.py:45-46`) — IsaacLab/torch와 VRAM 경합. import 후 allocator 교체 필요.
  2. `input_pointcloud`가 `t`를 **in-place 변경** — 복사본 전달 필수.
  3. `move()`와 `move_to()`의 grid shift 부호가 상반 — **`move_to`만 사용**.
  4. `param.update()` 필수 호출(cell_n 계산), weight/plugin 경로는 절대경로로.
  5. visibility cleanup은 점당 최대 `max_ray_length/resolution`회 반복 — 기본
     yaml값(10m/0.04m)이면 폭발. 끄거나 max_ray_length를 맵 크기 수준으로 축소.
  6. GT odometry면 `enable_drift_compensation=False` → error_counting 커널 생략 가능(~1/3 절약).
  7. traversability CNN이 업데이트 경로에 내장 — 불필요하므로 우회 패치 대상.
  8. `min_valid_distance`(기본 0.3/0.5m)가 유일한 내장 근거리 필터 — 캡슐 필터 t0=0.12와
     정합하도록 0.12 이하로 설정.

---

## 2. 설계

### 2.1 파이프라인 (per elevation-map 갱신 tick)

```
[매 tick, 기본 10Hz 제안 — §5-Q1]
1. lidar.data.ray_hits_w 읽기 (N,2160,3 world, miss=inf)
2. self-hit 마스크: ray_capsule_penetrates × 14 캡슐 (GT link pose, 부풀린 반경)
3. 유효점 = finite ∧ ¬self_hit  → sensor frame으로 변환: p_s = R_sᵀ(p_w − t_s)
4. per-env: em.move_to(base_pos_w, R_base); em.input_pointcloud(p_s, R_s, t_s.copy(), 0, 0)
   (주기적으로 update_variance/update_time)
5. 12×11 샘플: scandot 위치 = base_xy + yaw회전(그리드 오프셋) [ray_alignment="yaw" 동일]
   h_abs = bilinear(elevation) + center[2],  invalid 셀은 채움 규칙 적용(§5-Q3)
   h_obs = clip(base_z − h_abs − 0.3, −1, 1)  ← teacher 식과 동일
6. obs 캐시 갱신: obs[53:185] 자리에 h_obs 사용(다음 갱신까지 유지)
[매 policy step, 50Hz]
7. student action = depth_actor(obs_with_em_scan, hist_encoding=True, scandots_latent=None)
   → 자체 scan_encoder(=teacher init MLP)가 latent 계산. latent 갱신은 obs 갱신 주기를 따름.
   teacher action = policy.act_inference(원본 obs) (GT scandots) → L2 imitation.
```

ConvNet+GRU, `depth_camera` obs 그룹, depth camera 센서는 제거. `depth_encoder=None`
경로가 아니라 runner의 `learn_vision`을 단순화한 신규 학습 루프를 만든다(§2.5).

### 2.2 L1 센서 이식

`git show lidar:parkour_isaaclab/sensors/l1_scan_ray_caster.py` 등으로 아래를 master 기반
신규 브랜치에 복사:

- `parkour_isaaclab/sensors/{__init__.py, l1_scan_ray_caster.py}` (그대로)
- `default_cfg.py`에 `GO2_LIDAR_CFG` 블록 (그대로: 마운트 (0.28,0,0.10) 하향, 2160 ray, 10m)
- `parkour_student_cfg.py`: `camera` → `lidar = GO2_LIDAR_CFG` 교체, `update_period` 설정
- 검증 도구: `scripts/lidar_sim/viz_l1_scan.py`, `parkour_test/test_map_encoder_cpu.py`의
  `test_l1_pattern()` (선택)

주의: 센서는 `.data` 접근 시점 기준 직전 0.1s 프레임을 반환. 소비 주기와 프레임 경계가
겹치지 않도록 소비 코드에서만 읽는다(§5-Q1의 주기 결정에 종속).

### 2.3 Self-hit 필터

lidar 브랜치의 `GO2_SELF_FILTER_CAPSULES` + `ray_capsule_penetrates`를 그대로 사용.
요구사항("proprioception으로 primitive 구성, 부풀려서 보수적 제거")과 정확히 일치하며
검증된 구현(t0=0.12m 근거리 스킵 포함)이다. 캡슐 끝점은 GT link pose에서 매 tick 계산.
로봇 collision mesh를 raycast 타깃에 유지해 물리적 self-occlusion(가림)도 재현 —
필터는 "body에 맞은 점 제거", occlusion은 "body 뒤 지형이 안 보임"으로 역할이 다르며
둘 다 실기와 일치한다.

### 2.4 elevation_mapping_cupy 통합

래퍼 모듈 `parkour_isaaclab/envs/mdp/elevation_map_backend.py`(신규)를 만들어 격리:

```python
Parameter 설정(제안 초기값, §5-Q2 결정 반영):
  resolution=0.1, map_length=3.2          # scandot 최원점 1.415m + 여유; cell_n=34
  sensor_noise_factor=0.05, mahalanobis_thresh=2.0
  min_valid_distance=0.10                 # 캡슐 필터 t0=0.12보다 작게
  max_height_range/ramped_*: 하향 라이다에 맞게 재조정(§5-Q4)
  enable_drift_compensation=False         # GT odometry
  enable_visibility_cleanup=True, max_ray_length=3.0, cleanup 파라미터 축소  # §5-Q3와 연동
  enable_overlap_clearance=False          # 맵이 3.2m로 작음
  plugin_config: 빈 목록, subscriber_cfg={}, additional_layers=[], fusion=[]
```

- import 직후 `cp.cuda.set_allocator(...)`로 managed-memory pool 해제(함정 1).
- traversability CNN 우회: `ElevationMap`을 서브클래스해 traversability 호출을 no-op으로
  오버라이드(포크 수정 없이 가능하면 그 방법, 아니면 vendored 최소 패치).
- per-env `ElevationMap` 인스턴스 리스트. **env reset 시 해당 인스턴스 `clear()`**
  (teleport로 맵이 무효화되므로 필수).
- 좌표계: GT라 odom≡world. env별 맵은 world 좌표에서 base를 따라다니면 됨(env origin
  분리 불필요 — move_to가 recenter).
- 성능 전략은 2단계(§3 Phase 3/3′): stock 인스턴스 루프로 먼저 정확성을 확보하고
  벤치마크 후, 병목이면 커널 3개(add_points/error_counting/average_map)에 env 배치
  차원을 추가한 batched 포트로 교체. 조사 결과 기계적 변경으로 판단됨
  (ElementwiseKernel + atomicAdd 구조, maps (B,7,H,W) + 점별 env_id 컬럼).

### 2.5 관측/학습 통합

- obs: `ExtremeParkourObservations`의 scandot 슬라이스를 유지하되(teacher용 GT),
  **student 입력용 132-vector를 별도 버퍼로 계산**하고 runner에서 obs를 복제해
  [53:185]만 교체해 depth_actor에 공급. (teacher는 원본 obs 사용 — 현행 구조와 동일하게
  runner 안에서 두 벡터를 구분.)
- 학습 루프: `learn_vision`을 기반으로 한 `learn_em`(신규 또는 분기):
  - depth encoder/GRU 관련 로직 제거 → BPTT 불필요, hidden state 워밍업 불필요
    (resume 스파이크 이슈 소멸).
  - 학습 파라미터 = `depth_actor.parameters()` (scan_encoder 포함 — teacher init 후
    fine-tune, §5-Q6). 손실/옵티마이저/수집 길이(120)는 현행 유지.
  - latent 캐시는 "132-vector 캐시"로 대체: 갱신 tick에만 EM 샘플 재계산, 사이 step은
    직전 값 재사용(현행 `depth_latent` 패턴과 동일 — 단, tick 이전 미초기화 NameError
    버그는 초기값 세팅으로 함께 수정).
- play/eval도 동일 패턴으로 수정.
- 기존 depth 파이프라인은 삭제하지 않고 cfg 분기(신규 task 이름, 예:
  `Isaac-Extreme-Parkour-Student-EM-...-v0`)로 공존시킨다.

### 2.6 타이밍 (11Hz 요구 vs 50Hz 정수배)

사실관계: 1/11s = 4.545 policy step이라 50Hz와 정수배가 아니고, 센서 프레임은 0.1s(2160
ray)가 자연 단위이며 0.1s에 이미 yaw 1.1회전이 들어 있다. 선택지:

- **(A안, 권장) 10Hz(5 step)**: 프레임(=1.1회전) 전체를 매 tick 입력. 모든 점이 정확히
  1회 사용, 기존 `%5` 인프라·teacher scandots 갱신 주기와 일치, 구현 최소.
  요구사항 "1회전 pointcloud"에서 "1.1회전"으로의 편차만 존재.
- (B안) 정확 11Hz: 갱신 step 간격 5,4,5,4,…(평균 4.545), `valid_frame_times()`로 각 tick의
  [t−1/11, t] 방출점만 절취. 간격이 5 step(0.1s)인 tick에서는 프레임(0.1s)보다 윈도(0.0909s)가
  짧아 ~9%의 점이 버려지고, 4 step(0.08s) tick에서는 일부 점이 두 번 입력됨.
  구현 복잡도·엣지 케이스 대비 이득이 불분명.

→ §5-Q1에서 결정 필요. 이하 일정은 A안 기준.

---

## 3. 구현 단계

| Phase | 내용 | 검증 게이트 |
|---|---|---|
| 0 | em_cupy 단독 구동: sys.path, allocator 교체, `tests/test_elevation_mapping.py` 통과, 우리 Parameter로 init→input→move_to→직접 조회 smoke | 합성 점군에서 기대 높이 재현 |
| 1 | L1 센서 이식(§2.2) + viz/test 재실행 | `test_l1_pattern` 통과, viz에서 스캔 패턴 확인 |
| 2 | 단일 env 통합: 필터→변환→EM→12×11 샘플. play.py에 scandots(GT) vs EM 샘플 동시 시각화 | 정적/보행 중 유효 셀 오차 \|h_em−h_gt\| 통계 (기준치 §5-Q7), invalid 비율 곡선 |
| 3 | 192 env 확장(stock 인스턴스 루프) + 벤치마크 | tick당 EM 총 시간, 전체 step 시간 실측 → 3′ 진행 여부 판단 |
| 3′ | (조건부) batched 커널 포트 | Phase 2와 동일 출력 일치(회귀 테스트) |
| 4 | 학습 통합(§2.5): runner/cfg/task 등록, 짧은 학습(수백 iter)으로 loss 하강·teacher 대비 성능 확인 | distillation loss 곡선, play 정성 평가 |
| 5 | 본학습 5k iter | 기존 depth student 대비 성공률 |

각 Phase 종료 시 커밋. 검증 방법은 headless 원칙(memory: env cfg 검증은
`env_isaaclab python` + `SimulationApp(headless)` 선행, print는 파일로) 준수.

## 4. 리스크

1. **인스턴스 루프 성능** (최대 리스크): 192 env × 커널 ~6launch가 python 직렬.
   대략 tick당 수십~수백 ms 가능 → 5k iter 기준 수 시간 추가. Phase 3 벤치마크로
   조기 판정, 3′(batched port)로 해소. 최악의 경우 num_envs 축소도 병행 검토.
2. **VRAM**: 현 student 192 env가 이미 11.1GB/12GB. depth camera(RayCasterCamera,
   106×60=6,360 ray) 제거로 ray 수는 2160으로 감소하지만 cupy 컨텍스트/커널 캐시가
   수백 MB 추가될 수 있음. Phase 3에서 실측.
3. **미관측 셀**: 하향 라이다 특성상 전방 1.2m 원거리 셀·낙차 뒤 그늘은 관측 지연/공백.
   reset 직후 맵이 비어 있는 초기 구간 포함, 채움 규칙(§5-Q3)이 학습 품질을 좌우.
4. **teacher와의 분포 차이**: teacher scandots는 즉시·완전 관측, EM은 지연·부분 관측.
   scan_encoder를 teacher init에서 fine-tune하므로 완화되지만, 초기 loss가 depth student
   보다 높을 수 있음(정상).
5. em_cupy는 유지보수 중인 외부 코드 — 필요한 최소 수정(traversability 우회 등)은
   vendored 패치로 격리하고 upstream 파일 직접 수정 최소화.

## 5. 사용자 결정 필요 사항

- **Q1. 갱신 주기**: A안 10Hz(프레임=1.1회전, 권장) vs B안 정확 11Hz(1.0회전, 점 9% 손실
  또는 중복). — §2.6
- **Q2. EM 그리드 사양**: 제안 resolution 0.1m / map_length 3.2m. scandot 간격 0.15m
  기준 0.1이면 충분하다고 판단하나, 계단 모서리 선명도가 중요하면 0.05m(비용 ↑,
  특히 visibility cleanup). 어느 쪽?
- **Q3. invalid(미관측) 셀 채움 규칙**: (a) `upper_bound` 대체(visibility cleanup 필요,
  "최소한 이 높이보다 낮진 않다" 보수 추정) (b) 0으로(=평지 가정) (c) 최근접 유효값.
  lidar 브랜치는 [h|var|ub] 3채널로 회피했으나 이번엔 teacher와 동일한 132×h 단일채널이
  제약이므로 채움 규칙이 필요. 제안: (a).
- **Q4. 노이즈**: range σ=0.02m(전 브랜치 값) 적용 여부, `sensor_noise_factor`(EM 내부
  거리²비례 분산) 값, 캡슐 inflate 오차 σ=0.01 적용 여부. v1.3 1차 학습은 GT(무노이즈)로
  가고 노이즈는 후속으로 미뤄도 됨. 어느 쪽?
- **Q5. 지연 모델링**: EM 계산·전송 지연 모사 여부(전 브랜치는 60ms 고정 지연).
  1차는 지연 0 제안.
- **Q6. encoder 학습**: teacher init 후 fine-tune(제안) vs freeze(latent 분포 보존,
  actor backbone만 학습).
- **Q7. Phase 2 정량 기준**: EM 샘플 vs GT scandots 허용 오차(예: 유효 셀 RMSE < 0.05m,
  invalid 비율 < 20%)를 어느 수준으로 볼지.
- **Q8. 상부 구조물**: 하향 마운트라 상방 관측 없음 + EM의 ceiling 필터
  (`max_height_range`) 존재. parkour 지형에 overhang이 없다면 무시(기본값 유지) 제안.

## 6. 이번에 만들지 않는 것 (혼동 방지)

- `SimLegOdometry`/KF odometry — GT odometry 사용 (lidar 브랜치와의 차이점 1)
- 자체 KF elevation map(`LidarElevationMap`) — em_cupy로 대체 (차이점 2)
- `ExpandedScanEncoder`(396입력) — teacher 원형 132입력 MLP 사용 (차이점 3)
- depth camera 파이프라인 수정 — 기존 task는 그대로 두고 신규 task로 병행
