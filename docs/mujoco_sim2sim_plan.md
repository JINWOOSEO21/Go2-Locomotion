# MuJoCo sim2sim 이식 가이드라인 (trained_v1.3 / EM student)

대상 산출물: `logs/rsl_rl/unitree_go2_parkour/student_pretrained/trained_v1.3~30K/model_29998.pt`
목표: IsaacLab → MuJoCo sim2sim 으로 도메인 갭을 계량한 뒤 Go2 실기(sim2real)로 간다.
참조 프레임워크: `unitree_rl_lab` (로컬 클론 `~/workspace/codes/unitree_rl_lab`)

**작업 브랜치: `sim2sim`.** `master`(= tag `trained_v1.3`)는 IsaacSim 학습/재생 전용으로
그대로 둔다 — MuJoCo 이식물은 이 브랜치에만 쌓는다. master 에 들어간 것은
`elevation_mapping_cupy` submodule 고정(`76f14d5`) 하나뿐이고, 그건 원래 EM student 의
play/evaluation 이 돌아가려면 필요한 수정이라 여기 두지 않았다.

---

## 0. 요약 — 무엇을 하고 무엇을 하지 않는가

**순서를 고정한다. 두 트랙을 섞으면 원인 분리가 안 된다.**

| 트랙 | 목적 | 무엇을 검증하나 |
|---|---|---|
| **A. Python MuJoCo 패리티 하네스** | *물리/관측* 갭 측정 | 접촉·마찰·액추에이터·타이밍·지각 파이프라인 |
| **B. unitree_mujoco + C++ FSM** | *배포 스택* 리허설 | DDS 지연, 1kHz PD, 조이스틱 FSM, ONNX, 실기 전환 |

A 를 먼저 끝내고 B 로 간다. A 없이 B 로 가면 "정책이 못 걷는다"의 원인이 물리 갭인지 배포 배선인지 구분할 수 없다.

**확정된 작업 배치(2026-09-02, 사용자 결정):**
- 트랙 A·B **둘 다 포크한 `unitree_rl_lab` 안에서** 한다. 이 저장소(`Isaaclab_Parkour`)는
  **`policy.onnx` + `deploy.yaml` 을 뱉는 것까지만** 하고 이식 코드를 두지 않는다 (§6).
- 트랙 A 의 Python 하네스는 우회가 아니라 **트랙 B 의 기준값 생성기**다. 이미 검증된
  `elevation_map_backend.py`(순수 torch, isaaclab 비의존)를 복사해 그대로 쓰므로 EM 을 다시
  짜지 않고 물리 갭만 분리해 잴 수 있고, 그 수치가 C++ 포팅의 합격 기준이 된다.

### unitree_rl_lab 에서 가져올 것 / 못 가져올 것

**가져온다 (아키텍처):**
- 2-스레드 구조: 1kHz FSM 스레드가 `LowCmd` 를 계속 발행하고, `step_dt`(0.02s) 정책 스레드가 target q 만 갱신 (`deploy/include/FSM/State_RLBase.h:29-46`, `deploy/robots/go2/src/State_RLBase.cpp:28-31`).
- `joint_ids_map[isaaclab_idx] = sdk_motor_idx` 단일 재정렬 지점 (`deploy/include/unitree_articulation.h:37-40` 읽기, `deploy/robots/go2/src/State_RLBase.cpp:29-31` 쓰기).
- FSM 상태 기계 + 조이스틱 DSL 전환 (`deploy/include/FSM/CtrlFSM.h:21-46`, `deploy/include/FSM/FSMState.h:17-45`). Passive / FixStand / RL 3상태 안전 골격.
- 학습 cfg → `params/deploy.yaml` 자동 export 아이디어 (`source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py`).
- sim2sim 이 "같은 바이너리를 다른 DDS 피어에 붙이는 것"이라는 설계 (repo 안에 MuJoCo 코드가 전혀 없다 — domain id 0 고정, `deploy/robots/go2/main.cpp:36`).

**못 가져온다 (관측 매니저):**
- `deploy/include/isaaclab/manager/observation_manager.h` 는 등록된 term 이름만 받고 미등록이면 throw (`:134-136`). 등록된 것은 `base_ang_vel / projected_gravity / joint_pos / joint_pos_rel / joint_vel_rel / last_action / velocity_commands / gait_phase` 뿐이다 (`deploy/include/isaaclab/envs/mdp/observations/observations.h`).
- **exteroception 이 전무하다.** `deploy/` 전체에 height/scan/depth/lidar/raycast 문자열이 0회. `base_lin_vel` term 도 없다.
- 우리 obs 753 은 이 스키마로 표현 불가 → **커스텀 `State_Parkour` 상태를 새로 쓴다** (`REGISTER_FSM` + 자체 obs 조립).

> **핵심**: 이식의 무게 중심은 정책(MLP 몇 개)이 아니라 **정책 밖에 사는 지각 파이프라인**(L1 LiDAR → self-filter → elevation map → scandots 132)이다. 정책 자체는 하루면 옮긴다.

### 선결 과제 진행 상황 (2026-09-02 갱신)

| 항목 | 상태 |
|---|---|
| sim2real 전용 conda 환경 | **완료** — `env_sim2real` (Python 3.11.16, mujoco 3.12.0) |
| `elevation_mapping_cupy` 의존성 고정 | **완료** — submodule @ `20a8a26`, **master 커밋 `76f14d5`** (이 브랜치가 아니라 master 에 있다) |
| 관절 순서 3종 실측 | **완료** — `deploy/contract/deploy.yaml` |
| Go2 MJCF 확보 | 완료 — `mujoco_menagerie` @ `e4049d0` (submodule 아님, SHA 만 기록) |

**`elevation_mapping_cupy` 는 원래 깨져 있었다.** 로컬 클론에 `.py` 소스가 하나도 없이
`__pycache__/*.pyc` 만 남아 있었고 git 에 추적되지도 않아
`from elevation_mapping_cupy import ElevationMap, Parameter` 가 `ImportError` 를 냈다.
`BatchedElevationMapBackend.__init__` 이 런타임에 이걸 import 하므로
(`elevation_map_backend.py:50-69`) EM student 의 play/evaluation 자체가 불가능한 상태였다.

이제 submodule 로 고정했으므로 클론 후에는 다음이 필요하다:

```bash
git submodule update --init --recursive
```

확인된 맵 사양(실제 `Parameter` 로 재현): `cell_n = int(round(map_length/resolution)) + 2`
→ 3.2 m / 0.1 m 이면 **34×34 셀**.

> **메인 체크아웃 주의**: 기존 작업 복사본에는 소스가 사라진 `elevation_mapping_cupy/` 가
> untracked 로 남아 있다. master(`76f14d5` 이후)를 체크아웃하기 전에 그 디렉터리를 지워야
> submodule 이 깨끗하게 들어온다:
> ```bash
> rm -rf elevation_mapping_cupy && git checkout master && git submodule update --init --recursive
> ```

---

## 1. 동결해야 할 정책 계약 (contract)

이식 작업의 첫 산출물은 코드가 아니라 **이 계약을 못박은 표**다. 아래를 그대로 `deploy_contract.md` / `deploy.yaml` 로 떨어뜨리고, 양쪽 구현이 이 표만 보게 한다.

### 1.1 최상위 관측 (753)

`parkour_isaaclab/envs/mdp/observations.py:93-99`, 차원은 `parkour_tasks/.../agents/parkour_rl_cfg.py:24-29`.

| 구간 | 인덱스 | 크기 | 학습 시 내용 | **배포 시 내용** |
|---|---|---|---|---|
| prop | `[0:53]` | 53 | 아래 1.2 | 동일 (센서에서 조립) |
| scan | `[53:185]` | 132 | teacher GT height scan | **elevation map 샘플** |
| priv_explicit | `[185:194]` | 9 | GT base lin vel×2, 0, 0 | **estimator(prop) 출력** |
| priv_latent | `[194:223]` | 29 | 질량/COM/마찰/게인 | **사용되지 않음 → 0 채움** |
| history | `[223:753]` | 530 | 과거 prop 10프레임 | 동일 |

**priv_latent 이 무시되는 이유**: actor 를 `hist_encoding=True` 로 부르면 latent 가 `history_encoder(hist)` 에서 나오고 `priv_encoder` 는 호출되지 않는다 (`scripts/rsl_rl/modules/actor_critic_with_encoder.py:100-103`). 다만 history 슬라이스가 `obs[:, -530:]` 이라 **29칸의 자리는 반드시 있어야** 꼬리 정렬이 맞는다. 0 으로 채우고 무시할 것.

### 1.2 prop 53 상세

`parkour_isaaclab/envs/mdp/observations.py:70-86`.

| idx | 크기 | 값 | 비고 |
|---|---|---|---|
| 0:3 | 3 | `base_ang_vel_b * 0.25` | body frame 각속도 (IMU gyro) |
| 3:5 | 2 | `wrap_to_pi(roll), wrap_to_pi(pitch)` | **projected_gravity 가 아니다.** quat→euler_xyz 후 wrap |
| 5 | 1 | `0.0` | 상수 |
| 6 | 1 | `delta_yaw = target_yaw − yaw` | **10Hz 갱신** (`common_step_counter % 5`) |
| 7 | 1 | `delta_next_yaw` | 동상 |
| 8:10 | 2 | `0.0, 0.0` | 상수 |
| 10 | 1 | `commands[0]` = 전진속도 지령 | 학습 범위 0.3~0.8 m/s |
| 11 | 1 | **`1.0` 고정** | v1.3 규약: 항상 non-flat |
| 12 | 1 | **`0.0` 고정** | v1.3 규약: 항상 flat=0 |
| 13:25 | 12 | `joint_pos − default_joint_pos` | IsaacLab 관절 순서 |
| 25:37 | 12 | `joint_vel * 0.05` | |
| 37:49 | 12 | 직전 raw action | `action_history_buf[:, -1]` (지연·클립 이전 값) |
| 49:53 | 4 | `contact_filt − 0.5` | 발 4개, 아래 참조 |

- `contact_filt` = `‖F_now‖ > 2N` **OR** `‖F_prev‖ > 2N` (`observations.py:110-117`, contact history_length=2). 출력은 {−0.5, +0.5}.
- 전체 관측에 `clip=(-100,100)` 적용 (`parkour_mdp_cfg.py` ObsTerm).
- `empirical_normalization = False` → **런타임 정규화기 없음** (Identity).

### 1.3 history 530

`observations.py:101-108`.
- 매 step, `obs_buf` 의 **인덱스 6:8 을 0 으로 만든 뒤** 큐에 push (FIFO, 길이 10).
- 에피소드 시작(`episode_length_buf <= 1`)에는 현재 obs_buf 를 **10개 복제**해 채운다.
- 배치 레이아웃은 `view(N, -1)` = `[t-9 프레임 53개, t-8 …, …, t 53개]` (시간축 우선, term 우선 아님).

### 1.4 scan 132 (teacher 격자)

`parkour_tasks/.../parkour_teacher_cfg.py:16-24`, `GridPatternCfg(resolution=0.15, size=[1.65, 1.5])`, offset `pos=(0.375, 0, 20.0)`, `ray_alignment="yaw"`.

- x: `0.375 + [-0.825 … +0.825]` step 0.15 → **12점** (범위 −0.45 ~ +1.20 m)
- y: `[-0.75 … +0.75]` step 0.15 → **11점**
- 순서: IsaacLab `grid_pattern` 의 `meshgrid(x, y, indexing="xy")` → **y 바깥 루프(11), x 안쪽 루프(12)** 로 flatten. 132 = 11×12.
- 값: `clip(base_z − h_terrain − 0.3, −1, +1)`.
  `RayCaster._data.pos_w` 는 **body pose** 이고 cfg offset 은 `ray_starts` 에만 반영되므로 (`IsaacLab .../ray_caster.py:220-224, 241-248`) 기준 높이는 z=+20 이 아니라 **base link z** 다. EM 경로도 `sample(points_xy, base_z)` 로 동일 (`elevation_map_backend.py:sample_scan_heights`).

### 1.5 배포 시 실제 계산 그래프

`scripts/rsl_rl/play.py:616-631`, `scripts/rsl_rl/modules/on_policy_runner_with_extractor.py:502-521`.

```
prop(53) ──┬─────────────────────────────────────────► actor 입력 [0:53]
           └─► estimator MLP [53→128→64→9] ──────────► obs[185:194]
em_scan(132) ─► scan_encoder [132→128→64→32,tanh] ───► 32
history(530) ─► StateHistoryEncoder(tsteps=10,ch=10) ► 20
                                    ↓
        actor_backbone [53+32+9+20 = 114 → 512 → 256 → 128 → 12]
```
- 활성함수 ELU, 최종 tanh 없음(`tanh_encoder_output=False`).
- 체크포인트에서 쓸 텐서: `depth_actor_state_dict`(26개, = 학습된 student actor) + `estimator_state_dict`(6개). `model_state_dict` 는 teacher 라 배포에 쓰지 않는다.

### 1.6 액션

`parkour_isaaclab/envs/mdp/parkour_actions/joint_actions.py:36-55`, `parkour_mdp_cfg.py` ActionsCfg.

```
a_raw(12)  ─► [1-step 지연 큐]  ─► clip(±4.8) ─► ×0.25 + q_default  ─► q_target
```
- `use_delay=True`, `action_delay_steps=[1,1]` → **한 스텝(20ms) 지연된 액션이 적용**된다. 관측 37:49 에 들어가는 값은 지연·클립 **이전**의 raw action 이다.
- `q_default` = Go2 기본 자세 (hip ±0.1, thigh 0.8, calf −1.5).

### 1.7 타이밍 / 액추에이터

| 항목 | 값 | 출처 |
|---|---|---|
| sim dt | 0.005 s (200 Hz) | `parkour_teacher_cfg.py:62-64` |
| decimation | 4 → 정책 50 Hz | 동일 |
| EM tick / delta_yaw 갱신 | 5 step = 0.1 s (10 Hz) | `parkour_mdp_cfg.py` `update_interval=5` |
| PD | kp=40, kd=1 (전 관절) | `parkour_tasks/default_cfg.py:66-84` |
| effort limit | hip 35 / thigh 40 / calf 40 N·m | 동일 |
| saturation effort | hip 35 / thigh 45 / calf 45 | 동일 |
| velocity limit | hip 52.4 / thigh 30.1 / calf 30.1 rad/s | 동일 |

**액추에이터는 단순 PD 가 아니다.** `ParkourDCMotor` 가 속도 의존 토크 한계를 건다 (`parkour_isaaclab/actuators/parkour_actuator_pd.py:66-74`):
```
τ = kp·(q_des − q) + kd·(0 − dq)
τ_max = clip( τ_sat·(1 − dq/dq_lim),  0, τ_lim )
τ_min = clip( τ_sat·(−1 − dq/dq_lim), −τ_lim, 0 )
τ = clip(τ, τ_min, τ_max)
```
MuJoCo 기본 `position` actuator 에는 이 모델이 없다. **직접 계산해 `d.ctrl` 에 토크로 넣어야 한다** (motor actuator + 200Hz 수동 PD). 이걸 빼먹는 것이 sim2sim 실패의 흔한 1순위다.

---

## 2. 이식 분류표

| 구성요소 | 파일 | 판정 |
|---|---|---|
| elevation map 백엔드 | `parkour_isaaclab/envs/mdp/elevation_map_backend.py` | **그대로 재사용.** isaaclab 을 import 하지 않는다 (모듈 docstring 명시). `num_envs=1` 로 생성, `update(points_sensor, R_s, t_s, base_pos, R_base)` / `sample(points_xy, base_z)` 만 호출 |
| scandot 샘플링 | 같은 파일 `sample_scan_heights` | **그대로 재사용.** 순수 torch |
| self-filter 캡슐 + 판정 | `parkour_isaaclab/sensors/l1_scan_ray_caster.py:45-56, 57-88` (`GO2_SELF_FILTER_CAPSULES`, `ray_capsule_penetrates`) | **발췌 재사용.** 함수·테이블 자체는 순수 torch 지만 모듈 상단이 isaaclab 을 import 하므로 파일째 가져올 수는 없다 |
| L1 스캔 운동학 | 같은 파일 `:112-165` (`directions_from_angles`, `valid_frame_times`) | **static method 두 개만 발췌 재사용.** 순수 torch |
| L1 레이캐스팅 | 같은 파일 `_update_ray_infos` | **재구현.** IsaacLab warp → `mujoco.mj_multiRay` |
| 정책/estimator | `scripts/rsl_rl/modules/**` | **그대로 재사용** (torch) 또는 ONNX export |
| 노이즈/odometry drift 모델 | `observations.py:262-436` (`elevation_map_scan`) | **발췌 재구현.** IsaacLab 텐서 접근만 갈아끼우면 로직은 동일 |
| obs 조립 / 액션 / PD | env manager 전반 | **재구현** (MuJoCo 하네스) |
| 지형 | `parkour_isaaclab/terrains/**` | **export.** heightfield → trimesh 이미 존재 |

---

## 3. 단계별 실행 계획

### Phase 0 — ONNX 추출 + 골든 트레이스 ✅ **완료 (2026-09-02)**

산출물 3개가 `logs/rsl_rl/unitree_go2_parkour/student_pretrained/trained_v1.3~30K/exported/`
에 있다 (logs/ 는 gitignore 라 저장소에는 도구만 들어간다):
`policy.onnx`(1.1MB) · `onnx_reference.npz` · `golden_trace.npz`(100 스텝).
**골든 테스트 PASS**: IsaacLab 실관측 100 샘플에서 max|diff| = 4.77e-6 (허용 1e-5).

작업 중 걸린 함정 3개는 §4 의 17~19 번에 적어 두었다.

---

#### (원래 계획) — **이 저장소에서 하는 유일한 작업**

MuJoCo 코드를 한 줄도 쓰기 전에 한다. 이게 이후 모든 디버깅의 기준선이다.

1. **`policy.onnx` 추출** — §6.1 의 3입력 시그니처로. EM student 용 exporter 가 없으므로 새로 쓴다.
   `deploy/contract/deploy.yaml` 과 나란히 떨어뜨려 이 둘이 배포 repo 로 넘어가는 유일한 산출물이 되게 한다.
2. `play.py` 에 트레이스 덤프를 붙여 **IsaacLab 에서 100 step 분량**을 npz 로 저장:
   `obs(753)`, `em_scan(132)`, `actions(12)`, `q, dq, τ`, `root_pos/quat/lin_vel/ang_vel`, `foot_contact`.
3. **오프라인 골든 테스트**: 저장된 트레이스의 prop/scan/hist 를 `policy.onnx` 에 넣어
   IsaacLab 이 낸 action 과 **1e-5 이내로 일치**하는지 확인.
   → 정책 로딩·레이어 순서·estimator 배선·슬라이스 정렬 오류를 물리와 분리해서 잡는다.
   여기서 안 맞으면 뒤 단계는 전부 무의미하다.

Phase 1 부터는 포크한 `unitree_rl_lab` 안에서 진행한다.

### Phase 1 — MuJoCo 씬 구성 (1~2일)

1. **Go2 모델**: `mujoco_menagerie/unitree_go2/go2.xml` 또는 `unitree_mujoco` 동봉 MJCF.
   - `Head_upper` / `Head_lower` 바디가 있어야 한다 (학습 씬에 존재, self-filter 캡슐과 LiDAR self-occlusion 에 영향).
   - `enabled_self_collisions = True` (`default_cfg.py:65`) 와 동등하게 MJCF contact 설정.
   - base 질량/관성/COM 을 IsaacLab USD 값과 대조 (DR 이 base mass −1~+3kg, COM ±2cm 였으므로 오차 허용폭은 그 안).
2. **지형 export**: 지형 생성기는 heightfield → trimesh 다 (`parkour_terrain_generator.py:91-147`).
   - 방법 A: `terrain_generator.terrain_mesh` 를 OBJ 로 export → MuJoCo `<mesh>` geom.
   - 방법 B(권장): `height_field_raw` 를 그대로 MuJoCo `<hfield>` 로 → 충돌 저렴, 값 대조 쉬움.
   - PLAY 설정과 동일 조건으로 뽑을 것: **난이도 0.7 고정, num_rows=1, size (24, 4), `trapezoid_only`, noise 0.02** (`parkour_em_student_cfg.py:74-102`).
   - 스폰: 타일 시작점에서 x=+1.0 m (`parkour_mdp_cfg.py` `reset_root_state` offset).
3. **마찰**: 학습은 static/dynamic 1.0 지형 + 로봇 DR 0.6~2.0 (`physics_material` 이벤트). MuJoCo 기본값은 다르므로 명시적으로 맞춘다.

### Phase 2 — proprioception + GT scan 패리티 (2~3일)

**EM 을 아직 붙이지 않는다.** scan 132 를 MuJoCo 레이캐스트로 만든 **GT height scan** 으로 채운다(= teacher 등가 입력). 이 단계가 순수 물리 갭 측정이다.

- 132개 격자점에서 위→아래 `mj_ray` → `clip(base_z − h − 0.3, ±1)`.
- 200Hz 수동 PD + DC 모터 포화 + 1-step 액션 지연 구현.
- `delta_yaw` 는 아직 오라클: MuJoCo 지형에 동일한 goal waypoint 를 심고 같은 식으로 계산하거나, 우선 **전방 고정 heading** 으로 대체.
- 성공 기준: 난이도 0.7 사다리꼴에서 IsaacLab PLAY 와 **비슷한 통과율**. 여기서 크게 떨어지면 EM 이 아니라 물리/액추에이터 문제다.

### Phase 3 — L1 LiDAR + EM 파이프라인 (3~5일)

1. `L1ScanRayCaster.valid_frame_times` / `directions_from_angles` 를 그대로 써서 0.1s 프레임의 2160개 (α, β) → 센서 프레임 방향 생성. 마운트: `pos=(0.28, 0, 0.10)`, `rot=(0,1,0,0)` (X축 180° = 돔 하향) (`default_cfg.py` `GO2_LIDAR_CFG`).
2. **`mujoco.mj_multiRay`** 로 캐스트. L1 은 전 ray 가 **공통 원점**이라 `mj_multiRay` 시그니처와 정확히 맞는다. 로봇 자신도 캐스트 대상에 포함(= IsaacLab 이 `Robot/.*/collisions` 를 타깃에 넣은 것과 동일).
3. 동일한 self-filter 캡슐 적용 → `valid` 마스크.
4. 동일한 노이즈: `range_std=0.02`, `ray_dir_std_deg=0.2`.
5. odometry drift 모델(§1 의 `odom_*` 파라미터) 을 **그대로 적용**. MuJoCo GT pose 에 학습과 같은 분포의 오차를 얹어야 입력 분포가 일치한다.
6. `BatchedElevationMapBackend(num_envs=1, ...)` 에 `update()` → `sample()` → 132.
7. **회귀 게이트**: 동일 지형·동일 궤적에서 IsaacLab em_scan 과 MuJoCo em_scan 의 셀당 평균 |차이| 를 비교. `play.py --em_error_plot` 이 이미 GT 대비 오차를 뽑으므로 같은 지표로 두 시뮬을 나란히 놓는다.

> cupy 버전은 13.x 로 고정 (기존 제약). MuJoCo 하네스도 같은 conda 환경(`env_isaaclab`)에서 돌리는 편이 의존성 충돌이 적다. `mujoco` 는 아직 미설치 — `pip install mujoco` 필요.

### Phase 4 — 도메인 갭 계량 (1~2일)

동일 지형·동일 시드·동일 명령으로 IsaacLab vs MuJoCo 를 비교표로:
통과율 / 목표 도달 수 / 평균 전진속도 / 접촉력 / 토크 RMS / 넘어짐 원인 분포.
갭이 크면 되돌아갈 순서: 액추에이터 모델 → 마찰/접촉 solver → 관성 파라미터 → EM 오차.

### Phase 5 — 배포 리허설 (unitree_mujoco + FSM)

1. `unitree_mujoco` 설치, `simulate/config.yaml` 에 `robot: go2`, `domain_id: 0`, `use_joystick: 1`.
2. `deploy/robots/go2` 를 복제해 **`State_Parkour`** 를 새로 작성:
   - `REGISTER_FSM(State_Parkour)` + `config.yaml` 의 `FSM._` 에 등록, 전환은 기존 `Passive ↔ FixStand ↔ Parkour`.
   - obs 조립은 unitree_rl_lab 의 ObservationManager 를 **쓰지 않고** §1.2 를 직접 구현.
   - `joint_ids_map` 은 반드시 export 로 생성 (아래 함정 #1).
3. EM 파이프라인 배치 선택지:
   - (a) **Python 사이드카**: EM 을 별도 프로세스로 돌려 132-vector 를 공유메모리/DDS 로 전달. 가장 빠른 길.
   - (b) **CPU C++ 포팅**: 맵이 3.2m/0.1m = **34×34 셀**뿐이고 tick 당 2160점이라 CPU 로 충분히 실시간이다. 실기 최종형은 이쪽. cupy 판을 골든 레퍼런스로 두고 회귀 비교.
   - (c) Python 전체 배포 (`unitree_sdk2_python`). 실기 지터가 커서 권장하지 않음.
4. 안전: `bad_orientation(>1.0 rad)` → Passive 를 그대로 유지 (`deploy/include/isaaclab/envs/mdp/terminations.h:10-15`).

---

## 4. 함정 목록 (실패 원인 후보, 체감 위험 순)

1. **관절 순서 — 실측 완료, `deploy/contract/deploy.yaml` 참조.** 셋이 전부 다르다:
   - IsaacLab(PhysX): **BFS** — `hip×4(FL,FR,RL,RR) → thigh×4 → calf×4`
   - MJCF(menagerie): 다리별 — `FL,FR,RL,RR × (hip,thigh,calf)`
   - Unitree SDK: 다리별 — `FR,FL,RR,RL × (hip,thigh,calf)`

   매핑(`il_to_sdk` = `[3,0,9,6,4,1,10,7,5,2,11,8]`, `il_to_mj` = `[0,3,6,9,1,4,7,10,2,5,8,11]`)과
   역방향까지 왕복 검증해 yaml 에 못박아 두었다. 재측정은
   `deploy/tools/dump_isaaclab_joint_order.py` → `deploy/tools/build_joint_maps.py`.
   `default_joint_pos` 는 **IsaacLab 순서**, `kp/kd` 는 export 시 **SDK 순서**로 뒤집히는 비대칭이
   unitree_rl_lab 에 실제로 있으므로(`export_deploy_cfg.py:30-36`) 그대로 흉내 내지 말고 한 쪽으로 통일할 것.
   **순서가 맞아도 회전축 부호 규약(USD vs MJCF)은 별도 확인이 필요하다 — 아직 미검증.**

   부수 실측 2건:
   - **기본 자세가 앞뒤 비대칭**이다: thigh 가 앞다리 0.8 / 뒷다리 **1.0**
     (`[0.1,-0.1,0.1,-0.1, 0.8,0.8,1.0,1.0, -1.5×4]`). menagerie `home` 키프레임은
     hip 0 / thigh 0.9 / calf −1.8 로 **다르다** — action offset 은 반드시 IsaacLab 값을 쓸 것.
   - menagerie MJCF 에는 `Head_upper/Head_lower` 와 `*_foot` **바디가 없다**(발은 geom).
     self-filter 캡슐이 이 이름들을 참조하므로 Phase 3 에서 대체 지점이 필요하다.
   - 참고: `parkour_mdp_cfg.py` 의 `TeacherRewardsCfg` docstring 에 적힌 body 목록은 실측 순서와
     다르다(문서만 낡음, regex 로 해석되므로 기능 영향 없음).
2. **DC 모터 포화 미구현** (§1.7). 순수 PD 로 두면 고속 구간 토크가 과대해져 점프/착지가 달라진다.
3. **1-step 액션 지연 누락.** 학습이 지연 있는 상태로 수렴했다. MuJoCo 하네스에서 끄고 켜며 둘 다 측정할 것.
4. **`delta_yaw` 의 1.5배 불일치.** 학습(`learn_em`)은 obs[6],[7] 에 **raw** `delta_yaw` 를 넣는다 (`observations.py:75-77`). 그런데 `play.py --fixed_heading` 은 `1.5*delta` 를 넣는다 (`play.py:620-623`) — depth encoder 시절 스케일의 잔재다. **배포는 raw 를 쓸 것.** 두 값으로 PLAY 를 돌려 거동 차이를 먼저 확인하면 좋다.
5. **`delta_yaw` 는 10Hz 신호.** 매 step 갱신하면 학습 분포와 다르다. `%5` 게이트 유지.
6. **지형 타입 플래그.** obs[11]=1.0, obs[12]=0.0 **상수**. 지형에 따라 바꾸면 v1.3 규약 위반.
7. **priv_latent 29칸.** 값은 무시되지만 자리는 필요 (history 꼬리 정렬).
8. **priv_explicit 은 GT 가 아니라 estimator 출력.** GT 선속도를 넣으면 배포 조건과 달라진다.
9. **roll/pitch 는 euler + wrap_to_pi.** `projected_gravity` 로 대체하면 안 된다. IsaacLab `euler_xyz_from_quat` 이 [0,2π) 를 반환하는 버전이 있어 wrap 이 필수다.
10. **접촉 관측.** 임계 2N, 현재/직전 프레임 OR, 출력 −0.5 오프셋. MuJoCo 에서는 발 geom 접촉력 합으로 재현.
11. **history 초기화.** 리셋 직후 10프레임을 현재값으로 채운다. 0으로 두면 첫 0.2초 거동이 달라진다.
12. **scan 격자 순서** (y 바깥 11 × x 안쪽 12). 전치하면 조용히 망가진다.
13. **base_z 기준 정규화.** 실기에서는 절대 z 를 모른다. 다행히 elevation map 이 center z 상대값을 저장하고 tick 마다 Δz 로 시프트하므로 **상대 높이 변화**만 있으면 된다 — 그래도 z drift 는 직접 관측 오차로 들어간다. Phase 3 에서 z drift 민감도를 반드시 스윕할 것.
14. **DR 재현.** 마찰 0.6~2.0, base mass −1~+3kg, COM ±2cm, 8초 주기 push. sim2sim 을 "깨끗한 조건"에서만 하면 실기 갭을 과소평가한다.
15. **obs clip ±100**, 정규화기 없음(Identity).
16. **EM 리셋 시맨틱.** 텔레포트/리셋 때 맵을 비우고 h_obs=0. 배포에서는 리셋이 없으므로 "시작 시 맵이 비어 있는 동안 h_obs=0" 구간을 어떻게 다룰지 결정해야 한다(로봇이 제자리에서 한 바퀴 스캔 후 출발 등).

**아래 3개는 Phase 0 을 하면서 실제로 걸린 것들이다.**

17. **`simulation_app.close()` 는 예외와 종료코드를 통째로 삼킨다.**
    `try: main() finally: simulation_app.close()` 로 감싸면 실패가 "출력 한 줄 없이
    exit 0" 으로 나온다. 이것 때문에 두 번 헤맸다. 대응: 예외를 **먼저** 찍고 flush 한
    뒤 닫고, `[RESULT] OK|FAILED` 마커를 남긴다. **종료코드로 성공 판정하지 말 것** —
    산출물 파일이나 마커로 판정한다. 덧붙여 Isaac 은 C++ 쪽 stdout 을 따로 쓰므로
    파이썬 출력과 순서가 섞인다. `python -u` 로 돌려야 어디서 죽었는지 보인다.
18. **student 씬의 `record_camera` 는 상시로 켜져 있다.** TiledCamera 라
    `--enable_cameras` 없이는 `RuntimeError: A camera was spawned without the
    --enable_cameras flag` 로 죽고, 켜면 env 마다 렌더가 돌아 VRAM 을 먹는다.
    train.py 가 학습 경로에서 떼어내듯 스크립트에서 `scene.record_camera = None` 로
    비우고 시작할 것.
19. **수치 비교에 절대 오차만 쓰면 안 된다.** ONNX 변환은 연산 순서를 바꾸므로
    float32 반올림 차이가 **값 크기에 비례**해 생긴다(eps=1.2e-7). 표준정규 랜덤
    입력을 넣으면 액션이 |35| 까지 커지고, 그 상태의 1.1e-5 차이는 상대 3.2e-7 —
    그냥 반올림인데 절대 기준 1e-5 로는 FAIL 로 보인다. ① 판정은
    `|d| <= atol + rtol·|want|` 로, ② 테스트 입력은 **실제 관측 규모**로 만들 것
    (prop/hist ~0.3 스케일, scan 은 정의상 ±1).

---

## 5. sim2real 로 가기 전에 반드시 풀어야 할 것

| 항목 | 현재 (sim) | 실기에서 필요한 것 |
|---|---|---|
| **odometry (xy, yaw, z)** | GT + 주입 drift | **최대 리스크.** Go2 sport-mode odometry, LiDAR-inertial odometry(FAST-LIO 계열), 또는 leg+IMU. 학습이 가정한 drift 크기(scale bias ±3%, gyro bias 0.01~0.05 deg/s) 안에 들어오는지 실측 필요 |
| 점군 | IsaacLab raycast | `unilidar_sdk2` → 센서 프레임 점군. 마운트 extrinsic 실측 후 `GO2_LIDAR_CFG.offset` 교체 |
| base 선속도 | estimator(prop) | 그대로 사용 가능 (설계상 이미 실기 조건) |
| 발 접촉 | 접촉력 2N 임계 | LowState `foot_force` 로 임계 재보정 |
| heading 지령 | 지형 goal 오라클 | 조이스틱 yaw 지령 → `delta_yaw` |
| EM 실시간성 | cupy/GPU | 34×34 맵 · 2160점 → CPU C++ 로 충분. 온보드 GPU 불필요 |

---

## 6. 저장소 경계와 디렉토리 구조

**결정(2026-09-02): 저장소를 둘로 나눈다.** 학습 repo 는 산출물만 뱉고, 이식·배포 코드는
포크한 `unitree_rl_lab` 에만 쌓는다. 이건 unitree_rl_lab 자신의 규약이기도 하다 — 그쪽은
학습 repo 가 `exported/policy.onnx` + `params/deploy.yaml` 을 만들고 배포 프로그램이
그 둘을 읽는다(`export_deploy_cfg.py` 가 학습 스크립트에서 호출된다).

```
Isaaclab_Parkour  (이 repo — 학습 전용, 인터페이스는 아래 둘뿐)
└─ deploy/
   ├─ contract/deploy.yaml         # 관절 순서·게인·스케일·차원 (완료)
   ├─ tools/                       # 실측/덤프 도구
   │   ├─ dump_isaaclab_joint_order.py   (완료)
   │   ├─ build_joint_maps.py            (완료)
   │   ├─ export_em_student_onnx.py      # ★ 미구현 — 아래 §6.1
   │   ├─ dump_golden_trace.py           # IsaacLab 100 step 트레이스
   │   └─ export_terrain.py              # 지형 → OBJ/hfield
   └─ (그 외 이식 코드는 여기 두지 않는다)

unitree_rl_lab (포크)  — 이식·배포 전부
└─ deploy/
   ├─ parkour/
   │   ├─ common/                  # 시뮬레이터 무관, 세 곳에서 글자 그대로 같아야 하는 코드
   │   │   ├─ obs_assembler.py     # §1.2 조립 + history 큐 + 10Hz 게이팅
   │   │   ├─ policy_runner.py     # policy.onnx 로더
   │   │   ├─ em_pipeline.py       # self-filter + 노이즈 + odom drift + EM + 샘플
   │   │   └─ l1_scan.py           # L1 방향 생성 (raycast 백엔드는 주입)
   │   ├─ vendored/                # Isaaclab_Parkour 에서 복사 (순수 torch, isaaclab 비의존)
   │   │   ├─ elevation_map_backend.py
   │   │   └─ self_filter_capsules.py
   │   └─ mujoco/
   │       ├─ go2_scene.xml / terrain.obj|hfield
   │       ├─ raycast_mj.py        # mj_multiRay 백엔드
   │       └─ run_sim2sim.py       # 200Hz PD + DC 포화 + 1-step 지연
   └─ robots/go2_parkour/          # 트랙 B: 커스텀 State_Parkour (C++)
```

`common/` 이 핵심이다. **여기 있는 코드는 MuJoCo 하네스와 실기에서 글자 그대로 같은 것**이어야
하며, 차이는 raycast 백엔드와 상태 읽기 어댑터로만 주입한다. 그래야 "sim2sim 갭"이 물리 갭인지
구현 갭인지 구분된다.

`vendored/` 는 복사본이다. 학습 repo 를 파이썬 의존성으로 걸면 배포 쪽이 IsaacLab 설치를
요구하게 되므로(패키지 `__init__` 이 isaaclab 을 끌어온다) **파일 단위로 복사**하고, 원본 커밋
SHA 를 파일 머리에 적어 둔다.

### 6.1 ONNX export 인터페이스 (미구현)

현재 `play.py` 는 EM student 의 export 를 지원하지 않는다 —
`"[INFO] EMDistillation: JIT/ONNX export 는 지원하지 않는다"`. depth student 용 exporter 만 있다.

정책에 재귀 구조가 없음을 확인했으므로(`depth_actor` 는 `priv_encoder / history_encoder /
scan_encoder / actor_backbone` 뿐, GRU/LSTM 없음) **상태 없는 순수 함수 하나로 뽑을 수 있다.**
권장 시그니처는 753 통짜가 아니라 3입력이다:

```
inputs : prop(1,53)  scan(1,132)  hist(1,530)
outputs: actions(1,12)
내부   : estimator(prop)→9, scan_encoder(scan)→32, history_encoder(hist)→20,
         actor_backbone(concat 114)→12
```

- `priv_latent` 29 칸은 `hist_encoding=True` 경로에서 쓰이지 않으므로 인터페이스에서 뺀다.
  753 통짜로 뽑으면 배포 쪽이 슬라이스 정렬을 틀릴 여지만 남는다.
- unitree_rl_lab 의 `OrtRunner` 는 ONNX 입력을 **이름으로** 조회하므로 다입력이 그대로 맞는다
  (`deploy/include/isaaclab/algorithms/algorithms.h:40-83`).
- 정책 바깥에 남는 유일한 상태는 **history 링버퍼**다. 배포 코드가 관리한다.

---

## 7. 검증 체크리스트

- [x] **선결**: `elevation_mapping_cupy` submodule 고정(`20a8a26`) + import 확인 (`smoke_em_backend.py` 는 GPU 확보 후 별도 실행)
- [x] **선결**: `env_sim2real` 생성 + mujoco 3.12.0 설치
- [x] **선결**: 관절 순서 3종(IsaacLab / MJCF / SDK) 실측 → `deploy/contract/deploy.yaml`
- [ ] **선결**: 관절 회전축 부호 규약(USD vs MJCF) 대조, SDK 순서를 unitree_sdk2 헤더와 교차 검증
- [ ] **선결**: `unitree_rl_lab` 포크 + origin 교체 (이식 코드를 쌓을 곳)
- [x] **P0**: EM student `policy.onnx` 추출 (§6.1 3입력 시그니처) — `deploy/tools/export_em_student_onnx.py`
- [x] **P0**: 골든 트레이스 100 스텝 덤프 — `deploy/tools/dump_golden_trace.py`
- [x] **P0**: **골든 테스트 PASS** — IsaacLab 실관측 100 샘플에서 `max|diff| = 4.77e-6`
      (허용 1e-5), 평균 2.4e-7. torch 기준값 검사도 `max|diff| = 6.68e-6`.
      검증은 torch/IsaacLab 없이 onnxruntime+numpy 만으로 수행 (`check_onnx_against_trace.py`)
- [ ] **P1**: MuJoCo 정지 상태 GT scan 132 vs IsaacLab 동일 pose scan 차이 < 1e-3 m
- [ ] **P1**: 같은 q_target 스텝 입력에 대한 관절 궤적(200Hz) 이 IsaacLab 과 육안 일치, 토크 포화 구간 재현
- [ ] **P2**: GT scan 으로 난이도 0.7 사다리꼴 통과율이 IsaacLab PLAY 대비 −20%p 이내
- [ ] **P3**: MuJoCo em_scan vs GT scan 셀당 평균 오차가 IsaacLab 의 `--em_error_plot` 곡선과 같은 수준
- [ ] **P3**: EM 켠 상태 통과율이 GT scan 대비 크게 떨어지지 않음
- [ ] **P4**: DR 켠 상태(마찰/질량/push)에서도 통과율 유지
- [ ] **P5**: unitree_mujoco + FSM 에서 조이스틱 전환·Passive 폴백 동작, 50Hz 정책 주기 지터 < 2ms
