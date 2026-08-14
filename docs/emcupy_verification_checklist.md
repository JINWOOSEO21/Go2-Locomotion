# EM student (v1.3) 검증 체크리스트 — VRAM 확보 후 순서대로

> **2026-08-13 실행 결과**: 0~5단계 전부 통과. Phase 3′(batched 커널 포트)까지
> 완료되어 EM tick 250.8ms → 8.6ms(192 env), sim 포함 95.1 → 49.0 ms/step.
> 회귀 검증은 `scripts/emcupy_check/regression_batched.py` (em_cupy 의
> upper_bound 가 비원자적 write 경쟁으로 원래 비결정적이라 판정은
> 마스크 일치/tick0 일치/GT RMSE 동등성으로 한다). 본학습(5000 iter)은
> batched 백엔드 + 192 env 로 진행 (~15h 예상).

계획서: docs/emcupy_student_plan.md. 코드는 모두 구현·커밋된 상태이고,
아래는 GPU 가 빌 때 실행할 런타임 검증이다. 모든 명령은 env_isaaclab 환경에서.
CPU 단위 테스트(0단계)만 VRAM 없이 언제든 가능하다.

## 0. CPU 단위 테스트 (VRAM 불필요 — 이미 통과 확인됨)

```bash
python parkour_test/test_em_student_cpu.py
```
- L1 스캔 운동학(유효 반주기/프레임 구조/반구/결정론) 4개
- self-hit 캡슐 필터(관통 판정/t0 스킵) 2개
- 샘플링(teacher 식/bilinear/cascade/보더/clip) 7개
- **기준: 13/13 PASS** (2026-08-12 확인 완료)

## 1. 의존성 (1회)

```bash
python -c "import cupy; print(cupy.__version__)"
# 없으면 (CUDA 12.x 기준):
pip install cupy-cuda12x ruamel.yaml simple_parsing shapely scipy
```
- elevation_mapping_cupy 클론이 repo root 에 있어야 한다 (다른 곳이면
  `EMCUPY_ROOT=/path/to/elevation_mapping_cupy` 환경변수로 지정).

## 2. Phase 0 — em_cupy 백엔드 단독 smoke (VRAM ~1GB, Isaac 불필요)

```bash
python scripts/emcupy_check/smoke_em_backend.py --out /tmp/emcupy_smoke.txt
```
- 합성 지형(평지+0.2m 단차)에서 end-to-end: 인스턴스 생성 → 점군 입력 →
  scandots 격자 샘플 → 해석 정답 비교.
- **기준: RMSE(valid) < 0.05 m, invalid < 20 %** → 스크립트가 PASS/FAIL 출력.
- 첫 tick 은 NVRTC 커널 컴파일이 포함되어 수 초 걸림(정상, 이후 캐시).

## 3. Phase 1 — L1 센서 시각 확인 (Isaac 필요, GUI 권장)

```bash
python scripts/lidar_sim/viz_l1_scan.py   # teacher-Play 씬에 lidar 주입, 스캔 패턴 확인
```
- 확인: 하향 반구 커버리지, yaw 1/3/11회전 누적 형상, self-occlusion(로봇 그림자).

## 4. Phase 2 — 시뮬레이션 GT scandots vs EM 샘플 (Isaac headless)

```bash
python scripts/emcupy_check/compare_em_vs_scandots.py --num_envs 4 --steps 250 --headless \
    --out /tmp/emcupy_compare.txt
```
- zero action(제자리 서기) 5초. env cfg 검증 관례대로 headless + 결과는 파일로.
- **기준: RMSE(valid) < 0.05 m, invalid < 20 % (1s 워밍업 후)** — Q7 sanity 게이트.
- 보행 중 검증은 6단계의 play 로 정성 확인(움직이는 로봇 + odometry 노이즈 하의 맵).

## 5. Phase 3 — 192 env 벤치마크 (성능·VRAM 판정)

```bash
# 5a. 백엔드만 (Isaac 없이 인스턴스 루프 비용 분리 측정)
python scripts/emcupy_check/smoke_em_backend.py --num-envs 192 --out /tmp/emcupy_bench.txt
# 5b. 시뮬레이션 포함 총 step 시간 + VRAM
python scripts/emcupy_check/compare_em_vs_scandots.py --num_envs 192 --steps 300 --headless \
    --out /tmp/emcupy_bench_sim.txt
nvidia-smi --query-gpu=memory.used --format=csv   # cupy pool 포함 총 VRAM
```
- 판정 기준(계획서 §4-리스크 1,2):
  - EM tick 비용이 학습 스텝 시간을 지배하면(대략 tick 당 > 수십 ms) → batched
    커널 포트(Phase 3′) 착수. 그 전까지는 진행 가능.
  - 총 VRAM 이 12GB 에 근접하면 §7 의 대응 순서(expandable_segments → cupy 풀
    제한 → num_envs 축소) 적용.

## 6. Phase 4 — 짧은 학습 + 정성 확인

```bash
# 수백 iteration 만 — loss 하강 곡선 확인용 (teacher ckpt 규약은 depth student 와 동일)
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-v0 \
    --seed 1 --headless --max_iterations 300
# 정성 확인 (16 env, GUI)
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-Play-v0
```
- 확인: depth_actor_loss 가 depth student 초기 학습과 유사한 스케일에서 하강하는지,
  play 에서 계단/램프 접근 시 정지·회피가 아닌 등반 시도가 나오는지.
- 체크포인트는 `logs/rsl_rl/unitree_go2_parkour/student_pretrained/<timestamp>/`
  (depth student 와 같은 폴더 규약을 공유한다).

## 7. Phase 5 — 본학습 + 평가

```bash
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-v0 \
    --seed 1 --headless          # 5000 iter (기존 실측 기준 depth student ≈ 11.4h 이하 예상)
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-EM-Student-Unitree-Go2-Eval-v0 --headless
```
- 최종 지표: 기존 depth student 대비 장애물 8종 성공률 (RMSE/invalid 는 여기서는
  사용하지 않는다 — Q7 합의대로 사전 sanity 게이트 전용).
- 평가 전에 학습 산출물 중 1개를 `student_pretrained/model_*.pt` 최상위로 승격
  (depth student 의 weight_candidates 규약과 동일). 승격 전이라도 train --resume /
  play 는 최상위에 checkpoint 가 없으면 최근 하위 run 폴더로 내려가 찾는다.

## 알려진 제약 / 주의

- play.py 의 JIT/ONNX export 는 EM student 미지원 (EM 파이프라인이 정책 밖).
- 샘플링은 GT base pose 로 조회한다(odometry 노이즈는 맵 등록 오차로만 반영) —
  실기에서는 추정 pose 조회가 되므로 sim2real 단계에서 재검토.
- `update_mesh_ids` 는 켜지 말 것 (MultiMeshRayCaster upstream shape 버그).
