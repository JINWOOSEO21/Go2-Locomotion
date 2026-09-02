"""추출한 policy.onnx 가 IsaacLab 과 같은 액션을 내는지 검증한다 (Phase 0 골든 테스트).

**torch 도 IsaacLab 도 쓰지 않는다.** onnxruntime + numpy 뿐이다. 이건 의도된 제약이다 —
배포 환경(로봇/MuJoCo 하네스)이 가질 의존성만으로 검증해야 "여기선 되는데 저기선 안 되는"
상황을 미리 잡는다. 그래서 이 스크립트는 env_sim2real 에서 돌린다.

두 가지를 본다
--------------
1) `onnx_reference.npz` — exporter 가 torch 로 계산해 둔 기준값. 랜덤 입력이라
   수치 경로만 본다. 여기서 틀리면 ONNX 변환 자체가 깨진 것이다.
2) `golden_trace.npz` — IsaacLab 을 실제로 굴려 얻은 진짜 관측/액션.
   여기서 틀리면 변환은 됐는데 **의미**가 틀린 것이다(활성함수, 슬라이스 정렬,
   hist 프레임 순서 등). exporter 가 weights 로 복원할 수 없는 것들이 여기서 잡힌다.

실행
----
    python deploy/tools/check_onnx_against_trace.py \
        --onnx      logs/.../exported/policy.onnx \
        --reference logs/.../exported/onnx_reference.npz \
        --trace     logs/.../exported/golden_trace.npz
"""

import argparse
import sys

import numpy as np
import onnxruntime as ort


def load_session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  # 배포와 같은 설정
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    print(f"[onnx] {path}")
    for i in sess.get_inputs():
        print(f"  input  {i.name:8s} {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"  output {o.name:8s} {o.shape} {o.type}")
    return sess


def run_rows(sess, prop, scan, hist):
    """배치 1 고정으로 뽑았으므로 한 행씩 넣는다 (배포도 이렇게 부른다)."""
    out = []
    for i in range(prop.shape[0]):
        r = sess.run(
            ["actions"],
            {
                "prop": prop[i : i + 1].astype(np.float32),
                "scan": scan[i : i + 1].astype(np.float32),
                "hist": hist[i : i + 1].astype(np.float32),
            },
        )[0]
        out.append(r[0])
    return np.stack(out)


def report(name: str, got: np.ndarray, want: np.ndarray, atol: float, rtol: float) -> bool:
    """np.allclose 규약(|d| <= atol + rtol*|want|)으로 판정한다.

    순수 절대 오차로 재면 안 된다. ONNX 변환은 연산 순서를 바꾸므로 float32
    반올림 차이가 값 크기에 비례해서 생긴다(eps=1.2e-7). 액션이 O(1) 이면
    절대 오차가 곧 유효 판정이지만, 값이 커지면 반올림이 오차처럼 보인다.
    실제 액션은 ±4.8 로 clip 되므로 atol 이 사실상의 기준이 된다.
    """
    d = np.abs(got - want)
    limit = atol + rtol * np.abs(want)
    ok = bool((d <= limit).all())
    denom = np.maximum(np.abs(want), 1e-9)
    rel = d / denom
    print(f"\n[{name}]  샘플 {want.shape[0]}개 x {want.shape[-1]}차원")
    print(f"  max|diff|     = {d.max():.3e}   (허용 = {atol:.0e} + {rtol:.0e}·|want|)")
    print(f"  mean|diff|    = {d.mean():.3e}")
    print(f"  max relative  = {rel.max():.3e}   (float32 eps = 1.2e-07)")
    print(f"  액션 크기: |want| 평균 {np.abs(want).mean():.4f}, 최대 {np.abs(want).max():.4f}")
    j = np.unravel_index(np.argmax(d), d.shape)
    print(f"  최대차 지점 sample={j[0]} joint={j[1]}: got={got[j]:.6f} want={want[j]:.6f}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--reference", default=None, help="exporter 가 남긴 onnx_reference.npz")
    ap.add_argument("--trace", default=None, help="dump_golden_trace.py 가 남긴 golden_trace.npz")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    sess = load_session(args.onnx)
    results = []

    if args.reference:
        r = np.load(args.reference)
        got = run_rows(sess, r["prop"], r["scan"], r["hist"])
        results.append(report("1. torch 기준값 (실제 규모 랜덤 입력)", got, r["actions"], args.atol, args.rtol))

    if args.trace:
        t = np.load(args.trace, allow_pickle=True)
        # 트레이스는 (T, num_envs, D). env 축을 펴서 T*num_envs 개 샘플로 본다.
        def flat(k):
            a = t[k]
            return a.reshape(-1, a.shape[-1])

        prop, scan, hist, act = flat("prop"), flat("scan"), flat("hist"), flat("actions")
        print(f"\n[trace] task={t['task']}  스텝 {t['prop'].shape[0]} x env {t['prop'].shape[1]}")
        got = run_rows(sess, prop, scan, hist)
        results.append(report("2. IsaacLab 골든 트레이스 (실제 관측)", got, act, args.atol, args.rtol))

    if not results:
        raise SystemExit("--reference 나 --trace 중 하나는 줘야 한다.")
    print("\n" + ("전체 PASS" if all(results) else "실패한 검사가 있다"))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
