"""LiDAR PPO 정책을 배포용 ONNX로 내보낸다.

시그니처
-------------------------------------------
    inputs : prop(1,53)  scan(1,132)  hist(1,530)
    output : actions(1,12)

정책의 전체 observation은 753차원이지만, 배포 인터페이스는 세 입력으로 나눈다.
  - obs[194:223] 의 priv_latent 29 칸은 `hist_encoding=True` 경로에서 아예 읽히지
    않는 죽은 입력이다. 배포 코드가 채워 줄 이유가 없다.
  - 통짜 벡터는 배포 쪽이 슬라이스 경계를 틀릴 여지만 남긴다. 입력을 셋으로 쪼개면
    그 실수가 구조적으로 불가능해진다.
  - estimator(priv_explicit 9칸을 만드는 망)를 그래프 안으로 넣어 배포 쪽이
    "prop 을 넣으면 알아서 된다" 가 되게 한다.
그래프 안: estimator(prop)->9, scan_encoder(scan)->32, history_encoder(hist)->20,
actor_backbone(concat 114)->12.

정책에 재귀 구조(GRU/LSTM)가 없어서 상태 없는 순수 함수로 떨어진다. 배포 쪽에
남는 유일한 상태는 history 링버퍼이고 그건 obs 조립 코드가 관리한다.

구조를 cfg 가 아니라 체크포인트에서 읽는 이유
---------------------------------------------
agent cfg(`locomotion_rl_cfg.py`)는 isaaclab 을 import 한다. 그러면 export 에
Isaac Sim 이 필요해진다. 가중치 shape 만으로 거의 모든 치수가 복원되므로
체크포인트에서 읽고, 복원한 값들끼리 아귀가 맞는지 assert 로 검증한다.

다만 **weights 로 복원 불가능한 것이 둘** 있다 — 활성함수와 tanh_encoder_output.
둘 다 CLI 플래그로 두고 기본값을 학습 cfg 값(elu / False)으로 뒀다.
내장 검증은 이 설정이 학습 때와 같은지 확인하지 못하므로 학습 설정과 맞춰야 한다.

실행 (Isaac Sim 불필요)
----------------------
    python deploy/tools/export_em_student_onnx.py \
        --checkpoint logs/rsl_rl/unitree_go2_locomotion/<timestamp>_lidar/model_29999.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

# scripts/rsl_rl 를 import 경로에 넣어야 modules 패키지가 보인다.
# (modules 는 rsl_rl 만 쓰고 isaaclab 을 건드리지 않는다 — 그래서 Isaac Sim 없이 돈다.)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "scripts", "rsl_rl"))

from modules.actor_critic_with_encoder import Actor  # noqa: E402
from modules.feature_extractors.estimator import DefaultEstimator  # noqa: E402
from rsl_rl.utils import resolve_nn_activation  # noqa: E402


# ---------------------------------------------------------------------------
# 1) PPO 체크포인트에서 actor 가중치를 분리하고 망 구조를 복원한다
# ---------------------------------------------------------------------------
def validate_checkpoint(ckpt: dict) -> None:
    """Fail fast for checkpoints that this deployment graph cannot represent."""
    required = {"model_state_dict", "estimator_state_dict"}
    missing = sorted(required.difference(ckpt))
    if missing:
        raise ValueError(f"PPO checkpoint is missing required keys: {missing}")
    if ckpt.get("algorithm") != "PPOWithExtractor":
        raise ValueError("checkpoint algorithm must be 'PPOWithExtractor'")
    if ckpt.get("input_mode") != "lidar_input":
        raise ValueError("checkpoint input_mode must be 'lidar_input'")
    if "obs_norm_state_dict" in ckpt or "privileged_obs_norm_state_dict" in ckpt:
        raise ValueError("empirical observation normalization is not supported by this split-input exporter")
    if "depth_actor_state_dict" in ckpt or "depth_encoder_state_dict" in ckpt:
        raise ValueError("legacy non-PPO checkpoints are not supported")


def extract_actor_state_dict(model_sd: dict) -> dict:
    """Extract ``Actor`` parameters from an ``ActorCriticRMA`` state dictionary."""
    prefix = "actor."
    actor_sd = {key[len(prefix) :]: value for key, value in model_sd.items() if key.startswith(prefix)}
    if not actor_sd:
        raise ValueError("model_state_dict contains no 'actor.*' parameters")
    return actor_sd


def infer_arch(actor_sd: dict, est_sd: dict) -> dict:
    def linear_out_dims(sd, prefix):
        """nn.Sequential 안의 Linear 출력 폭을 인덱스 순서대로 모은다."""
        idxs = sorted(
            {
                int(k.split(".")[len(prefix.split(".")) - 1])
                for k in sd
                if k.startswith(prefix) and k.endswith(".weight")
            }
        )
        return [sd[f"{prefix}{i}.weight"].shape[0] for i in idxs], idxs

    a = {}
    a["num_prop"] = int(est_sd["estimator.0.weight"].shape[1])
    a["num_scan"] = int(actor_sd["scan_encoder.0.weight"].shape[1])
    a["num_priv_latent"] = int(actor_sd["priv_encoder.0.weight"].shape[1])

    est_outs, _ = linear_out_dims(est_sd, "estimator.")
    a["estimator_hidden_dims"] = est_outs[:-1]  # 마지막 층은 출력이라 hidden 이 아니다
    a["num_priv_explicit"] = est_outs[-1]

    scan_outs, _ = linear_out_dims(actor_sd, "scan_encoder.")
    a["scan_encoder_dims"] = scan_outs

    priv_outs, _ = linear_out_dims(actor_sd, "priv_encoder.")
    a["priv_encoder_dims"] = priv_outs

    actor_outs, _ = linear_out_dims(actor_sd, "actor_backbone.")
    a["actor_hidden_dims"] = actor_outs[:-1]
    a["num_actions"] = actor_outs[-1]

    # history encoder: 첫 Linear 가 num_prop -> 3*channel_size
    a["channel_size"] = int(actor_sd["history_encoder.encoder.0.weight"].shape[0]) // 3

    # num_hist(tsteps) 는 폭이 아니라 conv 커널 크기로만 구분된다.
    # StateHistoryEncoder 는 tsteps 10/20/50 에서 각각 커널 (4,2) / (6,4) / (8,5,5) 를 쓴다.
    k0 = int(actor_sd["history_encoder.conv_layers.0.weight"].shape[2])
    a["num_hist"] = {4: 10, 6: 20, 8: 50}[k0]

    # 아귀 검사: actor_backbone 입력 폭 == prop + scan_latent + priv_explicit + hist_latent
    expected_in = a["num_prop"] + a["scan_encoder_dims"][-1] + a["num_priv_explicit"] + a["priv_encoder_dims"][-1]
    actual_in = int(actor_sd["actor_backbone.0.weight"].shape[1])
    assert expected_in == actual_in, (
        f"actor_backbone 입력 폭 불일치: 복원값 {expected_in} vs 실제 {actual_in}. "
        "구조 복원이 틀렸다는 뜻이므로 여기서 멈춘다."
    )
    # history_encoder 출력은 priv_encoder 출력과 같은 폭이어야 한다 (Actor 가 그렇게 만든다)
    assert int(actor_sd["history_encoder.linear_output.0.weight"].shape[0]) == a["priv_encoder_dims"][-1]
    assert int(est_sd["estimator.0.weight"].shape[1]) == a["num_prop"]
    return a


# ---------------------------------------------------------------------------
# 2) 배포용 래퍼: 입력 셋을 받아 학습 때와 '같은' Actor.forward 를 부른다
# ---------------------------------------------------------------------------
class LidarPolicy(nn.Module):
    """prop/scan/hist -> actions.

    forward 를 새로 구현하지 않고 obs 벡터를 재조립해 `Actor.forward` 를 그대로
    부른다. 재구현하면 concat 순서 같은 걸 틀릴 수 있는데, 그런 버그는 조용하고
    사후에 잡기 어렵다. 여기서는 학습/재생과 문자 그대로 같은 코드가 돈다.

    priv_latent 자리는 0 으로 채운다 — hist_encoding=True 면 Actor 가 priv_encoder
    를 아예 부르지 않으므로 값이 무엇이든 출력에 영향이 없고, 트레이스에도 안 남는다.
    """

    def __init__(self, actor: Actor, estimator: DefaultEstimator, num_priv_latent: int):
        super().__init__()
        self.actor = actor
        self.estimator = estimator
        self.num_priv_latent = num_priv_latent

    def forward(self, prop: torch.Tensor, scan: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        priv_explicit = self.estimator(prop)
        priv_latent = torch.zeros(prop.shape[0], self.num_priv_latent, dtype=prop.dtype, device=prop.device)
        obs = torch.cat([prop, scan, priv_explicit, priv_latent, hist], dim=1)
        return self.actor(obs, hist_encoding=True, scandots_latent=None)


def build_policy(ckpt: dict, arch: dict, activation: str, tanh_encoder_output: bool) -> LidarPolicy:
    act_module = resolve_nn_activation(activation)
    actor = Actor(
        arch["num_actions"],
        arch["scan_encoder_dims"],
        arch["actor_hidden_dims"],
        arch["priv_encoder_dims"],
        act_module,
        tanh_encoder_output=tanh_encoder_output,
        num_prop=arch["num_prop"],
        num_scan=arch["num_scan"],
        num_hist=arch["num_hist"],
        num_priv_latent=arch["num_priv_latent"],
        num_priv_explicit=arch["num_priv_explicit"],
        state_history_encoder={
            "class_name": "StateHistoryEncoder",
            "channel_size": arch["channel_size"],
        },
    )
    estimator = DefaultEstimator(
        num_prop=arch["num_prop"],
        num_priv_explicit=arch["num_priv_explicit"],
        hidden_dims=arch["estimator_hidden_dims"],
        activation=activation,
    )
    # strict=True: 키가 하나라도 남거나 모자라면 실패한다. 조용한 부분 로드를 막는다.
    actor.load_state_dict(extract_actor_state_dict(ckpt["model_state_dict"]), strict=True)
    estimator.load_state_dict(ckpt["estimator_state_dict"], strict=True)

    policy = LidarPolicy(actor, estimator, arch["num_priv_latent"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


# ---------------------------------------------------------------------------
# 3) 학습/재생 경로와 수치가 같은지 확인 (torch 안에서)
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_against_production_path(policy: LidarPolicy, arch: dict, n: int = 8):
    """Compare the split-input graph with the PPO actor inference path."""
    P, S, H = arch["num_prop"], arch["num_scan"], arch["num_hist"] * arch["num_prop"]
    PE, PL = arch["num_priv_explicit"], arch["num_priv_latent"]
    g = torch.Generator().manual_seed(0)

    # 입력 크기를 실제 관측 규모로 맞춘다. prop 성분은 대부분 |x| < 1 이고
    # (ang_vel×0.25, joint_vel×0.05, contact ±0.5, joint_pos 편차 ~0.3) scan 은
    # 정의상 ±1 로 clip 된다. 표준정규를 그냥 넣으면 MLP 활성이 실제보다 한참
    # 커져서 액션이 |35| 같은 값이 나오고, 그 상태로 절대 오차를 재면 float32
    # 반올림이 오차처럼 보인다. 검증은 실제로 돌 구간에서 해야 의미가 있다.
    prop = torch.randn(n, P, generator=g) * 0.3
    scan = torch.rand(n, S, generator=g) * 2.0 - 1.0
    hist = torch.randn(n, H, generator=g) * 0.3

    # 학습 경로: 753 통짜 obs 를 만들고 estimator 로 priv_explicit 자리를 덮어쓴다.
    obs = torch.cat([prop, scan, torch.randn(n, PE, generator=g), torch.randn(n, PL, generator=g), hist], dim=1)
    obs[:, P + S : P + S + PE] = policy.estimator.inference(obs[:, :P])
    ref = policy.actor(obs, hist_encoding=True, scandots_latent=None)

    out = policy(prop, scan, hist)
    diff = (ref - out).abs().max().item()
    # The two paths assemble the same actor input. Float32 GEMM kernels can still
    # differ by an ULP when another test or caller changes PyTorch backend settings.
    torch.testing.assert_close(out, ref, rtol=1.0e-6, atol=1.0e-7)
    print(f"  [check] split 입력과 PPO actor 출력 일치 (max|diff|={diff:.1e})")
    return prop, scan, hist, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="LiDAR PPO checkpoint (.pt)")
    ap.add_argument("--out-dir", default=None, help="기본값: <체크포인트 폴더>/exported")
    ap.add_argument("--activation", default="elu", help="가중치로 복원 불가 — 학습 cfg 값 (기본 elu)")
    ap.add_argument("--tanh-encoder-output", action="store_true", help="가중치로 복원 불가 — 학습 cfg 는 False")
    ap.add_argument("--opset", type=int, default=11, help="unitree_rl_lab 의 onnxruntime 1.22 와 맞춘 값")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    try:
        validate_checkpoint(ckpt)
        actor_sd = extract_actor_state_dict(ckpt["model_state_dict"])
    except ValueError as exc:
        raise SystemExit(f"Unsupported checkpoint: {exc}") from exc

    arch = infer_arch(actor_sd, ckpt["estimator_state_dict"])
    print("체크포인트에서 복원한 구조:")
    for k, v in arch.items():
        print(f"  {k:24s} = {v}")

    policy = build_policy(ckpt, arch, args.activation, args.tanh_encoder_output)
    prop, scan, hist, ref_out = check_against_production_path(policy, arch)

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "exported")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "policy.onnx")

    # 배치 1 고정으로 뽑는다. unitree_rl_lab 의 OrtRunner 는 모델에서 정적 shape 를
    # 읽어 버퍼 크기를 계산하므로, 동적 축(-1)을 남기면 엉뚱한 크기를 잡는다
    # (deploy/include/isaaclab/algorithms/algorithms.h:47-53).
    dummy = (
        torch.zeros(1, arch["num_prop"]),
        torch.zeros(1, arch["num_scan"]),
        torch.zeros(1, arch["num_hist"] * arch["num_prop"]),
    )
    torch.onnx.export(
        policy,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        input_names=["prop", "scan", "hist"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"\nONNX 저장: {onnx_path}")

    # 배포 쪽(env_sim2real, torch 없음)이 대조할 수 있게 torch 기준값을 함께 남긴다.
    ref_path = os.path.join(out_dir, "onnx_reference.npz")
    import numpy as np

    np.savez(
        ref_path,
        prop=prop.numpy(),
        scan=scan.numpy(),
        hist=hist.numpy(),
        actions=ref_out.numpy(),
    )
    print(f"torch 기준값 저장: {ref_path}  (배치 {prop.shape[0]}개)")

    meta = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "iter": int(ckpt.get("iter", -1)),
        "activation": args.activation,
        "tanh_encoder_output": bool(args.tanh_encoder_output),
        "opset": args.opset,
        "inputs": {"prop": arch["num_prop"], "scan": arch["num_scan"], "hist": arch["num_hist"] * arch["num_prop"]},
        "outputs": {"actions": arch["num_actions"]},
        "arch": arch,
        "note": "hist 는 과거 prop 프레임 num_hist 개를 [t-9 ... t] 순서로 이어붙인 것. "
        "prop 인덱스 6:8(delta_yaw)은 history 에 넣기 전에 0 으로 만든다.",
    }
    meta_path = os.path.join(out_dir, "policy_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"메타 저장: {meta_path}")


if __name__ == "__main__":
    main()
