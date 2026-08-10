# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""eval_fixed_difficulty.py 가 남긴 episodes.csv 를 요약하고 그래프로 그린다.

Isaac Sim 이 필요 없다. 평가가 끝난 뒤(또는 도중에) 따로 돌리면 된다.

    python scripts/rsl_rl/plot_eval_fixed_difficulty.py --input_dir logs/eval/...

만드는 것:
    summary.csv     iteration 별 집계 (전체 / 완주 에피소드 한정)
    reward.png      iteration vs 초당 reward (클리핑 이전 가중합)
    diagnostics.png 완주율과 종료 원인 분해. reward 비교가 생존율 차이에
                    오염됐는지 확인하는 용도다.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dataviz 스킬의 검증된 기본 팔레트 (light). 순서 고정, 순환 금지.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedcd6"

REASON_ORDER = ["goal", "timeout", "fall_roll", "fall_pitch", "height"]
REASON_LABEL = {
    "goal": "완주 (goal 8/8)",
    "timeout": "시간 초과",
    "fall_roll": "전복 (roll)",
    "fall_pitch": "전복 (pitch)",
    "height": "추락",
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    """평균과 95% 신뢰구간 반쪽 폭. 에피소드끼리 독립이라고 보고 정규근사."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(n))


def build_summary(df: pd.DataFrame, raw_cols: list[str]) -> pd.DataFrame:
    rows = []
    for iteration, group in df.groupby("iteration", sort=True):
        done = group[group["success"] == 1]
        row = {"iteration": int(iteration), "n_episodes": len(group), "n_success": len(done)}
        row["success_rate"] = len(done) / len(group) if len(group) else float("nan")

        for label, subset in (("all", group), ("success", done)):
            mean, ci = mean_ci(subset["rew_total_raw_per_s"].to_numpy())
            row[f"rew_per_s_{label}_mean"] = mean
            row[f"rew_per_s_{label}_ci95"] = ci
            row[f"duration_s_{label}_mean"] = subset["duration_s"].mean() if len(subset) else float("nan")
            row[f"dist_from_start_{label}_mean"] = (
                subset["dist_from_start"].mean() if len(subset) else float("nan")
            )
            for col in raw_cols:
                row[f"{col}_{label}_mean"] = subset[col].mean() if len(subset) else float("nan")

        counts = group["term_reason"].value_counts()
        for reason in REASON_ORDER:
            row[f"frac_{reason}"] = counts.get(reason, 0) / len(group) if len(group) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)


def plot_reward(summary: pd.DataFrame, meta: dict, path: str):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    x = summary["iteration"].to_numpy()
    series = [
        ("success", "완주한 에피소드만", SERIES[0]),
        ("all", "전체 에피소드", SERIES[1]),
    ]
    for key, label, color in series:
        y = summary[f"rew_per_s_{key}_mean"].to_numpy()
        ci = summary[f"rew_per_s_{key}_ci95"].to_numpy()
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)

    difficulty = meta.get("difficulty", "?")
    terrains = ", ".join(meta.get("active_terrains", []))
    ax.set_title(
        f"고정 난이도 {difficulty} 에서의 초당 reward (클리핑 이전 가중합)",
        color=INK, fontsize=13, pad=32, loc="left",
    )
    ax.text(
        0.0, 1.012,
        f"지형 {terrains} · {meta.get('num_rows')}x{meta.get('num_cols')} 타일 · "
        f"env {meta.get('num_envs')} · seed {meta.get('seed')} · 음영은 95% 신뢰구간",
        transform=ax.transAxes, color=INK_MUTED, fontsize=9, va="bottom",
    )
    ax.set_xlabel("학습 iteration", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("reward [1/s]", color=INK_MUTED, fontsize=10)
    legend = ax.legend(frameon=False, fontsize=10, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_diagnostics(summary: pd.DataFrame, meta: dict, path: str):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=160, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    x = summary["iteration"].to_numpy()

    ax = axes[0]
    style_axes(ax)
    ax.plot(x, summary["success_rate"].to_numpy() * 100, color=SERIES[0], linewidth=2.0,
            marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_ylabel("완주율 [%]", color=INK_MUTED, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(
        "완주율 — reward 비교가 생존율 차이에 오염됐는지 확인용",
        color=INK, fontsize=12, pad=12, loc="left",
    )

    ax = axes[1]
    style_axes(ax)
    present = [r for r in REASON_ORDER if summary.get(f"frac_{r}", pd.Series(dtype=float)).fillna(0).sum() > 0]
    stack = [summary[f"frac_{r}"].fillna(0).to_numpy() * 100 for r in present]
    ax.stackplot(x, *stack, colors=SERIES[: len(present)],
                 labels=[REASON_LABEL[r] for r in present], edgecolor=SURFACE, linewidth=1.0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("에피소드 비율 [%]", color=INK_MUTED, fontsize=10)
    ax.set_xlabel("학습 iteration", color=INK_MUTED, fontsize=10)
    ax.set_title("종료 원인 분해", color=INK, fontsize=12, pad=12, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarise and plot fixed-difficulty evaluation results.")
    parser.add_argument("--input_dir", type=str, required=True, help="episodes.csv 가 있는 폴더.")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.input_dir, "episodes.csv"))
    if df.empty:
        raise SystemExit("episodes.csv 가 비어 있다.")
    meta_path = os.path.join(args.input_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    raw_cols = [c for c in df.columns if c.startswith("raw_")]
    summary = build_summary(df, raw_cols)

    summary_path = os.path.join(args.input_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    # 한글 라벨이 깨지지 않는 폰트가 있으면 쓴다. 없으면 기본 폰트로 두고 진행한다.
    # Noto Sans CJK 는 하나의 OTC 에 KR/JP/SC/TC 가 같이 들어 있어 matplotlib 이
    # 지역 변종 중 하나(여기서는 JP)만 이름으로 등록하는 경우가 있다. 어느 변종이든
    # 한글 글리프를 모두 포함하므로 JP 이름도 후보에 넣는다.
    for family in (
        "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP",
        "Noto Sans KR", "Malgun Gothic", "AppleGothic",
    ):
        if any(f.name == family for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            break
    else:
        print("[WARN] 한글 폰트를 못 찾았다. 그래프의 한글 라벨이 깨질 수 있다.")

    plot_reward(summary, meta, os.path.join(args.input_dir, "reward.png"))
    plot_diagnostics(summary, meta, os.path.join(args.input_dir, "diagnostics.png"))

    print(f"[INFO] {summary_path}")
    print(summary[["iteration", "n_episodes", "success_rate",
                   "rew_per_s_success_mean", "rew_per_s_all_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
