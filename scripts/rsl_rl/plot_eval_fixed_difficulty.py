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

# meta.json 의 terrain_params 를 사람이 읽는 문장으로 바꿀 때 쓰는 표. 여기 없는
# 필드는 필드명 그대로, 단위 없이 찍힌다 (지형이 늘어도 그래프가 깨지지 않게).
PARAM_LABEL = {
    "slope_angle": "경사각",
    "step_height": "단차",
    "gap_size": "간격",
    "hurdle_height_range": "장애물 높이",
    "incline_height": "경사 높이",
    "stone_len": "디딤돌 길이",
}
PARAM_UNIT = {
    "slope_angle": "°",
    "step_height": " m",
    "gap_size": " m",
    "hurdle_height_range": " m",
    "incline_height": " m",
    "stone_len": " m",
}
TERRAIN_LABEL = {
    "parkour_trapezoid_ramp": "경사로",
    "parkour_trapezoid_stairs": "계단",
    "parkour_flat": "평지",
}

REASON_ORDER = ["goal", "timeout", "fall_roll", "fall_pitch", "height"]
REASON_LABEL = {
    "goal": "완주 (goal 8/8)",
    "timeout": "시간 초과",
    "fall_roll": "전복 (roll)",
    "fall_pitch": "전복 (pitch)",
    "height": "추락",
}


def format_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return "~".join(f"{float(v):g}" for v in value)
    return f"{float(value):g}"


def terrain_param_line(meta: dict) -> str:
    """'경사로 경사각 28.9° · 계단 단차 0.176 m' 같은 한 줄을 만든다.

    meta.json 의 terrain_params 는 평가에 쓴 난이도에서 지형 수식을 실제로 푼
    값이다. 여기서 다시 계산하지 않는다 (계산하려면 Isaac Sim 이 필요하고,
    그래프와 실제 지형이 어긋날 여지도 생긴다).
    """
    params = meta.get("terrain_params") or {}
    parts = []
    for terrain, fields in params.items():
        name = TERRAIN_LABEL.get(terrain, terrain.replace("parkour_", ""))
        for field, value in fields.items():
            label = PARAM_LABEL.get(field, field)
            unit = PARAM_UNIT.get(field, "")
            parts.append(f"{name} {label} {format_value(value)}{unit}")
    return " · ".join(parts)


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
            row[f"dist_from_start_{label}_mean"] = subset["dist_from_start"].mean() if len(subset) else float("nan")
            for col in raw_cols:
                row[f"{col}_{label}_mean"] = subset[col].mean() if len(subset) else float("nan")

        counts = group["term_reason"].value_counts()
        for reason in REASON_ORDER:
            row[f"frac_{reason}"] = counts.get(reason, 0) / len(group) if len(group) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)


SERIES_SPEC = [
    ("success", "완주한 에피소드만", SERIES[0]),
    ("all", "전체 에피소드", SERIES[1]),
]


def _draw_reward(ax, summary: pd.DataFrame, legend: bool):
    x = summary["iteration"].to_numpy()
    for key, label, color in SERIES_SPEC:
        y = summary[f"rew_per_s_{key}_mean"].to_numpy()
        ci = summary[f"rew_per_s_{key}_ci95"].to_numpy()
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.15, linewidth=0)
        ax.plot(
            x,
            y,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=label,
        )
    ax.set_ylabel("reward [1/s]", color=INK_MUTED, fontsize=10)
    if legend:
        leg = ax.legend(frameon=False, fontsize=10, loc="lower right")
        for text in leg.get_texts():
            text.set_color(INK)


def plot_reward(summary: pd.DataFrame, meta: dict, path: str, zoom_from: int | None = None):
    # 초기 체크포인트가 y 범위를 통째로 잡아먹어서 정작 보고 싶은 수렴 구간이
    # 납작해진다. 전체 곡선과 확대 곡선을 위아래로 같이 둔다.
    #
    # zoom_from 을 생략하면 iteration 0 만 뺀다. 초반 상승이 여전히 가팔라서
    # 후반이 눌린다면 더 뒤에서 자르면 된다.
    if zoom_from is None:
        positive = summary.loc[summary["iteration"] > 0, "iteration"]
        zoom_from = int(positive.min()) if len(positive) else 0
    zoomed = summary[summary["iteration"] >= zoom_from]
    two_panel = len(zoomed) >= 3 and len(zoomed) < len(summary)

    if two_panel:
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), dpi=160)
    else:
        fig, ax_single = plt.subplots(figsize=(9, 5), dpi=160)
        axes = [ax_single]
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style_axes(ax)

    _draw_reward(axes[0], summary, legend=True)

    difficulty = meta.get("difficulty", "?")
    terrains = ", ".join(meta.get("active_terrains", []))
    param_line = terrain_param_line(meta)
    subtitle = (
        f"지형 {terrains} · {meta.get('num_rows')}x{meta.get('num_cols')} 타일 · "
        f"env {meta.get('num_envs')} · seed {meta.get('seed')} · 음영은 95% 신뢰구간"
    )
    if param_line:
        # 난이도 숫자만 적으면 그게 물리적으로 무엇인지 그래프만 보고는 알 수 없다.
        subtitle = f"난이도 {difficulty} → {param_line}\n{subtitle}"
    axes[0].set_title(
        f"고정 난이도 {difficulty} 에서의 초당 reward (클리핑 이전 가중합)",
        color=INK,
        fontsize=13,
        pad=48 if param_line else 32,
        loc="left",
    )
    axes[0].text(
        0.0,
        1.012,
        subtitle,
        transform=axes[0].transAxes,
        color=INK_MUTED,
        fontsize=9.5,
        va="bottom",
        linespacing=1.6,
    )

    if two_panel:
        _draw_reward(axes[1], zoomed, legend=False)
        axes[1].set_title(
            f"iteration {int(zoomed['iteration'].min())} 이후 확대 — 수렴 구간",
            color=INK,
            fontsize=12,
            pad=12,
            loc="left",
        )
        axes[1].set_xlabel("학습 iteration", color=INK_MUTED, fontsize=10)
    else:
        axes[0].set_xlabel("학습 iteration", color=INK_MUTED, fontsize=10)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_diagnostics(summary: pd.DataFrame, meta: dict, path: str):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=160, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    x = summary["iteration"].to_numpy()

    ax = axes[0]
    style_axes(ax)
    ax.plot(
        x,
        summary["success_rate"].to_numpy() * 100,
        color=SERIES[0],
        linewidth=2.0,
        marker="o",
        markersize=5.5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
    )
    ax.set_ylabel("완주율 [%]", color=INK_MUTED, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(
        "완주율 — reward 비교가 생존율 차이에 오염됐는지 확인용",
        color=INK,
        fontsize=12,
        pad=12,
        loc="left",
    )

    ax = axes[1]
    style_axes(ax)
    present = [r for r in REASON_ORDER if summary.get(f"frac_{r}", pd.Series(dtype=float)).fillna(0).sum() > 0]
    stack = [summary[f"frac_{r}"].fillna(0).to_numpy() * 100 for r in present]
    ax.stackplot(
        x,
        *stack,
        colors=SERIES[: len(present)],
        labels=[REASON_LABEL[r] for r in present],
        edgecolor=SURFACE,
        linewidth=1.0,
    )
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
    parser.add_argument(
        "--zoom_from",
        type=int,
        default=None,
        help="아래 확대 패널을 시작할 iteration. 생략하면 iteration 0 만 뺀다.",
    )
    parser.add_argument(
        "--reward_out",
        type=str,
        default="reward.png",
        help="reward 그래프 파일명. --zoom_from 을 바꿔 여러 장을 만들 때 쓴다.",
    )
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
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans CJK JP",
        "Noto Sans KR",
        "Malgun Gothic",
        "AppleGothic",
    ):
        if any(f.name == family for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            break
    else:
        print("[WARN] 한글 폰트를 못 찾았다. 그래프의 한글 라벨이 깨질 수 있다.")

    reward_path = os.path.join(args.input_dir, args.reward_out)
    plot_reward(summary, meta, reward_path, zoom_from=args.zoom_from)
    plot_diagnostics(summary, meta, os.path.join(args.input_dir, "diagnostics.png"))
    print(f"[INFO] {reward_path}")

    print(f"[INFO] {summary_path}")
    print(
        summary[["iteration", "n_episodes", "success_rate", "rew_per_s_success_mean", "rew_per_s_all_mean"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
