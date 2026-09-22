"""Find checkpoints in a selected run, with fallback for older nested log layouts.

New runs are written directly beneath the experiment directory. Existing nested
directories remain readable without moving their checkpoint files.
"""

import os
import re


def get_checkpoint_path_with_fallback(log_root_path: str, run_dir: str, checkpoint: str) -> str:
    """run 폴더 바로 아래의 checkpoint 를 우선하고, 없으면 하위 run 폴더로 내려간다.

    1) isaaclab 의 get_checkpoint_path 그대로: <log_root>/<run_dir>/ 바로 아래에서
       checkpoint 패턴에 맞는 가장 최근 파일을 찾는다.
    2) 없으면 run 폴더의 하위 폴더들(<timestamp>/ 산출물, trained_* 스냅샷) 중
       checkpoint 를 가진 가장 최근 수정된 폴더에서 찾는다.
       weight_candidates/ 는 규약상 스캔 대상이 아니라 제외한다.
    """
    from isaaclab_tasks.utils import get_checkpoint_path

    try:
        return get_checkpoint_path(log_root_path, run_dir, checkpoint)
    except ValueError:
        pass

    # run 폴더 해석은 get_checkpoint_path 와 같은 규칙: 정규식 매칭, 알파벳순 마지막.
    runs = sorted(entry.path for entry in os.scandir(log_root_path) if entry.is_dir() and re.match(run_dir, entry.name))
    if not runs:
        raise ValueError(f"No runs present in the directory: '{log_root_path}' match: '{run_dir}'.")
    run_path = runs[-1]

    pattern = checkpoint if checkpoint else "model_.*.pt"
    candidates = [
        entry
        for entry in os.scandir(run_path)
        if entry.is_dir()
        and entry.name != "weight_candidates"
        and any(re.match(pattern, f) for f in os.listdir(entry.path))
    ]
    if not candidates:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' (or its sub-runs) match '{pattern}'.")
    latest = max(candidates, key=lambda entry: entry.stat().st_mtime)
    resume_path = get_checkpoint_path(run_path, re.escape(latest.name), pattern)
    print(f"[INFO] '{run_path}' 바로 아래에 checkpoint 가 없어 최근 하위 run 에서 찾았다: {resume_path}")
    return resume_path
