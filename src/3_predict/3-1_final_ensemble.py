#!/usr/bin/env python
"""严格校验并等权融合 A 轮两份 store 提交。"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
BEFORE_ENSEMBLE_DIR = OUTPUT_DIR / "before_ensemble"
LOG_DIR = PROJECT_DIR / "logs"
INPUT_FILES = [
    BEFORE_ENSEMBLE_DIR / "submission_kaggle_recursive_store.csv",
    BEFORE_ENSEMBLE_DIR / "submission_kaggle_nonrecursive_store.csv",
]
OUTPUT_FILE = OUTPUT_DIR / "submission_store_only.csv"
F_COLUMNS = [f"F{day}" for day in range(1, 29)]
EXPECTED_COLUMNS = ["id", *F_COLUMNS]


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("store_ensemble")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(LOG_DIR / "store_ensemble.log", encoding="utf-8")]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = configure_logging()


def log_event(event: str, **values: object) -> None:
    LOGGER.info(json.dumps({"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **values}, ensure_ascii=False, default=str, sort_keys=True))


def validate_submission(path: Path, sample_ids: pd.Series) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少待融合文件: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path.name} 列不符合契约: {list(frame.columns)}")
    if len(frame) != 60980:
        raise ValueError(f"{path.name} 应有 60980 行，实际 {len(frame)}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path.name} 包含重复 ID")
    if not frame["id"].reset_index(drop=True).equals(sample_ids.reset_index(drop=True)):
        raise ValueError(f"{path.name} 的 ID 或顺序与 sample_submission.csv 不一致")
    values = frame[F_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} 包含 NaN 或 inf")
    if (values < 0).any():
        raise ValueError(f"{path.name} 包含负数预测")
    log_event("input_validated", path=str(path), rows=len(frame), prediction_sum=float(values.sum()))
    return frame


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    if list(sample.columns) != EXPECTED_COLUMNS or len(sample) != 60980:
        raise ValueError("sample_submission.csv 不符合预期的 60980 行、id + F1..F28 契约")
    inputs = [validate_submission(path, sample["id"]) for path in INPUT_FILES]
    output = sample[["id"]].copy()
    stacked = np.stack([frame[F_COLUMNS].to_numpy(dtype=np.float64) for frame in inputs])
    output[F_COLUMNS] = stacked.mean(axis=0)
    values = output[F_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("融合结果包含 NaN、inf 或负数")
    atomic_csv(output, OUTPUT_FILE)
    log_event("ensemble_finished", inputs=[str(path) for path in INPUT_FILES], rows=len(output), output=str(OUTPUT_FILE), output_mb=round(OUTPUT_FILE.stat().st_size / 1024**2, 2))


if __name__ == "__main__":
    main()
