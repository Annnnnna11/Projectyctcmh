#!/usr/bin/env python
"""使用 A 轮门店模型执行 28 天递归预测。"""

from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = RAW_DIR / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "output" / "before_ensemble"
LOG_DIR = PROJECT_DIR / "logs"
ALL_STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
END_TRAIN = 1941
TARGET = "sales"
F_COLUMNS = [f"F{day}" for day in range(1, 29)]
ROLLING_SPECS = [(shift, window) for shift in [1, 7, 14] for window in [7, 14, 30, 60]]


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("recursive_store_predict")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(LOG_DIR / "recursive_store_predict.log", encoding="utf-8")]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = configure_logging()


def log_event(event: str, **values: object) -> None:
    LOGGER.info(json.dumps({"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **values}, ensure_ascii=False, default=str, sort_keys=True))


def stores_from_env() -> list[str]:
    value = os.environ.get("M5_STORES", "").strip()
    requested = ALL_STORES if not value else [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(ALL_STORES))
    if unknown:
        raise ValueError(f"M5_STORES 包含未知门店: {unknown}")
    selected = set(requested)
    return [store for store in ALL_STORES if store in selected]


def load_base_test(stores: list[str]) -> pd.DataFrame:
    parts = []
    for store in stores:
        path = PROCESSED_DIR / f"test_{store}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"缺少递归测试集: {path}")
        part = pd.read_pickle(path)
        if part.duplicated(["id", "d"]).any():
            raise ValueError(f"{store} test pkl 的 id/d 不唯一")
        part = part.copy()
        part["_store"] = store
        parts.append(part)
    base = pd.concat(parts, ignore_index=True)
    base.sort_values(["id", "d"], inplace=True, ignore_index=True)
    expected_days = set(range(END_TRAIN + 1, END_TRAIN + 29))
    if set(base.loc[base["d"] > END_TRAIN, "d"].unique()) != expected_days:
        raise ValueError("递归测试集缺少完整的 28 个预测日")
    return base


def make_dynamic_rollings(base: pd.DataFrame) -> pd.DataFrame:
    dynamic = pd.DataFrame(index=base.index)
    ids = base["id"]
    for shift, window in ROLLING_SPECS:
        shifted = base[TARGET].groupby(ids, observed=True).shift(shift)
        values = shifted.groupby(ids, observed=True).rolling(window).mean().reset_index(level=0, drop=True)
        dynamic[f"rolling_mean_tmp_{shift}_{window}"] = values.reindex(base.index).astype(np.float32)
    return dynamic


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def format_submission(predictions: pd.DataFrame, stores: list[str]) -> pd.DataFrame:
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    full_run = stores == ALL_STORES
    if full_run:
        output = sample.copy()
        output[F_COLUMNS] = np.float32(0)
    else:
        suffixes = tuple(f"_{store}_evaluation" for store in stores)
        output = sample.loc[sample["id"].str.endswith(suffixes)].copy()
    mapped = predictions.set_index("id")
    evaluation_mask = output["id"].isin(mapped.index)
    output.loc[evaluation_mask, F_COLUMNS] = mapped.reindex(output.loc[evaluation_mask, "id"])[F_COLUMNS].to_numpy()
    if full_run and len(output) != 60980:
        raise ValueError(f"完整提交应有 60980 行，实际 {len(output)}")
    values = output[F_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("递归提交包含 NaN、inf 或负数")
    return output[["id", *F_COLUMNS]]


def main() -> None:
    stores = stores_from_env()
    if not stores:
        raise ValueError("M5_STORES 未选择任何门店")
    base_test = load_base_test(stores)
    prediction_days: list[pd.DataFrame] = []
    started = time.time()
    for horizon in range(1, 29):
        day = END_TRAIN + horizon
        dynamic = make_dynamic_rollings(base_test)
        grid = pd.concat([base_test, dynamic], axis=1)
        day_mask = base_test["d"] == day
        for store in stores:
            model_path = MODEL_DIR / f"lgb_model_{store}_v1.bin"
            if not model_path.exists():
                raise FileNotFoundError(f"缺少递归模型: {model_path}")
            with model_path.open("rb") as handle:
                estimator = pickle.load(handle)
            features = estimator.feature_name()
            missing = sorted(set(features) - set(grid.columns))
            if missing:
                raise ValueError(f"{store} 递归预测缺少模型特征: {missing}")
            mask = day_mask & (base_test["_store"] == store)
            values = estimator.predict(grid.loc[mask, features])
            base_test.loc[mask, TARGET] = values
            del estimator, values
            gc.collect()
        day_predictions = base_test.loc[day_mask, ["id", TARGET]].rename(columns={TARGET: f"F{horizon}"})
        prediction_days.append(day_predictions)
        log_event("day_predicted", horizon=horizon, rows=len(day_predictions), sales_sum=float(day_predictions[f"F{horizon}"].sum()), elapsed_seconds=round(time.time() - started, 1))
        del dynamic, grid
        gc.collect()

    predictions = prediction_days[0]
    for frame in prediction_days[1:]:
        predictions = predictions.merge(frame, on="id", how="inner", validate="one_to_one")
    submission = format_submission(predictions, stores)
    output_path = OUTPUT_DIR / "submission_kaggle_recursive_store.csv"
    atomic_csv(submission, output_path)
    log_event("prediction_finished", stores=stores, rows=len(submission), output=str(output_path), output_mb=round(output_path.stat().st_size / 1024**2, 2))


if __name__ == "__main__":
    main()
