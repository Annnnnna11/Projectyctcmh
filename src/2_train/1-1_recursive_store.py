#!/usr/bin/env python
"""训练 A 轮按门店递归 LightGBM 模型。"""

from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import random
import resource
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
LOG_DIR = PROJECT_DIR / "logs"
ALL_STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
FILES = ["grid_part_1.pkl", "grid_part_2.pkl", "grid_part_3.pkl", "lags_df_28.pkl", "mean_encoding_df.pkl"]
TARGET = "sales"
END_TRAIN = 1941
P_HORIZON = 28
SEED = 42
MEAN_FEATURES = [
    "enc_cat_id_mean", "enc_cat_id_std", "enc_dept_id_mean",
    "enc_dept_id_std", "enc_item_id_mean", "enc_item_id_std",
]
REMOVE_FEATURES = {"id", "state_id", "store_id", "date", "wm_yr_wk", "d", TARGET}
LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "subsample": 0.5,
    "subsample_freq": 1,
    "learning_rate": 0.015,
    "num_leaves": 2**11 - 1,
    "min_data_in_leaf": 2**12 - 1,
    "feature_fraction": 0.5,
    "max_bin": 100,
    "num_iterations": 3000,
    "boost_from_average": False,
    "verbosity": -1,
    "num_threads": 4,
    "seed": SEED,
}


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("recursive_store_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(LOG_DIR / "recursive_store_train.log", encoding="utf-8")]:
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


def atomic_pickle(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_store(store: str) -> tuple[pd.DataFrame, list[str]]:
    store_dir = PROCESSED_DIR / store
    frames = {name: pd.read_pickle(store_dir / name) for name in FILES}
    reference = frames[FILES[0]].index
    if not reference.is_unique or any(not frame.index.equals(reference) for frame in frames.values()):
        raise ValueError(f"{store} 五表索引不一致")
    missing = sorted(set(MEAN_FEATURES) - set(frames["mean_encoding_df.pkl"].columns))
    if missing:
        raise ValueError(f"{store} 缺少目标编码列: {missing}")
    frame = pd.concat(
        [
            frames["grid_part_1.pkl"],
            frames["grid_part_2.pkl"].drop(columns=["id", "d"]),
            frames["grid_part_3.pkl"].drop(columns=["id", "d"]),
            frames["mean_encoding_df.pkl"][MEAN_FEATURES],
            frames["lags_df_28.pkl"].drop(columns=["id", "d", TARGET]),
        ],
        axis=1,
    )
    features = [column for column in frame.columns if column not in REMOVE_FEATURES]
    frame = frame[["id", "d", TARGET, *features]]
    del frames
    gc.collect()
    return frame, features


def save_test_frame(store: str, frame: pd.DataFrame) -> None:
    prediction_mask = (frame["d"] > END_TRAIN - 100) & (frame["d"] <= END_TRAIN + P_HORIZON)
    test = frame.loc[prediction_mask].copy()
    test.drop(columns=[column for column in test.columns if "_tmp_" in column], inplace=True)
    test.loc[test["d"] > END_TRAIN, TARGET] = np.nan
    atomic_pickle(test, PROCESSED_DIR / f"test_{store}.pkl")
    log_event("test_frame_written", store=store, rows=len(test), path=str(PROCESSED_DIR / f"test_{store}.pkl"))
    del test


def train_store(store: str, force: bool) -> None:
    model_path = MODEL_DIR / f"lgb_model_{store}_v1.bin"
    test_path = PROCESSED_DIR / f"test_{store}.pkl"
    if model_path.exists() and test_path.exists() and not force:
        log_event("store_skipped", store=store, reason="model_and_test_exist")
        return
    frame, features = load_store(store)
    if force or not test_path.exists():
        save_test_frame(store, frame)
    if model_path.exists() and not force:
        log_event("training_skipped", store=store, reason="model_exists_test_rebuilt")
        return
    train_mask = frame["d"] <= END_TRAIN
    valid_mask = train_mask & (frame["d"] > END_TRAIN - P_HORIZON)
    if frame.loc[train_mask, TARGET].isna().any():
        raise ValueError(f"{store} 训练标签含 NaN")
    train_data = lgb.Dataset(frame.loc[train_mask, features], label=frame.loc[train_mask, TARGET])
    valid_data = lgb.Dataset(frame.loc[valid_mask, features], label=frame.loc[valid_mask, TARGET], reference=train_data)
    random.seed(SEED)
    np.random.seed(SEED)
    log_event("training_started", store=store, rows=int(train_mask.sum()), valid_rows=int(valid_mask.sum()), features=len(features), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
    started = time.time()
    estimator = lgb.train(LGB_PARAMS, train_data, valid_sets=[valid_data], callbacks=[lgb.log_evaluation(100)])
    atomic_pickle(estimator, model_path)
    log_event("training_finished", store=store, seconds=round(time.time() - started, 1), model_mb=round(model_path.stat().st_size / 1024**2, 1))
    del frame, train_data, valid_data, estimator
    gc.collect()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    force = os.environ.get("M5_FORCE", "0") == "1"
    for store in stores_from_env():
        train_store(store, force)


if __name__ == "__main__":
    main()
