#!/usr/bin/env python
"""训练 A 轮按门店非递归 LightGBM 模型。"""

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
FIRST_DAY = 710
END_TRAIN = 1941
SEED = 1995
GRID2_COLUMNS = [
    "sell_price", "price_max", "price_min", "price_std", "price_mean", "price_norm",
    "price_nunique", "item_nunique", "price_momentum", "price_momentum_m", "price_momentum_y",
]
GRID3_COLUMNS = [
    "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI",
    "tm_d", "tm_w", "tm_m", "tm_y", "tm_wm", "tm_dw", "tm_w_end",
]
LAG_COLUMNS = [*[f"sales_lag_{day}" for day in range(28, 43)], *[f"rolling_{kind}_{window}" for window in [7, 14, 30, 60, 180] for kind in ["mean", "std"]]]
MEAN_COLUMNS = [
    "enc_store_id_dept_id_mean", "enc_store_id_dept_id_std",
    "enc_item_id_state_id_mean", "enc_item_id_state_id_std",
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
    "num_leaves": 2**8 - 1,
    "min_data_in_leaf": 2**8 - 1,
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
    logger = logging.getLogger("nonrecursive_store_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(LOG_DIR / "nonrecursive_store_train.log", encoding="utf-8")]:
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
    required = {
        "grid_part_2.pkl": GRID2_COLUMNS,
        "grid_part_3.pkl": GRID3_COLUMNS,
        "lags_df_28.pkl": LAG_COLUMNS,
        "mean_encoding_df.pkl": MEAN_COLUMNS,
    }
    for name, columns in required.items():
        missing = sorted(set(columns) - set(frames[name].columns))
        if missing:
            raise ValueError(f"{store}/{name} 缺少列: {missing}")
    frame = pd.concat(
        [
            frames["grid_part_1.pkl"],
            frames["grid_part_2.pkl"][GRID2_COLUMNS],
            frames["grid_part_3.pkl"][GRID3_COLUMNS],
            frames["lags_df_28.pkl"][LAG_COLUMNS],
            frames["mean_encoding_df.pkl"][MEAN_COLUMNS],
        ],
        axis=1,
    )
    frame = frame.loc[frame["d"] >= FIRST_DAY]
    features = [column for column in frame.columns if column not in REMOVE_FEATURES]
    del frames
    gc.collect()
    return frame, features


def train_store(store: str, force: bool) -> None:
    model_path = MODEL_DIR / f"non_recur_model_{store}.bin"
    if model_path.exists() and not force:
        log_event("store_skipped", store=store, reason="model_exists")
        return
    frame, features = load_store(store)
    train_mask = frame["d"] <= END_TRAIN
    valid_mask = train_mask & (frame["d"] > END_TRAIN - 28)
    if frame.loc[train_mask, TARGET].isna().any():
        raise ValueError(f"{store} 训练标签含 NaN")
    train_data = lgb.Dataset(frame.loc[train_mask, features], label=frame.loc[train_mask, TARGET])
    valid_data = lgb.Dataset(frame.loc[valid_mask, features], label=frame.loc[valid_mask, TARGET], reference=train_data)
    random.seed(SEED)
    np.random.seed(SEED)
    log_event("training_started", store=store, rows=int(train_mask.sum()), valid_rows=int(valid_mask.sum()), features=len(features), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
    started = time.time()
    estimator = lgb.train(LGB_PARAMS, train_data, valid_sets=[valid_data, train_data], callbacks=[lgb.log_evaluation(100)])
    atomic_pickle(estimator, model_path)
    log_event("training_finished", store=store, seconds=round(time.time() - started, 1), model_mb=round(model_path.stat().st_size / 1024**2, 1))
    del frame, train_data, valid_data, estimator
    gc.collect()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    force = os.environ.get("M5_FORCE", "0") == "1"
    for store in stores_from_env():
        train_store(store, force)


if __name__ == "__main__":
    main()
