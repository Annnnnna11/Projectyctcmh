#!/usr/bin/env python
"""使用稳定 id/d 映射执行 A 轮门店非递归预测。"""

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
FILES = ["grid_part_1.pkl", "grid_part_2.pkl", "grid_part_3.pkl", "lags_df_28.pkl", "mean_encoding_df.pkl"]
END_TRAIN = 1941
F_COLUMNS = [f"F{day}" for day in range(1, 29)]
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


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nonrecursive_store_predict")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(LOG_DIR / "nonrecursive_store_predict.log", encoding="utf-8")]:
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


def load_prediction_rows(store: str) -> pd.DataFrame:
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
    base = frames["grid_part_1.pkl"]
    future_mask = (base["d"] > END_TRAIN) & (base["d"] <= END_TRAIN + 28)
    rows = pd.concat(
        [
            base.loc[future_mask],
            frames["grid_part_2.pkl"].loc[future_mask, GRID2_COLUMNS],
            frames["grid_part_3.pkl"].loc[future_mask, GRID3_COLUMNS],
            frames["lags_df_28.pkl"].loc[future_mask, LAG_COLUMNS],
            frames["mean_encoding_df.pkl"].loc[future_mask, MEAN_COLUMNS],
        ],
        axis=1,
    )
    if rows.duplicated(["id", "d"]).any():
        raise ValueError(f"{store} 预测行的 id/d 不唯一")
    counts = rows.groupby("id", observed=True)["d"].nunique()
    if len(counts) == 0 or not counts.eq(28).all():
        raise ValueError(f"{store} 每个 id 必须恰有 28 个预测日")
    del frames
    gc.collect()
    return rows


def predict_store(store: str) -> pd.DataFrame:
    model_path = MODEL_DIR / f"non_recur_model_{store}.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"缺少非递归模型: {model_path}")
    rows = load_prediction_rows(store)
    with model_path.open("rb") as handle:
        estimator = pickle.load(handle)
    features = estimator.feature_name()
    missing = sorted(set(features) - set(rows.columns))
    if missing:
        raise ValueError(f"{store} 非递归预测缺少模型特征: {missing}")
    rows = rows[["id", "d", *features]].copy()
    rows["prediction"] = estimator.predict(rows[features])
    wide = rows.pivot(index="id", columns="d", values="prediction")
    expected_days = list(range(END_TRAIN + 1, END_TRAIN + 29))
    if list(wide.columns) != expected_days:
        raise ValueError(f"{store} 预测日集合不完整")
    wide.columns = F_COLUMNS
    result = wide.reset_index()
    log_event("store_predicted", store=store, rows=len(result), sales_sum=float(result[F_COLUMNS].to_numpy().sum()))
    del rows, estimator, wide
    gc.collect()
    return result


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
        raise ValueError("非递归提交包含 NaN、inf 或负数")
    return output[["id", *F_COLUMNS]]


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
    stores = stores_from_env()
    if not stores:
        raise ValueError("M5_STORES 未选择任何门店")
    started = time.time()
    predictions = pd.concat([predict_store(store) for store in stores], ignore_index=True)
    if predictions["id"].duplicated().any():
        raise ValueError("多个门店预测包含重复 id")
    submission = format_submission(predictions, stores)
    output_path = OUTPUT_DIR / "submission_kaggle_nonrecursive_store.csv"
    atomic_csv(submission, output_path)
    log_event("prediction_finished", stores=stores, rows=len(submission), elapsed_seconds=round(time.time() - started, 1), output=str(output_path), output_mb=round(output_path.stat().st_size / 1024**2, 2))


if __name__ == "__main__":
    main()
