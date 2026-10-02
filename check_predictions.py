from pathlib import Path
import numpy as np
import pandas as pd

folder = Path("output/before_ensemble")
cols = [f"F{i}" for i in range(1, 29)]

sample = pd.read_csv("data/sample_submission.csv")
expected_ids = set(
    sample.loc[
        sample["id"].str.endswith("_CA_1_evaluation"), "id"
    ]
)

for name in ["nonrecursive_store", "recursive_store"]:
    path = folder / f"submission_kaggle_{name}.csv"
    assert path.exists(), f"缺少预测文件：{path}"

    df = pd.read_csv(path)
    assert list(df.columns) == ["id", *cols], "列名或顺序不正确"
    assert len(df) == 3049, f"行数不正确：{len(df)}"
    assert not df["id"].duplicated().any(), "存在重复 ID"
    assert set(df["id"]) == expected_ids, "商品 ID 不匹配"

    values = df[cols].to_numpy(dtype=float)
    assert np.isfinite(values).all(), "存在缺失值或无穷值"
    assert (values >= 0).all(), "存在负预测值"

    print(f"{name}：检查通过")
    print(f"  商品数：{len(df)}，预测天数：28")
    print(f"  28天预测总销量：{values.sum():.2f}")
    print(f"  每日预测总销量均值：{values.sum(axis=0).mean():.2f}")

print("两路预测文件均通过结构与数值检查。")


