# M5 时间验证 v3 运行结果

方案 A：development 与 test 各自独立选择轮数与融合权重；final 不选参，沿用 test 的选择结果。
选择阶段 WRMSSE 的美元权重取 select_cutoff 前 28 天、RMSSE 尺度只用 d<=select_cutoff 历史。
开发期/测试期/最终预测期分别为 d1886-1913、d1914-1941、d1942-1969。

## development（select_cutoff=1857, train_cutoff=1885）

- 选中轮数：{'nonrecursive': 500, 'recursive': 500}
- 融合权重：recursive=0.4, nonrecursive=0.6（选择窗 ensemble WRMSSE=0.616986）

|模型|WRMSSE|MAE|RMSE|Bias|
|---|---:|---:|---:|---:|
|nonrecursive|0.539185|1.019498|2.089731|-0.035198|
|recursive|0.588473|1.046468|2.084768|0.074428|
|ensemble|0.501571|1.024986|2.060413|0.008653|
|last_week|0.922781|1.232355|2.759459|-0.039584|
|weekday_mean_4weeks|0.654648|1.073854|2.278347|-0.008854|

完整明细、误差图和特征重要性：`development/results`

## test（select_cutoff=1885, train_cutoff=1913）

- 选中轮数：{'nonrecursive': 1000, 'recursive': 500}
- 融合权重：recursive=0.4, nonrecursive=0.6（选择窗 ensemble WRMSSE=0.502637）

|模型|WRMSSE|MAE|RMSE|Bias|
|---|---:|---:|---:|---:|
|nonrecursive|0.650349|1.035538|2.132510|-0.098349|
|recursive|0.505020|1.037916|2.062707|-0.024945|
|ensemble|0.576360|1.031382|2.081368|-0.068987|
|last_week|0.869701|1.243972|2.676895|-0.106180|
|weekday_mean_4weeks|0.752421|1.103654|2.313961|-0.056386|

完整明细、误差图和特征重要性：`test/results`

## final（select_cutoff=None, train_cutoff=1941）

- 无选参步骤：轮数与融合权重继承自 test 阶段的选择结果。

- 完整性检查通过：30490 条序列、28 天、无 NaN、无负数、无重复 ID。
- 最终窗口没有真实销量，不报告准确度。Kaggle 文件已在后处理阶段独立生成，见 `kaggle_semantics.json`。


## 归档内容

`all_scores.csv` 含两个回测窗口的简单基线、单模型、既定 60/40 融合和补充的等权融合；`model_inventory.csv` 含 60 个正式模型的参数摘要、特征数、耗时与模型 SHA256。每个模型的完整参数、特征列表及重要性另存于对应阶段的 `model_records/`。

`manifest.json` 记录实际 12 线程配置、训练源代码版本和输入哈希；早期 8 线程及 Mac 4 线程候选结果保留原始来源。`runtime_scripts/` 保留本机执行脚本，原共享计算模块未变更。执行适配器的本机路径已固定，跨机运行时应按本机路径调整执行层。

原始数据、缓存、模型文件、预测 CSV 和本地评价文档不上传 Git。Kaggle 格式和校验记录见 `kaggle_semantics.json`；validation 是 d1914–1941 的样本外预测，evaluation 是 d1942–1969 的最终预测。未向 Kaggle 自动提交。
