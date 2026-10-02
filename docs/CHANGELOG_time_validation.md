# 时间验证分支变更记录

本分支 `experiment/time-validation` 直接继承原仓库提交 `c904e55`，包含完整原代码历史。
无需额外复制仓库；原版代码可通过该提交或基线提交 `c75d089` 查看。
2026-10-01 按用户确认，将本分支发布到个人 Fork `Annnnnna11/Projectyctcmh`。
GitHub 返回原仓库的实际名称为 `meihan527/Projectyctcmh`，旧地址 `meihan527/m5-forecasting` 指向该仓库。
`origin` 指向个人 Fork，`upstream` 保留原仓库地址；只推送 `experiment/time-validation`，不修改 main。
此次发布仅包含 Git 跟踪的代码、文档、检查脚本和历史提交，不包含数据、模型、缓存或虚拟环境。

## 提交与主要文件

|提交|变更|
|---|---|
|c75d089|保存原 CA_1 产物 SHA-256 清单、已有 check_predictions.py，补充 Git 忽略规则|
|84bc54b|建立时间验证、特征、训练、评价与串行运行流程，新增中文操作说明|
|ba71134|检查点绑定阶段/截止日/门店/模式，增加独立训练缓存与原始数据核对|
|c0004a4|记录三阶段 smoke 测试和缓存恢复一致性验证|

- `src/time_validation/config.json`：开发训练截至 d1885、测试截至 d1913、最终截至 d1941；统一 28 天预测。
- `features.py`：复用原特征定义，按截止日重算全部门店目标编码；限制日历/价格至预测窗口；验证五表索引和类别映射。
- `model.py`：删除训练/验证重叠；非递归取消 d710 历史截断；递归遮蔽未来标签并逐日回填；显式浮点类型；保存模式独立二进制缓存。
- `metrics.py`、`evaluate.py`：完整 12 层 WRMSSE、MAE/RMSE/偏差、两个简单基线、等权融合、门店/品类/步长误差及重要性图。
- `common.py`、`run.py`、`launch.py`、`status.py`：配置/输入/代码/依赖指纹、产物哈希、串行子进程、持久日志、状态与阶段报告。
- `tests.py`、`external_reference_check.py`、`src/check_raw_inputs.py`：手算及外部指标参考、未来标签遮蔽、原始数据与恢复测试。
- 原两个训练和两个预测入口改为新流程兼容入口；原预处理模块取消导入时写日志。
- `docs/README_time_validation.md`：运行、恢复、结果解读、Kaggle 日期语义和局限。

## 与原版保持一致及改变之处

保留按店非递归/递归 LightGBM、Tweedie、主要树参数、学习率、seed 和 3000 轮。
使用 4 CPU 线程，增加 force_col_wise 控制内存；仍保留原版按价格首次上市周过滤上市前训练行。
开发完成后冻结原参数及 50:50 融合，不用最后测试反复调参。
原版 valid RMSE 来自重叠样本，只保留为运行日志，不用于准确度比较。

原始 CSV、原版 models/、data/processed/ 和 CA_1 归档预测未覆盖。
原产物 26 文件哈希清单在 `docs/baseline_CA_1_manifest.json`，每阶段核对。
正式新产物在 `experiments/time_validation_v2/`；早期 v1 预运行已被替代，不用于正式结果。
虚拟环境、原始数据、缓存、大模型、预测和运行日志均未提交 Git。

## 已验证与当前状态

8 项单元测试通过；48 条序列、5 轮的开发/测试/最终闭环通过，未出现 FutureWarning。
二进制训练缓存恢复后预测一致；WRMSSE 外部参考逐层最大差异 1.11e-16。
原始数据为 10 店各 3049 条序列，两个销量 CSV 的共有历史完全一致。
证据见 `docs/{smoke_validation,wrmsse_reference_check,raw_input_audit}.json`。

截至 2026-10-01 16:56（中国时间），正式任务已完成开发窗口 CA_1 非递归模型，正在训练递归模型。
这是状态快照，不代表 60 个模型完成；正式全量评分和最终预测尚未生成。
运行结束将自动生成阶段评分、误差图、重要性、耗时、REPORT.md 和最终预测；不自动提交 Kaggle。

## GPU 检查

本机 WSL 可见 NVIDIA GeForce RTX 5060 Laptop GPU，显存约 8GB。
独立极小样本测试显示当前 LightGBM 4.7.0 未编译 CUDA Tree Learner；OpenCL 路径返回 No OpenCL device found；PATH 中未找到 nvcc。
没有改变正在运行的 CPU 实验或依赖环境。
若后续验证 GPU，应使用独立环境与实验目录，先固定相同特征和超参数对比速度、显存与预测质量，不能承诺固定加速倍数。
LightGBM CUDA 路径与 OpenCL 路径不同，见 [官方安装说明](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)。

## 查看进展

```bash
cd ~/projects/m5-forecasting
.venv/bin/python src/time_validation/status.py
tail -f experiments/time_validation_v2/runner.log
```

文档提交不会修改当前模型代码/配置/依赖指纹，也不会重启训练。
