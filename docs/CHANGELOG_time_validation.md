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

## 2026-10-04 v3 代码改造（10.2 改进 Scheme 实施）

按定稿 `docs/10.2改进scheme.md` 实施 `time_validation_v3`（方案 A）：development（选 1857 / 重训 1885）与 test（选 1885 / 重训 1913）各自独立选轮数与融合权重，final（重训 1941）无选参步骤、直接沿用 test 阶段落盘的选择结果，1914–1941 只被 test 闭卷评分一次；final 仅完整性检查，不生成 Kaggle 提交文件。

- **前缀等价路径**：每个「选择阶段×店×模式」只训一次到候选上限（正式 3000、smoke 30），各候选 k 用 `model.predict(..., num_iteration=k)` 做完整 28 天预测（递归同样逐日回填）；新增 `prefix_fallback` 开关保留兜底分支（训至上限以 num_iteration=k 交付）；`equivalence_check` 动作在 smoke 阶段强制验证前缀等价前提并即时决策。
- **等价性验证边界**：冒烟探针为 30 轮上限（threads=4 与正式一致），两模式实测偏差 0.0；3000 轮全量规模的浮点累积未直接验证，属已接受偏差。正式全量前可在 WSL 环境先跑一次全尺寸预检：`run.py equivalence_check --cutoff 1857 --store CA_1`（代价约单店 2×3500 轮训练）。
- **选择阶段评分口径**：WRMSSE 美元权重取 select_cutoff 前 28 天（1857→d1830–1857）、RMSSE 尺度只用 d<=select_cutoff；`metrics.py` 新增 `selection_wrmsse`/`blend` 接口并断言历史宽度受 cutoff 约束。
- **config.json**：experiment 改 `time_validation_v3`，新增 `round_candidates`/`round_selection`/`ensemble_grid`/`early_stopping`(enabled=false)/`prefix_fallback`，删除旧 `selection` 字符串与固定 `ensemble` 0.5/0.5；smoke 用候选 [10,20,30]、CA_1 前 48 商品。
- **特征缓存**：按 cutoff 归置 `features/cutoff_{1857,1885,1913,1941}/<店>/`五表，cutoff_1885 同时服务 development 重训与 test 选择；select_cutoff 缓存在全部消费方结束后由 run.py 显式清理（train_cutoff 缓存与选择结果 json 保留）。
- **选择产物生命周期**：各候选预测/耗时/分数落盘后即删选择阶段模型文件（`keep_selection_models` 审计例外）；选择结果（rounds.json/weights.json 含预测文件哈希）落盘且 final 可读。
- **run.py 新 action**：prepare（按 --cutoff）/ train_candidates / select_rounds / select_weight / model（=retrain，按阶段已选轮数）/ evaluate / equivalence_check / cleanup；all/stage/check 语义保留；每阶段结束更新中文 REPORT.md 并在本文件追加运行时条目（smoke 不写 docs）。
- **launch.py / status.py / 兼容薄壳**：launch.py 改用带回退的解释器解析；status.py 增加六项链路旗标（选择类仅 dev/test，final 显示「继承自 test 的轮数与权重」）并支持 --smoke；`src/2_train`、`src/3_predict` 下 4 个兼容薄壳随新接口同步更新语义。
- **环境适配**：子进程 Python 改为 ROOT/.venv/bin/python 存在则用、否则回退 sys.executable（无 .venv 代码副本工作区；WSL 运行环境行为不变）；verify_preserved() 新增 `M5_BASELINE_MISSING=allow` 降级警告（默认仍严格抛错）；matplotlib 缺失时绘图降级跳过、csv 产物完整。
- **tests.py**：新增 11 项测试——mask 无交集、目标编码不含 cutoff 后销量、未来销量全遮蔽与每 id 恰 28 个未来日（build_base_grid 级）、递归仅预测回填（保留）、跨 cutoff 指纹隔离（保留）、候选完整 28 天评分、权重范围与 w=0/w=1 端点、test 选择窗止于 1913/权重窗止于 cutoff（选择窗口径边界）、final 无选参路径（stage_spec/selection_source_stage）、前缀等价容差断言（合成数据、30 轮取前 11、线程 1、两模式各一次）、select_cutoff 缓存清理决策（1857 于 dev 后删、1885 于 test 后删、train_cutoff 不删）、artifact_id 接受 grid 关键字且网格/轮数变更必变指纹（冒烟暴露的 TypeError 防回归，共 11 项新增、全套 19 项）。
- 本轮未动：`3-1_final_ensemble.py`、`check_predictions.py`、`1_preprocessing_by_store.py` 特征定义、v2 产物与 `docs/results_time_validation/`。

## 2026-10-04 评审修复（v3 交付前）

- **H1**：冒烟实验名由 `smoke_v3` 改为 `smoke_tv3`，避免与 v2 冒烟残留目录撞 manifest 指纹墙；`ensure_run` 与前缀等价失败指引均补「或删除 experiments/<实验名> 目录后重跑」。
- **M1**：select_cutoff 特征缓存清理从「每阶段完成后」延迟到「all 全部阶段成功完成后一次性执行」（含 smoke，清理路径每次冒烟都被回归）；中断重跑时缓存仍在、走 cache_verified 快速跳过，不再重建数小时特征。
- **L1**：选择窗评分构造改经 `metrics.make_selection_scorer`，宽度与列名（恰为 d_1..d_cutoff）在运行时强制。
- **最终复审补丁（L1 残留）**：初版两道防线实为恒真——`selection_scope` 的首末列断言与自身切片比较（列表索引下恒真），`selection_scorer` 传给工厂的列清单是独立重建值而非实际使用值，列清单整体漂移（宽度不变）抓不住。现改为：`selection_scope` 返回实际使用的列清单并以独立写出的首末日标签锚定（d_1/d_cutoff、d_(cutoff+1)/d_(cutoff+horizon)）；`selection_scorer` 把该实际清单传入工厂断言，内容↔标签绑定真正成立（平移探针实证：d_3..d_(cutoff+2) 漂移清单现抛 AssertionError，修复前两道防线均放行）；新增单测 `test_shifted_column_list_rejected` 覆盖工厂层与端到端层。
- **L2**：`select_rounds` 入口断言 scope/metric 仅支持 per_mode_global/WRMSSE，越界值立即报错。
- **L3**：status.py 增显候选轮数清单。
- **L4**：见上方「等价性验证边界」条目。
- **L5**：候选评估 metadata.json 补 `peak_rss_mb`，与 run_model 对齐。
- **README_time_validation.md**：按 v3 实际行为更新四处 v2 残留（日志路径、废除冻结 50:50、features/selection 按 cutoff 归置的目录结构、final 不生成 Kaggle 文件），顶部加 v3 生效说明。
