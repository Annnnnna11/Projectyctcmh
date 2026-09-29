# M5 A轮复现执行计划

## 目标与边界

- 本轮只完成 A 轮：store 粒度递归 10 模型、非递归 10 模型，以及两族等权集成。
- 最终产物为 `output/submission_store_only.csv`；暂不训练 store_cat、store_dept 等 B 轮模型。
- `src/M5-methods/` 作为官方只读基线，不在其中修改；改造脚本放在其同级工作目录。
- 成功标准是产出结构、ID、列名、行数和数值均合规的预测文件；本轮不实现 WRMSSE 评分器。

## 文档纠错与执行口径统一

- 修正 `docs/M5_第一名方案复现技术文档.md` 第 6 节：实际入口统一为 `src/1_preprocessing_by_store.py`，产物统一为 `data/processed/{store}/` 下五个 pickle，不再把官方全局预处理写成当前执行方式。
- 修正 `docs/M5_复现执行手册.md` 附录命令，全部显式使用 `/home/chenmeihan/miniconda3/envs/m5/bin/python`，避免误用 base 环境。
- 明确 A 轮输出名、递归与非递归不同 seed 和模型命名、LightGBM 固定轮数不使用早停、WRMSSE 指标权重与模型融合权重的区别。
- 在低内存说明中注明本项目为兼容官方脚本实际使用 pickle；Parquet 仅是通用建议，不改变本轮落盘格式。

## 工作副本与环境准备

- 保持 `src/M5-methods/Code of Winning Methods/A1/` 干净，只复制 A 轮所需官方脚本到工作区：
  - `src/2_train/1-1_recursive_store.py`
  - `src/2_train/2-1_nonrecursive_store.py`
  - `src/3_predict/1-1_recursive_store_PREDICT.py`
  - `src/3_predict/2-1_nonrecursive_store_PREDICT.py`
  - `src/3_predict/3-1_final_ensemble.py`
- 创建运行目录：`data/processed/`、`logs/`、`output/before_ensemble/`；已有 `models/` 和 `output/` 保留。
- 在 m5 环境内做 import 预检，只安装脚本实际需要且缺失的依赖。pickle 路线不强制安装 pyarrow；若代码使用 tqdm 再补装 tqdm。
- 固定解释器为 `/home/chenmeihan/miniconda3/envs/m5/bin/python`，LightGBM 线程数限制为 4。

## 低内存分店预处理

- 新建 `src/1_preprocessing_by_store.py`，以官方 `1. preprocessing.py` 的特征定义为基线，保持特征名、类型和训练语义一致。
- 全局阶段只读取必要列并生成：固定类别字典、日历特征、价格与 release 统计、全局唯一索引规则。
- 第一遍按 10 个 store 逐店执行：宽转长、追加 d_1942 至 d_1969、release 过滤、基础网格、价格特征、日历特征、15 个 lag、10 个 rolling、12 个递归临时 rolling，并累计 11 组 mean encoding 的 count、sum、sumsq。
- 汇总跨店统计后第二遍逐店回填 mean/std encoding；对 count 小于 2 的标准差安全置为 NaN，避免除零。
- 每店写入 `data/processed/{store}/` 下 `grid_part_1.pkl`、`grid_part_2.pkl`、`grid_part_3.pkl`、`lags_df_28.pkl`、`mean_encoding_df.pkl`。
- 写盘前强制检查五表索引完全一致、索引全局唯一、类别编码来自同一全局类别表、预测期 sales 已正确遮蔽。
- 加入按阶段和按门店断点续跑、原子写盘、内存释放与峰值日志，确保 9.7GB RAM 下可恢复执行。

## A轮训练脚本改造

- 两个训练脚本改为从 `data/processed/{store}/` 直接读取五表，不再先加载全局 pickle 后过滤。
- 保持官方模型特征和参数差异：递归 seed 42、非递归 seed 1995；递归与非递归各自的 mean encoding、lag 特征和树参数不混用。
- 将 LightGBM 参数改为 `num_iterations=3000`、`verbosity=-1`，日志改用 `lgb.log_evaluation` callback，适配 LightGBM 4.7。
- 递归训练生成 `models/lgb_model_{store}_v1.bin` 和 `data/processed/test_{store}.pkl`；非递归训练生成 `models/non_recur_model_{store}.bin`。
- 增加 `M5_STORES` 单店开关和模型存在即跳过的断点续跑；保存模型前后记录输入行数、特征数和文件大小。

## A轮预测与集成改造

- 递归预测用 `.loc` 完成逐日 sales 回填，每天重算临时 rolling，彻底消除 pandas 链式赋值导致的静默失效。
- 分店加载 `test_{store}.pkl` 与对应模型，输出 `output/before_ensemble/submission_kaggle_recursive_store.csv`。
- 非递归预测取消对全局 `grid_part_1.pkl` 和脆弱 `iloc` 位置索引的依赖，使用分店数据中稳定的 id、d 映射生成结果，输出 `output/before_ensemble/submission_kaggle_nonrecursive_store.csv`。
- 集成脚本接收明确输入文件列表，对两份 A 轮预测做等权平均，生成 `output/submission_store_only.csv`；不再硬编码六族文件。

## 分级验证与实际运行

- 静态验证：对新增和改造脚本执行 `py_compile`，检查 imports、路径、LightGBM API 与脚本参数。
- 预处理抽检：先只跑 CA_1，验证五表索引、列集合、dtype、预测期 NaN、mean/std 合理性和内存峰值。
- 单店闭环：仅对 CA_1 完成递归训练、非递归训练和两类预测；确认模型可重新加载，F1 与 F28 不是机械重复，递归每日回填确实生效。
- 单店闭环通过后运行全部 10 店预处理，再以可续跑方式训练 20 个模型；任何失败只重跑失败门店。
- 完成两族预测与集成后，对最终文件做硬校验：与 `sample_submission.csv` 的 ID 顺序完全一致，列严格为 id 与 F1 至 F28，共 60,980 行数据，无 NaN、无 inf、无负数、预测值非全零且各天不完全相同。
- 由独立验证代理执行代码检查与快速测试；实现完成后再做一次变更审查。若训练是长任务，则后台运行并保留日志、PID 和断点状态，完成后继续预测与最终校验。

## 假设与风险控制

- 当前 `src/M5-methods/` 已确认是完整且干净的官方仓库，作为 reference 使用，无需再次 clone 或复制整个仓库。
- 官方仓库没有显式 LICENSE，本轮仅按用户要求做本地学术复现，不发布衍生代码。
- 9.7GB 内存不允许官方全局 melt；任何实现若恢复全局五表加载均视为失败。
- 4 核 CPU 下完整 A 轮训练可能持续较久；通过单店先验、断点续跑和逐店产物校验降低长跑失败成本。
- 如 LightGBM 的 CUDA 构建在当前设备不可用，默认继续使用 CPU 设备，不将 GPU 作为运行前提。