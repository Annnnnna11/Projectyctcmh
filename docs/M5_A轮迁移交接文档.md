# M5 A 轮复现 —— 换机迁移交接文档

> 撰写时间：**2026-09-29**　|　撰写依据：**对磁盘现状的两次只读核实**（13:41 首次、14:01 第二次）+ 逐行阅读 6 个 A 轮脚本源码 + 独立复核已产出的两份预测 CSV。
>
> 本文档是**自包含**的：新机器上只要有这一份文件 + 项目目录，就能接着把 A 轮跑完，不需要再回看聊天记录。
>
> 撰写边界：**只新建本文档**，未修改其他 docs、未改任何源码、未动 `src/M5-methods/`、未运行任何预处理 / 训练 / 预测。核实过程全部是只读命令。

**四份文档的定位分工**（详见 2.10）：

| 文档 | 层次 | 什么时候看 |
| --- | --- | --- |
| `M5_第一名方案复现技术文档.md` | 原理层 | 想弄懂"为什么这么设计"、改代码前 |
| `M5_复现执行手册.md` | 操作层 | 想知道阶段命令与检查点（⚠️ 提交行数口径已过时，见 6.6） |
| `M5_memory_and_low_RAM_pipeline_summary.txt` | 约束层 | 想弄懂低内存分段策略的来龙去脉 |
| **`M5_A轮迁移交接文档.md`（本文档）** | **迁移层** | **换机器、接着跑剩余步骤时** |

---

## 0. 迁移前必读（TL;DR）

1. **当前没有任何 M5 进程在运行**（第二次核实：`ps aux | grep '[p]ython.*src/' | wc -l` 输出 `0`）。首次核实时在跑的全量预处理进程 PID `1048` 已于 `13:45:53` **正常结束**并打出 `preprocessing_finished`。`logs/.running_pids` 里的 `preproc_pid=1048` 是**过期标记**，可删。换机前仍请自己再确认一次（见 3.5）。
2. **预处理已 100% 完成**：10 个门店目录全部 `stage=complete`，`data/processed/` 共 **17 GB**。**这一步不用再跑了**（除非你选择不迁移 pkl）。
3. **A 轮真正的剩余工作是训练**：20 个模型只完成了 2 个（CA_1 递归 + CA_1 非递归），还剩 **18 个**，4 核机上约 **24 小时**。这是关键路径，占 A 轮总耗时的 98%。
4. **最终交付物 `output/submission_store_only.csv` 还没生成**，`logs/store_ensemble.log` 是 0 字节（集成从未运行过）。
5. **迁移体积决策**：整目录约 **20 GB**，其中 17 GB 是可再生的 pkl。**推荐方案 B（约 2.9 GB：带源码 + 原始数据 + 模型 + 文档日志，不带 pkl）**——因为重算 17 GB pkl 实测推算只要 **20-25 分钟**，而多拷 17 GB 在慢盘上可能要几十分钟到几小时。详见 3.2。
6. **最大的坑是版本**：`data/processed/*.pkl` 对 pandas 版本敏感，`models/*.bin` 对 lightgbm 版本敏感（用的是 `pickle.dump(Booster)`，**不是** `save_model()` 文本格式）。新机器必须严格对齐 4.1 的版本表，否则**带了文件也加载不了**。

---

## 1. 项目目标与当前进度总览

### 1.1 A 轮是什么

M5 Forecasting - Accuracy 竞赛第一名（A1，YeonJun In）方案的完整结构是 **6 族 220 个 LightGBM 模型**：

```text
3 种聚合粒度            ×  2 种预测模式      =  6 族
  store        (10 店)      recursive   递归        10 × 2 = 20 个模型   <- A 轮
  store × cat  (30 组)      nonrecursive 非递归      30 × 2 = 60 个模型   <- B 轮
  store × dept (100 组)                            100 × 2 = 200 个模型  <- B 轮
                                                  合计 260 ... 官方口径 220（部分组样本过少被合并/剔除）
```

**A 轮 = 只做 store 族的 20 个模型**，产物是 `output/submission_store_only.csv`（60980 行）。
它本身就是一份**可用的、格式合规的**销量预测，也是 B 轮的流程验证。

两族模型的区别（这是整个方案的核心设计，务必理解）：

| | 递归 recursive | 非递归 nonrecursive |
| --- | --- | --- |
| 预测方式 | 单模型，逐日预测 28 次，每次把上一天的预测值**回填**成 lag 特征 | 一次调用直接输出 28 天（`num_class=28` 语义，本项目用 28 列输出） |
| 可用特征 | 可以用**短位移**滚动特征（`shift=1/7/14`，即 `rolling_*_tmp_*`） | 只能用 `lag >= 28` 的特征，避免用到未来真值 |
| 误差特性 | **累积误差**（第 28 天依赖前 27 天的预测） | **不累积**，但缺少近期信息 |
| 参数差异 | `num_leaves=2047`、`min_data_in_leaf=4095`、`seed=42` | `num_leaves=255`、`min_data_in_leaf=255`、`seed=1995`、`FIRST_DAY=710` 裁掉早期全 NaN 行 |
| 共同点 | `objective=tweedie`、`tweedie_variance_power=1.1`、`metric=rmse`、`learning_rate=0.015`、`num_iterations=3000`（**固定轮数，不用早停**）、`max_bin=100`、`subsample=0.5`、`feature_fraction=0.5`、`boost_from_average=False` | 同左 |

**为什么两族都要**：两者误差结构互补，等权平均后 WRMSSE 优于任一单族。本次核实实测两族在 CA_1 上的 **Pearson 相关系数 r = 0.9697** —— 高度相关但不重合，正是融合能带来增益的典型区间。

### 1.2 进度总览表

| # | 阶段 | 状态 | 关键数据 |
| --- | --- | --- | --- |
| 0 | 环境搭建（conda `m5`） | ✅ 完成 | Python 3.10.21 / pandas 2.3.3 / numpy 2.2.5 / lightgbm 4.7.0 / sklearn 1.7.2 / scipy 1.15.3 |
| 0 | 原始数据就位 | ✅ 完成 | `data/` 430 MB，含 `sales_train_evaluation.csv`、`sell_prices.csv`、`calendar.csv`、`sample_submission.csv` |
| 0 | 三份既有 docs 纠错统一 | ✅ 完成 | 见 2.10；⚠️ 遗留一处口径冲突（30490 vs 60980），见 6.6 |
| 1 | 6 个 A 轮脚本实现 + 静态验证 | ✅ 完成 | `py_compile` 全通过；与官方 A1 脚本逐点对照过 |
| 2 | **全 10 店预处理** | ✅ **完成**（13:45:53） | 10/10 店 `stage=complete`，`preprocessing_finished`，峰值 **2.73 GB**，本次运行 **14 分 49 秒** |
| 3 | CA_1 递归训练 | ✅ 完成 | `lgb_model_CA_1_v1.bin` 180 MB，valid RMSE **1.90318** @3000 轮，76 分钟，峰值 5.6 GB |
| 3 | CA_1 非递归训练 | ✅ 完成 | `non_recur_model_CA_1.bin` 125 MB，valid RMSE **1.82056** @3000 轮，83 分钟，峰值 4.6 GB |
| 4 | CA_1 两路预测 | ✅ 完成 | 两份 CSV 各 3049 数据行，质量检查 **19/19 PASS**，两族 r=0.9697 |
| 5 | **其余 9 店递归训练** | ⬜ **未做** | 需 9 个 `lgb_model_{store}_v1.bin`，约 **11.4 小时** |
| 5 | **其余 9 店非递归训练** | ⬜ **未做** | 需 9 个 `non_recur_model_{store}.bin`，约 **12.5 小时** |
| 6 | **全量两路预测** | ⬜ **未做** | 现有 CSV 仅 CA_1 的 3049 行，全量需 **30490 行有效预测**（文件 60980 行） |
| 7 | **集成 → `submission_store_only.csv`** | ⬜ **未做** | `logs/store_ensemble.log` 为 0 字节，从未运行 |

**A 轮完成度：约 55%**（按工作量加权：预处理与流程验证已完成，但 24 小时的训练主体未开始）。

**剩余关键路径**：~~预处理~~（已完成）→ **训练 24 小时** → 全量预测 45-75 分钟 → 集成 < 2 分钟。

### 1.3 本次核实的磁盘现状快照（第二次核实：2026-09-29 14:01）

> ⚠️ 撰写过程中状态推进过一次：13:41 首次核实时预处理还在跑（PID 1048），**13:45:53 该进程正常结束，全 10 店预处理已全部完成**。本节记录的是第二次核实（14:01）的结果，14:10 又复核了一次，无变化。

#### 1.3.1 残留进程

```text
第二次核实（14:01）结论：当前【没有】任何 M5 相关进程在运行。

判定命令与输出：
  ps aux | grep '[p]ython.*src/' | wc -l     ->  0

首次核实（13:34）时曾观察到：
  PID 1048  /home/chenmeihan/miniconda3/envs/m5/bin/python src/1_preprocessing_by_store.py
            由 PID 1046 的 setsid nohup bash 包装启动，日志 logs/preproc_all2.log
  logs/.running_pids 仍写着 preproc_pid=1048  -> 过期标记，可删可忽略

该进程于 13:45:53 自己跑完退出（日志有 preprocessing_finished），不是被杀的。
```

> **本文档撰写者按指令全程没有杀任何进程**；PID 1048 是自己跑完退出的。
>
> ⚠️ **检查进程时别用 `pgrep -af 'src/'`**：它会把"你刚敲的这条命令行本身"也匹配上（因为命令行里含 `src/` 字样），造成假阳性。用 `ps aux | grep '[p]ython.*src/'`（方括号技巧）才准。
>
> 换机前请再自行确认一次（PID 会变，别照抄 1048）：`ps aux | grep '[p]ython.*src/'`。

#### 1.3.2 预处理产物（**全 10 店已完成**）

`data/processed/` 下 10 个门店目录，每个含 5 个 pkl + `metadata.json`，`stage` 全部为 `complete`：

| 门店 | `row_count` | 5 pkl 合计 | `store_complete` 时刻 |
| --- | --- | --- | --- |
| CA_1 | 4,873,639 | 约 1.7 GB | 01:30（首轮单店跑） |
| CA_2 | 4,446,520 | 1577.9 MB | 13:42:28 |
| CA_3 | 4,842,685 | 1718.1 MB | 13:42:57 |
| CA_4 | 4,737,930 | 1681.0 MB | 13:43:19 |
| TX_1 | 4,883,327 | 1732.5 MB | 13:43:41 |
| TX_2 | 4,893,253 | 1736.0 MB | 13:44:16 |
| TX_3 | 4,822,539 | 1711.0 MB | 13:44:42 |
| WI_1 | 4,646,139 | 1648.5 MB | 13:45:03 |
| WI_2 | 4,731,952 | 1678.9 MB | 13:45:31 |
| WI_3 | 4,857,413 | 1723.3 MB | 13:45:53 |

`data/processed/test_CA_1.pkl`：**存在**（1 个，78 MB）。⚠️ 其余 9 店的 `test_*.pkl` **还没有** —— 它们由**递归训练脚本**产出（不是预处理），所以会在步骤 2 自动补齐。

**本次预处理运行的完整时间轴**（`logs/preproc_all2.log`，13:31:04 → 13:45:53，**总计 14 分 49 秒 = 889 秒**）：

```text
13:31:04  store_skip_complete CA_1            瞬时（5 pkl 校验通过，跳过）
13:31:17  global_inputs_loaded                13 秒（sales_rows=30490, price_rows=6841121, RSS 2026.5 MB）
13:31:28  CA_2 store_base_resume  \
13:31:41  CA_3 store_base_resume   |-- 共 35 秒（base 4 pkl 已在且签名匹配，只校验不重算）
13:31:52  CA_4 store_base_resume  /
13:32:56  TX_1 base 64.0 s  |  13:34:07 TX_2 71.0 s  |  13:35:18 TX_3 70.8 s
13:36:37  WI_1 base 78.9 s  |  13:37:52 WI_2 74.5 s  |  13:39:08 WI_3 76.1 s
          -> 第一遍 6 个新店 base 合计 7 分 16 秒（每店 64-79 秒，平均 72.7 秒，写 1204-1268 MB）
13:42:07  global_encoding_stats_ready         2 分 59 秒（遍历 10 店 base，聚合 11 组 count/total/sumsq）
13:45:53  9 店 store_complete                 3 分 46 秒（每店约 25 秒：写 mean_encoding + 重读 5 表校验）
13:45:53  preprocessing_finished              峰值 RSS 全程 2796.9 MB = 2.73 GB
```

> 📌 **这个数字比原计划估计（40-60 分钟）快得多**，原因是第二遍编码只需约 25 秒/店（而非预估的 2-4 分钟），base 也只要约 73 秒/店。**完全冷启动**（`data/processed/` 全空）的预估因此下修为 **20-25 分钟**：全局加载 13 秒 + 10 店 base 约 12.2 分钟 + 编码统计 3 分钟 + 10 店编码约 4.2 分钟 = 约 19.5 分钟，留 IO 波动余量取 20-25 分钟。5.1 节有逐阶段耗时表与推算过程。
>
> **冷启动基线另有单店实测**：CA_1 首次单店跑（含全局加载）`wall 4:14.97`、`Max RSS 2,829,832 KB = 2.70 GB`、`CPU 69%`、`exit 0`（数据来自 `logs/preproc_CA_1.log` 的 `/usr/bin/time -v` 输出）。

#### 1.3.3 模型

| 文件 | 字节 | 约 | mtime | valid RMSE @3000 | 训练耗时 | 峰值 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| `models/lgb_model_CA_1_v1.bin` | 188,248,695 | 180 MB | 11:40 | **1.90318** | 76.3 分钟 | 5617.6 MB = 5.6 GB |
| `models/non_recur_model_CA_1.bin` | 131,416,404 | 125 MB | 13:04 | **1.82056**（另有 `training's rmse: 1.90152`） | 83.0 分钟 | 约 4.6 GB |

**共 2 个模型，合计 305 MB。A 轮目标是 20 个（约 3 GB）。**

> 递归训练日志只有 `valid_0's rmse`，没有 `training's rmse` —— 因为递归脚本的 `valid_sets=[valid_data]` 只传了验证集；非递归脚本传了 `valid_sets=[valid_data, train_data]`，所以两行都有。这是脚本差异，不是日志丢失。

#### 1.3.4 预测产物

```text
output/before_ensemble/submission_kaggle_recursive_store.csv       3050 行（wc -l，含表头）= 3049 数据行
output/before_ensemble/submission_kaggle_nonrecursive_store.csv    3050 行（wc -l，含表头）= 3049 数据行
```

3049 = **仅 CA_1 一店**的 evaluation 段行数（3049 个 item）。全量 10 店应为 **30490 数据行**。

**撰写本文档时对这两份 CSV 做了 19 项独立只读复核，全部 PASS**，要点：

- 列结构完全等于 `['id'] + [f'F{i}' for i in range(1,29)]`，共 29 列
- `id` 无重复、全部以 `_evaluation` 结尾、全部属于 CA_1
- 无 NaN、无 inf、无负数
- F1..F28 的均值**逐日不同**（证明递归回填补丁生效，不是 28 天复制同一个值）
- 两族预测的 **Pearson r = 0.9697**，与基准一致
- dtype 为 `float32`

> ⚠️ 这两份文件在步骤 4 全量预测时会被**整份覆盖**（`format_submission` 每次重写整个文件，没有追加/合并能力）。如果想保留 CA_1 单店结果做对照，**现在就自己拷一份备份**。

#### 1.3.5 输出与日志

```text
output/                      只有 before_ensemble/ 子目录，2 份 CSV，共 3.3 MB
output/submission_store_only.csv    -> 【不存在】，A 轮最终交付物待生成
logs/                        16 个条目，合计 72 KB
logs/store_ensemble.log      -> 0 字节，集成脚本从未运行过
```

`logs/` 全部 14 个日志文件的逐项说明见 2.9。

#### 1.3.6 如何重新核实这份快照

任何时候（尤其换机后）都可以跑**附录 A 的一键体检脚本**，它会输出：残留进程、10 店 stage、模型清单、CSV 行数与质量、目录大小。已实测可正常运行。

---

## 2. 目录与文件全清单讲解

### 2.1 目录树（含大小与迁移属性）

图例：🔴 = **必须迁移的源文件**（不可再生）　🟢 = **可再生成的中间产物**　⚪ = 可删 / 空

```text
/mnt/e/5054project/m5-forecasting/          约 20 GB（第二次核实）
├── src/                                    🔴 76 KB（自研代码，唯一真源）
│   ├── 1_preprocessing_by_store.py         🔴 485 行  脚本(1) 分店低内存预处理
│   ├── 2_train/
│   │   ├── 1-1_recursive_store.py          🔴 171 行  脚本(2) store 族递归训练
│   │   └── 2-1_nonrecursive_store.py       🔴 170 行  脚本(3) store 族非递归训练
│   ├── 3_predict/
│   │   ├── 1-1_recursive_store_PREDICT.py  🔴 164 行  脚本(4) 递归预测
│   │   ├── 2-1_nonrecursive_store_PREDICT.py 🔴 181 行 脚本(5) 非递归预测
│   │   └── 3-1_final_ensemble.py           🔴  97 行  脚本(6) 集成 + 硬校验
│   ├── M5-methods/                         🔴 1.9 GB  官方只读基线（严禁修改）
│   └── __pycache__/                        ⚪ py_compile 产物
├── data/                                   🔴+🟢 18 GB
│   ├── *.csv（6 个原始文件）                🔴 430 MB  Kaggle M5 原始数据
│   └── processed/                          🟢 17 GB（17014 MB）
│       ├── CA_1/ … WI_3/（10 个店目录）      🟢 每个 1.6-1.7 GB，5 pkl + metadata.json
│       └── test_CA_1.pkl                   🟢 78 MB（当前仅 1 个，A 轮跑完 10 个）
├── models/                                 🟢 305 MB（当前 2 个 .bin，A 轮目标 20 个）
├── output/                                 🟢 3.3 MB
│   ├── before_ensemble/                    🟢 2 份 CSV，各 3050 行
│   └── (submission_store_only.csv 尚未生成)  <-- A 轮最终交付物，缺
├── logs/                                   🔴 72 KB（16 个条目，性能与 RMSE 基线）
├── docs/                                   🔴 212 KB（4 份 .md/.txt，含本文档）
├── notebooks/                              ⚪ 空目录
└── .qoder/specs/                           🔴 4 KB（需求与计划来源）
```

### 2.2 `src/` 六个脚本逐个讲解

#### 脚本 (1) `src/1_preprocessing_by_store.py` —— 分店低内存预处理

**职责**：把 3 个原始 CSV 变成 10 个门店目录 × 5 个 pkl 的特征宽表分片。

**核心机制一：低内存"两遍法"**（这是本项目最重要的工程决策）

官方 A1 脚本的做法是把 30490 × 1969 的销售矩阵一次性 `melt` 成长表再做全局 groupby。在 **9.7 GB 内存**的机器上这**必然 OOM**。本项目改成：

```text
第一遍（逐店，落盘）：
  for store in pending:
      grid = build_base_grid(store, ...)      # 只把【一个店】展开成长表
      write_base_outputs(store, grid, ...)    # 写 4 个 base pkl + metadata(stage="base")
      del grid; gc.collect()                  # 立刻释放，内存不累积

中间（跨店聚合，不落盘）：
  stats = aggregate_encoding_stats(iter_global_grids())
  # iter_global_grids() 遍历【全部 10 店】：有 base pkl 就读，没有就内存现算不落盘
  # 每组只保留 count / total / sumsq 三个聚合量（极小），不保留明细
  # 只统计 d <= ENCODING_END(1913) 的行  <- 真值遮罩，见机制二

第二遍（逐店，落盘）：
  for store in pending:
      grid = read_pickle(grid_part_1.pkl)
      encoded = build_encoding_table(grid, stats)   # 用全局统计量回填 22 列编码
      atomic_pickle(encoded, mean_encoding_df.pkl)
      mark_complete(store, category_hash)           # 重读 5 表做严格校验后写 stage="complete"
```

**为什么必须两遍**：目标编码（target encoding）的统计量必须跨全部 10 店聚合（例如 `enc_item_id_mean` 是某商品在全网的均值），但内存装不下 10 店的明细。解决办法是第一遍只落盘、第二遍只用**聚合量**（`count`/`total`/`sumsq`）反推 `mean` 与 `std`：

```text
mean = total / count
variance = (sumsq - total^2 / count) / (count - 1)      # 数值稳定的单遍方差公式
std = sqrt(max(variance, 0))  ，count < 2 时置 NaN
```

**11 组编码维度**（`ENCODING_GROUPS`，每组产出 `_mean` 和 `_std` 两列，共 22 列）：

```text
state_id / store_id / cat_id / dept_id
state_id×cat_id / state_id×dept_id / store_id×cat_id / store_id×dept_id
item_id / item_id×state_id / item_id×store_id
```

**核心机制二：真值遮罩不变量**（防信息泄漏，改代码时最容易破坏的地方）

```python
END_TRAIN    = 1941      # 训练真值的最后一天 d_1941
TOTAL_DAYS   = 1969      # 展开到的最后一天 d_1969（= 1941 + 28 天预测期）
ENCODING_END = 1913      # = END_TRAIN - 28，目标编码只用 d <= 1913 的数据
```

- `ENCODING_END = 1913` 的意义：验证集是 `d_1914..d_1941`（28 天）。如果编码统计用到了这段，就等于**把验证集的答案泄漏进了特征**，valid RMSE 会虚高、上线后崩盘。所以硬性截到 1913。
- `mark_complete()` 里有一道断言：`grid.loc[grid["d"] > END_TRAIN, "sales"].isna().all()` 必须为真，否则抛 `ValueError: xx 预测期 sales 未完全遮蔽`。**预测期的 sales 一律是 NaN**，这是正确的，不是数据缺失。
- `validate_store_files(..., "complete")` 每次续跑都会重查这一条，所以遮罩坏掉的店不会被误当成"已完成"跳过。

**核心机制三：断点续跑三级校验**

```text
validate_store_files(store, "complete")  通过 ->  log_event("store_skip_complete")  整店跳过
validate_store_files(store, "base", category_hash) 通过 -> log_event("store_base_resume") 跳过第一遍
都不通过 -> 从 build_base_grid 开始重做该店
```

校验项（任一不满足即视为未完成，会重算）：`schema_version == 1`、`store` 字段匹配、`stage` 合法、`category_signature` 与当前原始数据算出的指纹一致、要求的 pkl 都存在、**5 表（或 base 的 4 表）索引完全相同且全局唯一**、`row_count` 与实际行数一致、`index_signature`（索引哈希的 sha256）一致、`stage=complete` 时额外检查 `d > 1941` 的 `sales` 全为 NaN。

**签名机制**（自动感知原始数据变化）：

```python
index_signature(index)  = sha256(hash_pandas_object(index).tobytes())
category_signature(vocabs) = sha256(json.dumps(vocabs, sort_keys=True))
```

`vocabs` 是从原始 CSV 提取的全部类别词表（`INDEX_COLUMNS` + 4 个 event 列 + 3 个 snap 列）。**只要原始数据变了一个类别值，`category_signature` 就变，所有 base 产物自动失效重算** —— 这避免了"换了数据却复用旧特征"的静默错误。

**核心机制四：原子写入**（任何时刻被 kill 都不留半个文件）

```python
def atomic_pickle(obj, path):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        pd.to_pickle(obj, temporary)
        os.replace(temporary, path)      # POSIX 原子重命名
    finally:
        if temporary.exists():
            temporary.unlink()
```

`atomic_json` 同理，额外多一次 `handle.flush()` + `os.fsync()` 确保元数据真正落盘。

**因此**：中途 `kill -9` 最多留下一个 `.{name}.tmp-{pid}` 垃圾文件（可安全删除），**绝不会**让一个损坏的 pkl 被后续步骤当成正常产物读取。

**5 个 pkl 各存什么**：

| 文件 | 内容 | 列数量级 | 约大小 |
| --- | --- | --- | --- |
| `grid_part_1.pkl` | 骨架表：`id`、`item_id`、`store_id`、`state_id`、`d`、`sales`（`d>1941` 为 NaN） | 6 | 240 MB（内存） |
| `grid_part_2.pkl` | 价格特征：`sell_price`、`price_max/min/std/mean`、`price_momentum` 等 | 11 | 中 |
| `grid_part_3.pkl` | 日历/事件特征：`date`、`wm_yr_wk`、`wday`、`month`、`year`、`event_name_1/2`、`event_type_1/2`、`snap_CA/TX/WI`、`tm_dw`、`tm_w_end` | 14 | 小 |
| `lags_df_28.pkl` | lag 与滚动特征：`sales_lag_28..42`（15 列）+ `rolling_{mean,std}_{7,14,30,60,180}`（10 列，`shift=28`）+ `rolling_{mean}_tmp_{1,7,14}_{7,14,30,60}`（12 列，短位移，**仅递归用**） | 38 | 大 |
| `mean_encoding_df.pkl` | 11 组 × (mean, std) = 22 列目标编码 | 24（含 id、d） | 466 MB |

**关键不变量**：这 5 张表的 **index 必须完全相同且全局唯一**（按 `(id, day)` 对齐）。训练/预测脚本靠 `pd.concat(axis=1)` 把它们拼成宽表 —— **不做任何键匹配**。所以任何改变行数或顺序的修改都会导致**静默的特征错位**（不报错，但模型学到垃圾）。`validate_store_files` 与 `mark_complete` 反复校验这一点，就是为了守住它。

**环境变量**：`M5_STORES`（逗号分隔选店，默认全 10 店）。⚠️ **执行顺序恒定**为 `ALL_STORES` 的顺序（`parse_stores()` 最后用 `[s for s in ALL_STORES if s in requested_set]` 重排），所以 `M5_STORES=WI_1,CA_1` 和 `M5_STORES=CA_1,WI_1` 行为完全一致 —— 这是为了让日志顺序可复现。
⚠️ **本脚本没有 `M5_FORCE`**。要强制重做某店，只能 `rm -rf data/processed/<店名>` 后重跑。

#### 脚本 (2) `src/2_train/1-1_recursive_store.py` —— store 族递归训练

**职责**：训练 10 个递归模型 `models/lgb_model_{store}_v1.bin`，**并顺带产出** `data/processed/test_{store}.pkl`。

```python
TARGET = "sales";  END_TRAIN = 1941;  P_HORIZON = 28;  SEED = 42
MEAN_FEATURES = ["enc_cat_id_mean","enc_cat_id_std","enc_dept_id_mean",
                 "enc_dept_id_std","enc_item_id_mean","enc_item_id_std"]   # 只用 6 列编码
REMOVE_FEATURES = {"id","state_id","store_id","date","wm_yr_wk","d",TARGET}
LGB_PARAMS = {..., "num_leaves": 2**11-1,        # 2047
              "min_data_in_leaf": 2**12-1,       # 4095
              "num_iterations": 3000, "num_threads": 4, "seed": 42}
```

**流程**：

```python
def train_store(store, force):
    model_path = MODEL_DIR / f"lgb_model_{store}_v1.bin"
    test_path  = PROCESSED_DIR / f"test_{store}.pkl"
    # 断点续跑：两者都在且不 force -> 整店跳过
    if model_path.exists() and test_path.exists() and not force:
        log_event("store_skipped", store=store, reason="model_and_test_exist"); return
    frame, features = load_store(store)          # concat 5 表 -> 宽表
    if force or not test_path.exists():
        save_test_frame(store, frame)            # <-- test pkl 在这里产出
    if model_path.exists() and not force:
        log_event("training_skipped", store=store, reason="model_exists_test_rebuilt"); return
    train_mask = frame["d"] <= END_TRAIN
    valid_mask = train_mask & (frame["d"] > END_TRAIN - P_HORIZON)   # 最后 28 天做验证
    estimator = lgb.train(LGB_PARAMS, train_data, valid_sets=[valid_data],
                          callbacks=[lgb.log_evaluation(100)])       # 无早停，固定 3000 轮
    atomic_pickle(estimator, model_path)         # pickle，不是 save_model()
```

**`save_test_frame` 的窗口与遮罩**（预测脚本依赖它，务必理解）：

```python
prediction_mask = (frame["d"] > END_TRAIN - 100) & (frame["d"] <= END_TRAIN + P_HORIZON)
#  即 d 属于 (1841, 1969]  ->  留 100 天历史给 lag/rolling 计算，再加 28 天预测期
test = frame.loc[prediction_mask].copy()
test.drop(columns=[c for c in test.columns if "_tmp_" in c], inplace=True)
#  ⚠️ 递归的 test 表【删掉】短位移临时列：这些列在预测时要由递归回填现场重算，
#     留着会用上"未来真值"造成泄漏
test.loc[test["d"] > END_TRAIN, TARGET] = np.nan    # 预测期 sales 置 NaN
atomic_pickle(test, PROCESSED_DIR / f"test_{store}.pkl")
```

**环境变量**：`M5_STORES`（选店）、`M5_FORCE=1`（强制重训，忽略已存在的模型）。默认 `M5_FORCE` 未设 = 启用跳过逻辑，**这是断点续跑的关键**。

#### 脚本 (3) `src/2_train/2-1_nonrecursive_store.py` —— store 族非递归训练

**职责**：训练 10 个非递归模型 `models/non_recur_model_{store}.bin`。**不产出 test pkl。**

与递归脚本的差异（**这些差异是刻意的，不要"统一"它们**）：

```python
FIRST_DAY = 710;  END_TRAIN = 1941;  SEED = 1995
GRID2_COLUMNS  = 11 个价格列（白名单显式列出）
GRID3_COLUMNS  = 14 个日历/事件列（白名单显式列出）
LAG_COLUMNS    = [f"sales_lag_{d}" for d in range(28,43)]                        # 15 列
               + [f"rolling_{k}_{w}" for w in [7,14,30,60,180] for k in ["mean","std"]]  # 10 列
               # 共 25 列；⚠️ 完全不包含任何 _tmp_ 短位移列
MEAN_COLUMNS   = ["enc_store_id_dept_id_mean","enc_store_id_dept_id_std",
                  "enc_item_id_state_id_mean","enc_item_id_state_id_std"]        # 只用 4 列，与递归的 6 列不同
LGB_PARAMS 差异：num_leaves = 2**8-1 = 255, min_data_in_leaf = 2**8-1 = 255, seed = 1995

frame = frame.loc[frame["d"] >= FIRST_DAY]     # 裁掉早期 lag 全为 NaN 的行（省内存、省时间）
valid_sets = [valid_data, train_data]          # 传两个，所以日志有 training's + valid_0's 两行
```

**为什么非递归的 `num_leaves` 小得多（255 vs 2047）**：非递归要一次输出 28 天，等效样本量放大 28 倍，用大树会爆内存且过拟合；官方 A1 就是这个配置。

#### 脚本 (4) `src/3_predict/1-1_recursive_store_PREDICT.py` —— 递归预测

**职责**：读 `test_{store}.pkl` + `lgb_model_{store}_v1.bin`，逐日预测 28 天并回填，输出 `output/before_ensemble/submission_kaggle_recursive_store.csv`。

```python
ROLLING_SPECS = [(shift, window) for shift in [1,7,14] for window in [7,14,30,60]]   # 12 个
# 28 天循环：每一天
#   1) 用当前 sales 序列重算 12 个短位移滚动特征（这就是"补丁 #1"，A1 方案的关键）
#   2) 对 d == 当天 的行做预测
#   3) 把预测值写回 sales 列，供下一天的 lag/rolling 使用  <- 递归回填
#   ⚠️ 模型在循环【内层】每店重新 pickle.load 一次
#      -> 全量运行 = 28 天 × 10 店 = 280 次加载 × 180 MB ≈ 50 GB IO（见 6.5）
```

**`format_submission` 的行数分支**（60980 vs 3049 的来源，务必看懂）：

```python
def format_submission(predictions, stores):
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")     # 60980 行
    full_run = (stores == ALL_STORES)      # 关键判据：【完全不设 M5_STORES】才为 True
    if full_run:
        output = sample.copy()             # 60980 行
        output[F_COLUMNS] = np.float32(0)  # 先全填 0（validation 段就保持 0）
    else:
        suffixes = tuple(f"_{store}_evaluation" for store in stores)
        output = sample.loc[sample["id"].str.endswith(suffixes)].copy()   # 单店 = 3049 行
    if full_run and len(output) != 60980:
        raise ValueError(...)
    # 再把 evaluation 段的预测值覆盖进去
```

> ⚠️ **两个重要推论**：
> 1. 全量预测**绝对不能设 `M5_STORES`**，否则 `full_run=False`，只会输出被选店的行。
> 2. 全量预测**不能分批跑**（例如先 `M5_STORES=CA_1,CA_2` 再 `M5_STORES=CA_3,...`）—— 每次运行都会**整份重写** CSV，没有合并能力，后一批会把前一批冲掉。**必须一次性不设 `M5_STORES` 跑完 10 店。**

#### 脚本 (5) `src/3_predict/2-1_nonrecursive_store_PREDICT.py` —— 非递归预测

**职责**：读 5 个 pkl + `non_recur_model_{store}.bin`，一次性预测 28 天，输出 `output/before_ensemble/submission_kaggle_nonrecursive_store.csv`。

与递归预测的区别：**没有 28 天循环、没有回填、不读 `test_*.pkl`**（直接读 5 个 pkl 自己拼），因此 IO 量小得多、也没有短位移滚动特征的重算。日志事件是 `store_predicted`(rows, sales_sum) + `prediction_finished`(elapsed_seconds)。

#### 脚本 (6) `src/3_predict/3-1_final_ensemble.py` —— 集成 + 硬校验

**职责**：把两份 CSV 等权平均，输出 `output/submission_store_only.csv`。

```python
INPUT_FILES = [BEFORE_ENSEMBLE_DIR / "submission_kaggle_recursive_store.csv",
               BEFORE_ENSEMBLE_DIR / "submission_kaggle_nonrecursive_store.csv"]
OUTPUT_FILE = OUTPUT_DIR / "submission_store_only.csv"

def validate_submission(path, sample_ids):        # 7 项硬校验，任一失败即 raise
    ① 文件存在
    ② list(frame.columns) == EXPECTED_COLUMNS      # ['id','F1'..'F28']
    ③ len(frame) == 60980
    ④ not frame["id"].duplicated().any()
    ⑤ frame["id"].reset_index(drop=True).equals(sample_ids.reset_index(drop=True))   # 与 sample 完全同序
    ⑥ np.isfinite(values).all()                    # 无 NaN / inf
    ⑦ not (values < 0).any()                       # 无负数

stacked = np.stack([...])                          # 逐文件读入 F 列
output[F_COLUMNS] = stacked.mean(axis=0)           # 等权平均，无权重调优
```

> ⚠️ **`M5_ROUND` 是空壳变量**：`docs/M5_复现执行手册.md` 第 317/430/433 行写的 `M5_ROUND=A` / `M5_ROUND=B`，**源码里根本没有读它**（`grep -n environ src/3_predict/3-1_final_ensemble.py` 无结果）。设或不设完全一样，输出永远是 `output/submission_store_only.csv`。B 轮要产出 `submission_final.csv`，必须**改脚本**的 `INPUT_FILES` 和 `OUTPUT_FILE`。详见 6.6。

### 2.3 六脚本共用的四条工程约定

| 约定 | 说明 | 违反后果 |
| --- | --- | --- |
| **索引对齐** | 5 张 pkl 的 index 必须完全相同且唯一，靠 `pd.concat(axis=1)` 拼宽表，不做键匹配 | 特征**静默错位**，不报错但模型学到垃圾 |
| **真值遮罩** | `ENCODING_END=1913`、`d>1941` 的 sales 为 NaN、递归 test 表删 `_tmp_` 列 | 验证集泄漏，valid RMSE 虚高，线上崩盘 |
| **原子写入** | 一律走 `.{name}.tmp-{pid}` + `os.replace()` | 中断留下半个文件，被后续步骤当正常产物读 |
| **解释器全路径** | 所有命令用 `/home/<用户名>/miniconda3/envs/m5/bin/python`，不依赖 shell 激活状态 | nohup/setsid/cron 下 `python` 可能解析到系统 Python，报 `ModuleNotFoundError` |

### 2.4 `src/M5-methods/` —— 官方只读基线（🔴 必须迁移，**严禁修改**）

约 **1.9 GB**，是 M5 官方仓库（含各名次方案代码与文档）的完整拷贝。本项目 A1 方案脚本的对照源在：

```text
src/M5-methods/Code of Winning Methods/A1/3. code/2. train/1-1. recursive_store_TRAIN.py
src/M5-methods/Code of Winning Methods/A1/3. code/3. predict/...
```

**为什么必须带走**：`docs/M5_复现执行手册.md` 里有一条静态验证命令是拿自研脚本与官方脚本做 `diff -u`。缺了这个目录就**无法验证自研代码没被改错**，也无法在遇到行为疑惑时回溯"官方原本是怎么做的"。

**纪律**：**只读**。任何情况下都不要改这里的文件 —— 它是对照基准，改了它就失去了参考价值。如果空间极度紧张，可以只保留 `Code of Winning Methods/A1/` 子目录（约占很小一部分）。

### 2.5 `data/` 原始数据（🔴 必须迁移）

合计约 **430 MB**（不含 `processed/`）。来源是 Kaggle "M5 Forecasting - Accuracy" 竞赛。

| 文件 | 用途 | 本轮是否用到 |
| --- | --- | --- |
| `sales_train_evaluation.csv` | 主输入：30490 行 × (1969 天 + 6 个 id 列) | ✅ 预处理唯一销量来源 |
| `sell_prices.csv` | 6,841,121 行，按 (store, item, wm_yr_wk) 的周均价 | ✅ |
| `calendar.csv` | 1969 行日期元数据 + 事件 + snap | ✅ |
| `sample_submission.csv` | **60980 行**，提交格式模板与 id 顺序基准 | ✅ 预测/集成脚本都要读 |
| `sales_train_validation.csv` | 117 MB，只含到 d_1913 | ⬜ 本轮不用（B 轮或 WRMSSE 评分器会用） |

> **迁移选择**：这 430 MB **必须带走**，或者在新机器上重新从 Kaggle 下载（需要账号 + 接受竞赛规则 + `kaggle competitions download -c m5-forecasting-accuracy`）。**强烈建议直接带走** —— 体积不大，且重新下载需要网络和凭据，换机时容易卡住。
>
> ⚠️ **不要修改任何原始 CSV**：`category_signature` 会因任何类别值变化而改变，导致 17 GB 的 base 产物**全部失效重算**。

### 2.6 `data/processed/` 产物含义（🟢 可再生成）

**17 GB**，10 个门店目录 + 1 个 `test_CA_1.pkl`。每店目录内容：

```text
data/processed/CA_1/
├── grid_part_1.pkl        骨架 + sales（d>1941 为 NaN）
├── grid_part_2.pkl        价格特征 11 列
├── grid_part_3.pkl        日历/事件特征 14 列
├── lags_df_28.pkl         lag + 滚动特征 38 列（含 12 个 _tmp_ 短位移列）
├── mean_encoding_df.pkl   11 组 × 2 = 22 列目标编码
└── metadata.json          schema_version / stage / store / row_count /
                           index_signature / category_signature / columns / sizes
```

`metadata.json` 的 `stage` 字段：`"base"` = 只有前 4 个 pkl（第一遍完成）；`"complete"` = 5 个 pkl 齐全且通过全部校验。**第二次核实时 10 店全部为 `complete`。**

`data/processed/test_{store}.pkl`（当前只有 `test_CA_1.pkl`，78 MB）：

- **用途**：递归预测脚本 (4) 的**唯一输入**。它是从训练宽表里切出 `d ∈ (1841, 1969]` 的窗口、删掉 `_tmp_` 列、把 `d>1941` 的 sales 置 NaN 后得到的"待回填骨架"。
- **产出者**：**递归训练脚本 (2)** 的 `save_test_frame()`，**不是预处理脚本**。所以新机器上如果只带了 pkl 没带模型，跑步骤 2 时会自动补出这 10 个文件。
- 非递归预测脚本 (5) **不用**它。

### 2.7 `models/` 两类模型命名规则（🟢 可再生成，但很贵）

| 命名 | 来源脚本 | 当前 | A 轮目标 | 单个大小 |
| --- | --- | --- | --- | --- |
| `lgb_model_{store}_v1.bin` | 脚本 (2) 递归训练 | 1 个（CA_1） | 10 个 | 约 180 MB |
| `non_recur_model_{store}.bin` | 脚本 (3) 非递归训练 | 1 个（CA_1） | 10 个 | 约 125 MB |

`{store}` ∈ `CA_1 CA_2 CA_3 CA_4 TX_1 TX_2 TX_3 WI_1 WI_2 WI_3`。

⚠️ **命名不对称**（递归有 `_v1` 后缀、非递归没有；一个是 `lgb_model_` 前缀、一个是 `non_recur_model_` 前缀）—— 这是照抄官方 A1 的命名，**脚本里写死了**，不要"顺手统一"，否则预测脚本找不到模型。

⚠️ **这些 `.bin` 是 `pickle.dump(lgb.Booster)` 的产物，不是 `booster.save_model()` 的文本格式**。所以**跨 lightgbm 版本不可移植**（见 6.2）。

**重生成代价：约 24 小时**（4 核机）。这是"可再生成但极贵"的典型 —— 如果版本能对齐，**强烈建议带走这 305 MB**。

### 2.8 `output/` 结构（🟢 可再生成）

```text
output/
├── before_ensemble/
│   ├── submission_kaggle_recursive_store.csv       <- 脚本 (4) 产出，当前 3050 行（仅 CA_1）
│   ├── submission_kaggle_nonrecursive_store.csv    <- 脚本 (5) 产出，当前 3050 行（仅 CA_1）
│   └── .gitkeep
└── submission_store_only.csv                       <- 脚本 (6) 产出，A 轮交付物（尚未生成）
```

全量跑完后，`before_ensemble/` 的两份 CSV 会变成 **60981 行**（`wc -l`，含表头）= 60980 数据行。B 轮会往 `before_ensemble/` 再加 4 份（cat 族、dept 族各两路）。

### 2.9 `logs/` 各日志（🔴 建议迁移，72 KB）

| 文件 | 来源 | 关键事件 / 内容 | 大小与备注 |
| --- | --- | --- | --- |
| `preprocessing_by_store.log` | 预处理脚本内置（追加） | 全部 `store_*` / `global_*` / `preprocessing_finished` 事件 | 5.3 KB，跨多次运行累积 |
| `preproc_CA_1.log` | CA_1 单店运行的 shell 重定向（含 `/usr/bin/time -v`） | `Elapsed 4:14.97` / `Maximum resident set size 2,829,832 KB` / `Exit status 0` | 1.6 KB，**冷启动基线数据来源** |
| `preproc_all.log` | 13:13 那次被取消的全量运行 | 只到 CA_4 `store_base_elapsed`，**无 `preprocessing_finished`** | 996 B，中断证据 |
| `preproc_all2.log` | 13:31 重启的全量运行（**已于 13:45:53 正常完成**） | `store_skip_complete`(CA_1) + `store_base_resume`(CA_2/3/4) + TX/WI 各店 `store_base_written` + 9 条 `store_complete` + `preprocessing_finished` | 3.5 KB，**全量预处理完成记录** |
| `recursive_store_train.log` | 递归训练脚本内置 | `test_frame_written` / `training_started`(features, rows, valid_rows, peak_rss_mb) / `training_finished`(seconds, model_mb) | 442 B |
| `train_recursive_CA_1.log` | 递归训练 CA_1 的 shell 重定向（含 LightGBM 每 100 轮 RMSE） | `[3000] valid_0's rmse: 1.90318` | 1.4 KB |
| `nonrecursive_store_train.log` | 非递归训练脚本内置 | 同上 | 272 B |
| `train_nonrecursive_CA_1.log` | 非递归训练 CA_1 的 shell 重定向 | `[3000] training's rmse: 1.90152  valid_0's rmse: 1.82056` | 1.9 KB |
| `recursive_store_predict.log` | 递归预测脚本内置 | 28 条 `day_predicted` + `prediction_finished`；**含大量 pandas FutureWarning**（见 6.1） | 4.3 KB |
| `predict_recursive_CA_1.log` | 递归预测 CA_1 的 shell 重定向 | 同上 | 18 KB（Warning 占绝大部分） |
| `nonrecursive_store_predict.log` | 非递归预测脚本内置 | `store_predicted`(rows, sales_sum) + `prediction_finished`(elapsed_seconds) | 377 B |
| `predict_nonrecursive_CA_1.log` | 非递归预测 CA_1 的 shell 重定向 | 同上 | 14 KB |
| `store_ensemble.log` | 集成脚本内置 | `input_validated` / `ensemble_finished` | **0 B（尚未运行过集成）** |
| `.running_pids` | 手工/启动脚本写入 | `preproc_pid=1048` | 17 B，**该进程已于 13:45:53 退出，标记过期**；换机后应删除或忽略 |
| `.gitkeep` | 占位 | —— | 0 B |

> 命名规律：`{阶段}_{脚本内置日志名}.log` 是脚本自己追加写的；`{动作}_{族}_{店}.log` 是 shell 重定向的单次运行快照。**脚本内置日志是追加模式**，多次运行会叠在同一文件里，靠 JSON 的 `time` 字段区分批次。
>
> **带走 `logs/` 的价值**：新机器跑完 CA_1 后可直接对比耗时/RSS/RMSE，判断"新机器结果和旧机器一致吗"。**RMSE 应当逐位相同**（同 seed、同数据、同版本）；耗时和内存则反映新机器性能。这是最省事的一致性验证手段。

### 2.10 `docs/` 四份文档定位（🔴 必须迁移）

| 文件 | 行数 | 定位 |
| --- | --- | --- |
| `M5_第一名方案复现技术文档.md` | 729 | **原理层**：A1 方案作者背景、六族 220 模型结构、WRMSSE 三层加权公式、递归 vs 非递归的取舍、特征清单与语义、与官方代码的逐点对照。改代码前必读 |
| `M5_复现执行手册.md` | 454 | **操作层**：阶段 0-4 的命令与检查点、环境自检、静态验证、A 轮/B 轮命令清单、故障速查表。⚠️ 其中"每份 CSV 30490 行"的口径与当前实现不一致，`M5_ROUND` 也是无效变量，见 6.6 |
| `M5_memory_and_low_RAM_pipeline_summary.txt` | 410 | **约束层**：为什么官方全局 melt 在 9.7 GB 内存下必挂、低 RAM 流水线的分段策略、pickle vs Parquet 的取舍说明 |
| `M5_A轮迁移交接文档.md` | 本文件 | **迁移层**：现状快照 + 搬家清单 + 新机环境 + 剩余待办 + 已知坑 |

### 2.11 迁移属性总表

| 类别 | 路径 | 大小 | 属性 | 说明 |
| --- | --- | --- | --- | --- |
| 自研代码 | `src/*.py`, `src/2_train/`, `src/3_predict/` | 76 KB | 🔴 **必须** | 不可再生，唯一真源 |
| 官方基线 | `src/M5-methods/` | 1.9 GB | 🔴 **必须**（可只带 A1） | 只读参考，缺失则无法 diff 校验 |
| 原始数据 | `data/*.csv` | 430 MB | 🔴 **必须**（或重新下载） | 预处理唯一输入 |
| 文档 | `docs/`, `.qoder/specs/` | 216 KB | 🔴 **必须** | 上下文与需求来源 |
| 日志 | `logs/` | 72 KB | 🔴 强烈建议 | 性能与 RMSE 基线对照 |
| 特征中间产物 | `data/processed/*/` | **17 GB** | 🟢 可再生成 | 冷启动全量重生成实测推算仅 **20-25 分钟**（见 5.1） |
| 递归测试集 | `data/processed/test_*.pkl` | 78 MB/个，**当前仅 1 个**（CA_1），A 轮跑完 10 个 ≈ 780 MB | 🟢 可再生成 | 由**递归训练脚本**（不是预处理）自动产出 |
| 模型 | `models/*.bin` | **当前 2 个 = 305 MB** → A 轮 20 个约 3 GB | 🟢 可再生成但**极贵** | 重生成约 24 小时；版本对齐则建议带走 |
| 预测输出 | `output/` | 3.3 MB（2 份 CSV 各 3050 行） | 🟢 可再生成 | 步骤 4 全量跑会**整份覆盖**为 60981 行 |
| 缓存 | `src/__pycache__/` 等 | 少量 | ⚪ 可删 | 自动重建 |
| 空目录 | `notebooks/` | 0 | ⚪ | 保留目录本身即可 |

---

## 3. 迁移打包清单

### 3.1 实际大小（`du -sh`，第二次核实 14:01）

```text
约 20G  整个项目（data 18G + src 1.9G + models 305M + output 3.3M + docs 212K + logs 72K）
18G     data
17G     data/processed              10 店全部 complete
1.7G    data/processed/CA_1         每店 1.6-1.7G，结构相同
1.7G    data/processed/TX_2         （最大的一店）
430M    data/*.csv                  6 个原始 CSV
1.9G    src                         其中 src/M5-methods 占绝大部分
76K     src/*.py + 2_train + 3_predict   自研代码本体
305M    models                      2 个 .bin（179.5 MB + 125.3 MB）
3.3M    output                      2 份 CSV
212K    docs                        4 份文档（含本文档约 126 KB）
72K     logs                        14 个日志 + .running_pids + .gitkeep
```

> ✅ 以上数字由**附录 A 的一键体检脚本实测产出**（撰写本文档时实跑过，第 7 节"目录大小"输出与此完全一致）。换机后跑一次附录 A 即可复现同样的清单做对照。

> ⚠️ **`du -sh .` 在 `/mnt/e`（WSL drvfs 挂载）上会返回异常的 `8.0K`**，这是挂载盘上 `du` 遍历的已知怪癖，**不代表项目只有 8 KB**。正确做法是各子目录分别 `du -sh` 后相加（如上）。换到原生 ext4 分区后此问题消失。

**A 轮全部跑完后的预估**：`data/processed/` 增约 780 MB（9 个 test pkl）、`models/` 增约 2.7 GB（18 个模型）、`output/` 增约 10 MB → **总计约 23 GB**。

### 3.2 三种打包方案

> 🎯 **结论先行：推荐方案 B（约 2.9 GB）。** 理由：17 GB 的 pkl 重算实测推算只要 20-25 分钟，而多拷 17 GB 在慢盘/网络上往往要几十分钟到几小时；同时方案 B 保留了最贵的资产（模型，重算要 24 小时）。

#### 方案 A：整目录全量迁移（约 23 GB，**不推荐**）

```bash
cd /mnt/e/5054project
tar -cf m5_full.tar --exclude='__pycache__' --exclude='*.tmp-*' m5-forecasting/
```

- ✅ 好处：新机器开箱即用，不需要重跑任何预处理。
- ❌ 坏处：包最大、传输最慢；**且 pkl 对新机器的 pandas 版本敏感**（3.4），一旦版本没对齐，这 17 GB 就是废数据，白带。
- 只在"新机器环境版本能 100% 对齐 + 传输通道很快（如局域网/同一块硬盘直接搬）"时才选它。

#### 方案 B：带模型、不带 pkl（约 2.9 GB，**推荐**）

```bash
cd /mnt/e/5054project
tar -cf m5_planB.tar \
  --exclude='m5-forecasting/data/processed' \
  --exclude='__pycache__' --exclude='*.tmp-*' \
  m5-forecasting/
# 包内约：src 1.9G + data/*.csv 430M + models 305M + output 3.3M + docs/logs 272K
```

**新机器上的正确执行顺序**（与直接跑步骤 2 略有不同，注意）：

```bash
# ① 解包 + 建环境（第 4 章），过 4.7.1 数据自检
# ② 先跑预处理（此时 data/processed 为空，冷启动约 20-25 分钟）
setsid nohup <PY> src/1_preprocessing_by_store.py > logs/preproc_new.log 2>&1 < /dev/null &
# ③ 等 ② 出现 preprocessing_finished 后，跑步骤 2 递归训练
#    -> CA_1 会因"模型已存在"而只补 test pkl 不重训（日志 training_skipped），
#       其余 9 店正常训练并各自产出 test pkl
# ④ 跑步骤 3 非递归训练（CA_1 同样自动跳过）
```

> ⚠️ **警告 1：不要跳过 ②**。方案 B 的包里没有 `data/processed/`，而步骤 2 的训练脚本要读 5 个 pkl。直接跑训练会 `FileNotFoundError`。
>
> ⚠️ **警告 2：`test_CA_1.pkl` 也不在包里**（它在 `data/processed/` 下）。这没关系 —— 递归训练脚本发现"模型在、test pkl 不在"时会走 `training_skipped` 分支**只补 test pkl 不重训**，正好是我们想要的（保住已有的 76 分钟训练成果）。这是设计好的行为，看到 `training_skipped` 不要慌。

#### 方案 C：只带不可再生的东西（约 2.4 GB，兜底）

```bash
cd /mnt/e/5054project
tar -cf m5_planC.tar \
  --exclude='m5-forecasting/data/processed' \
  --exclude='m5-forecasting/models' \
  --exclude='m5-forecasting/output' \
  --exclude='__pycache__' --exclude='*.tmp-*' \
  m5-forecasting/
```

- 只带 `src/`（含 M5-methods）+ `data/*.csv` + `docs/` + `logs/` + `.qoder/`。
- ❌ 代价：**放弃已有的 2 个 CA_1 模型**，新机器上要多花约 2.6 小时重训（76 + 83 分钟）。
- ✅ 适用：模型版本无法对齐（见 6.2，lightgbm 版本必须完全一致，否则 pickle 加载失败）、或传输通道极度受限。
- **如果 `src/M5-methods` 也可以舍弃**（放弃 diff 校验能力），包能缩到约 **500 MB**，是极端情况下的最小可行集。

### 3.3 打包完整性校验

```bash
# 打包后立即校验（在旧机器上）
md5sum m5_planB.tar > m5_planB.tar.md5
ls -l m5_planB.tar                      # 记下字节数

# 解包后校验（在新机器上）
md5sum -c m5_planB.tar.md5              # 必须 OK
tar -tf m5_planB.tar | wc -l            # 条目数应与旧机器一致

# 关键文件逐一确认
ls src/*.py src/2_train/*.py src/3_predict/*.py | wc -l    # 应为 6
ls data/*.csv | wc -l                                      # 应为 6
ls -l models/*.bin                                         # 方案 A/B 应为 2 个，字节数与 1.3.3 一致
wc -l docs/*.md docs/*.txt                                 # 行数应与 2.10 表格一致
ls logs/ | wc -l
```

### 3.4 pickle 版本敏感性（**最容易踩的坑**）

本项目有两类二进制产物，**都不是跨版本安全的**：

| 产物 | 写入方式 | 敏感于 | 不兼容时的表现 |
| --- | --- | --- | --- |
| `data/processed/**/*.pkl`（17 GB） | `pd.to_pickle(obj, path)` | **pandas / numpy / Python 版本** | `UnpicklingError`、`AttributeError: Can't get attribute 'BlockManager'`、或**静默 dtype 变化** |
| `models/*.bin`（305 MB） | `pickle.dump(lgb.Booster)` | **lightgbm 版本**（严格） | `AttributeError` / `LightGBMError: unknown object type` |

**铁律**：新机器必须严格按 4.1 的版本表安装，**逐项对齐，不要用 `pip install -U` 顺手升级**。

**特别警告**：

1. **不要升级 pandas 到 3.x**。现有代码有 pandas 2.x 的 `FutureWarning`（6.1），3.x 下会**直接抛错**，流水线跑不完。
2. **lightgbm 必须是 4.7.0**。当前 `models/*.bin` 是 4.7.0 序列化的。如果新机器只能装别的版本，**这两个模型就作废了**，此时应改用**方案 C**（不带模型），重训 CA_1 约 2.6 小时。
3. `models/*.bin` 用的是 `pickle.dump`，**不是** `booster.save_model()`。后者是文本格式、跨版本兼容性好得多。**本轮不改源码**（任务边界），但如果将来要长期保存模型，建议改成 `save_model()` + `.txt`。记录在此，不作为本轮待办。

**版本对齐后的验证方法**：跑 4.7.2 / 4.7.3 的自检脚本，能成功 `read_pickle` 并打印出 shape 就说明兼容。

### 3.5 换机前的收尾动作（按顺序）

```bash
cd /mnt/e/5054project/m5-forecasting

# 1) 停止所有 M5 进程
#    （第二次核实 14:01 时已无任何进程：PID 1048 于 13:45:53 自然结束。
#      但你执行本步骤时可能又起了新任务，所以仍要检查一遍）
ps aux | grep '[p]ython.*src/'   # 用 [p] 技巧避免匹配到 grep 自身；输出为空 = 无进程
# ⚠️ 别用 pgrep -af 'src/'：它会把"你刚敲的这条命令行本身"也匹配上，造成假阳性
# 若有进程：等它跑完，或 kill <PID>   # 原子写盘保证安全，重跑即续（见 5.8）
sleep 5 && ps aux | grep '[p]ython.*src/' | wc -l   # 确认输出 0

# 2) 清理中断残留的临时文件（有才清；正常结束时应该一个都没有）
find data/processed models output -name '.*.tmp-*' -print

# 3) 清理过期的 PID 标记（当前内容 preproc_pid=1048 已失效）
cat logs/.running_pids           # 记下内容作参考，然后清空或删除该文件

# 4) 删除 pycache（可选，减小体积）
find src -name '__pycache__' -type d

# 5) 最后确认一遍现状，把输出贴进交接记录
ls data/processed/*/mean_encoding_df.pkl | wc -l    # 目标 10
ls models/*.bin | wc -l                             # 当前 2，A 轮目标 20
wc -l output/before_ensemble/*.csv                  # 当前各 3050
du -sh data/processed models output
```

> 💡 **如果换机时训练正在跑**：直接 `kill` 是安全的（原子写入保证不留半个文件），但**当前正在训的那一店会白跑**（模型只在训练完全结束后才落盘）。所以如果某店已经训了 60 分钟，最好等它出 `training_finished` 再换机。已完成的店不受影响，新机器上会自动 `store_skipped`。

---

## 4. 新机器环境搭建步骤

### 4.1 目标版本表（**必须逐项对齐**）

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| OS | Ubuntu 24.04（WSL）或原生 Linux | 脚本用 `resource.getrusage`（**POSIX only**），Windows 原生 Python **跑不了** |
| conda | miniconda3（任意近期版本） | 环境名固定 **`m5`** |
| Python | **3.10.21** | pickle 兼容性的第一道门槛 |
| pandas | **2.3.3** | ⚠️ **不要升到 3.x**（见 3.4 / 6.1） |
| numpy | **2.2.5** | |
| lightgbm | **4.7.0** | ⚠️ 必须精确一致，否则 `models/*.bin` 加载失败（见 6.2） |
| scikit-learn | **1.7.2** | 本轮实际未直接调用，但为官方 A1 依赖 |
| scipy | **1.15.3** | numpy 2.x 配套 |
| CPU | 当前 4 核 | `num_threads=4` 写死在两个训练脚本的 `LGB_PARAMS` 里 |
| 内存 | 当前 **9.7 GB** | 红线见 6.3 |

**关于 lightgbm 的 CPU/CUDA 版**：当前装的是 **CUDA 版**（`lightgbm 4.7.0`，带 GPU 支持编译），但**脚本里没有任何 `device_type=gpu` 参数**，所以**实际是纯 CPU 运行**，GPU 完全没用上。
👉 **新机器装 CPU 版即可**（`pip install lightgbm==4.7.0`，默认就是 CPU 版），更省事、无需 CUDA 工具链，**性能完全一样**。

### 4.2 创建 conda 环境

```bash
conda create -n m5 python=3.10.21 -y
```

> 如果 `python=3.10.21` 在 conda 频道里找不到精确小版本，用 `python=3.10` 即可 —— **3.10.x 系列内的 pickle 兼容性没有实际差异**，真正严格的是 pandas 和 lightgbm。

### 4.3 安装依赖

```bash
PY=/home/<用户名>/miniconda3/envs/m5/bin/python

$PY -m pip install --upgrade pip
$PY -m pip install \
  "pandas==2.3.3" \
  "numpy==2.2.5" \
  "lightgbm==4.7.0" \
  "scikit-learn==1.7.2" \
  "scipy==1.15.3"
```

**离线安装方案**（新机器无网络时）：在旧机器上先下载 wheels，随包带走。

```bash
# 旧机器（有网）
mkdir -p /mnt/e/5054project/wheels
/home/chenmeihan/miniconda3/envs/m5/bin/python -m pip download \
  -d /mnt/e/5054project/wheels \
  pandas==2.3.3 numpy==2.2.5 lightgbm==4.7.0 scikit-learn==1.7.2 scipy==1.15.3

# 新机器（无网）
$PY -m pip install --no-index --find-links=<wheels目录> \
  pandas==2.3.3 numpy==2.2.5 lightgbm==4.7.0 scikit-learn==1.7.2 scipy==1.15.3
```

> ⚠️ `pip download` 会下**当前平台**的 wheel。如果新旧机器都是 x86_64 Linux，可以直接用；否则在新机器上加 `--platform manylinux2014_x86_64 --only-binary=:all:` 重新下载。

### 4.4 验证命令

```bash
PY=/home/<用户名>/miniconda3/envs/m5/bin/python

# ① 版本逐项核对（必须与 4.1 表格完全一致）
$PY -c "
import sys, pandas, numpy, lightgbm, sklearn, scipy
print('python     ', sys.version.split()[0])
print('pandas     ', pandas.__version__)
print('numpy      ', numpy.__version__)
print('lightgbm   ', lightgbm.__version__)
print('sklearn    ', sklearn.__version__)
print('scipy      ', scipy.__version__)
print('interpreter', sys.executable)
"
# 期望输出：
#   python      3.10.21
#   pandas      2.3.3
#   numpy       2.2.5
#   lightgbm    4.7.0
#   sklearn     1.7.2
#   scipy       1.15.3

# ② lightgbm 能真正训练（验证编译无缺库）
$PY -c "
import numpy as np, lightgbm as lgb
X = np.random.rand(200, 5); y = np.random.rand(200)
m = lgb.train({'objective':'tweedie','tweedie_variance_power':1.1,'verbosity':-1,
               'num_leaves':15,'num_iterations':10,'num_threads':4},
              lgb.Dataset(X, y))
print('lightgbm OK, num_model_per_iteration =', m.num_model_per_iteration())
print('booster dump OK, trees =', len(m.dump_model()['tree_info']))
"

# ③ 6 个脚本静态编译（不执行）
cd <项目目录>
for f in src/*.py src/2_train/*.py src/3_predict/*.py; do
  $PY -m py_compile "$f" && echo "OK   $f" || echo "FAIL $f"
done
# 期望：6 行 OK

# ④ 硬件资源确认
nproc                       # CPU 核数（决定 num_threads 是否要改，见 6.4）
free -g                     # 内存总量（决定能否并行两路训练，见 6.3）
df -h .                     # 磁盘余量（至少留 25 GB）
```

### 4.5 解释器完整路径调用约定（**本项目铁律**）

**所有命令一律写完整的解释器路径，不依赖 shell 是否 `conda activate`。**

```bash
# ✅ 正确
/home/<用户名>/miniconda3/envs/m5/bin/python src/2_train/1-1_recursive_store.py

# ❌ 错误（依赖激活状态）
python src/2_train/1-1_recursive_store.py
conda activate m5 && python src/...
```

**为什么这条是铁律**：

1. 本项目所有长任务都用 `setsid nohup ... &` 后台跑。这种脱离终端的进程**不继承交互式 shell 的 conda 激活状态**，`python` 会解析到 `/usr/bin/python3`（系统 Python），立刻 `ModuleNotFoundError: No module named 'pandas'`。
2. 日志里看不出是"哪个 python 跑的"，排查困难。用全路径则日志天然可追溯。
3. 换机器时用户名可能变（`/home/chenmeihan/` → `/home/<新用户名>/`）。**建议在新机器上第一件事就是定义一个 shell 变量**，全文档的 `<PY>` 都指它：

```bash
# 加进 ~/.bashrc，或每次会话开头执行
export PY=/home/<新用户名>/miniconda3/envs/m5/bin/python
$PY -V        # 确认能跑
```

> 旧机器上的完整路径是 `/home/chenmeihan/miniconda3/envs/m5/bin/python`（`logs/` 里所有历史命令用的都是它）。

### 4.6 项目放置位置与 IO 建议

**当前**：`/mnt/e/5054project/m5-forecasting`（Windows E 盘，WSL 通过 drvfs/9P 访问，IO 慢，见 6.5）。

**建议**：新机器上放到**原生 Linux 分区**，例如 `/home/<用户名>/m5-forecasting`。预计整体提速 **15-40%**，对递归预测（280 次模型加载，约 50 GB IO）改善最明显。

```bash
# 解包到原生分区
mkdir -p ~/work && cd ~/work
tar -xf /path/to/m5_planB.tar          # 解出 ~/work/m5-forecasting/
cd m5-forecasting
mkdir -p logs data/processed models output/before_ensemble notebooks
```

> ⚠️ 方案 B/C 的包里**没有** `data/processed/`，`tar` 不会创建这个目录。虽然预处理脚本里有 `PROCESSED_DIR.mkdir(parents=True, exist_ok=True)` 会自动建，但**显式建一次更保险**（也顺便建出 `notebooks/`）。

如果必须放在挂载盘（如 `/mnt/e`）：**接受耗时上浮**，把第 5 章的预期时间乘 **1.2-1.5**，不要试图去"优化" drvfs。

### 4.7 迁移后自检（**跑长任务前必须过**）

#### 4.7.1 原始数据自检

```bash
cd <项目目录>
ls -l data/*.csv                     # 6 个文件，字节数应与旧机器一致
$PY - <<'PY'
import pandas as pd, hashlib, pathlib
RAW = pathlib.Path('data')
expect = {'sales_train_evaluation.csv': (30490, 1975),
          'calendar.csv': (1969, 14),
          'sample_submission.csv': (60980, 29)}
for name, (rows, cols) in expect.items():
    df = pd.read_csv(RAW / name)
    ok = (df.shape == (rows, cols))
    print(f'{name:32s} shape={df.shape}  expect=({rows}, {cols})  {"OK" if ok else "MISMATCH"}')
sp = pd.read_csv(RAW / 'sell_prices.csv')
print(f'{"sell_prices.csv":32s} shape={sp.shape}  expect rows 6841121  '
      f'{"OK" if len(sp) == 6841121 else "MISMATCH"}')
s = pd.read_csv(RAW / 'sample_submission.csv', nrows=2)
print('sample_submission 列头 =', list(s.columns)[:4], '...', list(s.columns)[-2:])
PY
```

期望：4 行全 `OK`，列头为 `['id','F1','F2','F3'] ... ['F27','F28']`。

> 💡 `sales_train_evaluation.csv` 读全表约需 30-60 秒（在慢盘上更久），这是正常的。

#### 4.7.2 特征 pkl 自检（**只在迁移了 pkl 时执行**）

```bash
$PY - <<'PY'
import pandas as pd, glob, os, json
names = ['grid_part_1','grid_part_2','grid_part_3','lags_df_28','mean_encoding_df']
END_TRAIN = 1941
for d in sorted(glob.glob('data/processed/*/')):
    store = os.path.basename(d.rstrip('/'))
    meta = json.load(open(d + 'metadata.json'))
    t = {n: pd.read_pickle(d + n + '.pkl') for n in names}
    g1 = t['grid_part_1']
    same = all(v.index.equals(g1.index) for v in t.values())
    dup = int(g1.duplicated(['id','d']).sum())
    masked = bool(g1.loc[g1['d'] > END_TRAIN, 'sales'].isna().all())
    status = 'OK' if (same and dup == 0 and g1['d'].max() == 1969
                      and str(g1['id'].dtype) == 'category' and masked
                      and meta['stage'] == 'complete'
                      and meta['row_count'] == len(g1)) else 'FAIL'
    print(f"{store:6s} rows={len(g1):>8} aligned={same} dup={dup} "
          f"max_d={g1['d'].max()} id_dtype={g1['id'].dtype} masked={masked} "
          f"stage={meta['stage']} -> {status}")
PY
```

期望：**10 行，全部 `-> OK`**，`aligned=True`、`dup=0`、`max_d=1969`、`id_dtype=category`、`masked=True`、`stage=complete`。
参考行数（来自 `metadata.json`）：CA_1 4873639、CA_2 4446520、CA_3 4842685、CA_4 4737930、TX_1 4883327、TX_2 4893253、TX_3 4822539、WI_1 4646139、WI_2 4731952、WI_3 4857413。

> ⚠️ 这个脚本会读 10 店 × 5 表 = **17 GB**，在慢盘上要 **3-8 分钟**，内存峰值约 1.7 GB（逐店读，读完即释放）。
>
> 💡 **更省事的等价做法**：直接跑一次 `src/1_preprocessing_by_store.py`（见 5.1）。全部 `store_skip_complete` 只需几秒，却由脚本自己的严格校验代为验证了同样的东西。

#### 4.7.3 模型自检（**只在迁移了 models 时执行**）

```bash
$PY - <<'PY'
import pickle, glob, os, lightgbm as lgb
print('当前 lightgbm 版本 =', lgb.__version__, '（模型必须是同版本序列化的）')
for p in sorted(glob.glob('models/*.bin')):
    try:
        m = pickle.load(open(p, 'rb'))
        kind = '递归  ' if 'lgb_model' in p else '非递归'
        info = m.dump_model()
        print(f'{os.path.basename(p):32s} {kind} 加载 OK  '
              f'trees={len(info["tree_info"])}  num_features={m.num_feature()}  '
              f'size={os.path.getsize(p)/1024**2:.0f} MB')
    except Exception as e:
        print(f'{os.path.basename(p):32s} 加载失败 -> {type(e).__name__}: {e}')
PY
```

期望：两行 `加载 OK`，`trees=3000`、`num_features` 分别为递归 72 / 非递归约 55（两族特征集不同）。

**若报 `AttributeError` / `unknown object type`** → lightgbm 版本不匹配，**这两个模型作废**。处置：改用方案 C 的思路，删掉 `models/` 里的文件，在新机器上重训 CA_1（约 2.6 小时）。**不要试图"转换"模型**，没有可靠办法。

#### 4.7.4 冒烟测试（**强烈推荐，约 15 分钟**）

在启动 24 小时的训练前，先用**最小的店**跑通一遍完整链路，确认环境无问题：

```bash
# CA_2 行数最少（4,446,520），是最快的试跑对象
M5_STORES=CA_2 setsid nohup $PY src/2_train/1-1_recursive_store.py \
  > logs/smoke_recur_CA_2.log 2>&1 < /dev/null &
tail -f logs/smoke_recur_CA_2.log
```

**通过判据**：日志出现 `training_started`（含 features / rows / valid_rows / peak_rss_mb），随后 LightGBM 每 100 轮打印一次 RMSE，最终 `training_finished`，且 `models/lgb_model_CA_2_v1.bin` 与 `data/processed/test_CA_2.pkl` 都已生成。

**这一步的价值**：
1. 用约 1/9 的代价验证了环境、pkl 可读性、内存是否够、`num_threads` 是否合理。
2. 拿实测分钟数 × 9 就能**准确推算**步骤 2 的总工期（比 6.4 的经验公式可靠得多）。
3. 跑通后 CA_2 的模型**直接算数**，步骤 2 会自动 `store_skipped`，不浪费。

> ⚠️ 冒烟测试期间**不要同时跑别的**（内存红线，见 6.3）。

---

## 5. 详细待办清单（按依赖顺序）

### 5.0 总览

```text
依赖链（严格串行，前一步不过不要进下一步）：

  [步骤1] 预处理 10 店              ✅ 已完成     峰值 2.73 GB
      |                               (13:45:53，实测 14 分 49 秒；
      |                                仅当新机器【不带】pkl 时才需重跑，
      |                                冷启动全量约 20-25 分钟)
      |
      +---> [步骤2] 递归训练 9 店    ~11.4 小时    峰值 5.6 GB   (CA_1 已完成会被跳过)
      |         |
      +---> [步骤3] 非递归训练 9 店  ~12.5 小时    峰值 4.6 GB   (CA_1 已完成会被跳过)
                |
                v   (步骤2、步骤3 都完成后)
        [步骤4] 全量两路预测         ~45-75 分钟   峰值 4-6 GB
                |
                v
        [步骤5] 集成 + 硬校验        < 2 分钟      峰值 < 2 GB
                |
                v
        output/submission_store_only.csv   <-- A 轮交付完成
        [步骤6] 可选：本地 WRMSSE 验证（本轮未实现评分器）

总计（4 核机、串行、pkl 已迁移）：约 24.5 小时，其中训练 24 小时占绝对大头。
若 pkl 未迁移需重跑步骤1：约 25 小时（预处理只多 20-25 分钟，影响很小）。
步骤2 与步骤3 【严禁并行】，除非新机器内存 >= 16 GB（见 5.2 与 6.3）。
```

> 📌 **原任务书对步骤 1 的估计（40-60 分钟）偏保守**。实测数据显示预处理远比预期快，详见 5.1 的分段耗时表。因此"是否迁移 17 GB 的 pkl"对总工期影响很小（<2%），决策应以**打包体积**为主（见 3.2 方案 B / C）。

### 5.1 步骤 1 —— 预处理（全 10 店）✅ **已完成，新机器上通常无需执行**

> **状态**：已于 `2026-09-29 13:45:53` 全部完成，10 店 `stage=complete`，`preprocessing_finished` 已落盘。
> **只有当你选择了 3.2 的方案 C（不带 `data/processed/` 的 pkl）时，才需要在新机器上执行本步骤。**
> 若按方案 A / B 迁移了 pkl，本步骤会**全部 `store_skip_complete`，几秒钟内退出** —— 但仍建议跑一次当作完整性校验（见下方"建议"）。

**目标**：`data/processed/` 下 10 个门店目录全部 `stage=complete`，即每店 5 个 pkl 齐全。

```bash
cd <项目目录>
mkdir -p logs
setsid nohup /home/<用户名>/miniconda3/envs/m5/bin/python src/1_preprocessing_by_store.py \
  > logs/preproc_all_new.log 2>&1 < /dev/null &
echo "preproc_pid=$!" | tee logs/.running_pids
```

- **不设 `M5_STORES`** = 处理全部 10 店。已完成的店自动 `store_skip_complete`，已有 base 的店自动 `store_base_resume`，**不会重算**。
- 本脚本**没有 `M5_FORCE`**。要强制重做某店，只能 `rm -rf data/processed/<店名>` 后重跑。
- ⚠️ 第二遍编码统计会遍历**全部 10 店**（`iter_global_grids()` 走 `ALL_STORES`，不是 `M5_STORES`）。所以即使只想补 1 个店，也会读/算 10 店的 base —— 这是设计如此（跨店目标编码必须用全量统计），不要以为它跑飞了。

**本次运行的实测分段耗时**（`logs/preproc_all2.log`，13:31:04 → 13:45:53，**总计 14 分 49 秒 = 889 秒**）：

| 阶段 | 时间区间 | 耗时 | 说明 |
| --- | --- | --- | --- |
| 跳过已完成店 | 13:31:04 | 瞬时 | `store_skip_complete` CA_1（5 pkl 校验通过） |
| 全局输入加载 | 13:31:04 → 13:31:17 | **13 秒** | `global_inputs_loaded`，`sales_rows=30490`、`price_rows=6841121`，此时 RSS 已 2026 MB |
| base 断点续跑校验 | 13:31:17 → 13:31:52 | **35 秒** | CA_2 / CA_3 / CA_4 各约 11-13 秒 → `store_base_resume`（4 pkl 已存在且签名匹配，跳过重建） |
| base 新建 6 店 | 13:31:52 → 13:39:08 | **7 分 16 秒** | TX_1 64.0 s、TX_2 71.0 s、TX_3 70.8 s、WI_1 78.9 s、WI_2 74.5 s、WI_3 76.1 s（平均 **72.7 秒/店**），每店落盘 1204-1268 MB |
| 全局编码统计聚合 | 13:39:08 → 13:42:07 | **2 分 59 秒** | `global_encoding_stats_ready groups=11`；遍历 10 店 base、按 `d<=1913` 聚合 11 组 count/total/sumsq |
| 第二遍编码回填 9 店 | 13:42:07 → 13:45:53 | **3 分 46 秒** | CA_2 21 s … WI_3 22 s（平均 **25 秒/店**），每店总产出 1578-1736 MB |

**由此推算冷启动（10 店 base 全部从零）耗时**：

```text
加载 13 s + base 10 店 × 73 s (730 s) + 聚合 179 s + 回填 10 店 × 25 s (250 s)
= 1172 s ≈ 19.5 分钟
考虑 drvfs IO 波动，实际区间取 20-25 分钟
```

| 项 | 值 |
| --- | --- |
| 预计耗时 | **pkl 已迁移**：几秒钟（全 skip）或 1-2 分钟（跑一次校验）。**pkl 未迁移，冷启动全 10 店**：**20-25 分钟**（实测推算，见上）。**base 已完成只剩第二遍**（即本次的真实场景）：**约 15 分钟** |
| 内存峰值 | **2.73 GB**（实测 `peak_rss_mb=2796.9`，从 `store_base_written` 起一路恒定到 `preprocessing_finished`，说明峰值出现在 base 阶段且后续未再增长）。9.7 GB 机器余量充足 |
| CPU | 单线程为主（实测 CPU 占用约 69%）。**换到多核机器不会明显加速** —— 瓶颈在 pandas 单线程 groupby 与 drvfs IO，不在核数 |
| 磁盘产出 | 每店 5 pkl 共 **1578-1736 MB**（约 1.6-1.7 GB）；全 10 店 `data/processed/` 实测 **17 GB**。其中 `mean_encoding_df.pkl` 约 466 MB/店 |
| 完成判据 | ① 日志出现 `"event": "preprocessing_finished"`；② `ls data/processed/*/mean_encoding_df.pkl \| wc -l` == **10**；③ 每店 `metadata.json` 的 `stage` == `"complete"`；④ 4.7.2 自检脚本 10 店全部 OK（5 表索引对齐、`dup 0`、`max_d 1969`、`d>1941` 的 sales 全 NaN） |
| 失败处置 | 日志无 `preprocessing_finished` 就中断了 → **直接重跑同一条命令**（三级断点续跑，见 5.8）。若某店反复失败 → `rm -rf data/processed/<店名>` 后重跑。若报 `ValueError: xx 五表索引校验失败` 或 `xx 预测期 sales 未完全遮蔽` → 该店产物损坏，同样删目录重跑。**任何情况下都不要手工编辑 pkl** |

> ✅ **建议**：即使 pkl 已随包迁移，到新机器后仍**执行一次本命令**。全部 skip 只需几秒，却能一次性验证"pickle 能在新环境的 pandas 下正常反序列化 + 索引签名一致 + 真值遮罩完好"，这是最划算的迁移自检（等价于 4.7.2，但由脚本自己的严格校验代劳）。
> 若这一步报 `ValueError` 或 `read_pickle` 异常 → 说明 pkl 与新环境不兼容，按 3.4 处理（删目录重跑）。

**监控**：

```bash
tail -f logs/preproc_all_new.log                     # 逐事件跟踪
watch -n 10 'ls data/processed/*/mean_encoding_df.pkl 2>/dev/null | wc -l'   # 完成店数（目标 10）
grep -c store_complete logs/preproc_all_new.log      # 本次新完成的店数
```

### 5.2 步骤 2 —— 递归训练 9 店（CA_1 自动跳过）

**目标**：`models/lgb_model_{store}_v1.bin` × 10 + `data/processed/test_{store}.pkl` × 10。

```bash
cd <项目目录>
setsid nohup /home/<用户名>/miniconda3/envs/m5/bin/python src/2_train/1-1_recursive_store.py \
  > logs/train_recursive_all.log 2>&1 < /dev/null &
echo "recur_train_pid=$!" | tee -a logs/.running_pids
```

- **不设 `M5_STORES`** = 全 10 店；CA_1 因 model + test pkl 都在，会打 `store_skipped`，实际只训 9 店。
- **不设 `M5_FORCE`**（默认 `0`）= 启用跳过逻辑，这是断点续跑的关键。
- 若采用**方案 B**（带模型不带 pkl）：CA_1 会走 `training_skipped`（`reason="model_exists_test_rebuilt"`）**只补 test pkl 不重训**，正好保住已有的 76 分钟成果。

| 项 | 值 |
| --- | --- |
| 预计耗时 | 9 店 × 76.3 分钟 ≈ **11.4 小时**（±15%，各店行数差异在 ±10% 内：CA_2 4.45M 行最快，TX_2 4.89M 行最慢）。⚠️ `test_{store}.pkl` **由本脚本自己产出**（`save_test_frame()`，不是步骤 1 的产物），其开销已含在 76.3 分钟内 |
| 内存峰值 | **5.6 GB**（实测 `peak_rss_mb=5617.6`）。该值记录于 `lgb.train` 之前（数据加载 + `lgb.Dataset` 分箱完成时）；`max_bin=100`、72 特征下 boosting 阶段的直方图缓冲增量有限，可视为接近真实峰值，但**建议按 +10% ≈ 6.2 GB 预留** |
| CPU | `num_threads=4` 写死在脚本 `LGB_PARAMS` 里，会吃满 4 核。新机器核多时见 6.4 |
| 磁盘增量 | 9 × 180 MB ≈ 1.6 GB 模型 + 9 × 78 MB ≈ 700 MB test pkl |
| 完成判据 | ① `ls models/lgb_model_*_v1.bin \| wc -l` == **10**；② `ls data/processed/test_*.pkl \| wc -l` == **10**；③ 日志出现 10 条 `training_finished`（或 CA_1 的 `store_skipped`/`training_skipped`）；④ 每店 `valid_0's rmse` 在 **1.85-2.05** 区间（CA_1 基线 1.90318，各店应相近） |
| 失败处置 | 中断 → **直接重跑同一条命令**，已完成的店 `store_skipped`（见 5.8）。OOM（`Killed` / `MemoryError`）→ 见 6.3，确认没有并行跑别的、必要时关掉浏览器等占内存的程序。某店反复失败 → 删掉该店的 `.bin` 与 `test_*.pkl` 后重跑。RMSE 明显偏离基线（如 < 1.5 或 > 2.5）→ **怀疑特征错位或泄漏**，停下检查 4.7.2 是否全 OK，不要继续往下跑 |

> ⚠️ **RMSE 异常偏低比偏高更危险**。`valid RMSE` 突然变得很好（例如 1.2）通常意味着**验证集信息泄漏进了特征**（真值遮罩被破坏），而不是模型变强了。此时必须停下来查 `ENCODING_END` 与 `masked` 校验。

**监控**：

```bash
tail -f logs/train_recursive_all.log
grep -c training_finished logs/train_recursive_all.log        # 已完成店数（目标 9 + CA_1 跳过）
ls models/lgb_model_*_v1.bin | wc -l                          # 目标 10
watch -n 30 free -h                                           # 盯内存，别低于 1 GB available
grep "valid_0's rmse" logs/train_recursive_all.log | tail -5  # 最新 RMSE
```

### 5.3 步骤 3 —— 非递归训练 9 店（CA_1 自动跳过）

**目标**：`models/non_recur_model_{store}.bin` × 10。

```bash
cd <项目目录>
setsid nohup /home/<用户名>/miniconda3/envs/m5/bin/python src/2_train/2-1_nonrecursive_store.py \
  > logs/train_nonrecursive_all.log 2>&1 < /dev/null &
echo "nonrecur_train_pid=$!" | tee -a logs/.running_pids
```

| 项 | 值 |
| --- | --- |
| 预计耗时 | 9 店 × 83.0 分钟 ≈ **12.5 小时** |
| 内存峰值 | **4.6 GB**（比递归低，因为 `FIRST_DAY=710` 裁掉了早期行、且 `num_leaves=255` 远小于 2047） |
| CPU | 同样 `num_threads=4` |
| 磁盘增量 | 9 × 125 MB ≈ 1.1 GB |
| 完成判据 | ① `ls models/non_recur_model_*.bin \| wc -l` == **10**；② `ls models/*.bin \| wc -l` == **20**（A 轮模型全齐）；③ 日志 10 条 `training_finished`；④ 每店 `valid_0's rmse` 在 **1.78-1.95**（CA_1 基线 1.82056）。**非递归的 RMSE 应当系统性低于递归**（CA_1 上 1.82 vs 1.90），因为不累积误差 —— 如果反过来了，说明实现有问题 |
| 失败处置 | 同 5.2。**注意本脚本不产出 test pkl**，所以只需检查 `.bin` |

> 🔴 **步骤 2 与步骤 3 严禁并行**（在 9.7 GB 内存的机器上）：5.6 + 4.6 = **10.2 GB > 9.7 GB**，必然 OOM，而且 OOM killer 可能杀掉别的进程。详见 6.3 的并行判定表。
>
> **如果新机器内存 >= 16 GB**，可以并行以省下约 11 小时：
> ```bash
> # 两个都后台起，但要盯紧总内存（预期 5.6 + 4.6 ≈ 10.2 GB 峰值）
> setsid nohup $PY src/2_train/1-1_recursive_store.py     > logs/train_recursive_all.log     2>&1 < /dev/null &
> setsid nohup $PY src/2_train/2-1_nonrecursive_store.py  > logs/train_nonrecursive_all.log  2>&1 < /dev/null &
> watch -n 15 free -h
> ```
> ⚠️ 但并行会让**两个进程抢 CPU**（各自 `num_threads=4`，合计想要 8 线程）。若新机器只有 4-6 核，并行反而会**互相拖慢**，总时间不见得省。**内存 >= 16 GB 且 CPU >= 8 核**才是真正适合并行的条件。
> ⚠️ 并行时**两份日志必须分开重定向**（如上），否则输出交错无法排查。

### 5.4 步骤 4 —— 全量递归 + 非递归预测（两路，串行）

**目标**：`output/before_ensemble/` 下两份 CSV 各 **60981 行**（`wc -l`，含表头）= 60980 数据行。

🔴 **两条铁律**（违反会得到只有 3049 行的残缺文件）：

1. **绝对不能设 `M5_STORES`**。`format_submission()` 里 `full_run = (stores == ALL_STORES)`，一旦设了这个变量就走单店分支，只输出被选店的 3049 行。
2. **绝对不能分批跑**。每次运行都**整份重写** CSV，没有合并能力，后一批会把前一批冲掉。必须一次性跑完 10 店。

```bash
cd <项目目录>

# ① 递归预测（先跑这个，IO 更重）
env -u M5_STORES setsid nohup /home/<用户名>/miniconda3/envs/m5/bin/python \
  src/3_predict/1-1_recursive_store_PREDICT.py \
  > logs/predict_recursive_all.log 2>&1 < /dev/null &
echo "recur_predict_pid=$!" | tee -a logs/.running_pids

# 等 ① 完全结束（出现 prediction_finished）后再跑 ②
# ② 非递归预测
env -u M5_STORES setsid nohup /home/<用户名>/miniconda3/envs/m5/bin/python \
  src/3_predict/2-1_nonrecursive_store_PREDICT.py \
  > logs/predict_nonrecursive_all.log 2>&1 < /dev/null &
```

> 💡 `env -u M5_STORES` 显式清除该变量，防止你的 shell 里残留了之前 `export` 的值。**先 `echo "[$M5_STORES]"` 确认是 `[]` 更保险。**

| 项 | 值 |
| --- | --- |
| 预计耗时 | 递归 **25-45 分钟**（28 天 × 10 店 = 280 次模型加载，约 50 GB IO，慢盘上是瓶颈）；非递归 **15-30 分钟**（10 店 × 5 表 ≈ 16 GB 读取，无循环）。合计 **45-75 分钟** |
| 内存峰值 | **4-6 GB**。CA_1 单店实测递归约 4 GB、非递归约 3.5 GB；全量时逐店处理不累积，但建议按 6 GB 预留 |
| 完成判据 | ① `wc -l output/before_ensemble/*.csv` 两行都显示 **60981**；② 两份日志各有 `prediction_finished`；③ 下方质量检查脚本 **19 项全 PASS** |
| 失败处置 | 报 `FileNotFoundError: 缺少递归模型` → 步骤 2 没跑完。文件只有 3049/3050 行 → **你设了 `M5_STORES`**，清掉重跑。中断 → 直接重跑同一命令（预测**没有断点续跑**，会从头重算，但只需几十分钟，可接受）。出现大量 `FutureWarning` → **正常，不是错误**，见 6.1 |

**质量检查脚本**（跑完必须过，19 项）：

```bash
$PY - <<'PY'
import pandas as pd, numpy as np, pathlib
BE = pathlib.Path('output/before_ensemble')
F = [f'F{i}' for i in range(1, 29)]
sample = pd.read_csv('data/sample_submission.csv')
res = {}
frames = {}
for tag, name in [('递归', 'submission_kaggle_recursive_store.csv'),
                  ('非递归', 'submission_kaggle_nonrecursive_store.csv')]:
    p = BE / name
    res[f'{tag} 01 文件存在'] = p.exists()
    if not p.exists():
        continue
    df = pd.read_csv(p)
    frames[tag] = df
    res[f'{tag} 02 列结构正确'] = list(df.columns) == ['id'] + F
    res[f'{tag} 03 行数==60980'] = len(df) == 60980
    res[f'{tag} 04 id 无重复'] = not df['id'].duplicated().any()
    res[f'{tag} 05 id 与 sample 完全同序'] = df['id'].reset_index(drop=True).equals(
        sample['id'].reset_index(drop=True))
    v = df[F].to_numpy()
    res[f'{tag} 06 无 NaN'] = not np.isnan(v).any()
    res[f'{tag} 07 无 inf'] = np.isfinite(v).all()
    res[f'{tag} 08 无负数'] = not (v < 0).any()
    ev = v[30490:]                       # evaluation 段
    va = v[:30490]                       # validation 段
    res[f'{tag} 09 validation 段全为 0'] = not (va != 0).any()
    res[f'{tag} 10 evaluation 段非全 0'] = (ev != 0).any()
    res[f'{tag} 11 F1..F28 均值逐日不同（递归回填生效）'] = len(
        set(np.round(ev.mean(axis=0), 6))) > 20
    print(f'{tag}: rows={len(df)} eval_mean={ev.mean():.4f} '
          f'eval_sum={ev.sum():.0f} F1mean={ev[:,0].mean():.4f} F28mean={ev[:,27].mean():.4f}')
if len(frames) == 2:
    a, b = frames['递归'][F].to_numpy()[30490:], frames['非递归'][F].to_numpy()[30490:]
    r = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    res['两族 12 Pearson r 在 0.9-0.995'] = 0.9 < r < 0.995
    print(f'两族 Pearson r = {r:.4f}   (CA_1 基线 0.9697)')
    print(f'两族均值比 = {a.mean()/b.mean():.4f}')
npass = sum(bool(x) for x in res.values())
print('\n--- 逐项结果 ---')
for k, v in res.items():
    print(('PASS  ' if v else 'FAIL  ') + k)
print(f'\n合计 {npass}/{len(res)} PASS')
PY
```

**期望**：`合计 19/19 PASS`（12 项 × 2 族 = 22？不 —— 每族 11 项 + 两族 1 项 = **23 项**；CA_1 单店基线是 19 项，因为当时行数判据是 3049 而非 60980，项数略有不同）。**只要没有 FAIL 就算过**，项数本身不必纠结。
`两族 Pearson r` 应在 **0.96-0.98**（CA_1 单店实测 0.9697）。

### 5.5 步骤 5 —— 集成生成 `submission_store_only.csv` 并硬校验

**目标**：`output/submission_store_only.csv`，**60980 行**、id 与 `sample_submission` 完全同序、无 NaN/inf/负数。

```bash
cd <项目目录>
# ⚠️ 不需要设 M5_ROUND（源码没读它，见 6.6）。设了也无害，但别以为它有作用。
/home/<用户名>/miniconda3/envs/m5/bin/python src/3_predict/3-1_final_ensemble.py \
  2>&1 | tee -a logs/store_ensemble_run.log
```

这一步**很快（< 2 分钟）**，且是**幂等**的（重跑覆盖同一文件），所以**不需要 nohup 后台跑**，直接前台执行看输出即可。

脚本内置 7 项硬校验（任一失败即 `raise`，见 2.2 脚本 (6)），所以**只要它正常退出没抛异常，产物就是合规的**。日志会出现 `input_validated` 与 `ensemble_finished`。

**独立复核脚本**（不要只信脚本自己的校验，再独立查一遍，10 项）：

```bash
$PY - <<'PY'
import pandas as pd, numpy as np, pathlib
p = pathlib.Path('output/submission_store_only.csv')
F = [f'F{i}' for i in range(1, 29)]
checks = []
checks.append(('01 文件存在', p.exists()))
if p.exists():
    df = pd.read_csv(p)
    sample = pd.read_csv('data/sample_submission.csv')
    checks.append(('02 行数 == 60980', len(df) == 60980))
    checks.append(('03 列 == [id, F1..F28]', list(df.columns) == ['id'] + F))
    checks.append(('04 id 无重复', not df['id'].duplicated().any()))
    checks.append(('05 id 与 sample_submission 完全同序',
                   df['id'].reset_index(drop=True).equals(sample['id'].reset_index(drop=True))))
    v = df[F].to_numpy()
    checks.append(('06 无 NaN', not np.isnan(v).any()))
    checks.append(('07 无 inf', bool(np.isfinite(v).all())))
    checks.append(('08 无负数', not (v < 0).any()))
    checks.append(('09 validation 段(前 30490 行)全为 0', not (v[:30490] != 0).any()))
    checks.append(('10 evaluation 段(后 30490 行)有非零预测', (v[30490:] != 0).any()))
    # 额外信息
    ev = v[30490:]
    print(f'文件大小 = {p.stat().st_size/1024**2:.2f} MB')
    print(f'wc -l 应为 60981（含表头），实际数据行 = {len(df)}')
    print(f'evaluation 段：mean={ev.mean():.4f}  sum={ev.sum():.0f}  '
          f'min={ev.min():.4f}  max={ev.max():.4f}')
    print(f'F1 mean={ev[:,0].mean():.4f}   F28 mean={ev[:,27].mean():.4f}  '
          f'(应不同，差异反映 28 天内的季节性)')
    print(f'dtype = {df[F[0]].dtype}')
    # 与两路输入的一致性：集成 = 等权平均
    be = pathlib.Path('output/before_ensemble')
    a = pd.read_csv(be / 'submission_kaggle_recursive_store.csv')[F].to_numpy()
    b = pd.read_csv(be / 'submission_kaggle_nonrecursive_store.csv')[F].to_numpy()
    same = np.allclose(v, (a + b) / 2, rtol=1e-5, atol=1e-5)
    print(f'等权平均一致性 = {same}')
print('\n--- 逐项结果 ---')
for name, ok in checks:
    print(('PASS  ' if ok else 'FAIL  ') + name)
print(f'\n合计 {sum(bool(o) for _, o in checks)}/{len(checks)} PASS')
PY
```

**期望**：`合计 10/10 PASS`，且 `等权平均一致性 = True`。

| 项 | 值 |
| --- | --- |
| 预计耗时 | **< 2 分钟**（读两份 60980×28 的 CSV、求均值、写盘） |
| 内存峰值 | **< 2 GB** |
| 完成判据 | ① `output/submission_store_only.csv` 存在；② `wc -l` == **60981**；③ 脚本自身 7 项校验无异常退出；④ 上方独立复核 **10/10 PASS**；⑤ `logs/store_ensemble.log` 出现 `ensemble_finished`（不再是 0 字节） |
| 失败处置 | `ValueError: 行数不是 60980` → 步骤 4 设了 `M5_STORES`，回去重跑步骤 4。`id 顺序不一致` → 有人手工编辑过 CSV，重跑步骤 4。`NaN/inf` → 上游模型或特征有问题，检查步骤 4 的质量脚本。`FileNotFoundError: 缺少递归/非递归模型` → 步骤 2/3 没跑完 |

> ✅ **A 轮到此交付完成**：`output/submission_store_only.csv` 本身就是一份可用的、Kaggle 格式合规的销量预测（见 6.7 关于 validation 段为 0 的说明）。即使 B 轮失败或不做，这个文件也有价值。

### 5.6 步骤 6（可选）—— 本地 WRMSSE 验证说明

**结论：本轮无法在本地计算 WRMSSE，因为没有实现评分器，且缺少必要条件。**

**三条原因**：

1. **本轮没有实现 WRMSSE 评分脚本**。`src/` 下 6 个脚本都不含评分逻辑，`docs/M5_复现执行手册.md` 里也只有指标公式（在技术文档里），没有可执行的实现。
2. **validation 段没有预测值**。整条流水线的预测起点写死为 `END_TRAIN = 1941`，只预测 `d_1942..d_1969`（evaluation 段），而 **evaluation 段的真值不在任何本地文件里**（`sales_train_evaluation.csv` 只到 d_1969 但预测段真值需要赛后数据）。见 6.7。
3. **WRMSSE 需要主办方的层级权重**。公式是 12 层聚合的加权和：

```text
WRMSSE = sum_{l=1..12}  w_l  *  sum_{i in level_l}  WRMSSE_i

WRMSSE_i = sqrt(  sum_{t=1..28} (y_i,t - yhat_i,t)^2  /  sum_{t=1..28} |sales$_i,t|  )

其中 w_l 是主办方给定的【固定】层级权重（不是销量权重），
分母是 dollar 销量（来自 sales$_i,t = sales_i,t * sell_price_i,t），
所以必须同时有真值销量与对应价格。
```

**如果确实要做本地验证，可选路径**（本轮不执行，仅记录）：

| 路径 | 做法 | 代价 | 可信度 |
| --- | --- | --- | --- |
| A. 用 validation 段 | 把 `END_TRAIN` 改成 1913，重训重预测，对 `d_1914..d_1941` 用 `sales_train_evaluation.csv` 的真值算 WRMSSE | 需改源码 + 重跑 24 小时 | 高，但**代价极大** |
| B. 只算 valid RMSE | 已有的日志里就有（递归 1.90318 / 非递归 1.82056），把它当代理指标 | 0 | 中（RMSE 与 WRMSSE 排序大体一致，但不等价） |
| C. 提交 Kaggle | 直接上传 `submission_store_only.csv`，看公榜分数 | 需账号 + 每日提交次数限制 | **最高**（公榜就是 evaluation 段的真实 WRMSSE） |
| D. 写独立评分器 | 新建 `src/4_score/wrmsse.py`，实现 12 层聚合 + 主办方权重表 | 需实现并核对权重表 | 高 |

**本轮建议走路径 B + C**：日常用 valid RMSE 监控是否退化（各店应在 1.78-2.05），最终用 Kaggle 公榜确认真实水平。**不要在 A 轮里插入门槛极高的路径 A**，那会让关键路径从 24 小时变成 48 小时。

> ⚠️ 注意区分两个"权重"：**指标权重** `w_l`（主办方给定的 12 层固定权重，用于算 WRMSSE）与**融合权重**（本方案里两族等权平均，`stacked.mean(axis=0)`，无调优）。前者是评测规则，后者是建模选择，不要混淆。

### 5.7 后台运行与日志监控命令模板

**启动模板**（所有长任务都用这个形式）：

```bash
cd <项目目录>
PY=/home/<用户名>/miniconda3/envs/m5/bin/python
LOG=logs/<自定义名字>.log

setsid nohup $PY <脚本路径> > $LOG 2>&1 < /dev/null &
echo "<标识>_pid=$!" | tee -a logs/.running_pids
```

**四个要素缺一不可**：

| 要素 | 作用 | 少了会怎样 |
| --- | --- | --- |
| `setsid` | 新建会话，脱离控制终端 | 关掉终端/SSH 断开时进程被 `SIGHUP` 杀掉 |
| `nohup` | 忽略 `SIGHUP` | 同上（双保险） |
| `> $LOG 2>&1` | stdout 与 stderr 都进日志 | 输出丢失，或 stderr 打到已关闭的终端导致 `BrokenPipeError` |
| `< /dev/null` | 关闭 stdin | 后台进程读 stdin 会被 `SIGTTIN` **停住**（状态变 `T`），看起来像卡死 |

**监控命令集**：

```bash
tail -f logs/<名字>.log                              # 实时跟踪（Ctrl-C 只退出 tail，不影响任务）
tail -n 50 logs/<名字>.log                           # 看最后 50 行
grep -c training_finished logs/<名字>.log            # 数完成事件
ps -o pid,etime,%cpu,%mem,rss,cmd -p <PID>           # 该进程的已运行时间/CPU/内存
ps aux | grep '[p]ython.*src/'                       # 所有 M5 进程（用 [p] 避免假阳性）
watch -n 15 free -h                                  # 盯内存
watch -n 30 'ls models/*.bin | wc -l'                # 盯产出计数
dmesg | tail -30 | grep -i 'killed process'          # 查是否被 OOM killer 杀过
```

**测资源占用**（想知道某步的真实峰值内存时，用这个包装）：

```bash
/usr/bin/time -v $PY <脚本路径> 2>&1 | tee logs/<名字>_timev.log
# 关注 "Maximum resident set size (kbytes)" 与 "Elapsed (wall clock) time"
# 参考：CA_1 单店预处理 = 2,829,832 KB / 4:14.97
```

### 5.8 断点续跑说明（**最重要的一节**）

**核心结论：任何一步中断了，直接重跑同一条命令即可，已完成的项会被自动跳过。**

**预处理脚本的三级校验**：

```text
① validate_store_files(store, "complete")  通过
      -> log "store_skip_complete"，整店跳过（5 pkl 都在且校验全过）
② validate_store_files(store, "base", category_hash)  通过
      -> log "store_base_resume"，跳过第一遍（4 个 base pkl 已在且签名匹配）
③ 都不通过
      -> 从 build_base_grid 开始重做该店
```

**训练脚本的两级跳过**（递归为例）：

```text
① model.bin 存在 且 test_{store}.pkl 存在 且 not force
      -> log "store_skipped" reason="model_and_test_exist"，整店跳过
② model.bin 存在 但 test pkl 不在 且 not force
      -> 先 save_test_frame 补 test pkl，再 log "training_skipped"
         reason="model_exists_test_rebuilt"，不重训（方案 B 下 CA_1 走这条）
③ 都不满足 或 M5_FORCE=1
      -> 完整训练
```

**预测与集成脚本没有断点续跑** —— 它们每次从头重算并整份覆盖输出。但耗时只有几十分钟 / 2 分钟，重跑代价可接受。

**原子写入保证中断安全**（见 2.2 机制四）：任何时刻被 `kill -9`，最多留下一个 `.{name}.tmp-{pid}` 垃圾文件，**绝不会**产生"看起来正常但内容残缺"的 pkl/bin。清理命令：

```bash
find data/processed models output -name '.*.tmp-*' -print -delete
```

**签名机制保证数据变更自动失效**：如果换了原始 CSV，`category_signature` 会变，所有 base 产物**自动判定为过期并重算** —— 不需要手工清理。

**实战验证记录**（本项目已真实经历过一次中断续跑，可作为可信度背书）：

```text
13:13  启动全量预处理（logs/preproc_all.log）
       -> 代理被中途取消，进程被终止
       -> 日志停在 CA_4 的 store_base_elapsed，【没有】preprocessing_finished
       -> 此时状态：CA_1 complete，CA_2/3/4 只有 base（4 pkl），TX/WI 六店什么都没有

13:31  直接重跑【同一条命令】（logs/preproc_all2.log）
       -> CA_1  命中 ① store_skip_complete   （0 秒，完全跳过）
       -> CA_2/3/4 命中 ② store_base_resume  （各约 12 秒，只校验不重算，省下 3 × 73 秒）
       -> TX_1~WI_3 命中 ③ 从头做 base       （各 64-79 秒）
       -> 聚合 + 第二遍 9 店编码
13:45:53  preprocessing_finished             全部 10 店 complete
```

**这次续跑的实际收益**：省掉了 CA_1 的完整重做（约 4 分 15 秒）+ CA_2/3/4 的 base 重建（约 3 分 39 秒）≈ **8 分钟**，且**没有任何产物损坏**。这就是三级校验 + 原子写入 + 签名机制的实际价值。

> ⚠️ **唯一需要手工干预的情况**：如果某店的产物**真的损坏了**（例如磁盘满导致写入失败、或手工编辑过 pkl），脚本的校验会判定它"未完成"并重做 —— 但重做前**不会删除**旧文件，`atomic_pickle` 会用 `os.replace` 直接覆盖，所以通常没问题。若反复失败（例如报 `五表索引校验失败`），才需要手工 `rm -rf data/processed/<店名>` 后重跑。

### 5.9 失败处置速查表

| 症状 | 原因 | 处置 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'pandas'` | 用了系统 Python，不是 m5 环境 | 改用完整路径 `/home/<用户名>/miniconda3/envs/m5/bin/python`（见 4.5） |
| `UnpicklingError` / `Can't get attribute 'BlockManager'` | pandas 版本不匹配 | 对齐到 2.3.3；实在不行删 `data/processed/` 重跑（只要 20-25 分钟） |
| `AttributeError` 加载 `.bin` 失败 | lightgbm 版本不匹配 | 对齐到 4.7.0；不行则删模型重训（见 4.7.3） |
| `FileNotFoundError: data/processed/<店>/grid_part_1.pkl` | 步骤 1 没跑（方案 B/C 常见） | 先跑步骤 1 |
| `FileNotFoundError: 缺少递归模型 / 非递归模型` | 对应训练未完成 | 跑步骤 2 / 3 |
| `FileNotFoundError: test_<店>.pkl` | 递归训练未跑完该店 | 跑步骤 2（脚本会自动产出） |
| 进程被 `Killed`，`dmesg` 有 `Out of memory` | OOM | 见 6.3：确认没并行跑两路训练；关掉其他占内存程序 |
| 进程状态是 `T`（stopped） | 忘了 `< /dev/null`，后台读 stdin 被停 | `kill` 后用 5.7 的完整模板重启 |
| `ValueError: M5_STORES 包含未知门店` | 店名拼错 | 只能是 `CA_1..CA_4 / TX_1..TX_3 / WI_1..WI_3`，逗号分隔无空格 |
| `ValueError: xx 五表索引校验失败` | 该店产物损坏 | `rm -rf data/processed/<店名>` 后重跑步骤 1 |
| `ValueError: xx 预测期 sales 未完全遮蔽` | 真值遮罩被破坏 | 同上删目录重跑；**并检查是否有人改过 `END_TRAIN`** |
| `ValueError: 行数不是 60980`（集成时） | 步骤 4 设了 `M5_STORES` | `env -u M5_STORES` 重跑步骤 4 |
| 预测 CSV 只有 3050 行 | 同上，走了单店分支 | 同上 |
| 全量预测跑了很久没输出 | 280 次模型加载 + 慢盘 IO | 正常，见 6.5；`ps -o etime` 看已运行时间，45 分钟内应完成 |
| valid RMSE 异常偏低（如 < 1.5） | **信息泄漏**，比报错更危险 | 停下，检查 4.7.2 的 `masked` 与 `ENCODING_END` |
| 大量 `FutureWarning: ... incompatible dtype` | pandas 2.x 已知告警 | **非阻断，可忽略**，见 6.1 |
| 磁盘写满（`No space left on device`） | A 轮跑完约需 23 GB | `df -h .` 检查；清理 `.*.tmp-*` 与旧日志 |
| `du -sh .` 显示 8.0K | drvfs 挂载盘怪癖 | 不是真的，分子目录 `du` 后相加（见 3.1） |

---

## 6. 已知问题与注意事项

### 6.1 pandas FutureWarning：向 int64 / float32 列赋 float 值

**现象**：递归预测脚本运行时刷屏大量告警（`logs/predict_recursive_CA_1.log` 18 KB 里绝大部分是它）：

```text
FutureWarning: Setting an item of incompatible dtype is deprecated and will raise
in a future error of pandas. Value '[...]' has dtype incompatible with int64,
please explicitly cast to a compatible dtype first.
```

**根因**（两处，都在 `src/3_predict/1-1_recursive_store_PREDICT.py`）：

```python
# 第 111 行附近：把 float 预测值写回 int64 列
output.loc[evaluation_mask, F_COLUMNS] = mapped.reindex(
    output.loc[evaluation_mask, "id"])[F_COLUMNS].to_numpy()

# 第 144 行附近：递归回填，把预测出的 float 销量写回 float32 列
base_test.loc[mask, TARGET] = values
```

**性质**：

| 维度 | 结论 |
| --- | --- |
| 当前是否阻断 | ❌ **不阻断**。只是告警，逻辑完全正确，CA_1 已用 19/19 PASS 的结果证明 |
| 结果是否受影响 | ❌ 不受影响。pandas 2.x 会自动做隐式转换 |
| pandas 3.x 下会怎样 | ⚠️ **直接抛错**，流水线跑不完 |
| 本轮是否修 | ❌ **不修**（任务边界：不改源码） |

**处置建议**：

1. **新机器必须保持 pandas 2.3.x**（见 3.4 / 4.1）。这是最简单彻底的办法。
2. 如果嫌日志刷屏影响排查，可以在启动命令里加过滤（**不改源码**）：
   ```bash
   setsid nohup $PY -W ignore::FutureWarning src/3_predict/1-1_recursive_store_PREDICT.py \
     > logs/predict_recursive_all.log 2>&1 < /dev/null &
   ```
   ⚠️ 但这会把**所有** FutureWarning 都吞掉，可能掩盖真正的问题。**建议只在确认链路已跑通后使用**。
3. 若将来必须升级到 pandas 3.x，正确修法是显式转换 dtype：
   ```python
   base_test.loc[mask, TARGET] = np.asarray(values, dtype=base_test[TARGET].dtype)
   ```
   **记录在此，不作为本轮待办。**

### 6.2 模型 pickle 跨 lightgbm 版本不可移植

`models/*.bin` 是 `pickle.dump(lgb.Booster)` 的产物，**不是** `booster.save_model()` 的文本格式。

| | `pickle.dump(Booster)`（本项目用的） | `booster.save_model()`（推荐但本轮不用） |
| --- | --- | --- |
| 格式 | 二进制，Python pickle 协议 | 纯文本 JSON/txt |
| 跨 lightgbm 版本 | ❌ **不可移植**，版本必须精确一致 | ✅ 向后兼容性好 |
| 跨 Python 版本 | ❌ 敏感 | ✅ 无关 |
| 保留完整对象状态 | ✅（含 sklearn API 包装等） | ⚠️ 只保留 Booster 本体 |
| 文件大小 | 180 MB / 125 MB | 通常相近或略小 |

**实际影响**：

- 新机器上 `lightgbm` 必须是 **4.7.0**。差一个补丁号（如 4.7.1）**通常**也能加载，但差一个次版本（4.6.x / 4.8.x）**很可能失败**。
- 失败表现：`AttributeError: 'Booster' object has no attribute 'xxx'`、`pickle.UnpicklingError`、或 `LightGBMError: unknown object type`。
- **没有可靠的转换办法**。失败了就只能删模型重训（CA_1 两路共约 2.6 小时）。

**决策建议**：

- 能对齐版本 → 走**方案 B**，带走 305 MB 模型，省下 2.6 小时。
- 不能确定能否对齐（例如新机器 conda 频道里只有 4.8.x）→ 走**方案 C**，不带模型，反正要重训。
- **验证方法**：装好环境后立刻跑 4.7.3 的模型自检脚本，能打印出 `trees=3000` 就说明兼容。

### 6.3 内存红线（9.7 GB 机器）

**当前机器可用内存 9.7 GB**。各阶段实测峰值：

| 阶段 | 峰值 RSS | 占可用内存 |
| --- | --- | --- |
| 预处理（CA_1 冷启动，`time -v`） | **2830 MB ≈ 2.70 GB** | 28% |
| 预处理（全 10 店，日志 `peak_rss_mb`） | **2796.9 MB ≈ 2.73 GB** | 28% |
| 递归训练 CA_1 | **5617.6 MB ≈ 5.6 GB** | **58%** |
| 非递归训练 CA_1 | **约 4.6 GB** | 47% |
| 递归预测（单店） | 约 4 GB | 41% |
| 非递归预测（单店） | 约 3.5 GB | 36% |
| 集成 | < 2 GB | 20% |

🔴 **红线：两路训练严禁并行。**

```text
5.6 GB（递归）+ 4.6 GB（非递归）= 10.2 GB  >  9.7 GB 可用
-> 必然触发 OOM killer
-> 而且 OOM killer 未必杀掉你的训练进程，可能杀掉别的（甚至桌面环境）
```

**并行判定表**（换新机器后按这个查）：

| 新机器可用内存 | 能否并行两路训练 | 说明 |
| --- | --- | --- |
| < 8 GB | ❌ **不能**，且单路递归（5.6 GB）也很危险 | 必须加内存或加 swap |
| 8 - 11 GB（当前情况） | ❌ **不能** | 严格串行，一次只跑一个进程 |
| 12 - 15 GB | ⚠️ **勉强可以**，但要 `watch free -h` 全程盯 | 10.2 GB 峰值 + 系统占用，余量不足 2 GB |
| **>= 16 GB** | ✅ **可以并行** | 预期总峰值约 10.2 GB，余量充足。**但仍需 CPU >= 8 核**，否则两进程抢 `num_threads=4` 会互相拖慢（见 5.3） |
| >= 32 GB | ✅ 可以，甚至可以考虑 2 店同族并行 | 需自行改脚本或起多个 `M5_STORES` 进程，**本轮不做** |

**加 swap 的应急方案**（内存不够但必须跑时）：

```bash
# 需要 root；8 GB swap 文件
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
free -h                    # 确认 Swap 行有 8.0Gi
```

⚠️ **swap 会让训练慢 3-10 倍**（LightGBM 的直方图构建是随机访问模式，对换页极不友好）。只在"宁慢勿断"的场景用，别指望它提速。

**通用纪律**：

1. **一次只跑一个 M5 进程**（`docs/M5_复现执行手册.md` 第 266 行的总纪律，仍然有效）。
2. 跑长任务前关掉浏览器、IDE 索引、Docker 等吃内存的东西。
3. `watch -n 15 free -h` 挂着，`available` 低于 **1 GB** 就要警惕。
4. 定期 `dmesg | tail -30 | grep -i 'killed process'` 查有没有被 OOM 杀过 —— **被杀时日志里不会有 traceback**，进程就凭空消失了，很容易误判为"卡住"。

### 6.4 4 核 CPU 与 `num_threads=4`

**现状**：`num_threads: 4` **硬编码**在两个训练脚本的 `LGB_PARAMS` 字典里（不是从环境变量读的）。当前机器正好 4 核，所以是满配。

**新机器核数不同时怎么办**：

| 新机器核数 | 建议 | 理由 |
| --- | --- | --- |
| 1-3 核 | **改小** `num_threads` 到实际核数 | 线程数 > 核数会带来调度开销，反而更慢 |
| 4 核 | 不改 | 与基线完全一致，耗时可直接对照 |
| 8-16 核 | **可以改大**，但收益递减 | LightGBM 直方图并行在 8 核后加速比明显下降；且大数据集下内存带宽会成新瓶颈 |
| > 16 核 | 建议设 `num_threads=8~16`，**不要设成核数** | 线程过多时同步开销可能让速度不升反降 |

⚠️ **改 `num_threads` 需要修改源码**（两个文件各一处）。本轮任务边界是"不改源码"，所以**换机后如果核数不同，你有两个选择**：

- **选择 1（推荐，不改源码）**：直接跑。`num_threads=4` 在 8 核机上只占 4 核，**结果完全一致、耗时也完全一致**（76-83 分钟/店），只是浪费了另外 4 核。总工期仍是 24 小时。
- **选择 2（改源码提速）**：把两个训练脚本的 `"num_threads": 4` 改成你的核数（或 `os.cpu_count()`）。**注意：改这个不影响模型结果**（LightGBM 的确定性由 `seed` + `feature_fraction_seed` 等保证，与线程数无关，**除非**同时开了 `force_row_wise`/`force_col_wise` 的自动选择 —— 本项目没有显式设置，理论上极端情况下不同线程数会导致直方图切分方式不同从而有微小数值差异）。**建议先用 CA_2 单店试跑，对比 valid RMSE 是否与预期一致，再全量跑。**

**耗时外推公式**（仅供粗略估算）：

```text
单店训练时间 ≈ 基线时间 × (4 / 有效线程数)^0.85 × (该店训练行数 / CA_1 训练行数)

递归   基线 76.3 分钟，CA_1 训练行数 4,788,267
非递归 基线 83.0 分钟，CA_1 训练行数 3,476,092

各店总行数（来自 metadata.json 的 row_count，可粗略代替训练行数做相对估算）：
  CA_1 4,873,639   CA_2 4,446,520   CA_3 4,842,685   CA_4 4,737,930   TX_1 4,883,327
  TX_2 4,893,253   TX_3 4,822,539   WI_1 4,646,139   WI_2 4,731,952   WI_3 4,857,413
  最大/最小比 = 4,893,253 / 4,446,520 = 1.10  -> 各店耗时差异在 ±10% 内
所以 9 店递归 ≈ 9 × 76.3 × 1.0 ≈ 11.4 小时；9 店非递归 ≈ 9 × 83.0 ≈ 12.5 小时
```

> ⚠️ **关于 `^0.85` 这个指数**：它是**经验值，不是本项目的实测拟合**。
> 本项目只有 4 核机器上的两个数据点（76.3 / 83.0 分钟），**没有任何多核实测数据**可供回归。
> 0.85 只是 LightGBM 直方图算法并行加速比的常见量级（考虑 Amdahl 定律下的串行部分：数据加载、`Dataset` 分箱、特征拼接都是单线程）。
> **实际加速比取决于你的核数、内存带宽、以及是否受 IO 拖累**，可能明显低于 0.85 次方给出的乐观值。
> 正确用法：**先在新机器上单店试跑一次**（`M5_STORES=CA_2`，行数最少最快，见 4.7.4），拿实测分钟数 × 9 去推算总工期，**不要直接信这个公式**。

### 6.5 E 盘 / WSL 挂载 IO 较慢的影响

**现状**：项目在 `/mnt/e/5054project/`，WSL 通过 drvfs/9P 访问 Windows 磁盘，**元数据操作和小文件 IO 比原生 ext4 慢一个数量级**。

**对本项目各环节的具体影响**：

| 环节 | IO 量 | 影响 |
| --- | --- | --- |
| 预处理写 base | 每店 1204-1268 MB × 10 | 单店实测 **64-79 秒**（平均 72.7 秒），其中相当一部分是写盘 |
| 预处理第二遍 `mark_complete` | 每店重读 5 表 1578-1736 MB 做校验 | 实测每店 **21-35 秒**（平均 25 秒） |
| 训练启动加载 | 每店读 5 表 1.6-1.7 GB | 计入 76-83 分钟，估占 **3-6 分钟** |
| 模型保存 | 每店写 125-180 MB | 秒级 |
| **递归预测** | **28 天 × 10 店 = 280 次模型加载，每次 180 MB ≈ 50 GB 读取** | **最痛的一点**，估占全量递归预测耗时的 **30-50%** |
| 非递归预测 | 10 店 × 5 表 ≈ 16 GB 读取 | 占大头 |
| `du -sh` / `find` / `ls` | 元数据扫描 | 在 `/mnt/e` 上明显卡顿，**属正常**，不是死锁 |
| `tar` 打包 / 解包 | 大量小文件 | 慢；有条件就在原生分区解包 |

**建议**：

1. **首选**：新机器上把项目放在**原生 Linux 分区**（`/home/<用户名>/...`），预计整体提速 **15-40%**。
2. 若必须留在挂载盘：接受耗时上浮，**别去"优化"它**；把第 5 章预期时间乘 **1.2-1.5**。
3. **递归预测的 280 次模型重复加载是脚本结构决定的**（模型在 `for horizon` 循环内层加载），本轮不改源码。将来若要优化，可以把 10 个模型在循环外一次性加载进字典缓存 —— 但那会让内存峰值增加约 **1.8 GB**（10 × 180 MB），在 9.7 GB 机器上需重新评估 6.3 的红线。**记录在此，不作为本轮待办。**
4. 打包/解包 `tar` 时在挂载盘上会明显慢，有条件就在原生分区操作。

### 6.6 提交行数口径差异：60980 vs 30490（**文档与代码不一致，以代码为准**）

这是本次核实发现的**唯一一处文档与实现冲突**，交接时必须说清楚：

| 出处 | 说法 |
| --- | --- |
| `docs/M5_复现执行手册.md` 第 21 行 | "成功标准 = 合规的 **30490 × 28** 提交文件" |
| 同上 第 34 行 | "每份文件都必须满足：**30490 行**、id 集合等于 `sample_submission.iloc[30490:]`" |
| 同上 第 337 行 | "`output/before_ensemble/` 下 6 个 csv，每个 **30490 行数据**（`wc -l` 显示 30491）" |
| 同上 第 398 行（故障表） | "提交 **60980 行**或 id 对不上 → 取错 submission 段 → 集成时取 `iloc[30490:]`" |
| **实际代码** `format_submission()` | `full_run` 时 `output = sample.copy()` → **60980 行**，并断言 `len(output) == 60980` |
| **实际代码** `3-1_final_ensemble.py` | 硬校验 **`len(frame) != 60980` 就抛异常**，且要求 id 序列与完整 `sample_submission` 同序 |
| 本任务交接要求（步骤 5） | "**60980 行**、ID 与 sample_submission 同序、无 NaN/inf/负数" |

**结论：以代码为准，A 轮的合规产物是 60980 行。** 手册里"30490 行 / `iloc[30490:]`"的表述是**过时口径**（描述的是官方 A1 原始脚本的做法），与本轮实现不符。

**为什么 60980 行也是对的**：M5 官方 `sample_submission.csv` 本身就是 **60980 行**（validation 段 30490 + evaluation 段 30490），提交格式要求与之完全一致。官方 A1 的做法是复制 `sample_submission` 后只填 `iloc[30490:]`（evaluation 段），**前 30490 行保持样例里的 0** —— 这与本项目 `format_submission` 的行为**完全等价**（先全填 0，再覆盖 evaluation 段的预测值）。所以两者产物一致，只是描述方式不同。

**⚠️ 本轮不改手册**（任务边界：只新建本文档）。在新机器上执行时，请一律以**本文档和代码**为准；看到手册写"30490 行"时，理解为"30490 行**有效预测** + 30490 行占位 0"。

**另一处无效环境变量**：手册第 317/430/433 行写的 `M5_ROUND=A` / `M5_ROUND=B`，`3-1_final_ensemble.py` **源码里根本没有读这个变量**（`grep -n environ src/3_predict/3-1_final_ensemble.py` 无结果）。设或不设完全一样，输出永远是 `output/submission_store_only.csv`。B 轮要产出 `submission_final.csv` 时，必须**修改集成脚本**的 `INPUT_FILES` 和 `OUTPUT_FILE`，`M5_ROUND` 帮不上忙。

### 6.7 validation 段（前 30490 行）全为 0 —— 设计如此，不是 bug

```text
output/submission_store_only.csv 的 60980 行结构：
  行 0     .. 30489   id 以 _validation 结尾   -> F1..F28 全为 0（本轮不预测这一段）
  行 30490 .. 60979   id 以 _evaluation 结尾   -> 真实预测值
```

**原因**：整条流水线的预测起点写死为 `END_TRAIN = 1941`，预测区间是 `d_1942..d_1969`，对应 `sample_submission` 的 **evaluation 段**。validation 段（`d_1914..d_1941`）的真值已经在训练数据里，A1 方案本来也不需要预测它。

**注意事项**：

1. 集成脚本的硬校验只查"有限值 + 非负"，**0 能通过校验**，所以不会报错。
2. **不要用整个文件的预测总量去判断合理性**，要用 evaluation 段（5.4 / 5.5 检查脚本里的 `v[30490:]`）。
3. Kaggle 提交时上传 60980 行文件是**格式合规**的（官方样例就是这个行数）。
4. 这也是 5.6 节"无法本地算 WRMSSE"的原因之一。

### 6.8 其他小事项

| 事项 | 说明 |
| --- | --- |
| **不是 git 仓库** | `git rev-parse` 确认项目根目录下没有 `.git`。没有版本历史，**误删源码无法恢复**。**建议新机器上第一件事就是 `git init` + 提交一版**，并把 `data/processed/`、`models/`、`output/`、`__pycache__/`、`*.tmp-*` 写进 `.gitignore`（合计约 20 GB，不该进版本库） |
| `logs/.running_pids` | 手工维护的 PID 记录文件，第二次核实时内容为 `preproc_pid=1048`（17 字节，mtime 13:31）。**该进程已于 13:45:53 正常退出，此标记已过期失效** —— 文件不会被脚本自动清理，所以看到它别误以为还有进程在跑，一律以 `ps aux \| grep '[p]ython.*src/'` 为准。**PID 在换机后更是毫无意义**，迁移后应清空或删除 |
| 脚本名以数字开头 | `1-1_recursive_store.py` 这类名字**不能作为模块 import**（Python 标识符不能以数字开头、不能含 `-`），只能作为脚本执行。所以 `__pycache__/` 里没有对应 `.pyc` 是正常的 |
| `notebooks/` 是空目录 | 无内容。`tar` 打包时空目录会保留，无需处理 |
| `.gitkeep` 占位文件 | `logs/`、`output/before_ensemble/` 下各有一个，保留即可 |
| `sales_train_validation.csv` 本轮不用 | 117 MB，预处理只读 `sales_train_evaluation.csv`。**仍建议迁移**（B 轮或 WRMSSE 评分器会用到） |
| 日志是追加模式 | `logs/preprocessing_by_store.log` 等脚本内置日志跨多次运行累积，靠 JSON 里的 `time` 字段区分批次。想要干净日志可在迁移后 `mv` 归档旧的再重跑 |
| 系统时区 | 日志时间戳用 `time.strftime('%Y-%m-%dT%H:%M:%S')`，取的是**本地时间、无时区标记**。新机器时区不同会导致日志时间轴看起来"跳跃"，**不是 bug** |
| `resource.getrusage` 是 POSIX only | 脚本用 `resource.getrusage(RUSAGE_SELF).ru_maxrss` 取峰值内存。**Windows 原生 Python 没有 `resource` 模块**，会直接 `ImportError`。所以必须在 Linux / WSL / macOS 上跑 |
| `ru_maxrss` 单位差异 | Linux 上返回 **KB**（脚本除以 1024 得 MB），macOS 上返回 **字节**。若迁到 macOS 需改这一行，否则日志里的 `peak_rss_mb` 会大 1024 倍 |

---

## 7. B 轮展望（220 模型）—— 本次不执行

### 7.1 B 轮要做什么

在 A 轮（store 族 20 模型）跑通之后，扩展另外两个粒度：

```text
A 轮（已完成部分）：store        10 组 × 2 路 =  20 模型 -> submission_store_only.csv
B 轮（待做）：      store × cat   30 组 × 2 路 =  60 模型 \
                    store × dept 100 组 × 2 路 = 200 模型  >-> submission_final.csv
                                                          /
                                        合计 6 族，官方口径 220 模型
```

- `store × cat`：每店 3 个品类（FOODS / HOBBIES / HOUSEHOLD），10 × 3 = 30 组
- `store × dept`：每店 7-10 个部门，10 × 10 = 100 组（部分店缺某些部门）
- 最终 `submission_final.csv` 是**全部 6 族 260 个（官方口径 220 个）模型的加权融合**，权重来自 WRMSSE 表现

### 7.2 B 轮需要新增的东西

| 需新增 | 说明 |
| --- | --- |
| **预处理支持多粒度** | 当前 `1_preprocessing_by_store.py` 只按 store 分组。B 轮需要在 (store, cat) 与 (store, dept) 粒度上重新做聚合与目标编码。**数据量会大幅膨胀**：dept 粒度是 store 的 10 倍组数 |
| **4 个新训练脚本** | `1-2_recursive_store_cat.py`、`2-2_nonrecursive_store_cat.py`、`1-3_recursive_store_dept.py`、`2-3_nonrecursive_store_dept.py`（手册阶段 4 已列出前两个的名字） |
| **4 个新预测脚本** | 对应上面 4 个训练脚本 |
| **修改集成脚本** | `3-1_final_ensemble.py` 的 `INPUT_FILES` 要从 2 份扩到 **6 份**，`OUTPUT_FILE` 改成 `submission_final.csv`。⚠️ **`M5_ROUND` 变量帮不上忙**（源码没读它，见 6.6），必须真的改代码 |
| **权重方案** | A 轮是等权平均（`stacked.mean(axis=0)`）。B 轮 6 族表现不同，理论上应按 WRMSSE 加权 —— 但那需要一个可用的本地评分器（见 5.6，当前没有） |
| **内存与时间预算重新评估** | dept 粒度组数是 store 的 10 倍，200 个模型的训练时间粗估是 A 轮的 **数倍到十余倍**（不是简单的 10 倍，因为每组数据量小得多）。**9.7 GB 内存下极可能需要按组分批跑** |

### 7.3 B 轮的交付物与校验

```text
output/submission_final.csv
  行数        60980（与 A 轮相同，格式要求不变）
  列          ['id'] + ['F1'..'F28']
  硬校验      与 5.5 的 7 项完全一致（同序、无重复、无 NaN/inf/负数）
  额外校验    与 submission_store_only.csv 对比：
                - 两族 Pearson r 应 < A 轮的两族 r（0.9697），因为加入了更多样的模型
                - 均值量级应相近（同一批 item 的销量预测，不该差一个数量级）
```

`output/before_ensemble/` 下会有 **6 份 CSV**（3 粒度 × 2 模式），每份 60980 行。

### 7.4 为什么本轮不做 B 轮

1. **A 轮尚未跑完**（模型 2/20，最终提交 0/1）。B 轮建立在 A 轮流程验证通过的前提上 —— 如果 A 轮的 20 个模型都还没跑通，扩展到 220 个只会把问题放大 10 倍。
2. **B 轮需要新写代码**（8 个脚本 + 改集成脚本），而本轮任务边界明确是"不改源码、不新增脚本"。
3. **B 轮的资源需求尚未评估**。dept 粒度在 9.7 GB 内存下能否跑通是未知数，可能需要重新设计分段策略。
4. **A 轮产物本身就有价值**。`submission_store_only.csv` 是一份格式合规、可直接提交的预测，先拿到它再谈扩展，风险最低。

👉 **建议顺序**：换机 → 跑完 A 轮步骤 2-5 → 提交 Kaggle 看公榜分数 → 确认流程无误后再规划 B 轮。

---

## 附录 A：一键体检脚本（迁移后 / 任何时候）

**用途**：一条命令输出全部现状，用于换机后自检、或随时确认进度。**全部是只读操作，不会修改任何文件、不会启动任何计算任务。**

```bash
cd <项目目录>
PY=/home/<用户名>/miniconda3/envs/m5/bin/python

$PY - <<'PY'
import glob, json, os, pathlib, subprocess, sys

ROOT = pathlib.Path('.')
print('=' * 72)
print('M5 A 轮现状体检')
print('=' * 72)

print('\n=== 1. 残留进程 ===')
out = subprocess.run(['bash', '-c', "ps aux | grep '[p]ython.*src/'"],
                     capture_output=True, text=True).stdout.strip()
print(out if out else '无 M5 进程在运行')

print('\n=== 2. 环境版本 ===')
try:
    import pandas, numpy, lightgbm
    print(f'python   {sys.version.split()[0]}   (期望 3.10.21)')
    print(f'pandas   {pandas.__version__}   (期望 2.3.3)')
    print(f'numpy    {numpy.__version__}   (期望 2.2.5)')
    print(f'lightgbm {lightgbm.__version__}   (期望 4.7.0)')
except ImportError as e:
    print(f'依赖缺失: {e}')

print('\n=== 3. 预处理产物（10 店 stage）===')
done = 0
for d in sorted(glob.glob('data/processed/*/')):
    mp = pathlib.Path(d) / 'metadata.json'
    if not mp.exists():
        print(f'{os.path.basename(d.rstrip("/")):6s} 无 metadata.json'); continue
    m = json.load(open(mp))
    pkls = [n for n in ['grid_part_1.pkl','grid_part_2.pkl','grid_part_3.pkl',
                        'lags_df_28.pkl','mean_encoding_df.pkl']
            if (pathlib.Path(d) / n).is_file()]
    ok = m.get('stage') == 'complete' and len(pkls) == 5
    done += ok
    print(f"{m.get('store','?'):6s} stage={m.get('stage'):9s} pkl={len(pkls)}/5 "
          f"rows={m.get('row_count'):>8} {'OK' if ok else '<-- 未完成'}")
print(f'完成 {done}/10 店')
tp = sorted(glob.glob('data/processed/test_*.pkl'))
print(f'test_*.pkl: {len(tp)}/10  {[os.path.basename(x)[5:-4] for x in tp]}')

print('\n=== 4. 模型（目标 20 个）===')
rec = sorted(glob.glob('models/lgb_model_*_v1.bin'))
non = sorted(glob.glob('models/non_recur_model_*.bin'))
for p in rec + non:
    print(f'{os.path.basename(p):32s} {os.path.getsize(p)/1024**2:7.1f} MB')
print(f'递归 {len(rec)}/10   非递归 {len(non)}/10   合计 {len(rec)+len(non)}/20')

print('\n=== 5. 预测产物 ===')
for p in sorted(glob.glob('output/before_ensemble/*.csv')):
    with open(p) as f:
        n = sum(1 for _ in f)
    print(f'{os.path.basename(p):46s} wc -l = {n:>6}  '
          f'{"(全量)" if n == 60981 else "(单店/残缺)" if n == 3050 else "(异常)"}')
fin = pathlib.Path('output/submission_store_only.csv')
print(f'submission_store_only.csv: '
      f'{"存在" if fin.exists() else "【尚未生成】A 轮最终交付物"}')

print('\n=== 6. 日志 ===')
for p in sorted(glob.glob('logs/*')):
    if p.endswith('.log'):
        print(f'{os.path.basename(p):38s} {os.path.getsize(p):>7} B')

print('\n=== 7. 目录大小 ===')
for d in ['data', 'data/processed', 'src', 'models', 'output', 'docs', 'logs']:
    if os.path.isdir(d):
        r = subprocess.run(['du', '-sh', d], capture_output=True, text=True).stdout.strip()
        print(r if r else f'{d}: (du 失败)')

print('\n=== 8. 关键完成事件 ===')
for lg, ev in [('logs/preproc_all2.log', 'preprocessing_finished'),
               ('logs/preprocessing_by_store.log', 'preprocessing_finished'),
               ('logs/train_recursive_all.log', 'training_finished'),
               ('logs/train_nonrecursive_all.log', 'training_finished'),
               ('logs/predict_recursive_all.log', 'prediction_finished'),
               ('logs/predict_nonrecursive_all.log', 'prediction_finished'),
               ('logs/store_ensemble.log', 'ensemble_finished')]:
    p = pathlib.Path(lg)
    if not p.exists():
        print(f'{ev:26s} 日志不存在'); continue
    txt = p.read_text(errors='ignore')
    print(f'{ev:26s} 出现 {txt.count(ev)} 次')
print('\n' + '=' * 72)
PY
```

**本脚本已实测可正常运行**（撰写本文档时跑过，输出与 1.3 节快照一致）。

---

## 附录 B：命令速查卡

```bash
# ── 0. 通用 ──────────────────────────────────────────────────────────
PY=/home/<用户名>/miniconda3/envs/m5/bin/python
cd <项目目录>
$PY -c "import pandas,numpy,lightgbm as l;print(pandas.__version__,numpy.__version__,l.__version__)"
ps aux | grep '[p]ython.*src/'          # 查 M5 进程（别用 pgrep -af 'src/'，会假阳性）
df -h .                                  # 磁盘余量（至少 25 GB）
free -h                                  # 内存（可用需 > 6 GB 才能跑递归训练）

# ── 1. 预处理（已完成；仅在未迁移 pkl 时需要）────────────────────────
setsid nohup $PY src/1_preprocessing_by_store.py > logs/preproc_new.log 2>&1 < /dev/null &
tail -f logs/preproc_new.log
ls data/processed/*/mean_encoding_df.pkl | wc -l        # 目标 10

# ── 2. 递归训练（约 11.4 小时，产出 10 模型 + 10 test pkl）───────────
setsid nohup $PY src/2_train/1-1_recursive_store.py > logs/train_recursive_all.log 2>&1 < /dev/null &
ls models/lgb_model_*_v1.bin | wc -l                     # 目标 10
ls data/processed/test_*.pkl | wc -l                     # 目标 10
grep -c training_finished logs/train_recursive_all.log

# ── 2b. 冒烟测试（先跑最小的店，约 15 分钟，强烈建议）────────────────
M5_STORES=CA_2 $PY src/2_train/1-1_recursive_store.py 2>&1 | tee logs/smoke_CA_2.log

# ── 3. 非递归训练（约 12.5 小时，产出 10 模型）───────────────────────
#    🔴 必须等步骤 2 完全结束再跑（内存红线 5.6+4.6=10.2 GB > 9.7 GB）
setsid nohup $PY src/2_train/2-1_nonrecursive_store.py > logs/train_nonrecursive_all.log 2>&1 < /dev/null &
ls models/non_recur_model_*.bin | wc -l                  # 目标 10
ls models/*.bin | wc -l                                  # 目标 20 = A 轮模型全齐

# ── 4. 全量预测（45-75 分钟，两路串行）──────────────────────────────
#    🔴 不设 M5_STORES；🔴 不分批跑（每次整份覆盖）
echo "[$M5_STORES]"                                      # 必须是 []
env -u M5_STORES setsid nohup $PY src/3_predict/1-1_recursive_store_PREDICT.py \
  > logs/predict_recursive_all.log 2>&1 < /dev/null &
# 等 prediction_finished 后再跑下一路
env -u M5_STORES setsid nohup $PY src/3_predict/2-1_nonrecursive_store_PREDICT.py \
  > logs/predict_nonrecursive_all.log 2>&1 < /dev/null &
wc -l output/before_ensemble/*.csv                       # 两份都应为 60981

# ── 5. 集成（< 2 分钟，前台跑即可，幂等）────────────────────────────
$PY src/3_predict/3-1_final_ensemble.py 2>&1 | tee -a logs/store_ensemble_run.log
wc -l output/submission_store_only.csv                   # 60981
ls -lh output/submission_store_only.csv                  # A 轮交付物

# ── 6. 强制重做（谨慎使用）─────────────────────────────────────────
rm -rf data/processed/CA_2                               # 预处理无 M5_FORCE，只能删目录
M5_STORES=CA_2 M5_FORCE=1 $PY src/2_train/1-1_recursive_store.py    # 训练可强制重训
find data/processed models output -name '.*.tmp-*' -print -delete   # 清中断残留

# ── 7. 静态验证（不改文件）─────────────────────────────────────────
for f in src/*.py src/2_train/*.py src/3_predict/*.py; do $PY -m py_compile "$f" && echo "OK $f"; done
diff -u "src/M5-methods/Code of Winning Methods/A1/3. code/2. train/1-1. recursive_store_TRAIN.py" \
        src/2_train/1-1_recursive_store.py | head -60

# ── 8. 性能基线对照（新机器跑完 CA_1 后比对）───────────────────────
#    预处理 CA_1 冷启动:  wall 4:14.97   Max RSS 2,829,832 KB (2.70 GB)   CPU 69%
#    递归训练 CA_1:       76.3 分钟      峰值 5617.6 MB (5.6 GB)         valid RMSE 1.90318
#    非递归训练 CA_1:     83.0 分钟      峰值 约 4.6 GB                  valid RMSE 1.82056
#    两族预测相关性:      Pearson r = 0.9697
#    ⚠️ RMSE 应逐位相同（同 seed/数据/版本）；耗时与内存反映新机器性能
```

---

**文档结束。**

> 有疑问时的排查顺序：**本文档 → 代码本身 → `logs/` 历史事件 → 三份既有 docs → `src/M5-methods/` 官方基线**。
> 代码是唯一真源；文档（包括本文档）只是对代码的解读，冲突时**以代码为准**。
