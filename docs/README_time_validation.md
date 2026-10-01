# 全门店时间验证流程

新入口：`src/time_validation/run.py`。在 WSL `~/projects/m5-forecasting` 中使用 `.venv/bin/python`。
基线代码在 `c75d089`（其父提交为 `c904e55`）；原模型、特征缓存、CA_1 预测及日志未搬移或覆盖。
基线 26 个文件的 SHA-256 在 `docs/baseline_CA_1_manifest.json`，每个阶段后自动核对。
旧版日志 valid RMSE 来自重叠训练样本，不能作为准确度对照。

## 运行与恢复

```bash
cd ~/projects/m5-forecasting
.venv/bin/python src/time_validation/tests.py
.venv/bin/python src/time_validation/run.py all --smoke
.venv/bin/python src/time_validation/run.py all
# 已有后台任务时只看状态，不再启动第二个任务
.venv/bin/python src/time_validation/status.py
tail -f experiments/time_validation_v2/runner.log
```

正式运行按门店串行，每家先非递归再递归；按模式隔离 LightGBM 二进制训练缓存，基础五表仅在同一截止日内共用；每个进程结束即释放内存。
Windows 隐藏的 WSL 客户端运行 `src/time_validation/launch.py`，Linux 训练子进程使用独立 session，
不依赖 IDE 终端存活。日志在实验根目录和各阶段 `logs/` 中。关机、休眠或 WSL shutdown 会中断计算；
恢复后执行相同 `run.py all`，先核对配置、代码内容、全部原始输入 SHA-256、依赖和已完成产物 SHA-256。
每个检查点另绑定阶段、截止日、门店和模式，防止跨目录误复用。互斥锁阻止同时启动两份。未完成的门店/模型会重新运行；不会仅因文件存在而跳过。
修改配置/代码/输入/依赖后必须换 `config.json` 的实验名，不允许混用产物。
若发生错误，`status.json` 保存异常和 traceback，运行停止，不带错进入后续阶段。

## 时间与信息范围

|阶段|训练|样本外预测/评价|
|---|---|---|
|development|d1–1885|d1886–1913|
|test|d1–1913|d1914–1941|
|final|d1–1941|d1942–1969，仅预测|

两模型均不再使用 FIRST_DAY=710。保留原方案依据价格首次上市周过滤上市前训练行的处理；
因此“d1 开始”指所有已知历史进入流程，单个商品只保留上市后的训练行。
两模型保留 Tweedie、学习率、树参数、seed、3000 轮；4 线程，增加 force_col_wise 控制内存，未使用早停。
开发期结束后按预先指定规则保留原参数和 50:50 融合，写入 `frozen_after_development.json`；
最后测试不用于改变参数或融合权重。两单模型与等权融合均报告。

跨店目标编码用所有 10 店截至当前 cutoff 的销量充分统计量重算，保留原有组别和样本标准差 ddof=1；
既不使用该窗口真实未来标签，也不复用原版编码。训练行使用训练期整体编码，包含其自身标签；
这是保留的原方案设定，不是交叉拟合编码，不能把训练误差当泛化误差。
类别表在当前窗口共用；五张表都检查索引、id/d 完全一致。原始 CSV 只读。

非递归 lag=28…42，rolling 都先按 id shift(28) 再在同一 id 中滚动；
预测第 28 天最晚只能取训练截止日的销量。递归短窗 shift=1/7/14，window=7/14/30/60；
先将未来 28 天 sales 置 NaN，再逐天仅用历史及此前预测回填并重算短窗。
回填显式转为 float32，导出预测 float64，保留小数，不屏蔽 FutureWarning。

采用比赛已知未来日历/价格设定。输入日历只保留至 cutoff+28，价格仅保留与该范围相交的周；
最大/最小/均价、月年均价等统计均在这个受限价格表计算。价格按店、商品、周排序后 shift，
不会使用预测窗口之后的新价格周。价格本身为周粒度，最后一周价格视作已知；
训练时允许使用预测起点已经提供的这些外生信息，不声称是逐历史日期实时可得的滚动价格统计。

## 评价与结果

`experiments/time_validation_v2/{development_d1885,test_d1913,final_d1941}/`：

- `cache/{store}/`：独立五表、全局类别映射、行数、范围与耗时。
- `{nonrecursive,recursive}/{store}/`：模型、预测、gain/split 重要性、参数/特征列表/耗时/RSS、校验检查点。
- `results/`：各模型和两个简单基线预测、scores.csv、12 层明细、error_analysis.csv、timings.csv 和 PNG 图。
- 实验根目录 `manifest.json`：配置、代码哈希、Git 提交、原始数据哈希、依赖；`REPORT.md` 汇总已完成阶段。

完整评价为 30,490 条底层序列，12 层聚合共 42,840 条；所有预测先按 id、商品/店和明确的日序列对齐。
RMSSE 分母从第一次非零销量开始计算相邻日差平方均值，不包含跳入首次非零的那一步。
权重使用当前训练截止日前最后 28 日销量×对应周价格，各层归一到 1，再对 12 层等权平均。
零 scale 且误差零记 0，零 scale 且误差非零记无穷；零权重贡献明确为 0，不用 epsilon 偷改官方尺度；
全局销售额为零会报错。正销量找不到价格会报错。各层输出边界样本数，若出现正权重零尺度应解释无穷分数。
MAE、RMSE、平均/总偏差和相对偏差属于底层诊断指标，不是层级加权得分。
简单基线为最后 7 日重复 4 次、最后 28 日的同星期均值重复 4 次。
小规模 smoke 的层级评价只用于技术检查，不能当作全量 WRMSSE。

手算测试：历史 `[0,0,1,2,3]` 的 scale=1，预测每步多 1，RMSSE=1。
测试另含稀疏/零尺度、跨店编码、分组 lag、ID 置换、毒化未来销量及小数回填。
独立参考使用 Nixtla `M5Evaluation.aggregate_levels` 与独立标量 scale，
逐层核对误差在 1e-12 内；参考源码 SHA-256 和实际结果在 `docs/wrmsse_reference_check.json`。
参考：[比赛组织方](https://github.com/Mcompetitions/M5-methods)、
[Nixtla 源码](https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/m5.py)。

## 最终文件与局限

`final_d1941/results/ensemble.csv` 是最终 30,490×28 预测；回测保存在各自阶段，绝不混用。
完成所有阶段后才生成 `final_d1941/results/kaggle_ensemble.csv`，严格按 sample_submission 排序：
`_validation` 段为训练截至 d1913 后对 d1914–1941 的真实样本外融合预测；
`_evaluation` 段为训练截至 d1941 后对 d1942–1969 的最终融合预测。没有占位零。
不向 Kaggle 提交，不推送 GitHub。最终窗口没有提供标签，无法本地评价其准确度。
本轮只有每窗口 20 个按店模型，不等于冠军全部 220 模型复现；仅两个样本外窗口，无法覆盖全年稳定性。
