# M5 v3 两机任务与结果交接

分支：`experiment/two-machine-v3`。基于 v3 `3e7d78e`，保留原特征与主要参数；正式本机 8 线程，Mac 4 线程，均按门店/模式串行。这个分支包含完整仓库，不使用之前的代码预检 ZIP 作为正式运行目录。

## 任务分配

|工作端|线程|门店|职责|
|---|---:|---|---|
|windows（WSL）|8|CA_1、CA_2、TX_1、TX_2、TX_3、WI_1、WI_2、WI_3|本机训练；接收 Mac 预测；全店 WRMSSE、统一选轮数与融合权重、评分及最终汇总|
|mac|4|CA_3、CA_4|执行分配门店任务，导出预测和证据；保留自己的模型与缓存|

正式配置仍包含全部 10 店。两台都必须保留全部 5 张原始 CSV，不能删除未分配门店的数据；跨店目标编码仍按全部门店历史计算。运行分配由 `src/time_validation/two_machine_plan.json` 控制，不能把 config.json 的 stores 改成各自子集。

Mac 截止为 **2026-10-06 17:00（UTC+8）**。每个新任务前预留预计耗时加 5 分钟：特征准备 5 分钟、非递归 30 分钟、递归 60 分钟；余量不足则停止启动后续任务，写 stopped_before_deadline。不会自动杀掉正在训练的模型，实际耗时超出预算仍可能越过 17:00。如必须按时关机，保存已完成结果，本机重做当前未完成模型。截止日期过后 Mac worker 不会开始新训练；不要自行删除日期检查来复用本轮计划。

## 时间和阶段

|阶段|候选训练截止日/预测窗|正式训练截止日/预测窗|
|---|---|---|
|development|1857 / 1858–1885|1885 / 1886–1913|
|test|1885 / 1886–1913|1913 / 1914–1941|
|final|无选择，沿用 test|1941 / 1942–1969|

development 与 test 各先收齐 10 店两种模式全部候选，主机选全店统一轮数和权重，再让两台重训；正式阶段不按单店 RMSE 选参。test 开始需要 development 已完成评分的 gate；final 开始需要 test 完成的 gate。最终窗口无真值，只检查完整性；不生成 Kaggle 文件，不自动提交。

## 拉取与准备

```bash
git clone --branch experiment/two-machine-v3 https://github.com/Annnnnna11/Projectyctcmh.git m5-two-machine
cd m5-two-machine
```

Mac 激活已经预检通过的 `mini` 环境，使用其中的 `python`，不需要重装 Python。把已有原始 CSV 放到新仓库 data/（或逐文件建立链接）。本机使用 `.venv/bin/python`。代码/数据/依赖更改后不要继续原实验；正式阶段工作端必须使用同一提交并保持核心计算文件一致。

```bash
# Mac，用已通过预检的环境
conda activate mini
python src/time_validation/tests.py
python src/time_validation/tests_handoff.py
python src/time_validation/collaborate.py show --node mac
```

Mac 运行此完整新分支前，在其独立 smoke 目录检查平台能运行新入口：

```bash
python src/time_validation/collaborate.py worker --node mac --stage development --phase candidates --smoke
```

这个检查仅 CA_3 前48商品，不是正式候选任务。之后正式命令不加 --smoke。本机已做单机双工作端三阶段真实小规模交接演练；不是 Mac 原生演练的替代。

## 每阶段操作（不要直接使用旧 run.py all / launch.py）

下例为 development；test 替换阶段名，但必须先导入测试 gate。每条命令完成后才执行依赖它的下一条命令。

### 1. 两台分别候选训练

```bash
# WSL
.venv/bin/python src/time_validation/collaborate.py worker --node windows --stage development --phase candidates

# Mac：使用 mini 环境，保持开机、插电；需要后台运行可由 Mac 端自行安排
caffeinate -i python src/time_validation/collaborate.py worker --node mac --stage development --phase candidates
```

Mac 完成后导出证据目录：

```bash
python src/time_validation/collaborate.py export --node mac --stage development --phase candidates --output transfers/mac_dev_candidates
```

`transfers/mac_dev_candidates` 是普通目录，完整传到本机；不要把两个文件夹混着覆盖，也不要只传 CSV。可以压缩整个目录进行传输，但先在新目录解压再导入，程序不直接解压未知档案。

若只有 CA_3 完成，加入 `--stores CA_3`；若某个模式尚未完成，加入 `--modes nonrecursive` 或 `--modes recursive`。没有通过源工作端本地检查点的任务不能导出。已有同名输出目录不能覆盖，另选新名字。

### 2. 本机校验导入、统一选参

```bash
.venv/bin/python src/time_validation/collaborate.py import --node windows --bundle transfers/mac_dev_candidates
.venv/bin/python src/time_validation/collaborate.py select --node windows --stage development
.venv/bin/python src/time_validation/collaborate.py export --node windows --stage development --phase selection --output transfers/dev_selection
```

10店预测未收齐时 select 会拒绝，不能把缺店的分数当正式 WRMSSE。将 `transfers/dev_selection` 传给 Mac：

```bash
python src/time_validation/collaborate.py import --node mac --bundle transfers/dev_selection
```

### 3. 两台分别按统一配置重训

```bash
# WSL
.venv/bin/python src/time_validation/collaborate.py worker --node windows --stage development --phase retrain
# Mac
caffeinate -i python src/time_validation/collaborate.py worker --node mac --stage development --phase retrain
python src/time_validation/collaborate.py export --node mac --stage development --phase retrain --output transfers/mac_dev_retrain
```

把导出目录交回 WSL：

```bash
.venv/bin/python src/time_validation/collaborate.py import --node windows --bundle transfers/mac_dev_retrain
.venv/bin/python src/time_validation/collaborate.py score --node windows --stage development
```

### 4. 开放下一阶段

```bash
# development 评分完成后，主机开放 test
.venv/bin/python src/time_validation/collaborate.py gate --node windows --stage test
.venv/bin/python src/time_validation/collaborate.py export --node windows --stage test --phase gate --output transfers/test_gate
# Mac 收到后
python src/time_validation/collaborate.py import --node mac --bundle transfers/test_gate
```

随后按上面步骤，把 development 改成 test，完成测试。test 完成后同样 gate/export/import `final`；final **只执行 retrain → Mac 导出 → 本机导入 → score**，不执行 candidates、select 或新的 selection 导出。final 读取已导入的 test 选择结果。

## 交接校验与指纹

bundle.json 记录全部文件 SHA-256、原始数据 SHA-256、核心代码哈希、计算配置、计划、核心库版本，以及源工作端原始 manifest（包含依赖和提交来源）。核心库版本、代码、数据、规则必须匹配；环境构建、系统、线程数分别记录，允许不同。Mac 新增本地脚本请放 src/time_validation/ 之外，不要覆盖 common.py、model.py 等共享文件。

导入前本机须已完成同截止日、同模式的一个本地任务，作为实际特征顺序参考。导入先检查所有文件和全部冲突，再复制；检查 3049 个正确 ID、28 列、有限且非负值、日期、候选轮数、模型参数、类别映射和特征证据。相同 bundle 可重复导入，不同内容不能覆盖既有结果。

只传候选预测/元数据/特征证据，或重训预测/重要性/元数据；不传训练缓存和大模型。Mac 的模型保留在 Mac。跨机产物保留 received.json 与源 manifest，不篡改 foreign complete.json 来伪装本机训练完成。收到的正式预测可在通过校验后参与评分，但本机没有因此获得 Mac 的模型文件。

文件哈希和声明检查用于防止损坏/误配，并非数字签名；只接受你控制的两台工作端结果。

## 中断、接手与本机后台

相同 worker 命令可恢复，只跳过指纹和文件哈希匹配的已完成本地任务；未完成模型重新训练。同一工作端一次只允许一个 worker。Mac 未完成或截止后的任务，本机可明确接手：

```bash
.venv/bin/python src/time_validation/collaborate.py worker --node windows --stage development --phase candidates --stores CA_3 --takeover
```

按实际缺失阶段/门店替换；可加 --modes 指定缺失模式。已导入且完整的任务不必接手重做，接手后不要再导入同一任务旧结果。已完成本地检查点优先于旧 received 证据。

在本机 PowerShell 后台启动一个阶段（不会自动跨越选参屏障）：

```powershell
Start-Process -FilePath wsl.exe -WindowStyle Hidden -ArgumentList @('-d','Ubuntu-24.04','--','/home/cyangcj/projects/m5-two-machine/.venv/bin/python','-u','/home/cyangcj/projects/m5-two-machine/src/time_validation/collaborate.py','worker','--node','windows','--stage','development','--phase','candidates')
```

worker 自身写持续日志，不依赖展示终端。关机、睡眠和 wsl --shutdown 仍会中断任务。这里提供命令，不表示已启动正式训练。

## 结果位置

- WSL：`experiments/time_validation_v3_two_machine_windows/`
- Mac：`experiments/time_validation_v3_two_machine_mac/`
- 每阶段 `worker_<阶段>_<阶段任务>.json`：完成列表和状态。
- `worker_logs/`：持续日志。
- `selection/cutoff_1857`、`cutoff_1885`：候选、全店评分、轮数与权重。
- `development/results`、`test/results`：回测；`final/results/ensemble.csv`：最终30490×28融合预测。
- `REPORT.md`：主机阶段汇总。

本机8线程只在CA_1前300轮做过速度对比；本分支另做8线程小规模前缀与完整交接检查。完整3000轮8线程前缀等价尚未直接实测，正式运行存在这一验证范围限制。Mac 当前预检为旧入口版本，也要检查新入口。仅两个回测窗口，不能证明全年稳定性。

仓库只推送代码、说明和小型检查证据。原始数据、虚拟环境、缓存、大模型、预测、日志、transfers 和 experiments 不上传 GitHub。
