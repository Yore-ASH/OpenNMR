# 配置说明（settings.json）

所有可调参数都在项目根目录的 **`settings.json`** 里

```text
settings.json          主配置
settings.local.json    可选覆盖层，同名键优先
src/setup.py DEFAULTS  不建议修改，但可作为最后保证
src/setup.py           类别数 / 输入维度 / 1H 总维度 / 各目录路径等自动计算
```

改完想确认有没有生效：

```bash
python main_cli.py settings          # 打印最终生效的配置 + 派生值
python main_cli.py settings --check  # 检查拼写错误 / 派生字段是否被硬写
```

> **派生值不要写进 JSON**：`model.n_classes`、`model.n_features`、
> `features.h1.total_dim`、`features.h1.spec_dim` 这几个是由别的参数算出来的，
> 写进 `settings.json` 会被忽略（`--check` 会提示你）。

---

## 改了要重新生成特征的参数

下面这些改了之后，`data/PTFiles/` 里的特征就和配置不符了，**必须重跑**
`python main_cli.py features`（20–60 分钟）：

| 键 | 说明 |
|---|---|
| `features.c13.buckets` | 13C 分桶数，默认 880（≈0.25 ppm/桶） |
| `features.c13.ppm_min` / `ppm_max` | 13C 化学位移范围，默认 0–220 ppm |
| `features.h1.buckets` | 1H 分桶数，默认 280（≈0.05 ppm/桶） |
| `features.h1.ppm_min` / `ppm_max` | 1H 化学位移范围，默认 0–14 ppm |
| `labels.*` | 官能团增删改 —— **类别数会自动跟着变** |

其他参数（模型结构、训练超参）改了**不需要**重新生成特征，直接重训即可。

---

## paths —— 目录（一般不用改）

| 键 | 默认 | 说明 |
|---|---|---|
| `parquet_dir` | `data/OriginalData` | 原始 parquet |
| `csv_dir` | `data/CSVFiles` | parquet 转出的 CSV |
| `pt_dir` | `data/PTFiles` | 特征张量 |
| `model_dir` | `data/ModelData` | 训练权重 |
| `log_data_dir` | `data/LogData` | 逐 epoch CSV / 运行汇总 / 状态文件（看板读这里） |
| `train_data_dir` | `data/TrainData` | 训练痕迹归档 |
| `console_log_dir` | `logs` | 看板后台进程的控制台日志 |

值为 `null` 表示用项目内的默认目录；填相对路径按项目根解析，也可以填绝对路径（例如把数据放 D 盘）。

---

## device

| 键 | 默认 | 说明 |
|---|---|---|
| `train_device` | `"auto"` | `auto` = 有 CUDA 用显卡、否则 CPU；也可强制 `"cuda"` / `"cuda:0"` / `"cpu"` |

---

## features —— 特征与标签

| 键 | 默认 | 说明 |
|---|---|---|
| `splits` | `["test","train","validation"]` | 处理哪几个 split |
| `c13.buckets` | `880` | 13C 桶数 |
| `c13.ppm_min` / `ppm_max` | `0.0` / `220.0` | 13C 范围 |
| `h1.buckets` | `280` | 1H 桶数 |
| `h1.ppm_min` / `ppm_max` | `0.0` / `14.0` | 1H 范围 |
| `h1.dtype` | `"float16"` | 1H 特征存储精度（`float16` 省一半磁盘和读取带宽） |
| `labels` | 14 个官能团 | **名称 → SMARTS**。类别数 = 条目数，自动传到模型输出层 |

`features.h1.channels`（3）和 `features.h1.scalars`（12）**不是配置项** ——
它们由 `src/parser.py` 的 1H 编码实现固定（3 个通道 = 全部峰 / 复杂多重峰 / J 加权；
12 个全局标量），改布局必须同时改编码代码，所以放在 `src/setup.py` 顶部当常量。

---

## model —— 模型结构

| 键 | 默认 | 说明 |
|---|---|---|
| `arch` | `"r4"` | `r4` = 0.8.0 原架构（19.07M 参数，已验证）；`r5` = 优化架构（5.02M 参数，实验性，受控 A/B 未跑赢 r4） |
| `c13.kernel_sizes` | `[5, 5, 3]` | 13C 分支 3 层 Conv1d 的卷积核（必须 3 个值） |
| `c13.channels` | `[32, 64, 128]` | 13C 分支 3 层通道数（必须 3 个值） |
| `h1.kernel_sizes` | `[5, 3]` | 1H 分支 2 层卷积核（必须 2 个值） |
| `h1.out_channels` | `[32, 64]` | 1H 分支 2 层通道数（必须 2 个值） |
| `h1.scalar_hidden` | `32` | 12 个全局标量走的小 MLP 隐层宽度 |
| `r5.head_channels` | `32` | 仅 r5：13C 主干 1×1 瓶颈宽度（128 → 32） |
| `r5.head_h1_channels` | `16` | 仅 r5：1H 主干 1×1 瓶颈宽度（64 → 16） |
| `head.hidden` | `512` | 分类头隐层宽度 |
| `head.dropout` | `0.2` | Dropout 比例 |

改结构后**旧权重要么加载不了、要么被忽略**，需要重新训练。
（`f1_test.py` 评估时会以权重文件里记录的架构和输入维度为准。）

---

## training —— 训练超参

| 键 | 默认 | 说明 |
|---|---|---|
| `batch_size` | `64` | 批大小 |
| `lr` | `0.001` | Adam 学习率 |
| `weight_decay` | `0.0` | Adam 权重衰减（0 = 不衰减） |
| `epochs` | `0` | **0 或负数 = 不限制**，一直训练到 Ctrl+C |
| `workers` | `2` | DataLoader 子进程数，**Windows 建议 0**（spawn + IPC 开销大） |
| `seed` | `2026` | 随机种子；A/B 对比时必须固定 |
| `focal.alpha` | `0.25` | Focal Loss 正负样本平衡 |
| `focal.gamma` | `2.0` | 越大越关注难分样本 |
| `focal.per_class_alpha` | `true` | 按类别正例率自适应 alpha（稀有类权重更高） |
| `lr_scheduler.name` | `"plateau"` | `none` / `plateau` |
| `lr_scheduler.factor` | `0.5` | plateau：每次衰减倍率 |
| `lr_scheduler.patience` | `0` | plateau：等几轮没改善才衰减 |

---

## data

| 键 | 默认 | 说明 |
|---|---|---|
| `mmap` | `true` | 用内存映射读 `.pt`（3.3 GB 的 X_train 不会整块进内存）。只有文件系统不支持 mmap 时才关 |

---

## archive —— 训练痕迹

| 键 | 默认 | 说明 |
|---|---|---|
| `save_every_epoch` | `false` | `false` = 只归档验证集变好的轮次（每份约 73 MB）；`true` = 每轮都存 |
| `epoch_metrics` | `true` | 每轮是否在验证集上算逐类指标（117k 验证集约多花 1–2 秒） |

产物在 `data/TrainData/run_<时间戳>/`：
`run.json`（含本次完整配置）、`epochs.csv`（逐轮汇总）、`epoch_XXXX.json`（逐类指标）、
`checkpoints/epoch_XXXX_valX.XXXXXX.pth`。

---

## 命令行覆盖

命令行参数优先级最高，会调用 `SETTINGS.apply({...})` 覆盖对应字段（和改 JSON 等价）：

```bash
python main_cli.py train --epochs 5 --batch-size 128 --lr 0.0005 --arch r5 --seed 2027
python main_cli.py train --device cpu
python main_cli.py eval --split test --threshold 0.35
```

注意：命令行覆盖**只在本次进程内有效**，不会写回 `settings.json`。
想长期生效就改 `settings.json`。
