# OpenNMR — 13C / 1H NMR 官能团预测（1.1.0）

从分子的核磁共振谱图预测 **14 个官能团**的多标签分类项目。

> **⚙️ 所有参数都在项目根目录的 [`settings.json`](settings.json) 里，改完直接跑，不用动 `.py`。**
> 逐字段说明见 [SETTINGS.md](SETTINGS.md)。派生值（类别数 / 输入维度 / 1H 总维度 / 各目录路径）
> 由 `src/setup.py` 自动计算，**不要**写进 JSON。
>
> ```bash
> python main_cli.py settings          # 看当前生效的配置与派生值
> python main_cli.py settings --check  # 检查配置有没有写错
> ```
>
> `settings.local.json`（可选，不入 git）可覆盖 `settings.json` 的同名字段，
> 适合放只在本机成立的配置（换数据盘、强制 cpu、临时小规模试跑）。

> **🚀 换机器部署请看 [DEPLOY.md](DEPLOY.md)** —— 从「已装 Python」到「跑起来」的完整步骤，
> 含依赖安装顺序（Pascal 显卡必须 `torch 2.14.1+cu126`）、`cuda-info` 自检、
> 从原始 parquet 重建特征、训练/评估/看板命令，以及一张排查表。

> **可选架构 R5**（`model.arch`）：BatchNorm + 1×1 通道瓶颈 + 全局 max/avg 上下文，
> 参数量从 **19.07 M 降到 5.02 M（26.3%）**。旧的 **R4** 架构完整保留，行为与 0.8.0 逐字节一致，
> 0.8.0 训练出来的权重也能原样加载。
>
> ⚠️ **但受控 A/B 显示 R5 并没有跑赢 R4**（80k/2 轮：macro-AP 0.6745 vs 0.6918，
> macro-F1 0.6106 vs 0.6124；micro-F1 与 Hamming 略好）。
> 所以 **默认仍是 R4**，R5 作为实验性选项保留。详见 [AB_RESULT.md](AB_RESULT.md)。

- 输入：分子 SMILES（用于生成训练标签）、13C NMR 峰、1H NMR 峰（含积分、多重峰、耦合常数 J）
- 输出：14 个官能团的 0/1 概率
- 两套架构：**R4（默认，19.07 M 参数，已验证）** / R5（实验性，5.02 M 参数，`"arch": "r5"`）
- R4 参考成绩：验证集 macro-F1 **0.6215** @0.5、**0.7403** @0.35，测试集 0.7403 @0.35，macro AP 0.8107

## 快速开始

```bash
python main_cli.py settings       # 0. 看当前配置（可选）
python main_cli.py cuda-info      # 1. 确认显卡能用
python main_cli.py features       # 2. 从 parquet 重建特征（分发包/首次运行需要）
python main_cli.py train          # 3. 训练（Ctrl+C 停，会先保存再退出）
python main_cli.py eval --split test --threshold 0.35   # 4. 评估
```

## 配置

改参数只改 `settings.json` 一个文件。常见的几个：

| 想改什么 | 改哪个键 |
|---|---|
| 训练多少轮 | `training.epochs`（`0` = 不限制，一直训到 Ctrl+C） |
| 批大小 / 学习率 | `training.batch_size` / `training.lr` |
| 用哪套架构 | `model.arch`（`"r4"` / `"r5"`） |
| 用不用显卡 | `device.train_device`（`"auto"` / `"cuda"` / `"cpu"`） |
| 加一个官能团 | `labels` 里加一条 `名称: SMARTS`（类别数自动跟着变，需重新 `features`） |
| 每轮都保存权重 | `archive.save_every_epoch` |
| 数据放别的盘 | `paths.*`（填绝对路径） |

命令行参数优先级最高，但**只在本次进程有效**，不会写回 JSON。

## 目录结构

```
.
├── README.md
├── settings.json          # ★ 所有可调参数都在这里（改完即生效，不用动 .py）
├── SETTINGS.md            # 配置逐字段说明
├── DEPLOY.md              # 部署说明（换机器必看）
├── AB_RESULT.md           # R4 / R5 受控 A/B 完整报告
├── CHANGELOG.md           # 版本变更
├── UPDATELOG.md           # 改造说明书 + R1–R5 完整实验记录
├── LICENSE.txt
├── VERSION
├── requirements.txt
├── main_cli.py            # 命令行入口
├── scripts/
│   ├── build_features.py  # 一键生成 CSV 与特征
│   ├── check_settings.py  # 打印 / 校验当前生效的配置（main_cli.py settings）
│   ├── ab_test.py         # R4 / R5 架构 A/B 对比
│   ├── install_cuda.ps1   # Windows 一键装 Python 3.10 + torch cu126
│   └── install_cuda.sh    # Linux 同上
├── src/
│   ├── setup.py           # ★ 加载 settings.json → 校验 → 计算派生值 → 暴露 SETTINGS
│   ├── parser.py          # parquet→csv、13C 分桶、1H 特征（积分/多重峰/J）
│   ├── dataset.py         # mmap 加载 X/Y/X1H，按行拼接
│   ├── model.py           # R4 / R5 双架构（默认参数全部来自 SETTINGS）
│   ├── train.py           # 训练（FocalLoss、断点保存、日志、状态文件、痕迹归档）
│   ├── train_archive.py   # 训练痕迹归档：data/TrainData 下逐 epoch 记录 + 权重
│   ├── cuda_info.py       # CUDA / 显卡自检（对应 main_cli.py cuda-info）
│   ├── f1_test.py         # F1 / AP / 阈值扫描评估
│   ├── log_utils.py       # CSV/JSONL 日志工具
│   ├── dashboard.py       # Flask 训练看板
│   ├── test.py / test.ipynb
│   └── __init__.py
└── data/
    ├── README.md          # 数据目录说明
    ├── OriginalData/      # 原始 parquet
    ├── CSVFiles/          # parquet 转出的 CSV（features 生成）
    ├── PTFiles/           # 特征张量（features 生成）
    ├── ModelData/         # 训练权重
    ├── LogData/           # 逐轮日志（看板读这里）
    ├── TrainData/         # 训练痕迹归档
    └── ABTest/            # A/B 对比报告
```

## 配置系统是怎么分工的

一句话：**`settings.json` 存「用户能改的原始参数」，`src/setup.py` 负责「加载 + 校验 + 计算派生值」，
两者不是二选一。**

| 职责 | 放在哪 | 为什么 |
|---|---|---|
| 用户可调的原始参数 | `settings.json` | 不用碰 Python，不会因为缩进/逗号把项目跑崩；能被 `check_settings.py` 校验；能整体 diff；能用 `settings.local.json` 做本机覆盖 |
| 从别处推导的值 | `src/setup.py` 自动算 | 类别数 = `len(labels)`、输入维度 = `features.c13.buckets`、1H 总维度 = `buckets×channels+scalars`。写进 JSON 就是重复，改一处忘另一处 |
| 目录路径 | `src/setup.py` 从项目根推 | 写绝对路径不可移植 |
| 编码器实现固定的常量 | `src/setup.py` 顶部 | 例如 1H 的 3 通道 / 12 标量，改了要同时改编码代码，不该让用户以为可配 |

`src/setup.py` 里没有任何 `os.environ` 写入，模块之间也不再通过环境变量传配置 ——
所有模块 `from setup import SETTINGS` 读同一个对象，参数只存在一处。
（1.1.0 之前是 `ENVIRON` dict + `os.environ` 双份，每个参数有两份默认值，已经漂移过。）

## 环境

要求 **Python 3.10**（已在 **3.10.11** 上实测通过）。

### 方式 A：一键脚本（推荐）

Windows（PowerShell）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_cuda.ps1
```

Linux：

```bash
bash scripts/install_cuda.sh
```

脚本会：建 Python 3.10 虚拟环境 → 装 `torch 2.14.1+cu126` → 装其余依赖 → 自动跑 `cuda-info` 自检。
只想用 CPU 加 `-CpuOnly`（Windows）/ `CPU_ONLY=1`（Linux）。

### 方式 B：手动

```bash
# 1. 建虚拟环境（必须是 Python 3.10）
py -3.10 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux

# 2. 先装 CUDA 版 torch（顺序重要，不要用 requirements.txt 的默认源）
pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126

# 3. 再装其余依赖
pip install -r requirements.txt

# 4. 自检
python main_cli.py cuda-info
```

`requirements.txt` 是按 Python 3.10 实测通过的版本生成的，其中 `torch==2.14.1` 是基础版本号，
**CUDA 轮子必须单独按上面的 `--index-url` 安装**（见下一节）。

## CUDA / 显卡

### 三条硬约束（踩过坑，务必遵守）

1. **Pascal 架构（MX150 / GTX 1060，`sm_61`）必须用 `torch <= 2.14` + `cu126`。**
   PyTorch 2.15 起不再发布任何 CUDA 12.x 轮子，用新版本会报
   `CUDA error: no kernel image is available for execution on the device`。
2. **Windows 上从 PyPI 直接 `pip install torch` 装到的是 CPU-only 轮子。**
   看 `torch.__version__`，没有 `+cu126` 就是装错了源，`cuda 可用` 会是 `False`。
3. **CUDA 12.x 要求显卡驱动 R525 及以上。**
   看 `nvidia-smi` 右上角的 `CUDA Version`，必须 ≥ 12.0；只有 11.x 的话必须升级驱动。

本机实测的安装路径（可作参考）：驱动 `511.69`（CUDA 11.6）→ 升级到 `582.78`（CUDA 13.0）后，
`torch 2.14.1+cu126` + MX150（Pascal `sm_61`）自检通过。

### GPU 加速效果与注意事项

| 场景 | 吞吐 | 折算全量 937k 一个 epoch |
|---|---|---|
| CPU | ~190 样本/秒 | ~82 分钟 |
| GPU 短暂突发（前 ~50 秒） | ~1150 样本/秒 | ~14 分钟 |
| **GPU 持续跑超过 ~54 秒后** | **~323 样本/秒** | **~48 分钟** |

上表是**笔记本 MX150** 的实测结果。注意第二、三行的差异不是 bug：

- MX150 在轻薄本里功耗预算很小，持续负载约 54 秒后 SM 频率会被锁死到 139 MHz（上限 1911 MHz），
  温度稳定在 79°C 说明**不是过热**，而是 EC/固件层的功耗墙。此时 GPU 只比 CPU 快约 1.7 倍。
- 解决办法：NVIDIA 控制面板 →「管理 3D 设置」→「电源管理模式」改「最高性能优先」；
  安装 MyASUS 把性能模式切「性能」；笔记本垫高散热。根本限制在固件。
- **长时间训练建议用台式机显卡**（如 GTX 1060），功耗和散热预算完全不是一个量级，不会这样降频。

其它经验：

- Windows 上 `--workers` **不建议开多进程**（实测纯数据加载 40 批：`workers=0` 1.4 秒，
  `workers=4` 9.7 秒），spawn 启动 + IPC 开销盖过并行收益。默认 `DATALOADER_WORKERS=2`，
  在 Windows 上可以显式传 `--workers 0`。
- 显存占用很小（batch 64 时远低于 2 GB），瓶颈通常在数据加载和 GPU 功耗，而不是显存。

### 常用命令

```bash
python main_cli.py cuda-info                 # 先跑这个，确认 CUDA 真的可用
python main_cli.py train --epochs 3          # 自动用显卡
python main_cli.py train --device cpu --epochs 3    # 强制 CPU 做对比
python main_cli.py eval --split test --threshold 0.35
```

`cuda-info` 退出码：`0` = CUDA 可用且实测通过；`1` = 不可用（会打印具体原因和修复命令），
可以直接写进脚本做环境校验。

### 本机（MX150 + Python 3.10.11）实测结果

CUDA 路径与原 CPU 版本的**数值行为完全一致**，已用 R4 权重复现：

| 指标 | 本次 GPU 实测 | README 记录的 R4 验证集 |
|---|---|---|
| macro-F1 @0.5 | **0.6215** | 0.6215 |
| micro-F1 @0.5 | **0.8150** | 0.8150 |
| weighted-F1 | **0.7816** | 0.7816 |
| macro AP | **0.8107** | 0.8107 |
| Hamming loss | **0.0898** | 0.0898 |

训练侧也验证过：`13C=880 + 1H=852` 双分支都在工作，显存占用约 527 MiB，
batch 64 时约 384 样本/秒（受上文功耗墙影响）。

## 快速命令入口

除了直接运行 `src/` 下的脚本，也可以用统一的入口：

```bash
python main_cli.py cuda-info                # 检查 torch / CUDA / 显卡是否可用
python main_cli.py features                 # 生成 CSV 与特征
python main_cli.py train --epochs 3 --workers 2
python main_cli.py eval --split test --threshold 0.35
python main_cli.py dashboard --port 8090
```

`train` / `eval` / `dashboard` 后面的参数会原样传给对应脚本。

## 数据准备

把原始 parquet 放到：

```
data/OriginalData/train.parquet
data/OriginalData/validation.parquet
data/OriginalData/test.parquet
```

需要的列：

| 列 | 说明 |
|---|---|
| `id` | 样本 id |
| `smiles` | 分子 SMILES（用于生成 14 个官能团标签） |
| `nmr_13c` | 13C 化学位移列表，如 `[158.54, 139.82, ...]` |
| `nmr_1h` | 1H 峰列表，每峰 `[化学位移, _, 多重峰, 积分, 耦合常数列表]`，如 `[7.26, 0.0, "d", "2H", [8.1]]` |

然后一条命令生成全部 CSV 和特征：

```bash
python scripts/build_features.py
```

它会依次执行 `parse_parquet_file()` → `gen_XY_pt()` → `gen_1h_pt()`，生成：

```
data/CSVFiles/{train,validation,test}.csv
data/PTFiles/X_{split}.pt        13C 多热特征 [N, 880] float32
data/PTFiles/Y_{split}.pt        14 类标签 [N, 14] float32
data/PTFiles/X1H_{split}.pt      1H 特征 [N, 852] float16
data/PTFiles/X1H_meta.json       1H 特征布局（bins=280, channels=3, scalars=12）
```

数据规模：train 937,022 / validation 117,035 / test 117,205；
全量特征约 5 GB 磁盘（X 3.1 GB + X1H 1.5 GB + CSV 0.5 GB），生成过程是流式的，内存占用约 1 GB。

## 训练

```bash
# 固定 3 个 epoch（推荐先这样跑）
python src/train.py --epochs 3 --workers 2

# 一直训练，Ctrl+C 停止（会先保存权重、写日志再退出）
python src/train.py --workers 2

# 从已有权重热启动（结构一致时完整加载；否则只加载 13C 卷积）
python src/train.py --epochs 3 --workers 2 \
    --init-from data/ModelData/model_13c_v1.pth

# 调整 batch / 学习率
python src/train.py --epochs 3 --workers 2 --batch-size 64 --lr 0.0003
```

> 想让**每轮都归档权重**（默认只归档变好的轮次），把 `src/setup.py` 里的
> `SAVE_EVERY_EPOCH` 从 `0` 改成 `1`。每份权重约 73 MB，不限时长训练时磁盘会线性增长。

产物：

```
data/ModelData/model_13c_v1.pth      验证 loss 最好的一轮（覆盖写）
data/ModelData/model_13c_last.pth    每个 epoch 都更新（断点，覆盖写）
data/LogData/train_history.csv       逐 epoch 记录（所有运行共用一张表，看板读取）
data/LogData/run_summary.jsonl       每次运行一条汇总
data/LogData/train_status.json       实时状态（看板读取）
logs/train_web.log                   看板启动的训练控制台输出
data/TrainData/…                     每个 epoch 的完整痕迹 + 历史权重（见下一节）
```

## 训练痕迹 `data/TrainData`

每次训练都会在 `data/TrainData/run_<时间戳>/` 下留一份完整痕迹，
**即使一直训练跑很久、中途 Ctrl+C，已经写完的轮次都不会丢**：

```
data/TrainData/
├── runs_index.csv                      所有运行各一行（追加）
└── run_20261020-140000/
    ├── run.json                        本次训练的完整配置 + 结束汇总
    ├── epochs.csv                      逐 epoch 一行，每行写完立即落盘
    ├── epoch_0001.json                 该 epoch 的完整明细（含逐类指标）
    ├── epoch_0002.json
    └── checkpoints/
        ├── epoch_0002_val0.086779.pth  归档权重（名字带轮次和验证 loss）
        └── epoch_0005_val0.081233.pth
```

各文件内容：

| 文件 | 内容 |
|---|---|
| `runs_index.csv` | 每次运行一行：run_id / 起止时间 / 停止原因 / 完成轮数 / 最佳轮次 / 最佳 val_loss / 总耗时 / 设备 / 目录 |
| `run.json` | 本次运行的**全部配置**（batch、lr、Focal 参数、调度器、workers、特征维度…）+ 结束汇总 |
| `epochs.csv` | 逐轮一行：lr、train_loss、val_loss、best_val_loss、是否改善、耗时、macro/micro F1、子集准确率、Hamming、macro AP、归档权重路径 |
| `epoch_XXXX.json` | 该轮的**完整明细**：上面的汇总字段 + `metrics` 汇总 + `per_class` 逐类 `support/tp/fp/fn/precision/recall/f1` |
| `checkpoints/*.pth` | 归档权重，文件名形如 `epoch_0002_val0.086779.pth`，不打开文件就能看出哪轮最好 |

三个相关配置（`src/setup.py`）：

| 配置 | 默认 | 说明 |
|---|---|---|
| `TRAIN_DATA_PATH` | `data/TrainData` | 痕迹根目录 |
| `SAVE_EVERY_EPOCH` | `0` | `0` = **只在验证集比历史最好还低时归档权重**；`1` = 每轮都归档（每份约 73 MB，不限时长训练会吃盘） |
| `EPOCH_METRICS` | `1` | `1` = 每轮在验证集上算逐类指标（117k 验证集约多花 1~2 秒）；`0` = 只记 loss，更快 |

**中断也不会丢痕迹**：Ctrl+C 落在训练或验证中途时，会额外归档一份
`epoch_XXXX_train_interrupted.pth` / `epoch_XXXX_val_interrupted.pth`，
并在 `epochs.csv` 与 `epoch_XXXX.json` 里记一条 `interrupted` 记录。

> 说明：`data/LogData` 仍然照常写，因为训练看板（`dashboard.py`）读的是那里；
> `data/TrainData` 是额外的归档层，两者互不影响。

## 评估

```bash
# 验证集（阈值 0.5，并做 0.05–0.95 阈值扫描）
python src/f1_test.py

# 测试集，推荐阈值 0.35
python src/f1_test.py --split test --threshold 0.35

# 指定权重、不写日志、跳过阈值扫描
python src/f1_test.py --model data/ModelData/model_13c_v1.pth --no-save --no-sweep
```

结果会追加到 `data/LogData/`：`f1_by_class.csv`、`f1_threshold_sweep.csv`、`f1_summary.jsonl`。

## 训练看板

```bash
python src/dashboard.py --host 0.0.0.0 --port 8090
```

浏览器打开 `http://<机器IP>:8090`，可以看实时进度、loss 曲线、逐 epoch 记录、F1 报告，并直接开始/停止训练、触发 F1 评估。
看板关闭不影响已经启动的训练；停止训练 = 发送 SIGINT，训练会先保存再退出。

## 模型架构（R4 / R5）与 A/B 对比

0.9.0 把模型拆成**两套可切换的架构**，同一份代码、同一份数据随时对比：

```bash
python main_cli.py train                      # 默认 R4（已验证，19.07 M 参数）
python main_cli.py train --arch r5            # R5 优化版（5.02 M 参数，实验性）
python main_cli.py ab-test                    # 自动跑两个架构并生成对比报告
```

### 两套架构的差别

| | R4（0.8.0 原架构） | R5（0.9.0 优化） |
|---|---|---|
| 归一化 | 无 | **每个卷积后 BatchNorm1d**（7 层） |
| 13C 分支输出 | flatten 128×220 = **28,160** 维 | 1×1 卷积压到 32 通道 → **7,040** 维 + 全局 max/avg **256** 维 |
| 1H 分支输出 | flatten 64×140 = **8,960** 维 | 1×1 卷积压到 16 通道 → **2,240** 维 + 全局 max/avg **128** 维 |
| 分类头输入 | **37,152** 维 | **9,696** 维 |
| 分类头参数 | **19,029,518** | **4,972,046** |
| 总参数 | **19,071,854（19.07 M）** | **5,020,286（5.02 M）** |
| 训练速度 | 基准 | 更快（参数量少 74%） |
| 旧权重 | 完全兼容 | — |

**为什么这样改**：原架构 19.07 M 参数里有 19.02 M（99.8%）集中在一个
`Linear(37152, 512)` 上 —— 卷积提取的特征被直接展平后一口气喂进巨型全连接层，
既没有结构先验，也让优化很吃力。R5 用 1×1 卷积做通道瓶颈再展平，
同时并联一路全局池化特征：**展平保留「峰在哪个化学位移」**（NMR 里 175 ppm 和
55 ppm 的碳意义完全不同），**全局池化补充与位置无关的整体证据**，两路互补。

`_arch_id` 会写进 `state_dict`，所以 `f1_test.py` 加载权重时能自动还原训练时用的架构
（老的 R4 权重没有这个标记，默认按 R4 处理）。

### 怎么做 A/B 对比

```bash
python main_cli.py ab-test                          # 默认 80k/15k 切片、各 4 轮
python main_cli.py ab-test --epochs 6 --train 150000 --val 20000
python main_cli.py ab-test --archs r5               # 只跑一个
```

脚本保证**公平**：同一份真实数据切片、同一个随机种子（`--seed`，默认 2026）、
完全相同的 batch/lr/损失/调度器，顺序跑两个架构，然后生成：

```
data/ABTest/<时间戳>/
├── comparison.csv     # 一张表：参数量、耗时、各项指标、逐类 F1 及差值
├── comparison.md      # 人读的对比报告
├── summary.json       # 机器读的汇总
└── runs/<arch>/       # 每个架构完整的训练痕迹（TrainData 结构）
```

> ⚠️ 这是**受控子集实验**，不是全量训练结论。子集越小、轮数越少，越偏向「收敛快」的模型；
> BatchNorm 的主要收益恰好是收敛速度，所以小预算下 R5 的优势可能被**高估**。
> 要看准请用默认参数以上规模重跑。

### 实测结果（80k 训练 / 15k 验证，各 2 轮，seed=2026）

| 指标（第 2 轮） | R4（19.07 M） | R5（5.02 M） | Δ（R5−R4） |
|---|---:|---:|---:|
| val_loss | **0.028405** | 0.029094 | +0.000689 |
| macro-F1 @0.5 | **0.6124** | 0.6106 | −0.0018 |
| micro-F1 @0.5 | 0.7462 | **0.7480** | +0.0018 |
| macro-AP（阈值无关） | **0.6918** | 0.6745 | **−0.0173** |
| Hamming loss | 0.1634 | **0.1589** | **−0.0046**（更好） |
| 子集准确率 | 0.1213 | **0.1287** | +0.0074 |

逐类 F1 的变化也是混合的（14 类里 R5 只有 4 类变好）：

| 类别 | R4 | R5 | Δ |
|---|---:|---:|---:|
| alkyne | 0.5949 | 0.6198 | **+0.0249** |
| nitrile | 0.2712 | 0.2944 | **+0.0232** |
| nitro | 0.2328 | 0.2525 | **+0.0197** |
| amide | 0.6696 | 0.6743 | +0.0047 |
| benzene | 0.9809 | 0.9749 | −0.0061 |
| hydroxyl | 0.4237 | 0.4151 | −0.0086 |
| halogen | 0.6252 | 0.6063 | −0.0189 |
| carboxyl | 0.2459 | 0.2088 | **−0.0371** |

变好的 3 个主要是样本少的类（support 514–977），变差的却是 benzene / carboxyl / halogen
这类占比较大的类，这与 macro 指标略降一致。完整 14 类见 `data/ABTest/manual/2ep/comparison.md`。

**结论（诚实版）**：在这个受控预算下 **R5 没有跑赢 R4**。
R5 在 macro-F1 / macro-AP / val_loss 上略输，在 micro-F1 / Hamming / 子集准确率
和 nitro、nitrile 两个稀有类上略赢，整体差异都在 0.02 以内，**很可能落在随机波动范围**。
（注意：两套架构参数量不同，同一个 seed 也**不会**让它们的初始化权重一一对应，
所以单 seed 的差异本身就有噪声；严格结论需要多 seed 平均。）

因此 **R5 没有设为默认**。要下定论，请在算力足够的机器（如 GTX 1060）上做：

```bash
# 全量 937k、训练到收敛、跑至少 3 个 seed
python main_cli.py ab-test --train 937022 --val 117035 --epochs 8 --seed 2026
python main_cli.py ab-test --train 937022 --val 117035 --epochs 8 --seed 2027
python main_cli.py ab-test --train 937022 --val 117035 --epochs 8 --seed 2028
```

在拿到全量结果之前，**用默认的 R4**，它的成绩是已经验证过的。

## 特征与模型

| 项目 | 规格 |
|---|---|
| 13C 特征 | 0–220 ppm，0.25 ppm，880 桶，多热 |
| 1H 特征 | 3 通道 × 280 桶（0–14 ppm，0.05 ppm）+ 12 个全局标量 = 852 维，float16 |
| 1H 通道 | 0=全部峰积分；1=复杂多重峰积分；2=J 加权积分 |
| 1H 标量 | 总氢数、峰数、s/d/t/q/复杂/其他占比、带 J 氢占比、平均/最大 J、每峰 J 数 |
| 标签 | 14 个官能团，顺序见 `src/setup.py: FUNCTIONAL_GROUPS` |
| 模型 | 13C 分支 28160 + 1H 分支 8960 + 标量 MLP 32 → 512 → 14，约 19.07 M 参数 |
| 损失 | FocalLoss（alpha=0.25, gamma=2.0，可选用按类别自适应 alpha） |

验证集/测试集结果（主模型 R4）：

| 数据 | 阈值 | macro-F1 | micro-F1 | 子集准确率 | Hamming |
|---|---:|---:|---:|---:|---:|
| validation | 0.50 | 0.6215 | 0.8150 | 0.3160 | 0.0898 |
| validation | 0.35 | 0.7403* | 0.8611 | 0.3939* | 0.0781* |
| test | 0.50 | 0.6192 | 0.8154 | 0.3164 | 0.0898 |
| test | 0.35 | 0.7403 | 0.8611 | 0.3939 | 0.0781 |

\* 验证集 0.35 阈值下的 macro/子集/Hamming 与测试集数值接近；0.35 是验证集 micro-F1 最优阈值。

## 移植说明

- 所有路径都由 `src/setup.py` 的 `ROOT = Path(__file__).resolve().parent.parent` 推导，**不需要改代码**，解压到任意目录即可。
- 用 `python src/train.py` 这种方式运行时，Python 会自动把 `src/` 加入 `sys.path`，因此 `from setup import ...` 这类平铺导入可以正常工作；也可以 `cd src` 后直接运行脚本。
- 数据、特征、权重、日志都不在源码包里（体积原因），按上面的命令重新生成或单独拷贝。
- 如果拿到的是带模型的包：把 `checkpoints/model_13c_v1.pth` 放到 `data/ModelData/model_13c_v1.pth`，或评估时用 `--model checkpoints/model_13c_v1.pth` 指定。
- **换机器到 CUDA 环境只需要三步**（代码零改动，因为 `TRAIN_DEVICE` 默认就是 `auto`）：
  1. 装 Python 3.10，跑 `scripts/install_cuda.ps1`（或 `install_cuda.sh`）；
  2. 把 `data/`（约 6.7 GB）拷过去，或按「数据准备」重新生成；
  3. `python main_cli.py cuda-info` 看到 `结论: CUDA 可用` 就可以直接 `python main_cli.py train`。
- 若目标机器的 CUDA 版本不是 12.x，把脚本里的 `-CudaChannel` / `CUDA_CHANNEL` 换成对应版本即可；
  但 **Pascal（MX150/GTX 1060）不能用 12.8 及以上**，只能用 cu126。

## 文档

- [CHANGELOG.md](CHANGELOG.md)：v0.1.0–v0.8.0 的版本变更，以及本 CUDA 版（0.8.0-cuda）的改动清单。
- [UPDATELOG.md](UPDATELOG.md)：完整改造过程、R1–R5 每轮配置/逐 epoch/逐类指标、产物与复现命令。
