# INSTRUCTION 项目使用说明书 - TouchNMR

> **Project Leader 项目负责人** : [博凯 杨](mailto:yangbk1226@mails.jlu.edu.cn)  
> **Author 作者** : [博凯 杨](mailto:yangbk1226@mails.jlu.edu.cn)  
> **Cooperator 合作者** :   
> **Supervisor 指导老师** :   
> **Application Version 应用程序版本** : `5.0.1`  
> **Document Version 文件版本** : `1.0.2`

这份文件主要介绍了位于这个项目中的大部分脚本的使用方法和介绍，不同于自动生成的 API 文档，这篇文章会更加详尽地讲述开发者的开发过程和改进思路，或者可以改进的方向

## `console.py` - 控制台文件 - 模型的入口

主程序在输出 `INITIALIZE TEXT` 之后，会开始初始化程序，标准情况下的程序输出如下

```text
Welcome to use TouchNMR (From ASH).
欢迎使用 TouchNMR (来自 ASH).
```

随后程序打印出

```text
Open$   
```

则主程序加载完毕，可以开始键入命令

目前可用的命令如下

### `help`

控制台语句

```console
Open$ help
```

会返回主程序的说明文本

### `features`

控制台语句

```console
Open$ features
```

可以重新创建一次 X 和 Y 的特征矩阵或者从 parquet 原数据重新计算一次数据

### `settings`

与设置有关的控制台语句

#### 直接调用

```console
Open$ settings
```

会直接打印出正在生效的配置与派生配置

#### 检查

语句执行

```console
Open$ settings --check
```

或者在 命令行 中运行

```cmd
python console.py settings --check
```

会开始拼写检查和派生字段的无效改写

> 请注意:
>> 以下参数由程序直接运算获得，即使修改也无法影响程序.
>>
>> - `model.n_classes`
>> - `model.n_features`
>> - `features.h1.total_dim`
>> - `features.h1.spec_dim`
>>
>> 以下参数在修改后需重新生成特征向量
>>
>> - `features.c13.buckets` : ¹³C 分桶数,默认值为 `880`（$\approx 0.25 \text{ppm/桶}$）
>> - `features.c13.ppm_min` & `features.c13.ppm_max` : ¹³C 的化学位移范围,默认为 $0\sim 220 ppm$
>> - `features.h1.buckets` : ¹H 分桶数,默认值为 `280`（$\approx 0.05 \text{ppm/桶}$）
>> - `features.h1.ppm_min` & `features.h1.ppm_max` : ¹H 的化学位移范围,默认为 $0\sim 14 ppm$

---

### `ab-test`

执行一次 A/B 测试（R4/R5 架构 A/B 对比）

### `cuda-info`

检查 `torch` - `CUDA` - `显卡` 工具链是否可用

### `eval`

执行 F1 评估

### `dashboard`

在指定端口上开放训练看板

### `train`

开始训练，允许传入参数

- `--epochs [n]` : 指定训练的 Epoch 数量
- `--workers [n]` : 指定训练时工作的线程数

## `setup.py` - 程序依赖控制与变量管理模块

### 创新设计

在项目初期，程序配置写于 `setup.py` 的 `ENVIRON` 中，再通过迭代写入环境变量，同时在每个模块内置 `_env_[int/tuple/float/...](TARGET_NAME,DEFAULT_VALUE)`，这种方式相当于在每一次调用数据时都需要硬编码一次参数；以及发生过漂移的 `DATALOADER_WORKERS` 会存在二值从而形成歧义

在重新修改变量读取链路后，程序实现了

```mermaid
flowchart LR
    A["settings.json"] -->|加载与校验| B["SETTINGS(setup.py)"]
    B -->|计算派生值| C["各模块"]
```

而对于命令行参数，则会在各模块脚本内调用 `SETTINGS.apply({KEY:VALUE})` 实现覆盖

### 模块介绍

#### 默认变量

##### `DEFAULTS`

在模块头存在一个 `DEFAULTS` 字典作为 `settings.json` 失效之后替代其调用

#### `DEFAULT_PATH_KEYS`

各目录的默认相对路径，当 `settings.json` 中为 `null` 时使用

#### `VALID_ARCHS`

当前有效的架构

#### `VALID_DEVICES`

可以选择的设备类型

#### `VALID_LR_SCHEDULERS`

有效的学习率方案

#### `VALID_H1_DTYPES`

有效的 ¹H 的特征数据结构

#### `H1_CHANNELS` 和 `H1_SCALARS`

描述 ¹H 的数据布局。这两个数据由编码模块 `parser.py` 实现与决定，解析函数 `parser.encode_1h_features()` 固定设计了 3 个通道，分别是

- 全部峰
- 多重峰
- J 加权

请勿尝试修改这个常量，如需修改则需要修改 parser 的实现方法

### 模块函数

#### `load_settings(settings_file, local_file)`

构建一个 `Settings` 类，可通过 `settings_file` 指定读取的文件位置

#### `reload(**overides)`

重载 `Settings`，可通过 `**overides` 传入关于 `Settings` 的配置

### `class Settings`

一个可以访问指定属性的类

#### Method 方法

##### `apply()`

依照 dotted keys 修改指定参数，例如 `{"training.batch_size": 128}` 

**就地修改**（以保持 `from setup import SETTINGS` 引用有效），改完重新派生。值的类型按字段当前类型自动转换（int/float/bool/tuple/str/list）。

## `parser.py` - 解析器

- 将 `.patquet` 文件提纯为 `.csv`
- 将 `.csv` 文件依据项目需要生成 `X_[TYPE].pt/Y_[TYPE].pt` 的 `.pt` 文件

### 创新设计

该程序主要在低配置服务器上跑大模型运算，由于这个模型的训练集共 937022 行左右，稠密特征矩阵 X 为 `[N,880]` ，初始方案为单独计算每一条的特征向量，随后通过 `torch.stack()` 堆叠。为了防止 `OOM Kill` ，本项目采用 `分块流式` 实现，无论大小的 `.csv` 文件都只会占用常数级别的内存，实现方案如下:

1. 通过 `pandas.read_csv(chunksize = ...)` 分块读入；
2. 每块编码成 `numpy` 数组之后立即追加写入临时二进制文件
3. 全部读取完成后使用 `np.memmap` 映射这一组临时文件，通过 `torch.from_numpy()` 得到按页读取的 `torch.Tensor` ，随后通过 `torch.save()` 保存
4. 删除临时文件

达成峰值内存仅占用可自定义化的 `chunk` 大小，默认为 70 MiB 与 `pandas` 解析开销

### 可改进点

1. `_bucket_indices()` 直接将越界值丢弃，部分计算情况（如 ¹H 的化学位移 <0 和 >=15 的情况并未考虑，可以通过 BatchNorm 将极端值作**归一化**处理，而非直接丢弃；或者考虑

### 模块函数

#### `_string_to_Path(*target)`

将以字符串形式传入的路径全部转换为 `pathlib.Path` 对象

#### `parse_parquet_file()`

载入 `.parquet` 文件并解析成 `.csv` 文件

#### `get_FG_labels`

传入 smiles 分子式，返回关于官能团的多热向量（对外接口保持不变）(从 v0.2.3 开始不再主动调用)

#### `gen_1h_pt`

从 `.csv` 文件的 `nmr_1h` 列生成 1H 特征，保存为 `X1H_[name].pt`

#### `gen_XY_pt()`

从 `[csv_folder]` 中读取 `[name].csv` ，分析完成后保存至 `[pt_folder] 的 [name]_[X/Y].pt`

## `cuda_info.py` - CUDA 训练

用 CUDA 进行训练的支持性程序

### 用处

1. 打印 torch / CUDA / cuDNN 版本；
2. 查 nvidia-smi 拿到显卡型号、驱动版本、驱动支持的 CUDA 版本；
3. CUDA 不可用时明确区分原因，并给出修复命令；
4. CUDA 可用时通过 GPU 进行一次矩阵乘测试。

### 注意事项

脚本只依赖 torch 和标准库，不导入 src/setup.py，
因此即使 rdkit 等其他依赖还没装好，也可以先用它确认显卡是否能进行模型训练

## `train_archive.py` - 训练数据管理

训练痕迹归档程序

把每一次训练的「完整痕迹」写进 `data/TrainData`，事后可以逐轮追查：

```text
data/TrainData/
├── runs_index.csv                      所有运行各一行
└── run_20261020-140000/                每次训练一个独立目录
    ├── run.json                        本次训练的完整配置与结束汇总
    ├── epochs.csv                      逐 epoch 一行，每行写完立即保存
    ├── epoch_0001.json                 该 epoch 的完整记录（含逐类指标）
    ├── epoch_0002.json
    └── checkpoints/
        └── epoch_0002_val0.118900.pth  归档权重
```

### 创新设计

1. [原子写] `.json` 文件 : `_write_json` 先完成 `.tmp` 编写之后替换，避免中途打断；
2. `.csv` 的追加式原子读写 : `_append_csv_row` 先追加一行 `CSV` （不存在则先完成表头），随后立刻关闭

### `class TrainRun`

一次训练的痕迹归档器

#### 用法

```python
run = TrainRun(SETTINGS.paths.train_data_dir, make_run_id())
run.begin(config_dict)
run.record_epoch(epoch, row, detail)
run.save_checkpoint(model.state_dict(), epoch, val_loss)
run.finish(summary_dict)
```

- `begin()` : 记录的开始
- `record_epoch()` : 记录一次 Epoch 的完成，汇总一行写入 epochs.csv，完整明细写入 epoch_XXXX.json。
  - `row`     —— 扁平的汇总字段（对应 EPOCH_FIELDS）
  - `detail`  —— 任意可 JSON 化的完整明细（逐类指标、配置等）
- `save_checkpoint()` : 记录一次检查点权重
- `finish()` : 记录的完成
- `checkpoint_name()` : 生成一个检查点的权重文件名称(带 .pth 后缀)

## `f1_test.py` - F1 / 多标签分类指标评估

对 14 个官能团的多标签分类结果进行评估，输出：

- 每个类别的 precision / recall / F1 / AP / TP·FP·FN·TN / 支持样本数；
- 整体 macro-F1、micro-F1、weighted-F1、子集准确率、Hamming loss、macro 平均精度（AP）；
- 阈值扫描结果，包括 macro-F1 最优阈值及每个类别各自的最优阈值

结果追加写入 data/LogData/ 下的：

- f1_by_class.csv
- f1_threshold_sweep.csv
- f1_summary.jsonl

## `log_utils` - 日志

`train.py` 和 `f1_test.py` 共用 `open_csv_logger`, `append_jsonl`, `make_run_id` 以保证格式一致性

- `open_csv_logger(log_path, fields)` : 打开（必要时创建）CSV 日志，返回 (文件句柄, csv.DictWriter)。调用方负责在结束时 close() 句柄；每条记录写完后建议 flush()。

## `model.py` - 模型文件

### 框架更新

- `"r4"`：0.8.0 原架构，19.07 M 参数。**行为与原版逐字节一致**，用来当基线；0.8.0 训练出来的权重可以原样加载。
- `"r5"`：优化架构，约 5.02 M 参数。三处改动：
  1. 每个卷积后加 BatchNorm1d；
  2. 卷积主干后接 1×1 通道瓶颈再 flatten（分类头参数量 19.03 M → 4.97 M）；
  3. 展平的位置特征之外并联一路全局 max/avg 池化上下文。

| arch | 模型架构 |
| :---: | :-------: |
| c13.kernel_sizes / c13.channels | 13C 分支 3 层卷积 |
| h1.kernel_sizes / h1.out_channels | 1H 分支 2 层卷积 |
| h1.scalar_hidden | 12 个全局标量的 MLP 隐层 |
| r5.head_channels / r5.head_h1_channels | 仅 R5：1×1 瓶颈宽度 |
| head.hidden / head.dropout | 分类头|

### 注意事项

构造函数参数仅用于「按已有权重重建模型」（f1_test 从 checkpoint 里的 buffer恢复输入维度与架构时传入），不传则一律取 SETTINGS。

### `class NMRClassifier(torch.nn.Module)`

13C 多热特征 + 可选 1H 特征（3 通道直方图 + 全局标量）-> 官能团多标签分类

#### 输入

拼接向量:

```text
[x_13C (n_features) | x_1H 直方图 (bins * channels) | x_1H 标量 (scalars)]
```

#### 输出

(batch, n_classes) 的 logits，n_classes = settings.json 里 labels 的条目数。

## `train.py` - 训练脚本

### 注意事项

1. 每次运行的指标都会追加到 data/LogData/ 下：
   - train_history.csv   每个 epoch 一行（run_id / lr / batch / epoch / train_loss /val_loss / best_val_loss / 耗时 ...），所有运行共用一张表，用 run_id 区分，方便直接用 pandas 画曲线对比；
   - run_summary.jsonl   每次运行结束时追加一条 JSON 汇总（配置、最佳 epoch、最佳验证 loss、总耗时、停止原因）。
2. 模型权重：
   - model_13c_v1.pth     验证 loss 创新低时保存（最佳模型）；
   - model_13c_last.pth   每个 epoch 结束都保存（断点/最后一轮）。
3. 训练痕迹归档（CNMR-CUDA 版新增，细节见 src/train_archive.py）：每次训练在 data/TrainData/run_<时间>/ 下留完整痕迹——epochs.csv（逐轮一行）、epoch_XXXX.json（逐轮完整明细 + 逐类指标）、checkpoints/（归档权重，默认只存验证集变好的轮次）、run.json（配置与汇总）。

### 函数说明

#### `_get_device()`

解析训练设备

在`SETTINGS.device` 中 ：

| 值 | 采取措施 |
| :--- | :--- |
| "auto" | 自动（默认有可用的 CUDA 就用显卡，否则自动退回 CPU） |
| "cuda" / "cuda:0" | 强制使用显卡；不可用时抛出异常 |
| "cpu" | 强制使用CPU |

#### `_write_status(**fields)`

把训练进度写入 data/LogData/train_status.json 中供看板/外部程序实时读取。

#### `train_one_epoch()`

训练一个 epoch。返回 `([平均 loss],[是否被 Ctrl+C 中断])`，中断时返回已经跑完的 batch 的平均 loss。

`on_batch(batch_index, running_loss)` 每 `100` 个 batch 回调一次，用于把进度写入 `train_status.json` 供看板实时读取。tqdm 进度条设置 `leave=False`，多个 epoch 连续训练时不会在终端留下大量残留行。

#### `evaluate()`

评估一个 epoch。返回 `(平均 loss, 是否被 Ctrl+C 中断, 概率, 标签)`。

`collect=False` 时后两项为 `None`，不额外占内存，也不想算逐类指标时使用。`collect=True` 时把每个 batch 的 `sigmoid(logits)` 与标签转成 `float().cpu()` 后累积，最后 `torch.cat` 拼成整轮结果，避免在显存里堆积。

#### `compute_split_metrics()`

从一轮验证的预测概率算出汇总指标与逐类指标，供训练痕迹归档使用。

汇总指标包含 `macro_f1` / `micro_f1` / `hamming_loss` / `subset_accuracy` / `macro_ap`。逐类指标按 `FUNCTIONAL_GROUPS` 的顺序展开，包含 support / tp / fp / fn / precision / recall / f1。

`macro_ap` 依赖 `sklearn.metrics.average_precision_score`，缺失时置为 `None` 而不抛异常，保证训练流程不因可选依赖缺失而中断。

#### `_set_seed()`

固定 `random` / `numpy` / `torch` / `torch.cuda` 的随机种子。0.9.0 起才加：做架构 A/B 对比时，初始化权重与 shuffle 顺序必须可复现，否则两个架构的差异里会混进随机波动。

`numpy` 的导入包在 `try` 里，缺失时跳过，不影响其余随机源。

#### `parse_args()`

命令行参数全部默认 `None`，在 `main()` 里通过 `SETTINGS.apply()` 统一覆盖。默认值 `None` 表示“不覆盖”，此时沿用 `settings.json` 的配置，而不是把 `argparse` 的默认值硬编码进程序。

可传参数：

- `--epochs` : 训练轮数，`0` 表示一直训练到 Ctrl+C
- `--batch-size` : batch 大小
- `--lr` : 学习率
- `--init-from` : 用旧权重的 13C 卷积分支做热启动（只加载 `conv.*` 参数）
- `--workers` : DataLoader 预取进程数
- `--device` : `auto` / `cuda` / `cuda:0` / `cpu`
- `--arch` : `r4` / `r5`
- `--seed` : 随机种子

### `class FocalLoss`

多标签二分类的 `Focal Loss`。对每个标签独立计算 `sigmoid 交叉熵`，再乘 $(1 - p_t)^\gamma$ 与 $\alpha _t$。

$\gamma$ 越大，模型越关注难分样本，对 14 个官能团这种稀有正例更友好。$\alpha$ 用于平衡正负样本，可以是标量（所有类别共用），也可以是长度等于类别数的向量。

 $\alpha _\text{pos}$ 传入向量时按类别逐元素构造  $\alpha _t$，实现“按类别正例率自适应，稀有类权重更高”。`reduction` 支持 `mean` / `sum` / `none`，与 `torch.nn` 的损失函数保持一致的接口习惯。

### `main()`

#### 参数覆盖与随机种子

`SETTINGS.apply()` 将命令行参数写回配置，再读 `SETTINGS.training.seed` 调 `_set_seed()`。顺序不能反：`seed` 本身也可能由命令行覆盖。

#### 数据加载

`pin_memory` 只在 `device.type == "cuda"` 时开启，配合 `train_one_epoch` 里的 `non_blocking=True` 做异步传输。CPU 训练时开 `pin_memory` 没有收益，还会多占用一份锁页内存。

`mmap=SETTINGS.data.mmap` 透传给 `get_dataloader`，大特征矩阵按页读取，避免一次性将整个 `.pt` 读进内存。

#### 训练痕迹归档

`TrainRun` 在训练开始前 `begin()`，把配置、设备、特征维度、seed 等一次性写进 `run.json`。

每个 epoch 结束调 `record_epoch()`：扁平字段进 `epochs.csv`，完整明细（含逐类指标）进 `epoch_XXXX.json`。权重默认只在验证集变好时归档，`SAVE_EVERY_EPOCH=1` 时每轮都存。

`archive_partial()` 处理 Ctrl+C：训练或验证中途被打断时，把当时权重按 `tag` 归档，并写一条 `interrupted=True` 的痕迹。这样即使没有跑完一个 epoch，也能从 `TrainData` 里查到中断发生在哪个阶段、当时 loss 是多少。

#### 状态文件

`_write_status()` 写 `train_status.json`，先读旧内容再 `update`，最后 `.tmp` 原子替换。看板与外部程序读这个文件获取实时进度，字段包括 `state` / `epoch` / `batch` / `train_loss` / `val_loss` / `best_val_loss` / `stopped_reason`。

`state` 取值：`running` / `finished` / `stopped`。`finished` 表示按 `max_epochs` 正常结束，`stopped` 表示被 Ctrl+C 打断。

#### 中断处理

训练与验证各自捕获 `KeyboardInterrupt`，分别走 `train_interrupted` 与 `val_interrupted` 两条归档路径，保证“中断在哪一阶段”可追溯。

`finally` 块里再兜底一次：关闭 CSV 句柄、写最终状态、追加 `run_summary.jsonl`、调 `archive.finish()` 写 `runs_index.csv`。即使 Ctrl+C 恰好落在保存与写日志之间，已经写完的逐 epoch 记录不会丢。

#### 返回值

`main()` 返回 `summary` 字典，与写入 `run_summary.jsonl` 的内容一致，方便在 `console.py` 里直接拿返回值做后续处理，而不必再读一遍日志文件。