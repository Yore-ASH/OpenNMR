# UPDATELOG

> **CNMR 官能团预测项目 — 改造说明书与实验记录**
> 生成时间：2026-10-19 · 评估集：validation（117,035 条）
> 每轮的原始数据都能在 `data/LogData/` 和 `logs/` 里找到，本文末尾给出了索引。

## 0. 速览

| 项目 | 当前状态 |
|---|---|
| 任务 | SMILES / 13C NMR / 1H NMR → 14 个官能团多标签分类 |
| 训练集 / 验证集 / 测试集 | 937,022 / 117,035 / 117,205 |
| 当前最佳模型 | R4：13C + 1H（多重峰/J）；R5 为高召回变体 |
| 验证集 macro-F1（0.5 / 0.30） | **0.6215 / 0.7440** |
| 验证集 micro-F1（0.5 / 0.35） | **0.8150 / 0.8611** |
| 测试集 macro / micro（0.5） | **0.6192 / 0.8154** |
| 测试集 macro / micro（0.35，验证集选阈值） | **0.7403 / 0.8611** |
| 权重文件 | `data/ModelData/model_13c_v1.pth`（73 MB，R4） |
| 训练硬件 | 4 vCPU / 3.7 GiB RAM / 无 GPU，单 epoch 约 53–55 分钟 |

## 1. 背景与目标

项目最初只有一条 13C NMR 分桶特征 + 一个小卷积网络。实际跑起来遇到两个核心问题：

1. **内存**：`train` 937k 行、880 桶的稠密矩阵约 3.1 GiB，旧实现还要在 `list` 里再存一份，
   峰值约 6.7 GiB，而机器只有 3.7 GiB，必然 OOM。
2. **效果**：早期模型 macro-F1 只有 0.22，`hydroxyl`/`carboxyl`/`nitro` 等稀有类 F1 为 0。

这次改造沿着「先让流程稳定可观测，再逐步加信息」的顺序推进：
流式数据管线 → 训练/日志/看板 → 评估体系 → 模型结构 + Focal → 1H 积分 → 多重峰/J。

## 2. 运行环境

| 项 | 版本 / 配置 |
|---|---|
| Python | 3.10.12（项目 venv） |
| PyTorch | 2.14.1+cpu |
| NumPy / pandas / pyarrow | 2.2.6 / 2.3.3 / 25.0.1 |
| RDKit | 2026.09.1 |
| scikit-learn / SciPy | 1.7.2 / 1.15.3 |
| matplotlib / Flask | 3.10.9 / 3.1.3 |
| CPU / 内存 | 4 vCPU / 3.7 GiB，无 CUDA |
| 磁盘 | 40 GB ext4（写本文时剩余约 3.1 GB） |
| 看板 | http://127.0.0.1:8090 （Flask，长驻后台） |

## 3. 修改过程

### 3.1 阶段一：先解决 OOM，把数据管线做稳

- **根因定位**：`gen_XY_pt` 把每行编码结果 `append` 到 Python list，最后 `torch.stack`；
  `train` 需要 937k 个小 Tensor（约 3.6 GiB）+ stack 复制（3.1 GiB），峰值 6.7 GiB。
- **改法**：`pd.read_csv(chunksize=20000)` 分块读取 → 每块 numpy 编码 → 追加写临时二进制文件 →
  最后 `np.memmap` + `torch.save`。峰值降到约 0.7 GiB，且输出与旧实现**逐位一致**。
- `parse_parquet_file` 改为逐个文件写盘、只读需要的列，不再把 DataFrame 攒在内存。
- `dataset.py` 改用 `torch.load(mmap=True)`，训练时按 batch 从磁盘读页，不再整表进内存。
- **验证**：对 `test` 重新生成 X/Y，与旧实现 `torch.equal` 比较，X、Y 均完全一致；
  `train` 全量转换耗时 3 分 48 秒，`MemAvailable` 全程稳定在约 2.5 GiB。

### 3.2 阶段二：训练可长期运行、可中断、可观测

- 训练默认无限进行，`Ctrl+C` 时先保存 `model_13c_last.pth`、写日志、写汇总再退出；
- 每个 epoch 追加 `data/LogData/train_history.csv`（每行立即 flush）；每次运行追加 `run_summary.jsonl`；
- 每 100 个 batch 更新 `train_status.json`，供看板/外部程序实时读取；
- 新增 Flask 看板：实时进度、loss 曲线、逐 epoch 表、F1 报告、开始/停止训练、F1 评估按钮。

### 3.3 阶段三：修正并增强评估脚本

- 原 `f1_test.py` 有三个致命问题：数据文件名写错（`X_val_13c.pt`）、
  模型路径写错（`data/PTFiles/model_13c_v1.pth`）、**官能团名字漏了 `nitro` 多了 `thiol`**，
  导致第 7 类之后标签整体错位。现在类别名直接从 `FUNCTIONAL_GROUPS` 读取。
- 指标扩展为：每类 P/R/F1/AP/TP·FP·FN·TN/support、macro/micro/weighted F1、
  子集准确率、Hamming loss、macro AP；额外做 0.05–0.95 阈值扫描与每类最佳阈值。
- 增加「预测正例率」告警，避免“几乎全判正类”带来的虚假 macro-F1。
- 补齐依赖并修正 `requirements.txt`（原文件是 UTF-16，且有几个版本号在镜像上不存在）。

### 3.4 阶段四：模型 v2 + Focal Loss

- 去掉 `AdaptiveAvgPool1d(1)`，保留两个 `MaxPool1d(2)`：880→440→220，
  Flatten 维度变成 `128 × 220 = 28160`，再接 `Linear(28160, 512)`。
- 用 FocalLoss 替换 BCE：`alpha=0.25`、`gamma=2.0`，缓解稀有正例被淹没的问题。
- 加 `N_FEATURES` / `CLASSES_SUM` 到 `setup.py`，修复模型输出 15 类、标签 14 类的形状不匹配。

### 3.5 阶段五：加入 1H 积分（R3）

- 原始 parquet 的 `nmr_1h` 每个峰为 `[化学位移, 0.0, 多重峰, 积分, 耦合常数]`，
  积分就是第 4 个字段（如 `"2H"`）。抽样 3 万行：积分总和与分子式氢数 93.4% 完全相等。
- 编码为 0–14 ppm、0.05 ppm、280 桶的积分直方图（按总氢数归一化）；
  生成 `X1H_train/validation/test.pt`，与 `gen_XY_pt` 使用相同的行顺序和跳过规则。
- 模型改双分支；`train.py` 加 `--init-from` 复用旧 13C 卷积；
  DataLoader 加 `--workers`（默认 2），训练从 3.16 it/s 提升到 4.55 it/s。

### 3.6 阶段六：加入多重峰与耦合常数 J（R4）

1H 特征从 280 维升级为 **3 通道 × 280 桶 + 12 个全局标量 = 852 维**：

| 通道 | 含义 |
|---|---|
| 0 | 所有峰的积分直方图（0–14 ppm，0.05 ppm，按总氢数归一化） |
| 1 | 只统计复杂多重峰（`m` / `dd` / `dt` / `ddd` …，不含 `s/d/t/q`）的积分直方图 |
| 2 | J 加权积分直方图 `sum(积分 × 该峰平均 J)`，再归一化 |

12 个标量：总氢数、峰数、`s`/`d`/`t`/`q`/复杂/其他 的积分占比、
带 J 的氢占比、按氢加权的平均 J、最大 J、每峰平均 J 个数。

- 存储用 float16（train 约 1.5 GB），并写 `X1H_meta.json` 描述特征布局；
- 模型在 1H 卷积之外再加一个 `Linear(12, 32)` 标量 MLP 分支；
- 仍从 R3 热启动 13C 卷积，训练 3 个 epoch。

## 4. 数据与特征规格

### 4.1 数据规模

| 拆分 | 行数 | X（13C） | X1H（v2） | Y |
|---|---|---|---|---|
| train | 937,022 | (937022, 880) 3.1 GB | (937022, 852) float16 1.5 GB | (937022, 14) |
| validation | 117,035 | (117035, 880) 411 MB | (117035, 852) float16 191 MB | (117035, 14) |
| test | 117,205 | (117205, 880) 412 MB | (117205, 852) float16 191 MB | (117205, 14) |

### 4.2 13C 特征

化学位移 0–220 ppm、0.25 ppm 一个桶，共 880 桶；每个位移落桶置 1（多热），越界值丢弃。

### 4.3 1H 特征（v2，当前版本）

3 通道 × 280 桶（0–14 ppm，0.05 ppm）+ 12 个标量，float16 存储；
布局记录在 `data/PTFiles/X1H_meta.json`：

| 字段 | 值 |
|---|---|
| bins | 280 |
| channels | 3 |
| scalars | 12 |
| ppm_min / ppm_max | 0.0 / 14.0 |
| dtype | float16 |
| total_dim | 852 |

### 4.4 标签

14 个官能团，顺序与 `FUNCTIONAL_GROUPS` 一致：

`benzene`, `carbonyl`, `hydroxyl`, `carboxyl`, `ester`, `amine`, `nitro`, `nitrile`, `alkene`, `alkyne`, `ether`, `amide`, `halogen`, `sulfonyl`。

验证集各类正例数：

| class | support | 占比 |
|---|---|---|
| benzene | 105,267 | 89.94% |
| carbonyl | 66,561 | 56.87% |
| hydroxyl | 16,889 | 14.43% |
| carboxyl | 1,892 | 1.62% |
| ester | 30,138 | 25.75% |
| amine | 49,574 | 42.36% |
| nitro | 4,296 | 3.67% |
| nitrile | 6,554 | 5.60% |
| alkene | 35,962 | 30.73% |
| alkyne | 7,934 | 6.78% |
| ether | 56,631 | 48.39% |
| amide | 25,339 | 21.65% |
| halogen | 40,764 | 34.83% |
| sulfonyl | 12,921 | 11.04% |

## 5. 实验记录

### R1 — 13C + BCE + 旧结构

| 配置项 | 值 |
|---|---|
| run_id | 20261018-195011 |
| 特征 | 13C 多热 880 桶（0–220 ppm，0.25 ppm） |
| 模型 | 旧结构：MaxPool×2 + AdaptiveAvgPool1d(1)，FC 128→64→14 |
| 参数量 / 权重大小 | 约 0.04 M / 178 KB |
| 损失函数 | BCEWithLogitsLoss |
| 优化器 | Adam, lr=1e-3 |
| batch size | 64 |
| dropout | 0.2 |
| 训练轮数 | 8 个完成 + 第 9 个中断（Ctrl+C） |
| 热启动 | 无 |
| 权重文件 | `data/ModelData/model_13c_v1_bce_oldarch.pth` |
| 总耗时 | 157.1 分钟 |
| 日志 | `logs/gen_XY_train.log / logs/f1_validation.log` |
| 备注 | 早期基线，只用了 13C；稀有类几乎全部预测为 0。 |

**逐 epoch 记录**

| epoch | train loss | val loss | best val loss | 耗时(分钟) | 中断 |
|---|---|---|---|---|---|
| 1 | 0.42449 | 0.41773 | 0.41773 | 16.1 |  |
| 2 | 0.41924 | 0.41649 | 0.41649 | 16.4 |  |
| 3 | 0.41804 | 0.41606 | 0.41606 | 16.5 |  |
| 4 | 0.41753 | 0.41790 | 0.41606 | 16.4 |  |
| 5 | 0.41731 | 0.41621 | 0.41606 | 34.4 |  |
| 6 | 0.41712 | 0.41550 | 0.41550 | 15.6 |  |
| 7 | 0.41679 | 0.41497 | 0.41497 | 16.4 |  |
| 8 | 0.41654 | 0.41495 | 0.41495 | 25.2 |  |
| 9 | 0.41938 | —（中断） | 0.41495 | 0.1 | 是 |

**F1 评估（阈值 0.5）**

| F1 run_id | 阈值 | macro-F1 | micro-F1 | weighted-F1 | 子集准确率 | Hamming | macro AP | 最佳阈值 / macro-F1 | micro 最佳阈值 / F1 |
|---|---|---|---|---|---|---|---|---|---|
| 20261018-211658 | 0.50 | 0.2188 | 0.5832 | 0.4719 | 0.0414 | 0.1931 | 0.3654 | 0.05 / 0.3913 | 0.35 / 0.5003 |
| 20261018-221400 | 0.50 | 0.2184 | 0.5823 | 0.4711 | 0.0416 | 0.1928 | 0.3692 | 0.05 / 0.3938 | 0.35 / 0.5028 |

> 注：两次评估分别对应 epoch 5 和 epoch 7 的检查点；epoch 8 的最终权重在切换模型结构前没有单独评估。

**每类指标（F1 @ 0.5 / best F1）**

| class | 20261018-211658（F1 / best） | 20261018-221400（F1 / best） |
|---|---|---|
| benzene | 0.9496 / 0.9496 | 0.9497 / 0.9497 |
| carbonyl | 0.7329 / 0.7432 | 0.7326 / 0.7438 |
| hydroxyl | 0.0000 / 0.2601 | 0.0000 / 0.2586 |
| carboxyl | 0.0000 / 0.0000 | 0.0000 / 0.0222 |
| ester | 0.1856 / 0.4941 | 0.1918 / 0.4947 |
| amine | 0.4059 / 0.6011 | 0.4067 / 0.6025 |
| nitro | 0.0000 / 0.0768 | 0.0000 / 0.0795 |
| nitrile | 0.0000 / 0.1106 | 0.0000 / 0.1118 |
| alkene | 0.0008 / 0.4758 | 0.0020 / 0.4774 |
| alkyne | 0.0000 / 0.1724 | 0.0003 / 0.1899 |
| ether | 0.6799 / 0.7024 | 0.6777 / 0.7031 |
| amide | 0.0002 / 0.3939 | 0.0020 / 0.3951 |
| halogen | 0.1082 / 0.5187 | 0.0951 / 0.5192 |
| sulfonyl | 0.0000 / 0.2248 | 0.0000 / 0.2266 |

---

### R2 — 13C + Focal + 新结构

| 配置项 | 值 |
|---|---|
| run_id | 20261018-223211 |
| 特征 | 13C 多热 880 桶 |
| 模型 | 去全局平均池化；两个 MaxPool 后 Flatten 28160 → Linear(512) → 14 |
| 参数量 / 权重大小 | 14.46 M / 56 MB |
| 损失函数 | FocalLoss(alpha=0.25, gamma=2.0) |
| 优化器 | Adam, lr=1e-3 |
| batch size | 64 |
| dropout | 0.2 |
| 训练轮数 | 3 |
| 热启动 | 无（从头训练） |
| 权重文件 | `data/ModelData/model_13c_v1_13c_focal.pth` |
| 总耗时 | 134.3 分钟 |
| 日志 | `logs/train_focal_3ep.log / logs/f1_focal_3ep.log` |
| 备注 | 结构 + Focal 让所有 14 个类第一次都有了非零 F1。 |

**逐 epoch 记录**

| epoch | train loss | val loss | best val loss | 耗时(分钟) | 中断 |
|---|---|---|---|---|---|
| 1 | 0.02876 | 0.02665 | 0.02665 | 44.9 |  |
| 2 | 0.02697 | 0.02633 | 0.02633 | 44.6 |  |
| 3 | 0.02620 | 0.02575 | 0.02575 | 44.8 |  |

**F1 评估（阈值 0.5）**

| F1 run_id | 阈值 | macro-F1 | micro-F1 | weighted-F1 | 子集准确率 | Hamming | macro AP | 最佳阈值 / macro-F1 | micro 最佳阈值 / F1 |
|---|---|---|---|---|---|---|---|---|---|
| 20261019-004639 | 0.50 | 0.4600 | 0.7323 | 0.6638 | 0.1894 | 0.1220 | 0.7028 | 0.30 / 0.6480 | 0.35 / 0.7772 |

**每类指标（F1 @ 0.5 / best F1）**

| class | 20261019-004639（F1 / best） |
|---|---|
| benzene | 0.9832 / 0.9843 |
| carbonyl | 0.9185 / 0.9359 |
| hydroxyl | 0.0564 / 0.4562 |
| carboxyl | 0.0505 / 0.3733 |
| ester | 0.8048 / 0.8515 |
| amine | 0.5712 / 0.7420 |
| nitro | 0.0352 / 0.3313 |
| nitrile | 0.1366 / 0.4303 |
| alkene | 0.1513 / 0.5713 |
| alkyne | 0.6499 / 0.7872 |
| ether | 0.8341 / 0.8772 |
| amide | 0.5777 / 0.7241 |
| halogen | 0.1942 / 0.5988 |
| sulfonyl | 0.4761 / 0.5937 |

---

### R3 — 13C + 1H 积分（280 维）

| 配置项 | 值 |
|---|---|
| run_id | 20261019-075843 |
| 特征 | 13C 880 + 1H 积分直方图 280（0.05 ppm，float32） |
| 模型 | 双分支：13C 28160 + 1H 8960 → 37120 → 512 → 14 |
| 参数量 / 权重大小 | 19.05 M / 73 MB |
| 损失函数 | FocalLoss(alpha=0.25, gamma=2.0) |
| 优化器 | Adam, lr=1e-3 |
| batch size | 64 |
| dropout | 0.2 |
| 训练轮数 | 3 |
| 热启动 | 13C 卷积 from R2 |
| 权重文件 | `data/ModelData/model_13c_v1_1h_v1.pth` |
| 总耗时 | 162.3 分钟 |
| 日志 | `logs/gen_1h.log / logs/train_1h_3ep.log / logs/f1_1h_3ep.log` |
| 备注 | 1H 积分显著提升稀有类（OH/COOH/NO2 等）。 |

**逐 epoch 记录**

| epoch | train loss | val loss | best val loss | 耗时(分钟) | 中断 |
|---|---|---|---|---|---|
| 1 | 0.02524 | 0.02239 | 0.02239 | 53.5 |  |
| 2 | 0.02257 | 0.02142 | 0.02142 | 54.2 |  |
| 3 | 0.02146 | 0.02110 | 0.02110 | 54.6 |  |

**F1 评估（阈值 0.5）**

| F1 run_id | 阈值 | macro-F1 | micro-F1 | weighted-F1 | 子集准确率 | Hamming | macro AP | 最佳阈值 / macro-F1 | micro 最佳阈值 / F1 |
|---|---|---|---|---|---|---|---|---|---|
| 20261019-104119 | 0.50 | 0.5788 | 0.7872 | 0.7460 | 0.2675 | 0.1006 | 0.7891 | 0.35 / 0.7264 | 0.35 / 0.8466 |

**每类指标（F1 @ 0.5 / best F1）**

| class | 20261019-104119（F1 / best） |
|---|---|
| benzene | 0.9896 / 0.9909 |
| carbonyl | 0.9264 / 0.9450 |
| hydroxyl | 0.2134 / 0.5443 |
| carboxyl | 0.2156 / 0.4729 |
| ester | 0.8426 / 0.8928 |
| amine | 0.6675 / 0.7978 |
| nitro | 0.2757 / 0.4757 |
| nitrile | 0.2540 / 0.5415 |
| alkene | 0.6290 / 0.7618 |
| alkyne | 0.7634 / 0.8633 |
| ether | 0.8718 / 0.9088 |
| amide | 0.6211 / 0.7832 |
| halogen | 0.2867 / 0.6393 |
| sulfonyl | 0.5465 / 0.6705 |

---

### R4 — 13C + 1H 多重峰/J（852 维）

| 配置项 | 值 |
|---|---|
| run_id | 20261019-131226 |
| 特征 | 13C 880 + 1H 852（3×280 通道 + 12 标量，float16） |
| 模型 | 三路：13C 28160 + 1H 8960 + 标量 MLP 32 → 37152 → 512 → 14 |
| 参数量 / 权重大小 | 19.07 M / 73 MB |
| 损失函数 | FocalLoss(alpha=0.25, gamma=2.0) |
| 优化器 | Adam, lr=1e-3 |
| batch size | 64 |
| dropout | 0.2 |
| 训练轮数 | 3 |
| 热启动 | 13C 卷积 from R3 |
| 权重文件 | `data/ModelData/model_13c_v1.pth（当前最佳）` |
| 总耗时 | 161.3 分钟 |
| 日志 | `logs/gen_1h_v2.log / logs/train_mh_3ep.log / logs/f1_mh_3ep.log` |
| 备注 | 多重峰类型 + J 信息；macro-F1 与稀有类继续提升，carboxyl 略降。 |

**逐 epoch 记录**

| epoch | train loss | val loss | best val loss | 耗时(分钟) | 中断 |
|---|---|---|---|---|---|
| 1 | 0.02417 | 0.02106 | 0.02106 | 53.3 |  |
| 2 | 0.02118 | 0.02006 | 0.02006 | 53.9 |  |
| 3 | 0.01995 | 0.01963 | 0.01963 | 54.0 |  |

**F1 评估（阈值 0.5）**

| F1 run_id | 阈值 | macro-F1 | micro-F1 | weighted-F1 | 子集准确率 | Hamming | macro AP | 最佳阈值 / macro-F1 | micro 最佳阈值 / F1 |
|---|---|---|---|---|---|---|---|---|---|
| 20261019-155403 | 0.50 | 0.6215 | 0.8150 | 0.7816 | 0.3160 | 0.0898 | 0.8107 | 0.30 / 0.7440 | 0.35 / 0.8439 |

**每类指标（F1 @ 0.5 / best F1）**

| class | 20261019-155403（F1 / best） |
|---|---|
| benzene | 0.9912 / 0.9922 |
| carbonyl | 0.9333 / 0.9469 |
| hydroxyl | 0.2443 / 0.5873 |
| carboxyl | 0.1907 / 0.5005 |
| ester | 0.8807 / 0.9041 |
| amine | 0.7141 / 0.8158 |
| nitro | 0.3498 / 0.5175 |
| nitrile | 0.2887 / 0.5505 |
| alkene | 0.7207 / 0.7977 |
| alkyne | 0.8465 / 0.8795 |
| ether | 0.8902 / 0.9155 |
| amide | 0.6917 / 0.7988 |
| halogen | 0.3900 / 0.6707 |
| sulfonyl | 0.5696 / 0.6898 |

---

### R5 — 按类别 Focal 微调（优化尝试）

| 配置项 | 值 |
|---|---|
| run_id | 20261019-193209 |
| 特征 | 同 R4（13C 880 + 1H 852） |
| 模型 | 同 R4（19.07 M） |
| 损失函数 | FocalLoss + 按类别自适应 alpha（1 − 正例率，clamp 0.05–0.95） |
| 学习率 | 3e-4，ReduceLROnPlateau(factor=0.5, patience=0) |
| 训练轮数 | 3（从 R4 完整热启动） |
| 总耗时 | 163.2 分钟 |
| 权重文件 | `data/ModelData/model_13c_v1_highrecall.pth`（高召回变体） |
| 备注 | 目的是提高稀有类召回；结果主要是工作点平移，排序能力（AP）基本不变。 |

**逐 epoch 记录**

| epoch | train loss | val loss | lr |
|---|---|---|---|
| 1 | 0.02056 | 0.02139 | 0.0003 |
| 2 | 0.01962 | 0.02143 | 0.0003 |
| 3 | 0.01878 | 0.02153 | 0.00015 |

**F1 评估（验证集，阈值 0.5）**

| run_id | macro-F1 | micro-F1 | weighted-F1 | 子集准确率 | Hamming | macro AP | 最佳阈值 / macro-F1 | micro 最佳阈值 / F1 |
|---|---|---|---|---|---|---|---|---|
| 20261019-221538 | 0.7123 | 0.8325 | 0.8518 | 0.2982 | 0.1002 | 0.8127 | 0.60 / 0.7346 | 0.55 / 0.8465 |

**每类指标（F1 @ 0.5 / best F1）**

| class | support | F1@0.5 | best F1 | best 阈值 |
|---|---:|---:|---:|---:|
| benzene | 105267 | 0.9857 | 0.9924 | 0.30 |
| carbonyl | 66561 | 0.9488 | 0.9492 | 0.45 |
| hydroxyl | 16889 | 0.5657 | 0.6030 | 0.55 |
| carboxyl | 1892 | 0.3778 | 0.5059 | 0.65 |
| ester | 30138 | 0.9014 | 0.9094 | 0.55 |
| amine | 49574 | 0.8247 | 0.8247 | 0.50 |
| nitro | 4296 | 0.3572 | 0.5134 | 0.65 |
| nitrile | 6554 | 0.4101 | 0.5468 | 0.65 |
| alkene | 35962 | 0.8042 | 0.8082 | 0.55 |
| alkyne | 7934 | 0.7952 | 0.8785 | 0.65 |
| ether | 56631 | 0.9199 | 0.9199 | 0.50 |
| amide | 25339 | 0.7992 | 0.8095 | 0.55 |
| halogen | 40764 | 0.6768 | 0.6768 | 0.50 |
| sulfonyl | 12921 | 0.6054 | 0.6926 | 0.60 |

**结论**

- 阈值 0.5：macro-F1 0.6215 → **0.7123**，micro-F1 0.8150 → **0.8325**，weighted-F1 0.7816 → **0.8518**；
  但 Hamming loss 0.0898 → 0.1002，子集准确率 0.3160 → 0.2982（把更多样本判成正类）。
- 最优阈值：R4 macro 0.7440@0.30 / micro 0.8611@0.35；R5 macro 0.7346@0.60 / micro 0.8465@0.55；
  macro AP 0.8107 → 0.8127（几乎不变）。
- 结论：**R5 没有提升排序能力，只是把工作点移向了高召回**。
  因此 R4 继续作为主模型（`model_13c_v1.pth`），R5 保留为高召回变体（`model_13c_v1_highrecall.pth`）。

## 6. 横向对比（R1–R5）

### 6.1 整体指标（R1–R4，阈值 0.5，验证集）

| 轮次 | macro-F1 | micro-F1 | weighted-F1 | 子集准确率 | Hamming | macro AP | 最佳阈值 / macro-F1 | 最佳 micro-F1 |
|---|---|---|---|---|---|---|---|---|
| R1 | 0.2184 | 0.5823 | 0.4711 | 0.0416 | 0.1928 | 0.3692 | 0.05 / 0.3938 | 0.6356 |
| R2 | 0.4600 | 0.7323 | 0.6638 | 0.1894 | 0.1220 | 0.7028 | 0.30 / 0.6480 | 0.8010 |
| R3 | 0.5788 | 0.7872 | 0.7460 | 0.2675 | 0.1006 | 0.7891 | 0.35 / 0.7264 | 0.8466 |
| R4 | 0.6215 | 0.8150 | 0.7816 | 0.3160 | 0.0898 | 0.8107 | 0.30 / 0.7440 | 0.8611 |

### 6.2 每类 F1（阈值 0.5）

| class | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| benzene | 0.9497 | 0.9832 | 0.9896 | 0.9912 |
| carbonyl | 0.7326 | 0.9185 | 0.9264 | 0.9333 |
| hydroxyl | 0.0000 | 0.0564 | 0.2134 | 0.2443 |
| carboxyl | 0.0000 | 0.0505 | 0.2156 | 0.1907 |
| ester | 0.1918 | 0.8048 | 0.8426 | 0.8807 |
| amine | 0.4067 | 0.5712 | 0.6675 | 0.7141 |
| nitro | 0.0000 | 0.0352 | 0.2757 | 0.3498 |
| nitrile | 0.0000 | 0.1366 | 0.2540 | 0.2887 |
| alkene | 0.0020 | 0.1513 | 0.6290 | 0.7207 |
| alkyne | 0.0003 | 0.6499 | 0.7634 | 0.8465 |
| ether | 0.6777 | 0.8341 | 0.8718 | 0.8902 |
| amide | 0.0020 | 0.5777 | 0.6211 | 0.6917 |
| halogen | 0.0951 | 0.1942 | 0.2867 | 0.3900 |
| sulfonyl | 0.0000 | 0.4761 | 0.5465 | 0.5696 |

### 6.3 每类 best F1（每类单独选阈值）

| class | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| benzene | 0.9497 | 0.9843 | 0.9909 | 0.9922 |
| carbonyl | 0.7438 | 0.9359 | 0.9450 | 0.9469 |
| hydroxyl | 0.2586 | 0.4562 | 0.5443 | 0.5873 |
| carboxyl | 0.0222 | 0.3733 | 0.4729 | 0.5005 |
| ester | 0.4947 | 0.8515 | 0.8928 | 0.9041 |
| amine | 0.6025 | 0.7420 | 0.7978 | 0.8158 |
| nitro | 0.0795 | 0.3313 | 0.4757 | 0.5175 |
| nitrile | 0.1118 | 0.4303 | 0.5415 | 0.5505 |
| alkene | 0.4774 | 0.5713 | 0.7618 | 0.7977 |
| alkyne | 0.1899 | 0.7872 | 0.8633 | 0.8795 |
| ether | 0.7031 | 0.8772 | 0.9088 | 0.9155 |
| amide | 0.3951 | 0.7241 | 0.7832 | 0.7988 |
| halogen | 0.5192 | 0.5988 | 0.6393 | 0.6707 |
| sulfonyl | 0.2266 | 0.5937 | 0.6705 | 0.6898 |

### 6.4 关键观察

- **结构 + Focal（R2）**：macro-F1 0.218 → 0.460，所有类别第一次都有非零 F1。
- **加 1H 积分（R3）**：macro-F1 0.460 → 0.579，`hydroxyl` 0.056 → 0.213、
  `alkene` 0.151 → 0.629；说明 O–H / 烯氢等 1H 信息确实补上了 13C 看不到的部分。
- **加多重峰/J（R4）**：macro-F1 0.579 → 0.621，`halogen` 0.287 → 0.390、
  `nitrile` 0.254 → 0.289、`nitro` 0.276 → 0.350；只有 `carboxyl` 从 0.216 降到 0.191
  （验证集里只有 1,892 个正例，波动大）。
- **阈值影响很大**：R4 在阈值 0.30 时 macro-F1 0.744、micro-F1 0.844；
  实际使用建议 0.30–0.35，或按类使用 `best_threshold`。
- **micro 与 macro 的取舍**：R4 的 macro 明显更高，但 micro 最佳值（0.844）比 R3（0.847）
  略低，说明模型把更多注意力分给了稀有类。

### 6.5 测试集评估（117,205 条）

| 模型 | 阈值 | macro-F1 | micro-F1 | weighted-F1 | 子集准确率 | Hamming | macro AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| R4 | 0.50 | 0.6192 | 0.8154 | 0.7819 | 0.3164 | 0.0898 | 0.8102 |
| R4 | 0.30（验证集 macro 最优） | 0.7427 | 0.8436 | 0.8480 | 0.3148 | 0.0947 | 0.8102 |
| R4 | **0.35（验证集 micro 最优）** | **0.7403** | **0.8611** | **0.8560** | **0.3939** | **0.0781** | 0.8102 |
| R5 | 0.50 | 0.7128 | 0.8327 | 0.8519 | 0.2990 | 0.1002 | 0.8128 |
| R5 | 0.55（验证集 micro 最优） | 0.7346 | 0.8468 | 0.8513 | 0.3619 | 0.0852 | 0.8128 |

结论：R4 在验证集上选出的 0.35 阈值，在测试集上同样全面最好（macro 0.7403、micro 0.8611、子集 0.3939、Hamming 0.0781）。
测试集与验证集数字几乎一致，说明模型没有对验证集过拟合。

## 7. 产物清单

### 7.1 代码

| 文件 | 作用 |
|---|---|
| `src/parser.py` | parquet→csv、13C 分桶、1H 特征（积分/多重峰/J）、流式写 .pt |
| `src/dataset.py` | mmap 加载 X/Y/X1H，按行拼接 13C+1H |
| `src/model.py` | 13C/1H 双分支 + 标量 MLP + 512 FC；兼容旧权重 |
| `src/train.py` | 无限训练/Ctrl+C、FocalLoss、日志、状态文件、--init-from/--workers |
| `src/f1_test.py` | F1/AP/阈值扫描评估，自动还原输入维度 |
| `src/log_utils.py` | CSV 表头变更备份、JSONL 追加 |
| `src/setup.py` | 路径、超参、1H 分桶、Focal、worker 配置 |
| `src/dashboard.py` | Flask 训练看板（实时进度/曲线/停止/评估） |
| `CHANGELOG.md` | 版本变更记录 |
| `UPDATELOG.md` | 本文件 |
| `requirements.txt` | 依赖清单（已修正编码与版本） |

### 7.2 模型权重（按轮次）

| 文件 | 对应轮次 | 大小 | 说明 |
|---|---|---|---|
| `data/ModelData/model_13c_v1_bce_oldarch.pth` | R1 | 178 KB | 旧结构 + BCE |
| `data/ModelData/model_13c_last_bce_oldarch.pth` | R1 | 178 KB | 旧结构最后一轮 |
| `data/ModelData/model_13c_v1_13c_focal.pth` | R2 | 56 MB | 13C-only + Focal |
| `data/ModelData/model_13c_last_13c_focal.pth` | R2 | 56 MB | 13C-only 最后一轮 |
| `data/ModelData/model_13c_v1_1h_v1.pth` | R3 | 73 MB | 1H 积分 280 维 |
| `data/ModelData/model_13c_last_1h_v1.pth` | R3 | 73 MB | 1H v1 最后一轮 |
| `data/ModelData/model_13c_v1.pth` | R4（当前最佳） | 73 MB | 1H 多重峰/J 852 维 |
| `data/ModelData/model_13c_last.pth` | R4 | 73 MB | 最后一轮 |
| `data/ModelData/model_13c_v1_r4.pth` | R4 备份 | 73 MB | 微调前保存 |
| `data/ModelData/model_13c_last_r4.pth` | R4 备份 | 73 MB | 微调前保存 |
| `data/ModelData/model_13c_v1_highrecall.pth` | R5 | 73 MB | 按类别 alpha 微调的高召回变体 |
| `data/ModelData/model_13c_last_highrecall.pth` | R5 | 73 MB | 最后一轮 |

### 7.3 数据与日志

| 路径 | 内容 |
|---|---|
| `data/PTFiles/X_{split}.pt` | 13C 多热特征（float32） |
| `data/PTFiles/X1H_{split}.pt` | 1H v2 特征（float16，852 维） |
| `data/PTFiles/X1H_meta.json` | 1H 特征布局 |
| `data/PTFiles/Y_{split}.pt` | 14 类多热标签 |
| `data/LogData/train_history.csv` | 逐 epoch 记录（R3、R4） |
| `data/LogData/train_history.*.csv` | 历史表备份（R1、R2） |
| `data/LogData/run_summary.jsonl` | 每次运行一条汇总 |
| `data/LogData/f1_by_class.csv` | 每次评估的每类指标 |
| `data/LogData/f1_threshold_sweep.csv` | 阈值扫描完整数据 |
| `data/LogData/f1_summary.jsonl` | 每次评估的整体汇总 |
| `data/LogData/train_status.json` | 看板用的实时状态 |
| `logs/train_*.log` | 各轮训练控制台输出 |
| `logs/f1_*.log` | 各轮 F1 评估输出 |
| `logs/gen_1h*.log` | 1H 特征生成日志 |

## 8. 复现命令

```bash
cd /home/ubuntu/CNMR/src

# 1) parquet -> csv（保留 nmr_1h）
../.venv/bin/python -c "import parser; parser.parse_parquet_file()"

# 2) 13C 特征（X/Y）
../.venv/bin/python -c "import parser; parser.gen_XY_pt()"

# 3) 1H 特征（多重峰 + J，852 维）
../.venv/bin/python -c "import parser; parser.gen_1h_pt()"

# 4) 训练（示例：3 个 epoch，热启动 13C 卷积）
../.venv/bin/python train.py --epochs 3 --workers 2 --init-from ../data/ModelData/model_13c_v1_1h_v1.pth

# 5) F1 评估
../.venv/bin/python f1_test.py

# 6) 看板
../.venv/bin/python dashboard.py --host 0.0.0.0 --port 8090
```

## 9. 已知问题与限制

- **测试集已评估**：R4 在 test 上 0.5 / 0.35 阈值分别为 macro 0.6192 / 0.7403、micro 0.8154 / 0.8611，与 validation 基本一致。
- **R5 只是高召回变体**：按类别 alpha 微调后 AP 几乎不变，最优阈值下仍不如 R4；适合“宁可多报、不能漏报”的场景。
- **阈值调参偏乐观**：最佳阈值是在同一份验证集上选的，报告里已提示；正式使用应在独立校准集上选阈值。
- **稀有类仍弱**：`carboxyl`（support 1,892，1.6%）F1 0.19，`hydroxyl` 0.24；
  需要更多数据、类别加权或分层采样。
- **1H v1 数据已被 v2 覆盖**：`X1H_*.pt` 现在是 852 维；旧 280 维权重的对应特征文件已不存在，
  如需回看 R3，需要重新生成旧版特征。
- **CPU 训练慢**：单 epoch 约 53–55 分钟，3 轮约 2.7 小时；GPU 会快一个数量级。
- **磁盘紧张**：写本文时剩余约 3.1 GB（X 3.1 GB + X1H 1.5 GB + 权重 + CSV）。
- **看板无鉴权**：目前监听 0.0.0.0:8090，仅建议在受控网络使用。

## 10. 后续建议

1. 在独立 test 集上评估 R4，并按类校准阈值（不要在同一份数据上选阈值）。
2. 给 FocalLoss 增加按类别频率的 `alpha`（或 BCE 的 `pos_weight`），专门照顾 `carboxyl`、`nitro`。
3. 把多重峰/J 做成更细的通道（如按 J 区间分桶），或把多重峰类型作为嵌入。
4. 训练更久（val loss 仍在下降），或加学习率调度、早停。
5. 增加从 `model_13c_last.pth` 完整续训（目前 `--init-from` 只加载 13C 卷积）。

## 附录 A：操作与试错记录

- 最初 `train` 转换 OOM；改成流式后，`test` 产物与旧实现逐位一致，确认没有改坏。
- R3 第一次启动用了 `workers=0`（3.16 it/s），跑了 94 秒后停止并清理了那条未完成记录，
  改 `--workers 2` 后提升到 4.55 it/s，再正式训练。
- 看板“停止训练”最初会让子进程变僵尸，接口等 30 秒并返回 `stopped:false`；
  已改为按 argv 识别脚本进程 + `wait()` 回收。
- 原 `f1_test.py` 的模型路径、数据文件名、类别名都已修正，详见 3.3。
- `requirements.txt` 原为 UTF-16，且 `scikit-learn==1.9.1` / `scipy==1.18.1` 等版本不存在；
  已转 UTF-8 并改为实际可安装版本。
- 1H v2 生成时把 `X1H_*.pt` 从 float32/280 维升级为 float16/852 维，
  旧 1H 权重备份为 `*_1h_v1.pth`。
- R5 优化实验：从 R4 完整热启动、per-class Focal alpha + plateau，训练 3 个 epoch。
  结果主要是工作点从“低召回”移向“高召回”，AP 0.8107 → 0.8127，最优阈值下 macro/ micro
  仍略低于 R4，因此把 R5 保留为 `model_13c_v1_highrecall.pth`，主模型恢复为 R4。
- 修正 `f1_summary.jsonl` 的 `best_micro_f1` 字段：之前误用了 macro 最优阈值那一行的 micro 值，
  现在改为 micro 最优阈值对应的值；本文 6.1 的 R1/R2/R4 数值已按阈值扫描表更正。

## 附录 B：原始数据文件索引

- 逐 epoch：`data/LogData/train_history.csv`、`train_history.20261018-223211.csv`、
  `train_history.20261019-075654.csv`
- 每次运行汇总：`data/LogData/run_summary.jsonl`
- 每次评估汇总：`data/LogData/f1_summary.jsonl`
- 每类评估：`data/LogData/f1_by_class.csv`
- 阈值扫描：`data/LogData/f1_threshold_sweep.csv`
- 训练控制台：`logs/train_focal_3ep.log`（R2）、`logs/train_1h_3ep.log`（R3）、
  `logs/train_mh_3ep.log`（R4）
- 评估控制台：`logs/f1_validation.log`（R1）、`logs/f1_focal_3ep.log`（R2）、
  `logs/f1_1h_3ep.log`（R3）、`logs/f1_mh_3ep.log`（R4）
