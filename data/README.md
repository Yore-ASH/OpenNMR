# data 目录说明

## 分发包里带了什么

**`CNMR-1.0.0-src-data.zip` 里带了原始数据**：

```
data/
└── OriginalData/                 # ✅ 随分发包提供（约 364 MB）
    ├── train.parquet             #   937,022 行
    ├── validation.parquet        #   117,035 行
    └── test.parquet              #   117,205 行
```

parquet 的列（生成标签与特征所需的最小集合是 `smiles`、`nmr_13c`、`nmr_1h`）：

| 列 | 说明 |
|---|---|
| `id` | 样本 id |
| `smiles` | 分子 SMILES，用于生成 14 个官能团标签 |
| `nmr_13c` | 13C 化学位移列表 |
| `nmr_1h` | 1H 峰列表（化学位移 / 多重峰 / 积分 / 耦合常数） |
| `atom_count` / `molecular_formula` / `tokenized_input` | 原始数据里自带，本项目当前不使用 |

## 其余目录需要自己生成

下面这些**不随包分发**（体积原因），用一条命令从 `OriginalData/` 重建：

```
data/
├── CSVFiles/                     # ✅ python main_cli.py features 生成（约 0.5 GB）
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── PTFiles/                      # ✅ 同上，生成后约 6.5 GB
│   ├── X_{split}.pt              # 13C 多热 [N, 880] float32
│   ├── Y_{split}.pt              # 标签 [N, 14] float32
│   ├── X1H_{split}.pt            # 1H 特征 [N, 852] float16
│   └── X1H_meta.json             # bins=280, channels=3, scalars=12
├── ModelData/                    # 训练时自动创建
│   ├── model_13c_v1.pth
│   └── model_13c_last.pth
├── LogData/                      # 训练/评估时自动创建（训练看板读这里）
└── TrainData/                    # 训练时自动创建（逐 epoch 完整痕迹 + 归档权重）
```

重建命令（在项目根目录）：

```bash
python main_cli.py features
```

耗时约 **20–60 分钟**（要用 RDKit 逐个分子匹配 14 类官能团并解析 1H 峰），
内存占用约 1 GB，属于正常现象。

详细部署步骤见 [../DEPLOY.md](../DEPLOY.md)。
