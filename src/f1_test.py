"""F1 / 多标签分类指标评估"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from datetime import datetime

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from dataset import get_dataloader
from log_utils import append_jsonl, make_run_id, open_csv_logger
from model import ID_ARCHS, NMRClassifier
from setup import SETTINGS, FUNCTIONAL_GROUPS

# 类别顺序与标签生成顺序保持一致
CLASS_NAMES = list(FUNCTIONAL_GROUPS.keys())

# 阈值扫描网格
THRESHOLD_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)

SPLITS = ("train", "validation", "test")

BY_CLASS_FIELDS = [
    "run_id",
    "timestamp",
    "split",
    "model_path",
    "threshold",
    "class",
    "support",
    "precision",
    "recall",
    "f1",
    "average_precision",
    "tp",
    "fp",
    "fn",
    "tn",
    "best_threshold",
    "best_f1",
]

SWEEP_FIELDS = [
    "run_id",
    "timestamp",
    "split",
    "model_path",
    "threshold",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "subset_accuracy",
    "hamming_loss",
    "positive_rate",
]


def _get_device() -> torch.device:
    """解析评估设备：auto 优先使用 CUDA，否则使用 CPU"""
    raw = str(SETTINGS.device).strip().lower()

    if raw in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if raw.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TRAIN_DEVICE 指定了 cuda，但 torch.cuda.is_available() 为 False"
                "运行 `python main_cli.py cuda-info` 查看原因"
            )
        return torch.device(raw)

    return torch.device("cpu")


def _resolve_model_path(explicit=None) -> Path:
    """确定待评估的权重文件路径"""

    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"找不到模型权重: {path}")
        return path

    model_dir = Path(SETTINGS.paths.model_dir)

    for name in ("model_13c_v1.pth", "model_13c_last.pth"):
        path = model_dir / name
        if path.exists():
            return path

    raise FileNotFoundError(
        f"在 {model_dir} 下找不到 model_13c_v1.pth 或 model_13c_last.pth，"
        "请先用 train.py 训练模型，或用 --model 指定权重路径"
    )


def load_model(model_path: Path, device: torch.device) -> NMRClassifier:
    """加载模型权重并恢复模型结构参数"""

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 兼容整模型保存
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # 从 checkpoint 中恢复输入维度与架构信息
    n_13c = 880
    h1_bins = 0
    h1_channels = 1
    h1_scalars = 0
    arch = "r4"

    if isinstance(checkpoint, dict):
        if "_n_13c_bins" in checkpoint:
            n_13c = int(checkpoint["_n_13c_bins"].item())
        if "_n_1h_bins" in checkpoint:
            h1_bins = int(checkpoint["_n_1h_bins"].item())
        if "_n_1h_channels" in checkpoint:
            h1_channels = int(checkpoint["_n_1h_channels"].item())
        if "_n_1h_scalars" in checkpoint:
            h1_scalars = int(checkpoint["_n_1h_scalars"].item())
        if "_arch_id" in checkpoint:
            arch = ID_ARCHS.get(int(checkpoint["_arch_id"].item()), "r4")

    n_1h = h1_bins * h1_channels + h1_scalars

    model = NMRClassifier(
        n_features = n_13c,
        n_1h = n_1h,
        h1_bins = h1_bins if h1_bins else None,
        h1_channels = h1_channels,
        h1_scalars = h1_scalars,
        arch = arch,
    ).to(device)

    # 补齐旧 checkpoint 中缺失的 buffer
    if isinstance(checkpoint, dict):
        checkpoint.setdefault("_n_13c_bins", torch.tensor(n_13c))
        checkpoint.setdefault("_n_1h_bins", torch.tensor(h1_bins))
        checkpoint.setdefault("_n_1h_channels", torch.tensor(h1_channels))
        checkpoint.setdefault("_n_1h_scalars", torch.tensor(h1_scalars))
        checkpoint.setdefault("_arch_id", torch.tensor(0))

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as error:
        raise RuntimeError(
            f"权重与模型结构不匹配（{model_path}）：{error}\n"
            "常见原因：训练时的类别数（CLASSES_SUM）或 1H 特征维度与当前不一致，"
            "或拿错了权重文件"
        ) from error

    model.eval()

    return model


def predict(model: NMRClassifier, loader, device: torch.device):
    """执行推理，返回概率矩阵和标签矩阵"""

    n_samples = len(loader.dataset)

    probs = None
    labels = None
    offset = 0

    model.eval()

    with torch.inference_mode():
        for X_batch, Y_batch in tqdm(loader, desc="推理", leave=False):

            batch_probs = torch.sigmoid(model(X_batch.to(device))).cpu().numpy()

            if probs is None:
                probs = np.empty((n_samples, batch_probs.shape[1]), dtype=np.float32)
                labels = np.empty((n_samples, batch_probs.shape[1]), dtype=np.float32)

            end = offset + batch_probs.shape[0]
            probs[offset:end] = batch_probs
            labels[offset:end] = Y_batch.numpy()
            offset = end

    if probs is None:
        raise ValueError("数据集为空，无法评估")

    if offset != n_samples:
        probs = probs[:offset]
        labels = labels[:offset]

    return probs, labels


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float):
    """按指定阈值计算每类指标与整体指标"""

    n_classes = len(CLASS_NAMES)
    labels = np.arange(n_classes)

    y_pred = (probs >= threshold).astype(np.float32)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        average_precision = average_precision_score(y_true, probs, average=None)

    average_precision = np.nan_to_num(average_precision, nan=0.0)

    confusion = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    tn = confusion[:, 0, 0]
    fp = confusion[:, 0, 1]
    fn = confusion[:, 1, 0]
    tp = confusion[:, 1, 1]

    per_class = []

    for i, name in enumerate(CLASS_NAMES):
        per_class.append(
            {
                "class": name,
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "average_precision": float(average_precision[i]),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "tn": int(tn[i]),
            }
        )

    overall = {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "macro_average_precision": float(average_precision.mean()),
        "positive_rate_true": float(y_true.mean()),
    }

    return per_class, overall


def sweep_thresholds(y_true: np.ndarray, probs: np.ndarray):
    """在阈值网格上扫描，返回扫描表、整体最佳阈值行及每类最佳阈值"""

    rows = []

    for threshold in THRESHOLD_GRID:

        y_pred = probs >= threshold

        rows.append(
            {
                "threshold": float(threshold),
                "macro_f1": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "micro_f1": float(
                    f1_score(y_true, y_pred, average="micro", zero_division=0)
                ),
                "weighted_f1": float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "subset_accuracy": float(accuracy_score(y_true, y_pred)),
                "hamming_loss": float(hamming_loss(y_true, y_pred)),
                "positive_rate": float(y_pred.mean()),
            }
        )

    sweep_table = pd.DataFrame(rows)
    best_row = sweep_table.loc[sweep_table["macro_f1"].idxmax()]

    per_class_best = []

    for i in range(len(CLASS_NAMES)):

        y_col = y_true[:, i]

        if y_col.sum() == 0:
            per_class_best.append((float("nan"), float("nan")))
            continue

        scores = [
            f1_score(y_col, probs[:, i] >= threshold, zero_division=0)
            for threshold in THRESHOLD_GRID
        ]

        best_index = int(np.argmax(scores))
        per_class_best.append(
            (float(THRESHOLD_GRID[best_index]), float(scores[best_index]))
        )

    return sweep_table, best_row, per_class_best


def _print_report(per_class, overall, best_row=None, best_micro_row=None):
    """打印评估结果表格与整体指标"""

    columns = [
        "class", "support", "precision", "recall", "f1",
        "average_precision", "tp", "fp", "fn", "tn",
    ]

    if per_class and ("best_threshold" in per_class[0]):
        columns += ["best_threshold", "best_f1"]

    table = pd.DataFrame(per_class)[columns]

    print(f"\n按阈值 {overall['threshold']:.2f} 统计（每类）：")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n整体指标：")
    print(f"  macro-F1    : {overall['macro_f1']:.4f}")
    print(f"  micro-F1    : {overall['micro_f1']:.4f}")
    print(f"  weighted-F1 : {overall['weighted_f1']:.4f}")
    print(f"  macro P / R : {overall['macro_precision']:.4f} / {overall['macro_recall']:.4f}")
    print(f"  子集准确率  : {overall['subset_accuracy']:.4f}（14 个标签全部预测正确的样本比例）")
    print(f"  Hamming loss: {overall['hamming_loss']:.4f}")
    print(f"  macro AP    : {overall['macro_average_precision']:.4f}")

    if best_row is not None:
        print(
            f"\n阈值扫描：macro-F1 最高的阈值是 {best_row['threshold']:.2f}"
            f"（macro-F1 {best_row['macro_f1']:.4f}，"
            f"micro-F1 {best_row['micro_f1']:.4f}，"
            f"预测正例率 {best_row['positive_rate']:.1%}）"
        )

        true_rate = overall.get("positive_rate_true", float("nan"))

        if (true_rate == true_rate) and (true_rate > 0) and (
            best_row["positive_rate"] > 2 * true_rate
        ):
            print(
                f"警告：该阈值把 {best_row['positive_rate']:.1%} 的标签判为正类，"
                f"而真实正例率只有 {true_rate:.1%}，macro-F1 的提升主要来自召回，"
                "实际使用时请谨慎选择阈值"
            )

        if best_micro_row is not None:
            print(
                f"若以 micro-F1 为准，最佳阈值是 {best_micro_row['threshold']:.2f}"
                f"（micro-F1 {best_micro_row['micro_f1']:.4f}，"
                f"预测正例率 {best_micro_row['positive_rate']:.1%}）"
            )

        print("提示：阈值是在同一份数据上挑的，指标会偏乐观，换数据验证才可靠")


def _save_results(
    run_id,
    split,
    model_path,
    per_class,
    overall,
    per_class_best=None,
    sweep_table=None,
):
    """将评估结果写入日志目录"""

    log_dir = Path(SETTINGS.paths.log_data_dir)
    timestamp = datetime.now().isoformat(timespec="seconds")

    # 每类指标
    by_class_path = log_dir / "f1_by_class.csv"
    fh, writer = open_csv_logger(by_class_path, BY_CLASS_FIELDS)

    try:
        for i, row in enumerate(per_class):

            if per_class_best is None:
                best_threshold = ""
                best_f1 = ""
            else:
                best_threshold, best_f1 = per_class_best[i]
                best_threshold = "" if np.isnan(best_threshold) else best_threshold
                best_f1 = "" if np.isnan(best_f1) else best_f1

            writer.writerow(
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "split": split,
                    "model_path": str(model_path),
                    "threshold": overall["threshold"],
                    "class": row["class"],
                    "support": row["support"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "average_precision": row["average_precision"],
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "tn": row["tn"],
                    "best_threshold": best_threshold,
                    "best_f1": best_f1,
                }
            )
    finally:
        fh.close()

    # 阈值扫描表
    if sweep_table is not None:

        sweep_path = log_dir / "f1_threshold_sweep.csv"
        fh, writer = open_csv_logger(sweep_path, SWEEP_FIELDS)

        try:
            for _, row in sweep_table.iterrows():
                writer.writerow(
                    {
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "split": split,
                        "model_path": str(model_path),
                        "threshold": row["threshold"],
                        "macro_f1": row["macro_f1"],
                        "micro_f1": row["micro_f1"],
                        "weighted_f1": row["weighted_f1"],
                        "subset_accuracy": row["subset_accuracy"],
                        "hamming_loss": row["hamming_loss"],
                        "positive_rate": row["positive_rate"],
                    }
                )
        finally:
            fh.close()

    return log_dir


def parse_args(argv=None):
    """解析命令行参数"""

    parser = argparse.ArgumentParser(
        description="评估 NMR 多标签分类模型（F1 / AP / 阈值扫描）"
    )

    parser.add_argument(
        "--split",
        default="validation",
        choices=SPLITS,
        help="要评估的数据拆分，默认 validation",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="权重路径，默认 data/ModelData/model_13c_v1.pth（不存在时用 model_13c_last.pth）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="主报表使用的判定阈值，默认 0.5",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"推理 batch 大小，默认取 settings.json 的 training.batch_size={SETTINGS.training.batch_size}",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="跳过阈值扫描（更快）",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="只打印，不写入 data/LogData",
    )

    args = parser.parse_args(argv)

    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold 必须在 (0, 1) 之间")

    return args


def main(argv=None):
    """执行评估流程并返回汇总结果"""

    args = parse_args(argv)

    if args.split not in SPLITS:
        raise ValueError(f"未知的数据拆分: {args.split}")

    started = time.time()
    run_id = make_run_id()
    device = _get_device()
    batch_size = args.batch_size or SETTINGS.training.batch_size

    data_dir = Path(SETTINGS.paths.pt_dir)
    X_path = data_dir / f"X_{args.split}.pt"
    Y_path = data_dir / f"Y_{args.split}.pt"

    for path in (X_path, Y_path):
        if not path.exists():
            raise FileNotFoundError(f"找不到数据文件 {path}，请先运行 parser.gen_XY_pt()")

    model_path = _resolve_model_path(args.model)

    print(f"运行 ID: {run_id}")
    print(f"设备: {device} | batch_size={batch_size} | split={args.split}")
    print(f"数据: {X_path.name} / {Y_path.name}")
    print(f"权重: {model_path}")

    model = load_model(model_path, device)

    loader = get_dataloader(
        X_path,
        Y_path,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    probs, y_true = predict(model, loader, device)

    dataset = loader.dataset
    n_13c = int(dataset.n_13c)
    n_1h = int(dataset.n_1h)

    print(
        f"样本数: {len(y_true)} | 13C 桶: {n_13c} | 1H 桶: {n_1h} | "
        f"类别数: {probs.shape[1]}"
    )

    expected_features = model.n_features + model.n_1h

    if (n_13c + n_1h) != expected_features:
        raise ValueError(
            f"数据特征维度 {n_13c + n_1h}（13C {n_13c} + 1H {n_1h}）"
            f"与模型期望的 {expected_features}（13C {model.n_features} + 1H {model.n_1h}）"
            "不一致，请检查 BUCKETS_SUM / H1_BUCKETS 与 .pt 数据是否匹配"
        )

    if probs.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"模型输出 {probs.shape[1]} 个类别，但 FUNCTIONAL_GROUPS 有 {len(CLASS_NAMES)} 个，"
            "请检查 CLASSES_SUM 与权重文件是否匹配"
        )

    if y_true.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"标签有 {y_true.shape[1]} 列，但 FUNCTIONAL_GROUPS 有 {len(CLASS_NAMES)} 个，"
            "请重新生成 .pt 数据或检查 setup.py"
        )

    per_class, overall = compute_metrics(y_true, probs, args.threshold)

    sweep_table = None
    best_row = None
    best_micro_row = None
    per_class_best = None

    if not args.no_sweep:
        sweep_table, best_row, per_class_best = sweep_thresholds(y_true, probs)
        best_micro_row = sweep_table.loc[sweep_table["micro_f1"].idxmax()]

        for i, (best_threshold, best_f1) in enumerate(per_class_best):
            per_class[i]["best_threshold"] = best_threshold
            per_class[i]["best_f1"] = best_f1

    _print_report(per_class, overall, best_row, best_micro_row)

    log_dir = None

    if not args.no_save:
        log_dir = _save_results(
            run_id,
            args.split,
            model_path,
            per_class,
            overall,
            per_class_best,
            sweep_table,
        )
        print(f"\n结果已追加到: {log_dir}")

    elapsed = time.time() - started

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "model_path": str(model_path),
        "device": str(device),
        "batch_size": batch_size,
        "n_samples": int(len(y_true)),
        "n_classes": int(probs.shape[1]),
        "n_13c": n_13c,
        "n_1h": n_1h,
        **overall,
        "best_threshold": None if best_row is None else float(best_row["threshold"]),
        "best_macro_f1": None if best_row is None else float(best_row["macro_f1"]),
        "best_micro_f1": None if best_micro_row is None else float(best_micro_row["micro_f1"]),
        "best_micro_threshold": (
            None if best_micro_row is None else float(best_micro_row["threshold"])
        ),
        "elapsed_seconds": round(elapsed, 2),
    }

    if not args.no_save:
        append_jsonl(Path(SETTINGS.paths.log_data_dir) / "f1_summary.jsonl", summary)

    return summary


if __name__ == "__main__":
    main()