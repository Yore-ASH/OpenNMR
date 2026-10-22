"""训练入口"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from dataset import get_dataloader
from log_utils import append_jsonl, make_run_id, open_csv_logger
from model import NMRClassifier
from setup import SETTINGS, FUNCTIONAL_GROUPS
from train_archive import TrainRun

LOG_FIELDS = [
    "run_id",
    "timestamp",
    "epoch",
    "lr",
    "batch_size",
    "dropout",
    "loss_fn",
    "n_1h",
    "device",
    "n_train",
    "n_val",
    "train_loss",
    "val_loss",
    "best_val_loss",
    "improved",
    "epoch_seconds",
    "elapsed_seconds",
    "interrupted",
    "model_path",
]


def _get_device() -> torch.device:
    """解析训练设备"""

    raw = str(SETTINGS.device).strip().lower()

    if raw in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if raw.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TRAIN_DEVICE 指定了 cuda，但 torch.cuda.is_available() 为 False。\n"
                "请运行 `python main_cli.py cuda-info` 可以看到具体原因"
            )
        return torch.device(raw)

    return torch.device("cpu")


def _write_status(**fields):
    """把训练进度写入 data/LogData/train_status.json"""

    path = Path(SETTINGS.paths.log_data_dir) / "train_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    current = {}

    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}

    current.update(fields)
    current["updated_at"] = datetime.now().isoformat(timespec="seconds")

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class FocalLoss(nn.Module):
    """多标签二分类的 Focal Loss

    对每个标签独立计算 sigmoid 交叉熵，再乘上 (1 - p_t)^gamma 与 alpha_t：
    - gamma 越大，模型越关注"难分"的样本，对 14 个官能团这种稀有正例更友好；
    - alpha 用于平衡正负样本。可以是标量（所有类别共用），
      也可以是长度 = 类别数的向量（按类别正例率自适应，稀有类权重更高）。

    输入是 logits（未过 sigmoid），targets 是 0/1 的多热标签。
    """

    def __init__(self, alpha=0.25, gamma=2.0, alpha_pos=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if alpha_pos is None:
            self.alpha_pos = None
        else:
            self.alpha_pos = torch.as_tensor(alpha_pos, dtype=torch.float32).flatten()

    def forward(self, logits, targets):
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        loss = bce * (1.0 - p_t).pow(self.gamma)

        if self.alpha_pos is not None:
            alpha_t = self.alpha_pos.to(device=logits.device, dtype=logits.dtype)
            alpha_t = alpha_t * targets + (1.0 - alpha_t) * (1.0 - targets)
            loss = alpha_t * loss
        elif self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        epoch=0,
        on_batch=None,
        rest_time : float = 0.0
        ):
    """训练一个 epoch。"""

    model.train()

    total_loss = 0.0
    n_batches = 0

    bar = tqdm(dataloader, desc=f"训练 epoch {epoch}", leave=False)

    try:
        for X_batch, Y_batch in bar:

            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            bar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

            if (on_batch is not None) and (n_batches % 100 == 0):
                on_batch(n_batches, total_loss / n_batches)

            # 设备休眠时间，我的顶配电脑啊....
            if rest_time > 0:
                time.sleep(rest_time)

    except KeyboardInterrupt:
        bar.close()
        return (total_loss / n_batches if n_batches else float("nan")), True

    bar.close()

    return (total_loss / n_batches if n_batches else float("nan")), False


def evaluate(model, dataloader, criterion, device, desc="验证", collect=False):
    """评估一个 epoch"""

    model.eval()

    total_loss = 0.0
    n_batches = 0

    probs_list = []
    target_list = []

    bar = tqdm(dataloader, desc=desc, leave=False)

    try:
        with torch.no_grad():
            for X_batch, Y_batch in bar:
                X_batch = X_batch.to(device, non_blocking=True)
                Y_batch = Y_batch.to(device, non_blocking=True)

                logits = model(X_batch)
                loss = criterion(logits, Y_batch)

                total_loss += loss.item()
                n_batches += 1

                if collect:
                    probs_list.append(torch.sigmoid(logits).float().cpu())
                    target_list.append(Y_batch.float().cpu())

                bar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    except KeyboardInterrupt:
        bar.close()
        return (total_loss / n_batches if n_batches else float("nan")), True, None, None

    bar.close()

    mean_loss = total_loss / n_batches if n_batches else float("nan")

    if not collect or not probs_list:
        return mean_loss, False, None, None

    return mean_loss, False, torch.cat(probs_list, dim=0), torch.cat(target_list, dim=0)


def _f1_parts(pred, target):
    """逐类的 (tp, fp, fn) 与 F1"""

    tp = (pred * target).sum(dim=0)
    fp = (pred * (1 - target)).sum(dim=0)
    fn = ((1 - pred) * target).sum(dim=0)

    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom.clamp_min(1e-12), torch.zeros_like(denom))

    return tp, fp, fn, f1


def compute_split_metrics(probs, targets, threshold=0.5):
    """从一轮验证的预测概率算出汇总指标 + 逐类指标（用于训练痕迹归档）"""

    pred = (probs >= threshold).float()
    targets = targets.float()

    tp, fp, fn, f1_per_class = _f1_parts(pred, targets)

    tp_all = float(tp.sum().item())
    fp_all = float(fp.sum().item())
    fn_all = float(fn.sum().item())
    micro_denom = 2 * tp_all + fp_all + fn_all

    summary = {
        "threshold": threshold,
        "macro_f1": float(f1_per_class.mean().item()),
        "micro_f1": float(2 * tp_all / micro_denom) if micro_denom > 0 else 0.0,
        "hamming_loss": float((pred != targets).float().mean().item()),
        "subset_accuracy": float((pred == targets).all(dim=1).float().mean().item()),
        "macro_ap": None,
    }

    try:
        from sklearn.metrics import average_precision_score

        summary["macro_ap"] = float(
            average_precision_score(targets.numpy(), probs.numpy(), average="macro")
        )
    except Exception:
        pass

    precision = tp / (tp + fp).clamp_min(1e-12)
    recall = tp / (tp + fn).clamp_min(1e-12)

    class_names = list(FUNCTIONAL_GROUPS.keys())
    per_class = {}

    for index, name in enumerate(class_names[: targets.shape[1]]):
        per_class[name] = {
            "support": int(targets[:, index].sum().item()),
            "tp": int(tp[index].item()),
            "fp": int(fp[index].item()),
            "fn": int(fn[index].item()),
            "precision": round(float(precision[index].item()), 6),
            "recall": round(float(recall[index].item()), 6),
            "f1": round(float(f1_per_class[index].item()), 6),
        }

    return summary, per_class


def _set_seed(seed: int):
    """设置随机种子"""

    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="训练 NMR 多标签分类模型")

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数；0 表示一直训练到 Ctrl+C（默认取 settings.json 的 training.epochs）",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="batch 大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument(
        "--init-from",
        default=None,
        help="用旧权重的 13C 卷积分支做热启动（只加载 conv.* 参数）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader 预取进程数（默认取 settings.json 的 training.workers）",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='训练设备: "auto" / "cuda" / "cuda:0" / "cpu"'
             '（默认取 settings.json 的 device.train_device，即 auto）',
    )
    parser.add_argument(
        "--arch",
        choices=["r4", "r5"],
        default=None,
        help='模型架构（默认取 settings.json 的 model.arch，即 r4）: '
             'r4 = 0.8.0 原架构（19.07M 参数，基线）; '
             'r5 = 优化架构（BatchNorm + 1×1 通道瓶颈 + 全局上下文，约 5.0M 参数）',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（默认取 settings.json 的 training.seed，即 2026）；A/B 对比时固定它才有可比性",
    )

    # 控制台加入参数
    parser.add_argument(
        "--rest",
        default = None,
        help = "单个 Batch 设备休息时间（秒，默认取 settings.json 的 training.default_batch_rest）",
        type = float
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    SETTINGS.apply(
        {
            "training.epochs": args.epochs,
            "training.batch_size": args.batch_size,
            "training.lr": args.lr,
            "training.workers": args.workers,
            "training.seed": args.seed,
            "training.default_batch_rest": args.rest,
            "device.train_device": args.device,
            "model.arch": args.arch,
        }
    )

    seed = SETTINGS.training.seed
    _set_seed(seed)

    data_dir = Path(SETTINGS.paths.pt_dir).resolve()

    batch_size = SETTINGS.training.batch_size
    lr = SETTINGS.training.lr
    weight_decay = SETTINGS.training.weight_decay
    dropout = SETTINGS.model.dropout
    focal_alpha = SETTINGS.training.focal_alpha
    focal_gamma = SETTINGS.training.focal_gamma
    focal_per_class_alpha = SETTINGS.training.focal_per_class_alpha
    lr_scheduler_name = SETTINGS.training.lr_scheduler
    lr_factor = SETTINGS.training.lr_factor
    lr_patience = SETTINGS.training.lr_patience
    max_epochs = SETTINGS.training.max_epochs
    workers = SETTINGS.training.workers
    rest_time = SETTINGS.training.default_batch_rest

    device = _get_device()

    log_dir = Path(SETTINGS.paths.log_data_dir)
    history_path = log_dir / "train_history.csv"
    summary_path = log_dir / "run_summary.jsonl"

    model_dir = Path(SETTINGS.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_path = model_dir / "model_13c_v1.pth"
    last_path = model_dir / "model_13c_last.pth"

    run_id = make_run_id()
    started_at = datetime.now().isoformat(timespec="seconds")
    run_start = time.time()

    pin_memory = device.type == "cuda"

    train_loader = get_dataloader(
        data_dir / "X_train.pt",
        data_dir / "Y_train.pt",
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        mmap=SETTINGS.data.mmap,
    )
    val_loader = get_dataloader(
        data_dir / "X_validation.pt",
        data_dir / "Y_validation.pt",
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        mmap=SETTINGS.data.mmap,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_1h = int(getattr(train_loader.dataset, "n_1h", 0))
    h1_bins = int(getattr(train_loader.dataset, "h1_bins", 0))
    h1_channels = int(getattr(train_loader.dataset, "h1_channels", 1))
    h1_scalars = int(getattr(train_loader.dataset, "h1_scalars", 0))

    # 训练痕迹归档
    save_every_epoch = SETTINGS.archive.save_every_epoch
    epoch_metrics = SETTINGS.archive.epoch_metrics

    archive = TrainRun(SETTINGS.paths.train_data_dir, run_id)
    archive.begin(
        {
            "script": "src/train.py",
            "device": str(device),
            "epochs_max": max_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
            "loss_fn": "focal",
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "focal_per_class_alpha": focal_per_class_alpha,
            "lr_scheduler": lr_scheduler_name,
            "lr_factor": lr_factor,
            "lr_patience": lr_patience,
            "workers": workers,
            "n_train": n_train,
            "n_val": n_val,
            "n_13c": int(train_loader.dataset.n_13c),
            "n_1h": n_1h,
            "h1_bins": h1_bins,
            "h1_channels": h1_channels,
            "h1_scalars": h1_scalars,
            "arch": SETTINGS.model.arch,
            "head_channels": SETTINGS.model.r5_head_channels,
            "head_h1_channels": SETTINGS.model.r5_head_h1_channels,
            "weight_decay": weight_decay,
            "settings": SETTINGS.summary(),
            "seed": seed,
            "save_every_epoch": save_every_epoch,
            "epoch_metrics": epoch_metrics,
        }
    )
    print(f"训练痕迹目录: {archive.run_dir}")

    _write_status(
        state = "running",
        pid = os.getpid(),
        run_id = run_id,
        started_at = started_at,
        finished_at = "",
        epochs_done = 0,
        epochs_max = max_epochs,
        epoch = 0,
        batch = 0,
        total_batches = len(train_loader),
        batch_size = batch_size,
        lr = lr,
        loss_fn = "focal",
        focal_alpha = focal_alpha,
        focal_gamma = focal_gamma,
        n_13c = int(train_loader.dataset.n_13c),
        n_1h = n_1h,
        h1_bins = h1_bins,
        h1_channels = h1_channels,
        h1_scalars = h1_scalars,
        train_loss = None,
        val_loss = None,
        best_val_loss = None,
        stopped_reason = "",
    )

    model = NMRClassifier(
        n_1h = n_1h,
        h1_bins = h1_bins if h1_bins else None,
        h1_channels = h1_channels,
        h1_scalars = h1_scalars,
        arch = SETTINGS.model.arch,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"模型: arch={model.arch} | 参数 {n_params:,} | "
        f"分类头输入 {model.head_input_dim:,} 维 | seed={seed}"
    )

    if args.init_from:
        init_path = Path(args.init_from)

        if not init_path.exists():
            raise FileNotFoundError(f"找不到初始化权重: {init_path}")

        old_state = torch.load(init_path, map_location=device, weights_only=True)

        if isinstance(old_state, dict) and "state_dict" in old_state:
            old_state = old_state["state_dict"]

        try:
            model.load_state_dict(old_state)
            print(f"已从 {init_path.name} 完整热启动（结构完全一致，用于微调）")
        except RuntimeError:
            current = model.state_dict()
            loaded = 0

            for key, value in old_state.items():
                if key.startswith("conv.") and (key in current) and (current[key].shape == value.shape):
                    current[key] = value
                    loaded += 1

            model.load_state_dict(current)
            print(f"已从 {init_path.name} 热启动 13C 卷积分支（{loaded} 个参数张量）")

    class_alpha = None

    if focal_per_class_alpha:
        y_train = train_loader.dataset.Y
        pos_rate = (y_train.sum(dim=0).float() / max(len(y_train), 1)).clamp(0.0, 1.0)
        class_alpha = (1.0 - pos_rate).clamp(0.05, 0.95)
        print(
            "Focal 每类 alpha（按正例率自适应）: "
            f"min={class_alpha.min():.3f}, max={class_alpha.max():.3f}, "
            f"苯环={class_alpha[0]:.3f}, 羧基={class_alpha[3]:.3f}"
        )

    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, alpha_pos=class_alpha)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None

    if lr_scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience
        )
        print(f"LR scheduler: ReduceLROnPlateau(factor={lr_factor}, patience={lr_patience})")
    else:
        print("LR scheduler: 无（固定学习率）")

    print(f"运行 ID: {run_id}")
    device_desc = str(device)

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        index = device.index if device.index is not None else torch.cuda.current_device()
        device_desc = (
            f"cuda:{index} | {props.name} | sm_{props.major}{props.minor} | "
            f"{props.total_memory / 1024 ** 3:.1f} GiB"
        )

    print(f"设备: {device_desc}")
    print(
        f"batch_size={batch_size} | lr={lr} | dropout={dropout} | "
        f"loss=Focal(alpha={focal_alpha}, gamma={focal_gamma}, "
        f"per_class_alpha={focal_per_class_alpha}, scheduler={lr_scheduler_name})"
    )
    print(
        f"输入特征: 13C={train_loader.dataset.n_13c} 桶 | "
        f"1H={n_1h} 维 (bins={h1_bins}, channels={h1_channels}, scalars={h1_scalars}) | "
        f"workers={workers}"
    )
    print(f"训练样本: {n_train} | 验证样本: {n_val}")
    print(f"数据目录: {data_dir}")
    print(f"日志表格: {history_path}")
    print(f"运行汇总: {summary_path}")

    if max_epochs > 0:
        print(f"最多训练 {max_epochs} 个 epoch")
    else:
        print("训练将一直进行，按 Ctrl+C 结束（会先保存权重、写入日志再退出）")

    history_fh, history_writer = open_csv_logger(history_path, LOG_FIELDS)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_done = 0
    stopped_reason = "max_epochs"

    def log_row(epoch, train_loss, val_loss, improved, epoch_seconds, interrupted):
        """把当前 epoch 的数据追加进 CSV 并立即 flush，保证 Ctrl+C 也不丢记录"""

        history_writer.writerow(
            {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "batch_size": batch_size,
                "dropout": dropout,
                "loss_fn": "focal",
                "n_1h": n_1h,
                "device": str(device),
                "n_train": n_train,
                "n_val": n_val,
                "train_loss": train_loss,
                "val_loss": "" if val_loss is None else val_loss,
                "best_val_loss": "" if math.isinf(best_val_loss) else best_val_loss,
                "improved": int(bool(improved)),
                "epoch_seconds": round(epoch_seconds, 2),
                "elapsed_seconds": round(time.time() - run_start, 2),
                "interrupted": int(bool(interrupted)),
                "model_path": str(last_path),
            }
        )

        history_fh.flush()

    def archive_partial(epoch, stage, train_loss_value, epoch_seconds):
        """Ctrl+C 打断时也留一条痕迹：记下这一轮跑到哪，并把当时权重归档。"""

        timestamp = datetime.now().isoformat(timespec="seconds")
        safe_train_loss = (
            None
            if (train_loss_value is None or math.isnan(train_loss_value))
            else round(train_loss_value, 6)
        )

        archived = str(
            archive.save_checkpoint(model.state_dict(), epoch, None, torch, tag=stage)
        )

        archive.record_epoch(
            epoch,
            {
                "run_id": run_id,
                "epoch": epoch,
                "timestamp": timestamp,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": "" if safe_train_loss is None else safe_train_loss,
                "val_loss": "",
                "best_val_loss": "" if math.isinf(best_val_loss) else round(best_val_loss, 6),
                "improved": 0,
                "epoch_seconds": round(epoch_seconds, 2),
                "elapsed_seconds": round(time.time() - run_start, 2),
                "checkpoint": archived,
            },
            detail={
                "run_id": run_id,
                "epoch": epoch,
                "timestamp": timestamp,
                "stage": stage,
                "train_loss": safe_train_loss,
                "val_loss": None,
                "interrupted": True,
                "checkpoint": archived,
            },
        )

        return archived

    try:
        while True:

            if (max_epochs > 0) and (epochs_done >= max_epochs):
                stopped_reason = "max_epochs"
                break

            epoch = epochs_done + 1
            epoch_start = time.time()

            _write_status(
                state = "running",
                epoch = epoch,
                batch = 0,
                epochs_done = epochs_done,
                train_loss = None,
                val_loss = None,
            )

            def report_batch(batch_index, running_loss, _epoch=epoch):
                _write_status(epoch=_epoch, batch=batch_index, train_loss=running_loss)

            train_loss, interrupted = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch = epoch,
                on_batch = report_batch,
                rest_time = rest_time,
            )

            # 训练中途被打断
            if interrupted:
                torch.save(model.state_dict(), last_path)
                archive_partial(epoch, "train_interrupted", train_loss, time.time() - epoch_start)
                log_row(
                    epoch,
                    train_loss,
                    None,
                    False,
                    time.time() - epoch_start,
                    True,
                )
                stopped_reason = "keyboard_interrupt"
                print(f"\n收到 Ctrl+C：epoch {epoch} 训练到一半被中断，"
                      f"已保存权重 {last_path} 并记录日志")
                _write_status(
                    state = "stopped",
                    stopped_reason = "keyboard_interrupt",
                    epoch = epoch,
                    train_loss = train_loss,
                    epochs_done = epochs_done,
                    finished_at = datetime.now().isoformat(timespec="seconds"),
                )
                break

            val_loss, interrupted, val_probs, val_targets = evaluate(
                model,
                val_loader,
                criterion,
                device,
                desc = f"验证 epoch {epoch}",
                collect = epoch_metrics,
            )

            if interrupted:
                torch.save(model.state_dict(), last_path)
                archive_partial(epoch, "val_interrupted", train_loss, time.time() - epoch_start)
                log_row(
                    epoch,
                    train_loss,
                    None,
                    False,
                    time.time() - epoch_start,
                    True,
                )
                stopped_reason = "keyboard_interrupt"
                print(f"\n收到 Ctrl+C：epoch {epoch} 验证阶段被中断，"
                      f"已保存权重 {last_path} 并记录日志")
                _write_status(
                    state = "stopped",
                    stopped_reason = "keyboard_interrupt",
                    epoch = epoch,
                    train_loss = train_loss,
                    epochs_done = epochs_done,
                    finished_at = datetime.now().isoformat(timespec="seconds"),
                )
                break

            # 正常完成一个 epoch
            epoch_seconds = time.time() - epoch_start
            improved = val_loss < best_val_loss

            if improved:
                best_val_loss = val_loss
                best_epoch = epoch

            epochs_done = epoch

            torch.save(model.state_dict(), last_path)

            if improved:
                torch.save(model.state_dict(), best_path)

            # 逐类指标
            if epoch_metrics and (val_probs is not None):
                split_metrics, per_class = compute_split_metrics(val_probs, val_targets)
            else:
                split_metrics, per_class = {}, {}

            # 归档权重
            archived = ""

            if save_every_epoch or improved:
                archived = str(
                    archive.save_checkpoint(model.state_dict(), epoch, val_loss, torch)
                )

            # 完整痕迹写入 data/TrainData
            timestamp = datetime.now().isoformat(timespec="seconds")
            elapsed_seconds = round(time.time() - run_start, 2)

            archive.record_epoch(
                epoch,
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "timestamp": timestamp,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "best_val_loss": round(best_val_loss, 6),
                    "improved": int(bool(improved)),
                    "epoch_seconds": round(epoch_seconds, 2),
                    "elapsed_seconds": elapsed_seconds,
                    "macro_f1": split_metrics.get("macro_f1", ""),
                    "micro_f1": split_metrics.get("micro_f1", ""),
                    "subset_accuracy": split_metrics.get("subset_accuracy", ""),
                    "hamming_loss": split_metrics.get("hamming_loss", ""),
                    "macro_ap": "" if split_metrics.get("macro_ap") is None
                                  else split_metrics["macro_ap"],
                    "checkpoint": archived,
                },
                detail={
                    "run_id": run_id,
                    "epoch": epoch,
                    "timestamp": timestamp,
                    "device": str(device),
                    "batch_size": batch_size,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "n_train": n_train,
                    "n_val": n_val,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "best_val_loss": round(best_val_loss, 6),
                    "improved": bool(improved),
                    "epoch_seconds": round(epoch_seconds, 2),
                    "elapsed_seconds": elapsed_seconds,
                    "metrics": split_metrics,
                    "per_class": per_class,
                    "checkpoint": archived,
                    "interrupted": False,
                },
            )

            log_row(epoch, train_loss, val_loss, improved, epoch_seconds, False)

            f1_text = ""

            if split_metrics.get("macro_f1") is not None and split_metrics:
                f1_text = f" | macro-F1: {split_metrics['macro_f1']:.4f}"

            print(
                f"Epoch {epoch} | train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f}{f1_text} | {epoch_seconds:.1f}s"
                + ("  [最佳，已保存]" if improved else "")
            )

            if archived:
                print(f"  归档权重: {archived}")

            if scheduler is not None:
                scheduler.step(val_loss)

            _write_status(
                state = "running",
                epoch = epoch,
                epochs_done = epochs_done,
                batch = len(train_loader),
                train_loss = train_loss,
                val_loss = val_loss,
                best_val_loss = best_val_loss,
            )

    except KeyboardInterrupt:
        
        stopped_reason = "keyboard_interrupt"
        print("\n收到 Ctrl+C，正在保存并退出...")

    finally:
        history_fh.close()

        _write_status(
            state = "finished" if stopped_reason == "max_epochs" else "stopped",
            stopped_reason = stopped_reason,
            epochs_done = epochs_done,
            best_val_loss = None if math.isinf(best_val_loss) else best_val_loss,
            finished_at = datetime.now().isoformat(timespec="seconds"),
        )

        summary = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "stopped_reason": stopped_reason,
            "epochs_done": epochs_done,
            "best_epoch": best_epoch,
            "best_val_loss": None if math.isinf(best_val_loss) else best_val_loss,
            "total_seconds": round(time.time() - run_start, 2),
            "lr": lr,
            "batch_size": batch_size,
            "dropout": dropout,
            "loss_fn": "focal",
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "focal_per_class_alpha": bool(class_alpha is not None),
            "lr_scheduler": lr_scheduler_name,
            "device": str(device),
            "n_features": model.n_features,
            "n_1h": model.n_1h,
            "h1_bins": model.h1_bins,
            "h1_channels": model.h1_channels,
            "h1_scalars": model.h1_scalars,
            "n_classes": model.n_classes,
            "n_train": n_train,
            "n_val": n_val,
            "history_csv": str(history_path),
            "best_model_path": str(best_path) if best_epoch > 0 else "",
            "last_model_path": str(last_path), 

            "train_data_dir": str(archive.run_dir),
            "epochs_csv": str(archive.epochs_csv),
            "checkpoints_dir": str(archive.checkpoint_dir),
            "archived_checkpoints": len(list(archive.checkpoint_dir.glob("*.pth")))
            if archive.checkpoint_dir.exists()
            else 0,
        }

        try:
            append_jsonl(summary_path, summary)
        except KeyboardInterrupt:
            print("汇总写入被中断，但逐 epoch 的 CSV 已完整保存")

        # 把结束汇总写进 TrainData（run.json + runs_index.csv）
        try:
            archive.finish(
                {
                    "stopped_reason": stopped_reason,
                    "epochs_done": epochs_done,
                    "best_epoch": best_epoch,
                    "best_val_loss": None if math.isinf(best_val_loss) else best_val_loss,
                    "total_seconds": round(time.time() - run_start, 2),
                    "device": str(device),
                }
            )
        except KeyboardInterrupt:
            print("训练痕迹汇总写入被中断，但逐 epoch 记录已完整保存")

        if epochs_done > 0:
            print(
                f"本次运行结束：完成 {epochs_done} 个 epoch，"
                f"最佳验证 loss {best_val_loss:.4f} (epoch {best_epoch})"
            )
        else:
            print("本次运行没有完成任何 epoch")

        print(f"逐 epoch 记录: {history_path}")
        print(f"运行汇总: {summary_path}")
        print(f"训练痕迹目录: {archive.run_dir}")
        print(f"  逐轮记录: {archive.epochs_csv}")
        print(f"  归档权重: {archive.checkpoint_dir}")

    return summary


if __name__ == "__main__":
    main()
