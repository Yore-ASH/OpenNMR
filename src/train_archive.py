"""训练痕迹归档"""

import csv
import json

from datetime import datetime
from pathlib import Path


# 逐 epoch 汇总表的列
EPOCH_FIELDS = [
    "run_id",
    "epoch",
    "timestamp",
    "lr",
    "train_loss",
    "val_loss",
    "best_val_loss",
    "improved",
    "epoch_seconds",
    "elapsed_seconds",
    "macro_f1",
    "micro_f1",
    "subset_accuracy",
    "hamming_loss",
    "macro_ap",
    "checkpoint",
]

# 所有运行的索引表
INDEX_FIELDS = [
    "run_id",
    "started_at",
    "finished_at",
    "stopped_reason",
    "epochs_done",
    "best_epoch",
    "best_val_loss",
    "total_seconds",
    "device",
    "run_dir",
]


def _write_json(path, payload):
    """[原子写] JSON：先写 .tmp 再替换，避免中途被打断留下半个文件。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def _append_csv_row(path, fields, row):
    """追加一行 CSV（不存在则先写表头），写完立即关闭，保证落盘。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not path.exists()

    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
        fh.flush()


class TrainRun:
    """一次训练的痕迹归档器。"""

    def __init__(self, root, run_id):
        self.root = Path(root)
        self.run_id = str(run_id)
        self.run_dir = self.root / f"run_{self.run_id}"
        self.epochs_csv = self.run_dir / "epochs.csv"
        self.run_json = self.run_dir / "run.json"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.index_csv = self.root / "runs_index.csv"

        self.started_at = datetime.now().isoformat(timespec="seconds")

    def begin(self, config):
        """建目录、写 run.json（先记录配置，训练中断也能知道当时用的什么参数）。"""

        self.config = dict(config)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        _write_json(
            self.run_json,
            {
                "run_id": self.run_id,
                "started_at": self.started_at,
                "finished_at": "",
                "stopped_reason": "running",
                "epochs_done": 0,
                "epochs_max": self.config.get("epochs_max", 0),
                "best_epoch": 0,
                "best_val_loss": None,
                "config": self.config,
                "run_dir": str(self.run_dir),
            },
        )

    def finish(self, summary):
        """写结束汇总，并把这次运行追加到 runs_index.csv。

        注意：是在 begin() 写好的 run.json 上做合并更新，**不会丢掉 config**。
        """

        payload = {}

        try:
            payload = json.loads(self.run_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}

        payload.update(summary)
        payload["run_id"] = self.run_id
        payload["started_at"] = self.started_at
        payload["finished_at"] = summary.get(
            "finished_at", datetime.now().isoformat(timespec="seconds")
        )
        payload["run_dir"] = str(self.run_dir)
        payload.setdefault("config", dict(getattr(self, "config", {})))

        _write_json(self.run_json, payload)
        _append_csv_row(self.index_csv, INDEX_FIELDS, payload)

    def record_epoch(self, epoch, row, detail):
        """记录一个 epoch"""

        _append_csv_row(self.epochs_csv, EPOCH_FIELDS, row)
        _write_json(self.run_dir / f"epoch_{epoch:04d}.json", detail)

    def checkpoint_name(self, epoch, val_loss):
        """归档权重的文件名：epoch_0002_val0.118900.pth"""

        return f"epoch_{epoch:04d}_val{val_loss:.6f}.pth"

    def save_checkpoint(self, state_dict, epoch, val_loss, torch_module, tag=None):
        """将权重归档至 checkpoints/，返回文件路径。"""

        if tag:
            name = f"epoch_{epoch:04d}_{tag}.pth"
        else:
            name = self.checkpoint_name(epoch, val_loss)

        path = self.checkpoint_dir / name

        tmp = path.with_name(path.name + ".tmp")
        torch_module.save(state_dict, tmp)
        tmp.replace(path)

        return path
