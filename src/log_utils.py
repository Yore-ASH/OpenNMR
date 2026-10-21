"""日志写入工具

"""

import csv
import json

from pathlib import Path
from datetime import datetime


def open_csv_logger(log_path, fields):
    """打开（必要时创建）CSV 日志"""

    log_path = Path(log_path)
    fields = list(fields)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():

        with open(log_path, "r", encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh), [])

        if header != fields:

            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = log_path.with_name(f"{log_path.stem}.{stamp}{log_path.suffix}")

            log_path.rename(backup)
            print(f"[日志] 表头与现有文件不一致，旧日志已备份为 {backup}")

    is_new = not log_path.exists()

    fh = open(log_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)

    if is_new:
        writer.writeheader()
        fh.flush()

    return fh, writer


def append_jsonl(path, record):
    """向 JSONL 文件追加一条记录（每个对象一行）"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def make_run_id():
    """生成运行 ID，精确到秒"""

    return datetime.now().strftime("%Y%m%d-%H%M%S")
