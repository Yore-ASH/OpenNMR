"""CNMR-0.9.0 架构 A/B 对比脚本

在同一份**真实数据切片**、同一个随机种子、完全相同的训练超参下，
分别训练 `MODEL_ARCH=r4`（0.8.0 基线）和 `r5`（优化版），然后对比：

- 参数数量 / 每轮耗时（速度）
- val_loss、macro-F1@0.5、micro-F1、macro-AP（阈值无关）、Hamming
- 逐类 F1 的差异（哪些官能团受益、哪些受害）

用法（在项目根目录）：

    python scripts/ab_test.py                          # 默认 80k/15k、3 轮
    python scripts/ab_test.py --epochs 5 --train 150000 --val 20000
    python scripts/ab_test.py --archs r5                 # 只跑某一个

结果写到 `data/ABTest/<时间戳>/`：
    comparison.csv    两个架构逐轮指标，一张表
    comparison.md     人读的对比报告
    summary.json      机器读的汇总
    runs/             每个架构完整的训练痕迹（TrainData 结构）

注意：这是**受控子集实验**，不是全量训练结论。子集越小、轮数越少，
越偏向「收敛快」的模型；BN 主要收益就是收敛速度，所以小预算下 R5 的优势
可能被高估。要看准请用默认参数以上规模重跑。
"""

import argparse
import csv
import json
import sys
import tempfile
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from setup import SETTINGS  # noqa: E402

DEFAULT_ARCHS = ("r4", "r5")


# ---------------------------------------------------------------- 数据切片


def slice_real_data(src_pt: Path, dst_pt: Path, n_train: int, n_val: int):
    """从完整特征里切出前 n_train / n_val 条，保持 13C、1H、标签三者对齐。"""

    dst_pt.mkdir(parents=True, exist_ok=True)

    plan = (("train", n_train), ("validation", n_val))

    for name, n in plan:
        for prefix in ("X_", "Y_", "X1H_"):
            src = src_pt / f"{prefix}{name}.pt"
            if not src.exists():
                raise FileNotFoundError(f"缺少特征文件: {src}")

            tensor = torch.load(src, mmap=True, weights_only=True)[:n].clone()
            torch.save(tensor, dst_pt / f"{prefix}{name}.pt")

    meta = src_pt / "X1H_meta.json"
    if meta.exists():
        (dst_pt / "X1H_meta.json").write_text(
            meta.read_text(encoding="utf-8"), encoding="utf-8"
        )


# ---------------------------------------------------------------- 单次训练


def train_one_arch(arch, work_root, data_dir, epochs, batch_size, seed):
    """跑一个架构，返回 {arch, epochs:[...], run_json, seconds}。"""

    run_root = work_root / arch
    run_root.mkdir(parents=True, exist_ok=True)

    # 每次训练都用独立的输出目录，避免互相覆盖
    # 用 dotted-key 覆盖，和改 settings.json 完全等价；每个架构跑完会重置
    SETTINGS.apply(
        {
            "paths.pt_dir": data_dir,
            "paths.model_dir": run_root / "ModelData",
            "paths.log_data_dir": run_root / "LogData",
            "paths.train_data_dir": run_root / "TrainData",
            "model.arch": arch,
            "training.epochs": epochs,
            "training.workers": 0,
            "device.train_device": "auto",
        }
    )

    import train as train_module

    started = time.time()
    train_module.main(
        [
            "--epochs", str(epochs),
            "--workers", "0",
            "--batch-size", str(batch_size),
            "--arch", arch,
            "--seed", str(seed),
        ]
    )
    seconds = time.time() - started

    # ---------- 读回训练痕迹 ----------
    train_data = run_root / "TrainData"
    run_dirs = sorted(p for p in train_data.iterdir() if p.is_dir())
    if not run_dirs:
        raise RuntimeError(f"{arch}: 没找到训练痕迹目录")

    run_dir = run_dirs[-1]
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    rows = []
    with open(run_dir / "epochs.csv", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    # 逐类指标从 epoch json 里取最后一轮
    per_class = {}
    last_epoch = int(rows[-1]["epoch"])
    detail = json.loads(
        (run_dir / f"epoch_{last_epoch:04d}.json").read_text(encoding="utf-8")
    )
    per_class = detail.get("per_class", {})

    return {
        "arch": arch,
        "seconds": seconds,
        "epochs": rows,
        "run_json": run_json,
        "per_class": per_class,
        "run_dir": str(run_dir),
        "n_params": None,   # 由调用方填充
    }


# ---------------------------------------------------------------- 报告

FIELDS = ["epoch", "val_loss", "macro_f1", "micro_f1", "macro_ap", "hamming_loss"]


def _float(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def write_report(results, out_dir, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    archs = [r["arch"] for r in results]

    # ---------- comparison.csv ----------
    with open(out_dir / "comparison.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        header = ["metric"]
        for arch in archs:
            header += [f"{arch}"]
        writer.writerow(header)

        writer.writerow(["n_params"] + [r["n_params"] for r in results])
        writer.writerow(["total_seconds"] + [round(r["seconds"], 1) for r in results])
        writer.writerow(
            ["seconds_per_epoch"]
            + [round(r["seconds"] / max(1, len(r["epochs"])), 1) for r in results]
        )

        for field in FIELDS:
            if field == "epoch":
                continue
            row = [field]
            for result in results:
                if not result["epochs"]:
                    row.append("")
                    continue
                # 最后一行 = 最后一轮
                row.append(round(_float(result["epochs"][-1], field), 6))
            writer.writerow(row)

        # 逐类 F1
        writer.writerow([])
        writer.writerow(["per_class_f1"] + archs + ["delta(r5-r4)" if set(archs) == {"r4", "r5"} else "delta"])
        classes = sorted({c for r in results for c in r["per_class"]})
        for cls in classes:
            values = [r["per_class"].get(cls, {}).get("f1", "") for r in results]
            delta = ""
            if len(values) == 2 and values[0] != "" and values[1] != "":
                delta = round(float(values[1]) - float(values[0]), 4)
            writer.writerow([cls] + values + [delta])

    # ---------- comparison.md ----------
    lines = []
    lines.append("# CNMR-0.9.0 架构 A/B 对比\n")
    lines.append("## 实验配置\n")
    for key, value in config.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    lines.append("## 汇总\n")
    lines.append("| 指标 | " + " | ".join(archs) + " |")
    lines.append("|---" * (len(archs) + 1) + "|")
    lines.append("| 参数量 | " + " | ".join(f"{r['n_params']:,}" for r in results) + " |")
    lines.append(
        "| 总耗时(s) | " + " | ".join(f"{r['seconds']:.0f}" for r in results) + " |"
    )
    lines.append(
        "| 每轮耗时(s) | "
        + " | ".join(f"{r['seconds'] / max(1, len(r['epochs'])):.0f}" for r in results)
        + " |"
    )
    for field in FIELDS:
        if field == "epoch":
            continue
        cells = []
        for result in results:
            if result["epochs"]:
                cells.append(f"{_float(result['epochs'][-1], field):.4f}")
            else:
                cells.append("-")
        lines.append(f"| {field}（最后一轮） | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 逐轮 macro-AP（阈值无关，最可靠）\n")
    lines.append("| epoch | " + " | ".join(archs) + " |")
    lines.append("|---" * (len(archs) + 1) + "|")
    max_epochs = max((len(r["epochs"]) for r in results), default=0)
    for index in range(max_epochs):
        cells = []
        for result in results:
            if index < len(result["epochs"]):
                cells.append(f"{_float(result['epochs'][index], 'macro_ap'):.4f}")
            else:
                cells.append("-")
        lines.append(f"| {index + 1} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 逐类 F1（最后一轮）\n")
    lines.append("| 官能团 | " + " | ".join(archs) + " | delta |")
    lines.append("|---" * (len(archs) + 2) + "|")
    classes = sorted({c for r in results for c in r["per_class"]})
    for cls in classes:
        values = [r["per_class"].get(cls, {}).get("f1", "") for r in results]
        delta = ""
        if len(values) == 2 and values[0] != "" and values[1] != "":
            delta = f"{float(values[1]) - float(values[0]):+.4f}"
        cells = [f"{v:.4f}" if v != "" else "-" for v in values]
        lines.append(f"| {cls} | " + " | ".join(cells) + f" | {delta} |")
    lines.append("")
    lines.append(
        "> 这是受控子集实验，不是全量训练结论。轮数越少越偏向收敛快的模型。\n"
    )

    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------- summary.json ----------
    summary = {
        "config": config,
        "archs": {},
    }
    for result in results:
        summary["archs"][result["arch"]] = {
            "n_params": result["n_params"],
            "total_seconds": round(result["seconds"], 1),
            "final": {
                field: _float(result["epochs"][-1], field)
                for field in FIELDS
                if field != "epoch" and result["epochs"]
            },
            "per_class_f1": {
                cls: data.get("f1") for cls, data in result["per_class"].items()
            },
            "run_dir": result["run_dir"],
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return summary


# ---------------------------------------------------------------- main


def main(argv=None):
    parser = argparse.ArgumentParser(description="CNMR-0.9.0 架构 A/B 对比")
    parser.add_argument("--train", type=int, default=80000, help="训练集切片数量")
    parser.add_argument("--val", type=int, default=15000, help="验证集切片数量")
    parser.add_argument("--epochs", type=int, default=3, help="每个架构训练几轮")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--archs", nargs="+", default=list(DEFAULT_ARCHS),
                        choices=list(DEFAULT_ARCHS))
    args = parser.parse_args(argv)

    from model import NMRClassifier

    # 用真实数据布局算两套架构的参数量，用于报告
    param_counts = {
        arch: sum(
            p.numel()
            for p in NMRClassifier(
                n_features=880, n_1h=852, h1_bins=280, h1_channels=3, h1_scalars=12,
                arch=arch,
            ).parameters()
        )
        for arch in args.archs
    }

    print("=" * 76)
    print("CNMR-0.9.0 架构 A/B 对比")
    print("=" * 76)
    print(f"  训练切片: {args.train:,} | 验证切片: {args.val:,}")
    print(f"  轮数: {args.epochs} | batch: {args.batch_size} | seed: {args.seed}")
    for arch, count in param_counts.items():
        print(f"  {arch}: {count:,} 参数")
    print("=" * 76)

    src_pt = ROOT / "data" / "PTFiles"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = ROOT / "data" / "ABTest" / stamp

    results = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        data_dir = tmp_root / "pt"

        print("\n切分数据...")
        slice_real_data(src_pt, data_dir, args.train, args.val)

        for arch in args.archs:
            print("\n" + "=" * 76)
            print(f"训练 {arch} ...")
            print("=" * 76)

            result = train_one_arch(
                arch=arch,
                work_root=tmp_root / "runs",
                data_dir=data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            result["n_params"] = param_counts[arch]
            results.append(result)

            # 把该架构的完整训练痕迹拷到输出目录留档
            import shutil

            target = out_root / "runs" / arch
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                Path(result["run_dir"]),
                target / Path(result["run_dir"]).name,
                dirs_exist_ok=True,
            )

    config = {
        "train": args.train,
        "val": args.val,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "archs": ",".join(args.archs),
        "loss": "focal(per_class_alpha=1)",
        "lr": SETTINGS.training.lr,
        "device": str(SETTINGS.device.train_device),
    }

    summary = write_report(results, out_root, config)

    print("\n" + "=" * 76)
    print("对比结果（最后一轮）")
    print("=" * 76)
    for arch in args.archs:
        info = summary["archs"][arch]
        final = info["final"]
        print(
            f"  {arch}: 参数 {info['n_params']:,} | "
            f"val_loss {final.get('val_loss', float('nan')):.4f} | "
            f"macro-F1 {final.get('macro_f1', float('nan')):.4f} | "
            f"macro-AP {final.get('macro_ap', float('nan')):.4f} | "
            f"{info['total_seconds']:.0f}s"
        )
    print(f"\n报告目录: {out_root}")
    print(f"  对比表: {out_root / 'comparison.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
