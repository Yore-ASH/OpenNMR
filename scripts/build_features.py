"""一键生成 CSV 与特征文件。

流程：parquet -> csv（保留 nmr_1h） -> X/Y（13C） -> X1H（1H 积分/多重峰/J）

用法（在项目根目录）：
    python scripts/build_features.py
"""

import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

import parser  # noqa: E402  （需要先设置 sys.path）


def main():
    parser.parse_parquet_file()
    parser.gen_XY_pt()
    parser.gen_1h_pt()


if __name__ == "__main__":
    main()
