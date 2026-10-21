#!/usr/bin/env bash
# 在 Linux 上为 CNMR-CUDA 创建 Python 3.10 虚拟环境，并装好 Pascal（MX150 / GTX 1060）可用的 CUDA 版 PyTorch。
#
# MX150 和 GTX 1060 都是 Pascal 架构(sm_61)。PyTorch 2.15 起不再发布 Pascal 内核，
# 所以固定 torch 2.14.1 + cu126（最后一个覆盖 Pascal 的 CUDA 12.x 构建）。
#
# 用法（在项目根目录）:
#   bash scripts/install_cuda.sh
#   PYTHON_BIN=python3.10 bash scripts/install_cuda.sh
#   CPU_ONLY=1 bash scripts/install_cuda.sh          # 只装 CPU 版
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "== 项目根目录: $PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.14.1}"
CUDA_CHANNEL="${CUDA_CHANNEL:-cu126}"
VENV_DIR="${VENV_DIR:-.venv}"
CPU_ONLY="${CPU_ONLY:-0}"

# ---------------------------------------------------------------- 1. 检查 Python
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "找不到 $PYTHON_BIN。请先安装 Python 3.10.11，或用 PYTHON_BIN=... 指定。" >&2
    exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys;print("%d.%d" % sys.version_info[:2])')"
if [ "$PY_VERSION" != "3.10" ]; then
    echo "当前 $PYTHON_BIN 是 $PY_VERSION，本项目要求 3.10。" >&2
    exit 1
fi
echo "== 使用 Python: $("$PYTHON_BIN" -c 'import sys;print(sys.version.split()[0])')"

# ---------------------------------------------------------------- 2. 建虚拟环境
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "== 创建虚拟环境 $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "== 复用已有虚拟环境 $VENV_DIR"
fi
VENV_PY="$VENV_DIR/bin/python"

"$VENV_PY" -m pip install --upgrade pip

# ---------------------------------------------------------------- 3. 装 PyTorch
if [ "$CPU_ONLY" = "1" ]; then
    echo "== 安装 CPU 版 torch $TORCH_VERSION"
    "$VENV_PY" -m pip install "torch==$TORCH_VERSION"
else
    echo "== 安装 CUDA 版 torch $TORCH_VERSION ($CUDA_CHANNEL)"
    "$VENV_PY" -m pip install "torch==$TORCH_VERSION+$CUDA_CHANNEL" \
        --index-url "https://download.pytorch.org/whl/$CUDA_CHANNEL"
fi

# ---------------------------------------------------------------- 4. 其它依赖
echo "== 安装其余依赖"
"$VENV_PY" -m pip install -r requirements.txt

# ---------------------------------------------------------------- 5. 自检
echo
echo "== 自检: 显卡是否真的可用"
set +e
"$VENV_PY" main_cli.py cuda-info
CUDA_OK=$?
set -e

if [ "$CUDA_OK" -eq 0 ]; then
    cat <<EOF

== 安装完成，CUDA 已可用。以后这样运行：
   $VENV_DIR/bin/python main_cli.py train --epochs 3
   $VENV_DIR/bin/python main_cli.py eval --split test --threshold 0.35
   $VENV_DIR/bin/python main_cli.py dashboard --port 8090
EOF
else
    cat <<EOF

== 依赖装完了，但 CUDA 仍不可用（训练会自动回退 CPU）。
   请按上面 cuda-info 打印的原因处理，最常见的是显卡驱动太旧：
   升级驱动并重启后，再跑一次  $VENV_DIR/bin/python main_cli.py cuda-info
EOF
fi
