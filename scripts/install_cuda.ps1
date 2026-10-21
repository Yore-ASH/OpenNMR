<#
.SYNOPSIS
    在 Windows 上为 CNMR-CUDA 创建 Python 3.10 虚拟环境，并装好 Pascal（MX150 / GTX 1060）可用的 CUDA 版 PyTorch。

.DESCRIPTION
    MX150 和 GTX 1060 都是 Pascal 架构 (sm_61)。PyTorch 2.15 起不再发布 Pascal 内核，
    所以这里固定 torch 2.14.1 + cu126（最后一个覆盖 Pascal 的 CUDA 12.x 构建）。

    另外：Windows 上从 PyPI 直接 pip install torch 拿到的是 CPU-only 轮子，
    必须加 --index-url https://download.pytorch.org/whl/cu126 才能用上显卡。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_cuda.ps1

.EXAMPLE
    # 只装 CPU 版（没有 N 卡，或者只想先跑通）
    powershell -ExecutionPolicy Bypass -File scripts\install_cuda.ps1 -CpuOnly
#>
[CmdletBinding()]
param(
    [string]$PythonExe    = '',
    [string]$TorchVersion = '2.14.1',
    [string]$CudaChannel  = 'cu126',
    [string]$VenvDir      = '.venv',
    [switch]$CpuOnly
)

$ErrorActionPreference = 'Stop'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
Write-Host "== 项目根目录: $ProjectRoot" -ForegroundColor Cyan

# ---------------------------------------------------------------- 1. 找 Python 3.10
if ($PythonExe) {
    $baseExe = $PythonExe
    $baseArgs = @()
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.10 -c "import sys" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "py -3.10 不可用。请安装 Python 3.10.11，或用 -PythonExe 指定解释器路径。"
    }
    $baseExe = 'py'
    $baseArgs = @('-3.10')
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $baseVersion = (& python -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null)
    if ($baseVersion -ne '3.10') {
        throw "当前 python 是 $baseVersion，需要 3.10。请安装 Python 3.10.11，或用 -PythonExe 指定。"
    }
    $baseExe = 'python'
    $baseArgs = @()
} else {
    throw "找不到 python。请安装 Python 3.10.11（勾选 Add python.exe to PATH）。"
}

$baseVersionText = (& $baseExe @baseArgs -c "import sys;print(sys.version.split()[0])")
Write-Host "== 使用 Python: $baseVersionText" -ForegroundColor Cyan

# ---------------------------------------------------------------- 2. 建虚拟环境
$venvPython = Join-Path $ProjectRoot "$VenvDir\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "== 创建虚拟环境 $VenvDir" -ForegroundColor Cyan
    & $baseExe @baseArgs -m venv $VenvDir
} else {
    Write-Host "== 复用已有虚拟环境 $VenvDir" -ForegroundColor Cyan
}

& $venvPython -m pip install --upgrade pip

# ---------------------------------------------------------------- 3. 装 PyTorch
if ($CpuOnly) {
    Write-Host "== 安装 CPU 版 torch $TorchVersion" -ForegroundColor Cyan
    & $venvPython -m pip install "torch==$TorchVersion"
} else {
    Write-Host "== 安装 CUDA 版 torch $TorchVersion ($CudaChannel，约 2.6 GB，请耐心等待)" -ForegroundColor Cyan
    & $venvPython -m pip install "torch==$TorchVersion+$CudaChannel" `
        --index-url "https://download.pytorch.org/whl/$CudaChannel"
}

# ---------------------------------------------------------------- 4. 其它依赖
Write-Host "== 安装其余依赖" -ForegroundColor Cyan
& $venvPython -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')

# ---------------------------------------------------------------- 5. 自检
Write-Host "`n== 自检: 显卡是否真的可用" -ForegroundColor Cyan
& $venvPython main_cli.py cuda-info
$cudaOk = ($LASTEXITCODE -eq 0)

if ($cudaOk) {
    Write-Host @"

== 安装完成，CUDA 已可用。以后这样运行：
   $VenvDir\Scripts\python.exe main_cli.py train --epochs 3
   $VenvDir\Scripts\python.exe main_cli.py eval --split test --threshold 0.35
   $VenvDir\Scripts\python.exe main_cli.py dashboard --port 8090
"@ -ForegroundColor Green
} else {
    Write-Host @"

== 依赖装完了，但 CUDA 仍不可用（训练会自动回退 CPU）。
   请按上面 cuda-info 打印的原因处理，最常见的是显卡驱动太旧：
   去 NVIDIA 官网按显卡型号下载最新驱动，安装并重启后再跑一次
   $VenvDir\Scripts\python.exe main_cli.py cuda-info
"@ -ForegroundColor Yellow
}
