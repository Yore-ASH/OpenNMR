"""CUDA / 显卡自检（CNMR-CUDA 版新增）"""

import re
import shutil
import subprocess
import sys
import warnings

from pathlib import Path

import torch

# 替换无法解析字符为 "?" 避免 UnicodeError
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

NVIDIA_SMI_CANDIDATES = (
    "nvidia-smi",
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)


def _find_nvidia_smi():
    for candidate in NVIDIA_SMI_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
        if Path(candidate).exists():
            return candidate
    return None

def _windows_gpu_names():
    """从注册表读显卡完整型号"""
    try:
        import winreg
    except ImportError:
        return []

    names = []
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as key:
            index = 0
            while True:
                try:
                    sub_key = winreg.EnumKey(key, index)
                except OSError:
                    break
                index += 1
                if not sub_key.isdigit():
                    continue
                try:
                    with winreg.OpenKey(key, sub_key) as adapter:
                        names.append(str(winreg.QueryValueEx(adapter, "DriverDesc")[0]))
                except OSError:
                    continue
    except OSError:
        return []

    return names


def _query_nvidia_smi():
    """返回 {"name", "driver", "cuda"}，否则返回 None。"""
    executable = _find_nvidia_smi()
    if not executable:
        return None

    try:
        completed = subprocess.run(
            [executable],
            capture_output=True,
            text=True,
            timeout=30,
            errors="ignore",
        )
    except Exception:
        return None

    output = completed.stdout or ""
    match = re.search(r"Driver Version:\s*([\d.]+)\s+CUDA Version:\s*([\d.]+)", output)
    if not match:
        return None

    name = ""
    name_match = re.search(r"\|\s+\d+\s+(.+?)\s+WDDM", output)
    if name_match:
        name = name_match.group(1).strip()

    if not name or name in ("ERR!", "N/A") or "..." in name:
        for candidate in _windows_gpu_names():
            if "nvidia" in candidate.lower():
                name = candidate
                break

    return {"name": name or "NVIDIA GPU", "driver": match.group(1), "cuda": match.group(2)}


def main() -> int:
    # 初始化失败时判断原因

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cuda_available = torch.cuda.is_available()
    warning_text = " ".join(str(w.message) for w in caught)

    build_cuda = torch.version.cuda
    smi = _query_nvidia_smi()

    print("=" * 72)
    print(f"Python     : {sys.version.split()[0]}")
    print(f"torch      : {torch.__version__}")
    print(f"CUDA 构建  : {build_cuda or '(无 —— 这是 CPU 版 torch)'}")
    print(f"cuDNN      : {torch.backends.cudnn.version()}")

    if smi:
        print(f"显卡       : {smi['name']}")
        print(f"驱动       : {smi['driver']}  (驱动支持的 CUDA: {smi['cuda']})")
    else:
        print("显卡       : 未查询到 nvidia-smi（无 N 卡或驱动未安装正确）")

    print(f"cuda 可用  : {cuda_available}")

    arch_list = []
    try:
        arch_list = list(torch.cuda.get_arch_list())
        print(f"轮子含架构 : {', '.join(arch_list) if arch_list else '(未知)'}")
    except Exception:
        pass

    if not cuda_available:
        print("-" * 72)

        if build_cuda is None:
            print("原因：装的是 CPU 版 torch（没有 CUDA 支持），与显卡无关。")
            print("  修复：")
            print(
                'pip install \"torch==2.14.1+cu126\" --index-url https://download.pytorch.org/whl/cu126'
            )
        elif smi is None and not warning_text:
            print("原因：未检测到 NVIDIA 显卡（或驱动未安装）。")
            print("  如果是笔记本，请确认独显是否开启；台式机请确认显示器接在显卡上。")
        else:
            print(f"原因：torch 是 CUDA {build_cuda} 版，但显卡驱动太旧，无法初始化 CUDA。")
            if smi:
                print(
                    f"  你的驱动 {smi['driver']} 只支持到 CUDA {smi['cuda']}，"
                    f"而 torch {torch.__version__} 需要 CUDA {build_cuda}"
                    "（Pascal 架构需要 R525 及以上驱动）。"
                )
            elif "too old" in warning_text.lower():
                detail = warning_text.split("Triggered")[0].strip()
                print(f"  torch 自带报错：{detail[:200]}")
            print("  修复：去 NVIDIA 官网按显卡型号下载最新驱动，安装并重启，再重新运行此命令。")
            print("        升级后 nvidia-smi 里的 CUDA Version 会变成 12.x。")

        print("-" * 72)
        print("结论: CUDA 不可用。训练仍可运行，但会自动回退到 CPU。")
        print("=" * 72)
        return 1

    device_count = torch.cuda.device_count()
    print(f"显卡数量   : {device_count}")

    ok = True

    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        capability = f"sm_{props.major}{props.minor}"

        print("-" * 72)
        print(f"[{index}] {props.name}")
        print(f"    显存: {props.total_memory / 1024 ** 3:.2f} GiB")
        print(f"    计算能力: {capability}")

        if arch_list and capability not in arch_list:
            ok = False
            print(f"    这个 torch 轮子未编译 {capability} 内核（只有 {', '.join(arch_list)}）")
            print("      PyTorch 2.15 起不再提供 Pascal (sm_60/sm_61) 轮子，请降到 2.14 + cu126。")
            continue

        try:
            a = torch.randn(512, 512, device=f"cuda:{index}")
            result = (a @ a).sum().item()
            torch.cuda.synchronize(index)
            print(f"测试矩阵乘，结果 {result:.3f}")
        except Exception as exc:  # noqa: BLE001 - 把底层报错直接给用户看
            ok = False
            print(f"GPU 内核执行失败: {exc}")

    print("=" * 72)

    if ok:
        print("结论: CUDA 可用，训练会自动使用显卡。")
    else:
        print("结论: 显卡在，但内核缺失/执行失败，需要按 README 换正确的 torch 轮子。")

    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
