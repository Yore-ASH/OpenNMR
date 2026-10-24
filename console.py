"""OpenNMR 控制台入口"""
print("Thank you for using OpenNMR (From ASH).\n欢迎使用 OpenNMR (来自 ASH).")

import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "cuda-info":    ROOT / "src" / "cuda_info.py",
    "train":        ROOT / "src" / "train.py",
    "eval":         ROOT / "src" / "f1_test.py",
    "dashboard":    ROOT / "src" / "dashboard.py",
}

FEATURES_SCRIPT =   ROOT / "scripts" / "build_features.py"
AB_TEST_SCRIPT =    ROOT / "scripts" / "ab_test.py"
SETTINGS_SCRIPT =   ROOT / "scripts" / "check_settings.py"

HELP = """OpenNMR-console.py
python console.py settings [--check]       打印 / 校验当前生效的配置
python console.py cuda-info               s 检查 torch / CUDA / 显卡是否可用
python console.py features                 从 parquet 生成 CSV 与 13C/1H 特征
python console.py train [参数...]          训练模型
python console.py eval [参数...]            F1 评估
python console.py dashboard [参数...]      启动训练看板
python console.py ab-test [参数...]        R4 / R5 架构 A/B 对比
"""

def console_main():
    argv = sys.argv[1:]

    if (not argv) or argv[0] in ("-h", "--help"):
        print(HELP)
        return 0

    command = argv[0]

    match argv[0]:
        case "features":
            if len(argv) > 1:
                        print("features 不接受额外参数")
                        return 2
            return subprocess.call([sys.executable, str(FEATURES_SCRIPT)])

        case "ab-test":
            return subprocess.call([sys.executable, str(AB_TEST_SCRIPT), *argv[1:]])

        case "settings":
            return subprocess.call([sys.executable, str(SETTINGS_SCRIPT), *argv[1:]])
        
    script = COMMANDS.get(command)

    if script is None:
        print(f"未知命令: {command}\n")
        print(HELP)
        return 2

    return subprocess.call([sys.executable, str(script), *argv[1:]])

def interactive_main():
    while True:

        command = input("\033[1;33mOpen$ \033[0m").split()

        match command[0].lower():
            case "help":

                print(HELP)
                
            case "features":

                return subprocess.call([sys.executable, str(FEATURES_SCRIPT)])

            case "ab-test":
                
                return subprocess.call([sys.executable, str(AB_TEST_SCRIPT), *command[1:]])

            case "settings":

                return subprocess.call([sys.executable, str(SETTINGS_SCRIPT), *command[1:]])

            case "cuda-info"|"train"|"eval"|"dashboard":
                return subprocess.call([sys.executable, str(COMMANDS.get(command[0])), *command[1:]])

            case _:

                print(f"[Unknown Command]未知命令: {command[0]}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        raise SystemExit(console_main())
    else:
        interactive_main()