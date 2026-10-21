"""打印 / 校验当前生效的配置

用法（在项目根目录）：

    python main_cli.py settings              # 打印最终生效的配置（含派生值）
    python main_cli.py settings --check      # 额外检查 settings.json 与内置 DEFAULTS 是否一致

为什么需要它：配置现在分散在 `settings.json`（用户改）、`settings.local.json`（本机覆盖）
和 `src/setup.py` 的 `DEFAULTS`（兜底）三层。改完不确定有没有生效、或者想确认
「类别数/输入维度/1H 总维度」这些派生值算成了多少，跑这个脚本最直接。
"""

import argparse
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import setup as config  # noqa: E402


def _walk_leaf(node, prefix=""):
    """展开成 (dotted_key, value)，只保留叶子。"""

    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).startswith("//"):
                continue
            yield from _walk_leaf(value, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, node


def _flatten_defaults():
    flat = {}
    for key, value in _walk_leaf(config.DEFAULTS):
        flat[key] = value
    return flat


def _flatten_json(path: Path):
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    flat = {}
    for key, value in _walk_leaf(raw):
        flat[key] = value
    return flat


def check_files() -> int:
    """检查 settings.json / settings.local.json 是否与代码里的 DEFAULTS 漂移。"""

    print("=" * 74)
    print("配置一致性检查")
    print("=" * 74)

    problems = 0

    for path in (config.SETTINGS_FILE, config.LOCAL_SETTINGS_FILE):
        if not path.exists():
            kind = "（可缺省）" if path == config.LOCAL_SETTINGS_FILE else "（缺省时用 DEFAULTS 兜底）"
            print(f"  {path.name:<24} 不存在 {kind}")
            continue

        flat = _flatten_json(path)
        print(f"  {path.name:<24} 存在，{len(flat)} 项")

    # settings.json 里出现、但 DEFAULTS 里没有的键 -> 拼错或者已废弃
    defaults = _flatten_defaults()
    for path in (config.SETTINGS_FILE, config.LOCAL_SETTINGS_FILE):
        flat = _flatten_json(path)
        unknown = sorted(set(flat) - set(defaults))
        if unknown:
            problems += len(unknown)
            print(f"\n  [!] {path.name} 里有 DEFAULTS 不认识的键（可能拼错或已废弃）：")
            for key in unknown:
                print(f"        {key} = {flat[key]!r}")

    # settings.json 与 DEFAULTS 的值是否一致（两份默认值最容易在这里漂移）
    flat_main = _flatten_json(config.SETTINGS_FILE)
    drift = [
        (key, flat_main[key], defaults[key])
        for key in sorted(flat_main)
        if key in defaults and flat_main[key] != defaults[key]
    ]
    if drift:
        print(f"\n  [!] settings.json 与 DEFAULTS 有 {len(drift)} 项值不一致"
              f"（说明两份默认值已经漂移，改一处就必须同步另一处）：")
        for key, got, want in drift:
            print(f"        {key:<34} settings.json={got!r}   DEFAULTS={want!r}")
        problems += len(drift)
    else:
        print(f"  settings.json 与 DEFAULTS 的 {len(flat_main)} 项取值完全一致")

    # 派生的字段不该出现在 JSON 里
    derived_keys = (
        "features.h1.total_dim",
        "features.h1.spec_dim",
        "model.n_classes",
        "model.n_features",
        "labels",
    )
    for path in (config.SETTINGS_FILE, config.LOCAL_SETTINGS_FILE):
        flat = _flatten_json(path)
        for key in derived_keys:
            if key in flat:
                problems += 1
                print(f"\n  [!] {path.name} 里出现了派生字段 {key} —— 派生值由 setup.py 计算，写了会被忽略")

    if config.RDKIT_IMPORT_ERROR is not None:
        print(f"\n  [!] rdkit 导入失败：{config.RDKIT_IMPORT_ERROR}")
        problems += 1

    print()
    if problems:
        print(f"发现 {problems} 个问题")
        return 1

    print("没有发现问题：settings.json 与 DEFAULTS 的键一致，派生字段未被硬写。")
    return 0


def show_settings():
    summary = config.SETTINGS.summary()

    print("=" * 74)
    print("当前生效的配置")
    print("=" * 74)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    print()
    print("=" * 74)
    print("派生值（由 setup.py 算出，不要在 settings.json 里写）")
    print("=" * 74)
    print(f"  类别数            model.n_classes   = {config.SETTINGS.model.n_classes}"
          f"   (labels 里有 {len(config.SETTINGS.labels)} 个官能团)")
    print(f"  13C 输入维度      model.n_features  = {config.SETTINGS.model.n_features}"
          f"   (= features.c13.buckets)")
    print(f"  1H 直方图维度     h1.spec_dim       = {config.SETTINGS.features.h1.spec_dim}"
          f"   (= {config.SETTINGS.features.h1.buckets} × {config.SETTINGS.features.h1.channels})")
    print(f"  1H 特征总维度     h1.total_dim      = {config.SETTINGS.features.h1.total_dim}"
          f"   (= {config.SETTINGS.features.h1.spec_dim} + {config.SETTINGS.features.h1.scalars})")
    print(f"  模型输入总维度                      = "
          f"{config.SETTINGS.model.n_features + config.SETTINGS.features.h1.total_dim}")
    print(f"  已编译 SMARTS 规则                  = {len(config.SETTINGS.patterns)} 条")

    print()
    print("=" * 74)
    print("配置来源")
    print("=" * 74)
    for path in (config.SETTINGS_FILE, config.LOCAL_SETTINGS_FILE):
        state = "存在，会读取" if path.exists() else "不存在"
        note = "（覆盖层，优先级最高）" if path == config.LOCAL_SETTINGS_FILE else "（主配置）"
        print(f"  {path.name:<24} {state} {note}")
    print("  src/setup.py DEFAULTS      兜底：上面两层缺某键时用它")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="打印 / 校验当前生效的配置")
    parser.add_argument("--check", action="store_true", help="检查 settings.json 与 DEFAULTS 是否漂移")
    parser.add_argument("--json", action="store_true", help="只输出 JSON（方便管道处理）")
    args = parser.parse_args(argv)

    if args.json:
        print(json.dumps(config.SETTINGS.summary(), ensure_ascii=False, indent=2, default=str))
        return 0

    code = 0

    if args.check:
        code = check_files()

    show_settings()

    return code


if __name__ == "__main__":
    raise SystemExit(main())
