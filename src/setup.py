"""OpenNMR的程序依赖控制与变量管理模块"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.json"
LOCAL_SETTINGS_FILE = ROOT / "settings.local.json"

# 该变量仅在 settings.json 失效之后默认调用，请慎重修改
DEFAULTS = {
    "paths": {
        "parquet_dir": None,
        "csv_dir": None,
        "pt_dir": None,
        "model_dir": None,
        "log_data_dir": None,
        "train_data_dir": None,
        "console_log_dir": None,
    },
    "device": {
        "train_device": "auto",
    },
    "features": {
        "splits": ["test", "train", "validation"],
        "c13": {"buckets": 880, "ppm_min": 0.0, "ppm_max": 220.0},
        "h1": {
            "buckets": 280,
            "ppm_min": 0.0,
            "ppm_max": 14.0,
            "dtype": "float16",
        },
    },
    "labels": {
        "benzene": "c1ccccc1",
        "carbonyl": "[CX3]=[OX1]",
        "hydroxyl": "[OX2H]",
        "carboxyl": "C(=O)[OH]",
        "ester": "C(=O)O[C]",
        "amine": "[NX3]",
        "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
        "nitrile": "C#N",
        "alkene": "[CX3]=[CX3]",
        "alkyne": "C#C",
        "ether": "[OD2]([#6])[#6]",
        "amide": "C(=O)[NX3]",
        "halogen": "[F,Cl,Br,I]",
        "sulfonyl": "S(=O)(=O)",
    },
    "model": {
        "arch": "r4",
        "c13": {"kernel_sizes": [5, 5, 3], "channels": [32, 64, 128]},
        "h1": {"kernel_sizes": [5, 3], "out_channels": [32, 64], "scalar_hidden": 32},
        "r5": {"head_channels": 32, "head_h1_channels": 16},
        "head": {"hidden": 512, "dropout": 0.2},
    },
    "training": {
        "batch_size": 64,
        "default_batch_rest": 0.0,
        "lr": 0.001,
        "weight_decay": 0.0,
        "epochs": 0,
        "workers": 2,
        "seed": 2026,
        "focal": {"alpha": 0.25, "gamma": 2.0, "per_class_alpha": True},
        "lr_scheduler": {"name": "plateau", "factor": 0.5, "patience": 0},
    },
    "data": {"mmap": True},
    "archive": {"save_every_epoch": False, "epoch_metrics": True},
}

# 各目录的默认相对路径
DEFAULT_PATH_KEYS = {
    "parquet_dir": "data/OriginalData",
    "csv_dir": "data/CSVFiles",
    "pt_dir": "data/PTFiles",
    "model_dir": "data/ModelData",
    "log_data_dir": "data/LogData",
    "train_data_dir": "data/TrainData",
    "console_log_dir": "logs",
}

VALID_ARCHS = ("r4", "r5")
VALID_DEVICES = ("auto", "cpu", "cuda", "mps")
VALID_LR_SCHEDULERS = ("none", "plateau")
VALID_H1_DTYPES = ("float16", "float32")

H1_CHANNELS = 3
H1_SCALARS = 12

try:
    from rdkit import Chem as _Chem

    RDKIT_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - 取决于运行环境
    _Chem = None
    RDKIT_IMPORT_ERROR = _exc


class SettingsError(ValueError):
    """当 settings.json 中的内容不合法时抛出"""
    pass

@dataclass
class PathsCfg:
    root: Path
    parquet_dir: Path
    csv_dir: Path
    pt_dir: Path
    model_dir: Path
    log_data_dir: Path
    train_data_dir: Path
    console_log_dir: Path
    readme: Path


@dataclass
class C13Cfg:
    buckets: int
    ppm_min: float
    ppm_max: float


@dataclass
class H1Cfg:
    buckets: int
    ppm_min: float
    ppm_max: float
    channels: int
    scalars: int
    dtype: str

    @property
    def spec_dim(self) -> int:
        """1H 直方图部分维度（buckets × channels）"""
        return self.buckets * self.channels

    @property
    def total_dim(self) -> int:
        """1H 特征总维度（直方图 + 全局标量），默认 280×3+12 = 852"""
        return self.spec_dim + self.scalars


@dataclass
class FeaturesCfg:
    splits: list
    c13: C13Cfg
    h1: H1Cfg


@dataclass
class ModelCfg:
    arch: str
    c13_kernel_sizes: tuple
    c13_channels: tuple
    h1_kernel_sizes: tuple
    h1_out_channels: tuple
    h1_scalar_hidden: int
    r5_head_channels: int
    r5_head_h1_channels: int
    fc_hidden: int
    dropout: float

    # 以下字段请勿修改
    n_features: int = 0
    n_classes: int = 0


@dataclass
class TrainingCfg:
    batch_size: int
    default_batch_rest: float
    lr: float
    weight_decay: float
    epochs: int
    workers: int
    seed: int
    focal_alpha: float
    focal_gamma: float
    focal_per_class_alpha: bool
    lr_scheduler: str
    lr_factor: float
    lr_patience: int

    @property
    def max_epochs(self) -> int:
        """epochs <= 0 表示不限制训练轮数"""
        return 0 if self.epochs <= 0 else self.epochs


@dataclass
class DataCfg:
    mmap: bool


@dataclass
class ArchiveCfg:
    save_every_epoch: bool
    epoch_metrics: bool


@dataclass
class Settings:
    paths: PathsCfg
    device: str
    features: FeaturesCfg
    labels: dict
    label_names: tuple
    patterns: tuple
    model: ModelCfg
    training: TrainingCfg
    data: DataCfg
    archive: ArchiveCfg

    def _derive(self):
        """把所有能从别的参数算出来的值算一遍。唯一真相 → 派生，不再重复配置。"""

        self.label_names = tuple(self.labels.keys())
        self.model.n_classes = len(self.labels)
        self.model.n_features = self.features.c13.buckets

    def apply(self, overrides: dict) -> Settings:
        """按 dotted key 覆盖字段。"""

        for dotted, value in overrides.items():
            if value is None:
                continue

            node = self
            parts = dotted.split(".")

            for part in parts[:-1]:
                if not hasattr(node, part):
                    raise SettingsError(f"未知配置字段: {dotted}（在 {part} 处找不到）")
                node = getattr(node, part)

            attr = parts[-1]
            if not hasattr(node, attr):
                raise SettingsError(f"未知配置字段: {dotted}")

            setattr(node, attr, _coerce(value, getattr(node, attr), dotted))

        self._derive()
        return self

    def summary(self) -> dict:
        """导出成嵌套 dict（用于打印、写进 run.json 留档）。"""

        return {
            "paths": {
                "root": str(self.paths.root),
                "parquet_dir": str(self.paths.parquet_dir),
                "csv_dir": str(self.paths.csv_dir),
                "pt_dir": str(self.paths.pt_dir),
                "model_dir": str(self.paths.model_dir),
                "log_data_dir": str(self.paths.log_data_dir),
                "train_data_dir": str(self.paths.train_data_dir),
                "console_log_dir": str(self.paths.console_log_dir),
            },
            "device": self.device,
            "features": {
                "splits": list(self.features.splits),
                "c13": {
                    "buckets": self.features.c13.buckets,
                    "ppm_min": self.features.c13.ppm_min,
                    "ppm_max": self.features.c13.ppm_max,
                },
                "h1": {
                    "buckets": self.features.h1.buckets,
                    "ppm_min": self.features.h1.ppm_min,
                    "ppm_max": self.features.h1.ppm_max,
                    "channels": self.features.h1.channels,
                    "scalars": self.features.h1.scalars,
                    "dtype": self.features.h1.dtype,
                    "spec_dim": self.features.h1.spec_dim,
                    "total_dim": self.features.h1.total_dim,
                },
            },
            "labels": dict(self.labels),
            "model": {
                "arch": self.model.arch,
                "c13_kernel_sizes": list(self.model.c13_kernel_sizes),
                "c13_channels": list(self.model.c13_channels),
                "h1_kernel_sizes": list(self.model.h1_kernel_sizes),
                "h1_out_channels": list(self.model.h1_out_channels),
                "h1_scalar_hidden": self.model.h1_scalar_hidden,
                "r5_head_channels": self.model.r5_head_channels,
                "r5_head_h1_channels": self.model.r5_head_h1_channels,
                "fc_hidden": self.model.fc_hidden,
                "dropout": self.model.dropout,
                # 派生
                "n_features": self.model.n_features,
                "n_classes": self.model.n_classes,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "default_batch_rest": self.training.default_batch_rest,
                "lr": self.training.lr,
                "weight_decay": self.training.weight_decay,
                "epochs": self.training.epochs,
                "workers": self.training.workers,
                "seed": self.training.seed,
                "focal_alpha": self.training.focal_alpha,
                "focal_gamma": self.training.focal_gamma,
                "focal_per_class_alpha": self.training.focal_per_class_alpha,
                "lr_scheduler": self.training.lr_scheduler,
                "lr_factor": self.training.lr_factor,
                "lr_patience": self.training.lr_patience,
            },
            "data": {"mmap": self.data.mmap},
            "archive": {
                "save_every_epoch": self.archive.save_every_epoch,
                "epoch_metrics": self.archive.epoch_metrics,
            },
        }

def _read_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise SettingsError(f"无法读取 {path}: {exc}") from exc

    try:
        data = json.loads(text)
    except ValueError as exc:
        raise SettingsError(f"{path.name} 不是合法 JSON：{exc}") from exc

    if not isinstance(data, dict):
        raise SettingsError(f"{path.name} 顶层必须是一个对象 {{...}}")

    return data


def _strip_comments(node):
    """去除 "//" 开头的说明性键"""

    if isinstance(node, dict):
        return {k: _strip_comments(v) for k, v in node.items() if not str(k).startswith("//")}
    if isinstance(node, list):
        return [_strip_comments(v) for v in node]
    return node


def _merge(base: dict, override: dict) -> dict:
    """深合并：override 里的叶子值覆盖 base，dict 递归合并。"""

    result = dict(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value

    return result


def _get(cfg: dict, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _as_int(cfg, dotted, default):
    value = _get(cfg, dotted, None)
    if value is None:
        return int(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"{dotted} 应为整数，当前是 {value!r}")
    return int(value)


def _as_float(cfg, dotted, default):
    value = _get(cfg, dotted, None)
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"{dotted} 应为数字，当前是 {value!r}")
    return float(value)


def _as_bool(cfg, dotted, default):
    value = _get(cfg, dotted, None)
    if value is None:
        return bool(default)
    if not isinstance(value, bool):
        raise SettingsError(f"{dotted} 应为 true / false，而非 {value!r}")
    return value


def _as_str(cfg, dotted, default, choices=None):
    value = _get(cfg, dotted, None)
    if value is None:
        value = default
    if not isinstance(value, str):
        raise SettingsError(f"{dotted} 应为字符串，而非 {value!r}")
    value = value.strip()
    if choices and value not in choices:
        raise SettingsError(f"{dotted} 需要 {list(choices)} 之一，而非 {value!r}")
    return value


def _as_int_tuple(cfg, dotted, default, length=None):
    value = _get(cfg, dotted, None)
    if value is None:
        value = list(default)
    if not isinstance(value, (list, tuple)):
        raise SettingsError(f"{dotted} 应为数组，而非 {value!r}")
    try:
        result = tuple(int(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"{dotted} 里必须都是整数，而非 {value!r}") from exc
    if length is not None and len(result) != length:
        raise SettingsError(f"{dotted} 需要 {length} 个值，而非 {len(result)} 个")
    return result


def _as_str_list(cfg, dotted, default):
    value = _get(cfg, dotted, None)
    if value is None:
        return list(default)
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise SettingsError(f"{dotted} 应为字符串数组，而非 {value!r}")
    return [v.strip() for v in value]


def _coerce(value, current, dotted):
    """按字段当前值的类型转换（供 Settings.apply 使用）。"""

    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    if isinstance(current, int):
        return int(value)

    if isinstance(current, float):
        return float(value)

    if isinstance(current, tuple):
        if isinstance(value, str):
            value = [v for v in value.replace(";", ",").split(",") if v.strip()]
        return tuple(int(v) for v in value)

    if isinstance(current, list):
        return list(value) if isinstance(value, (list, tuple)) else [value]

    if isinstance(current, Path):
        return Path(str(value))

    return str(value)


def _as_path(cfg: dict, key: str) -> Path:
    raw = _get(cfg, f"paths.{key}", None)

    if raw is None:
        return ROOT / DEFAULT_PATH_KEYS[key]

    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def _load_labels(cfg: dict):
    raw = _get(cfg, "labels", None)

    if not isinstance(raw, dict) or not raw:
        raise SettingsError("labels 不能为空：至少需要一个「名称 -> SMARTS」")

    labels = {}

    for name, smarts in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise SettingsError(f"labels 里的名称不合法: {name!r}")
        if not isinstance(smarts, str) or not smarts.strip():
            raise SettingsError(f"labels.{name} 的 SMARTS 必须是非空字符串，当前是 {smarts!r}")
        labels[name.strip()] = smarts.strip()

    patterns = []

    if _Chem is not None:
        for name, smarts in labels.items():
            pattern = _Chem.MolFromSmarts(smarts)
            if pattern is None:
                raise SettingsError(f"labels.{name} 的 SMARTS 无法解析: {smarts!r}")
            patterns.append((name, pattern))

    return labels, tuple(patterns)


def load_settings(settings_file: Path = None, local_file: Path = None) -> Settings:
    """读取并构建 SETTINGS"""

    settings_file = Path(settings_file) if settings_file else SETTINGS_FILE
    local_file = Path(local_file) if local_file else LOCAL_SETTINGS_FILE

    merged = _merge(DEFAULTS, _strip_comments(_read_json(settings_file)))
    merged = _merge(merged, _strip_comments(_read_json(local_file)))

    paths = PathsCfg(
        root=ROOT,
        readme=ROOT / "README.md",
        **{key: _as_path(merged, key) for key in DEFAULT_PATH_KEYS},
    )

    c13 = C13Cfg(
        buckets=_as_int(merged, "features.c13.buckets", 880),
        ppm_min=_as_float(merged, "features.c13.ppm_min", 0.0),
        ppm_max=_as_float(merged, "features.c13.ppm_max", 220.0),
    )

    h1 = H1Cfg(
        buckets=_as_int(merged, "features.h1.buckets", 280),
        ppm_min=_as_float(merged, "features.h1.ppm_min", 0.0),
        ppm_max=_as_float(merged, "features.h1.ppm_max", 14.0),
        channels=H1_CHANNELS,
        scalars=H1_SCALARS,
        dtype=_as_str(merged, "features.h1.dtype", "float16", VALID_H1_DTYPES),
    )

    if c13.buckets <= 0:
        raise SettingsError("features.c13.buckets 必须大于 0")
    if h1.buckets <= 0 or h1.channels <= 0:
        raise SettingsError("features.h1.buckets / channels 必须大于 0")

    features = FeaturesCfg(
        splits=_as_str_list(merged, "features.splits", DEFAULTS["features"]["splits"]),
        c13=c13,
        h1=h1,
    )

    labels, patterns = _load_labels(merged)

    model = ModelCfg(
        arch=_as_str(merged, "model.arch", "r4", VALID_ARCHS),
        c13_kernel_sizes=_as_int_tuple(merged, "model.c13.kernel_sizes", (5, 5, 3), length=3),
        c13_channels=_as_int_tuple(merged, "model.c13.channels", (32, 64, 128), length=3),
        h1_kernel_sizes=_as_int_tuple(merged, "model.h1.kernel_sizes", (5, 3), length=2),
        h1_out_channels=_as_int_tuple(merged, "model.h1.out_channels", (32, 64), length=2),
        h1_scalar_hidden=_as_int(merged, "model.h1.scalar_hidden", 32),
        r5_head_channels=_as_int(merged, "model.r5.head_channels", 32),
        r5_head_h1_channels=_as_int(merged, "model.r5.head_h1_channels", 16),
        fc_hidden=_as_int(merged, "model.head.hidden", 512),
        dropout=_as_float(merged, "model.head.dropout", 0.2),
    )

    training = TrainingCfg(
        batch_size=_as_int(merged, "training.batch_size", 64),
        default_batch_rest=_as_float(merged, "training.default_batch_rest", 0.0),
        lr=_as_float(merged, "training.lr", 0.001),
        weight_decay=_as_float(merged, "training.weight_decay", 0.0),
        epochs=_as_int(merged, "training.epochs", 0),
        workers=_as_int(merged, "training.workers", 2),
        seed=_as_int(merged, "training.seed", 2026),
        focal_alpha=_as_float(merged, "training.focal.alpha", 0.25),
        focal_gamma=_as_float(merged, "training.focal.gamma", 2.0),
        focal_per_class_alpha=_as_bool(merged, "training.focal.per_class_alpha", True),
        lr_scheduler=_as_str(merged, "training.lr_scheduler.name", "plateau", VALID_LR_SCHEDULERS),
        lr_factor=_as_float(merged, "training.lr_scheduler.factor", 0.5),
        lr_patience=_as_int(merged, "training.lr_scheduler.patience", 0),
    )

    if training.batch_size <= 0:
        raise SettingsError("training.batch_size 必须大于 0")
    if training.workers < 0:
        raise SettingsError("training.workers 不能是负数（0 表示主进程读）")

    data = DataCfg(mmap=_as_bool(merged, "data.mmap", True))

    archive = ArchiveCfg(
        save_every_epoch=_as_bool(merged, "archive.save_every_epoch", False),
        epoch_metrics=_as_bool(merged, "archive.epoch_metrics", True),
    )

    settings = Settings(
        paths=paths,
        device=_as_str(merged, "device.train_device", "auto"),
        features=features,
        labels=labels,
        label_names=tuple(labels.keys()),
        patterns=patterns,
        model=model,
        training=training,
        data=data,
        archive=archive,
    )

    settings._derive()

    return settings


SETTINGS = load_settings()

# 遗留变量
ROOT_PATH = ROOT
FUNCTIONAL_GROUPS = SETTINGS.labels
LABEL_NAMES = SETTINGS.label_names
_COMPILED_PATTERNS = SETTINGS.patterns


def reload(**overrides) -> Settings:
    """重新从磁盘加载配置，返回新的 SETTINGS（同时替换全局对象）。"""

    global SETTINGS, FUNCTIONAL_GROUPS, LABEL_NAMES, _COMPILED_PATTERNS

    settings = load_settings()

    if overrides:
        settings.apply(overrides)

    SETTINGS = settings
    FUNCTIONAL_GROUPS = settings.labels
    LABEL_NAMES = settings.label_names
    _COMPILED_PATTERNS = settings.patterns

    return settings
