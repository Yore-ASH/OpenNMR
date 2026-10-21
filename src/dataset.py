import json

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from setup import SETTINGS


def _load_tensor(path, use_mmap=True):
    """加载 .pt 张量 """

    if use_mmap:

        try:
            return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        except (RuntimeError, TypeError, ValueError, OSError) as e:
            print(f"[dataset] 无法 mmap 加载 {path}（{e}），回退为普通加载")

    return torch.load(path, map_location="cpu")


def _find_1h_path(x_path):
    """X_train.pt -> 同目录下的 X1H_train.pt"""

    x_path = Path(x_path)

    if not x_path.name.startswith("X_"):
        return None

    candidate = x_path.with_name("X1H_" + x_path.name[2:])

    return candidate if candidate.exists() else None


def _read_1h_meta(pt_dir, total_dim):
    """读取 X1H_meta.json，得到 (bins, channels, scalars)；没有 meta 时按旧的单通道处理"""

    meta_path = Path(pt_dir) / "X1H_meta.json"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return total_dim, 1, 0

    bins = int(meta.get("bins", total_dim))
    channels = int(meta.get("channels", 1))
    scalars = int(meta.get("scalars", 0))

    if bins * channels + scalars != total_dim:
        return total_dim, 1, 0

    return bins, channels, scalars


class NMRDataset(Dataset):
    """13C 多热特征 + （可选）1H 积分直方图 + 官能团标签

    X_*.pt 存在对应的 X1H_*.pt 时，会自动把两者按行拼接成一个向量，
    这样训练/评估循环不用改：模型内部再按 n_13c / n_1h 切开。
    """

    def __init__(self, X_path, Y_path, mmap=None):
        # mmap=None 表示用 settings.json 里的 data.mmap
        self.use_mmap = SETTINGS.data.mmap if mmap is None else bool(mmap)

        self.X = _load_tensor(X_path, use_mmap=self.use_mmap)
        self.Y = _load_tensor(Y_path, use_mmap=self.use_mmap)

        self.X1H = None
        self.h1_bins = 0
        self.h1_channels = 1
        self.h1_scalars = 0

        path_1h = _find_1h_path(X_path)

        if path_1h is not None:

            self.X1H = _load_tensor(path_1h, use_mmap=self.use_mmap)

            if len(self.X1H) != len(self.X):
                raise ValueError(
                    f"{path_1h.name} 有 {len(self.X1H)} 行，"
                    f"{Path(X_path).name} 有 {len(self.X)} 行，两者无法对齐"
                )

            self.h1_bins, self.h1_channels, self.h1_scalars = _read_1h_meta(
                Path(X_path).parent, int(self.X1H.shape[1])
            )

        self.n_13c = int(self.X.shape[1])
        self.n_1h = int(self.X1H.shape[1]) if self.X1H is not None else 0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]

        if x.dtype == torch.bool:
            x = x.float()

        if self.X1H is not None:
            x = torch.cat([x, self.X1H[index].float()])

        return x, self.Y[index]


def get_dataloader(
    X_path,
    Y_path,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    mmap=None,
    pin_memory=False,
):
    """构建 DataLoader"""
    dataset = NMRDataset(X_path, Y_path, mmap=mmap)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
