"""13C / 1H NMR 官能团多标签分类模型"""

import torch
import torch.nn as nn

from setup import SETTINGS

ARCH_IDS = {"r4": 0, "r5": 1}
ID_ARCHS = {value: key for key, value in ARCH_IDS.items()}


class NMRClassifier(nn.Module):
    """13C 多热特征 + 可选 1H 特征（3 通道直方图 + 全局标量）-> 官能团多标签分类 """

    def __init__(
        self,

        n_features: int = None,
        n_classes: int = None,
        n_1h: int = 0,
        h1_bins: int = None,
        h1_channels: int = 1,
        h1_scalars: int = 0,

        arch: str = None,
        c13_kernel_sizes: tuple = None,
        c13_channels: tuple = None,
        h1_kernel_sizes: tuple = None,
        h1_out_channels: tuple = None,
        h1_scalar_hidden: int = None,
        head_channels: int = None,
        head_h1_channels: int = None,
        fc_hidden: int = None,
        dropout: float = None,
    ):
        super().__init__()

        model_cfg = SETTINGS.model
        features_cfg = SETTINGS.features

        arch = str(arch or model_cfg.arch).strip().lower()
        if arch not in ARCH_IDS:
            raise ValueError(f"model.arch 只能是 {list(ARCH_IDS)}，当前: {arch}")

        if n_features is None:
            n_features = model_cfg.n_features
        if n_classes is None:
            n_classes = model_cfg.n_classes
        if c13_kernel_sizes is None:
            c13_kernel_sizes = model_cfg.c13_kernel_sizes
        if c13_channels is None:
            c13_channels = model_cfg.c13_channels
        if h1_kernel_sizes is None:
            h1_kernel_sizes = model_cfg.h1_kernel_sizes
        if h1_out_channels is None:
            h1_out_channels = model_cfg.h1_out_channels
        if h1_scalar_hidden is None:
            h1_scalar_hidden = model_cfg.h1_scalar_hidden
        if head_channels is None:
            head_channels = model_cfg.r5_head_channels
        if head_h1_channels is None:
            head_h1_channels = model_cfg.r5_head_h1_channels
        if fc_hidden is None:
            fc_hidden = model_cfg.fc_hidden
        if dropout is None:
            dropout = model_cfg.dropout

        if len(c13_kernel_sizes) != 3:
            raise ValueError(f"model.c13.kernel_sizes 需要 3 个值，当前: {c13_kernel_sizes}")
        if len(c13_channels) != 3:
            raise ValueError(f"model.c13.channels 需要 3 个值，当前: {c13_channels}")

        k1, k2, k3 = c13_kernel_sizes
        c1, c2, c3 = c13_channels

        n_1h = int(n_1h or 0)
        h1_channels = int(h1_channels or 1)
        h1_scalars = int(h1_scalars or 0)
        h1_bins = int(h1_bins) if (h1_bins is not None) else (n_1h if n_1h else 0)

        if n_1h:

            if h1_channels < 1:
                raise ValueError(f"1H 通道数至少为 1，当前: {h1_channels}")

            if (h1_bins % 2) != 0:
                raise ValueError(f"1H 桶数需要能被 2 整除（1H 分支有一次池化），当前: {h1_bins}")

            if h1_bins * h1_channels + h1_scalars != n_1h:
                raise ValueError(
                    f"1H 特征维度不一致：bins({h1_bins}) × channels({h1_channels}) "
                    f"+ scalars({h1_scalars}) != n_1h({n_1h})"
                )

        self.arch = arch
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.n_1h = n_1h
        self.h1_bins = h1_bins
        self.h1_channels = h1_channels
        self.h1_scalars = h1_scalars
        self.head_channels = int(head_channels)
        self.head_h1_channels = int(head_h1_channels)
        self.fc_hidden = int(fc_hidden)
        self.head_input_dim = 0

        self.register_buffer("_n_13c_bins", torch.tensor(self.n_features), persistent=True)
        self.register_buffer("_n_1h_bins", torch.tensor(self.h1_bins), persistent=True)
        self.register_buffer("_n_1h_channels", torch.tensor(self.h1_channels), persistent=True)
        self.register_buffer("_n_1h_scalars", torch.tensor(self.h1_scalars), persistent=True)
        self.register_buffer("_arch_id", torch.tensor(ARCH_IDS[arch]), persistent=True)

        if arch == "r4":
            self._build_r4(
                c1, c2, c3, k1, k2, k3, dropout,
                h1_kernel_sizes, h1_out_channels, h1_scalar_hidden,
            )
        else:
            self._build_r5(
                c1, c2, c3, k1, k2, k3, dropout,
                h1_kernel_sizes, h1_out_channels, h1_scalar_hidden,
            )

    def _build_r4(self, c1, c2, c3, k1, k2, k3, dropout,
                  h1_kernel_sizes, h1_out_channels, h1_scalar_hidden):
        """0.8.0 原架构，模块名与 0.8.0 完全一致，保证旧权重可直接加载。"""

        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=k1, padding=(k1 - 1) // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=(k2 - 1) // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=k3, padding=(k3 - 1) // 2),
            nn.ReLU(),
        )

        # 展平
        flatten_dim = c3 * (self.n_features // 4)

        if self.n_1h:
            hk1, hk2 = h1_kernel_sizes
            hc1, hc2 = h1_out_channels

            self.conv1h = nn.Sequential(
                nn.Conv1d(self.h1_channels, hc1, kernel_size=hk1, padding=(hk1 - 1) // 2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(hc1, hc2, kernel_size=hk2, padding=(hk2 - 1) // 2),
                nn.ReLU(),
            )

            # 1H 池化
            flatten_dim += hc2 * (self.h1_bins // 2)

        if self.h1_scalars:
            self.h1_mlp = nn.Sequential(
                nn.Linear(self.h1_scalars, h1_scalar_hidden),
                nn.ReLU(),
            )
            flatten_dim += h1_scalar_hidden

        self.head_input_dim = flatten_dim

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, self.fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fc_hidden, self.n_classes),
        )

    def _build_r5(self, c1, c2, c3, k1, k2, k3, dropout,
                  h1_kernel_sizes, h1_out_channels, h1_scalar_hidden):
        hc = self.head_channels

        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=k1, padding=(k1 - 1) // 2),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=(k2 - 1) // 2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=k3, padding=(k3 - 1) // 2),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
        )

        # 压缩后展平
        self.conv_pool = nn.Sequential(
            nn.Conv1d(c3, hc, kernel_size=1),
            nn.BatchNorm1d(hc),
            nn.ReLU(),
        )

        head_dim = hc * (self.n_features // 4) + 2 * c3

        if self.n_1h:
            hk1, hk2 = h1_kernel_sizes
            hc1, hc2 = h1_out_channels
            hcb = self.head_h1_channels

            self.conv1h = nn.Sequential(
                nn.Conv1d(self.h1_channels, hc1, kernel_size=hk1, padding=(hk1 - 1) // 2),
                nn.BatchNorm1d(hc1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(hc1, hc2, kernel_size=hk2, padding=(hk2 - 1) // 2),
                nn.BatchNorm1d(hc2),
                nn.ReLU(),
            )

            self.conv1h_pool = nn.Sequential(
                nn.Conv1d(hc2, hcb, kernel_size=1),
                nn.BatchNorm1d(hcb),
                nn.ReLU(),
            )

            head_dim += hcb * (self.h1_bins // 2) + 2 * hc2

        if self.h1_scalars:
            self.h1_mlp = nn.Sequential(
                nn.Linear(self.h1_scalars, h1_scalar_hidden),
                nn.ReLU(),
            )
            head_dim += h1_scalar_hidden

        self.head_input_dim = head_dim

        self.fc = nn.Sequential(
            nn.Linear(head_dim, self.fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fc_hidden, self.n_classes),
        )

    def forward(self, x):
        """前向"""
        
        x_13c = x[:, : self.n_features].unsqueeze(1)
        f_13c = self.conv(x_13c)

        if self.arch == "r4":
            feature = f_13c.flatten(1)
        else:
            feature = torch.cat(
                [
                    self.conv_pool(f_13c).flatten(1),
                    f_13c.max(dim=2).values,
                    f_13c.mean(dim=2),
                ],
                dim=1,
            )

        if self.n_1h:
            x_1h = x[:, self.n_features : self.n_features + self.n_1h]

            n_spec = self.h1_bins * self.h1_channels

            if n_spec > 0:
                spec = x_1h[:, :n_spec].reshape(-1, self.h1_channels, self.h1_bins)
                f_1h = self.conv1h(spec)

                if self.arch == "r4":
                    feature = torch.cat([feature, f_1h.flatten(1)], dim=1)
                else:
                    feature = torch.cat(
                        [
                            feature,
                            self.conv1h_pool(f_1h).flatten(1),
                            f_1h.max(dim=2).values,
                            f_1h.mean(dim=2),
                        ],
                        dim=1,
                    )

            if self.h1_scalars:
                stats = x_1h[:, n_spec : n_spec + self.h1_scalars]
                feature = torch.cat([feature, self.h1_mlp(stats)], dim=1)

        return self.fc(feature)
