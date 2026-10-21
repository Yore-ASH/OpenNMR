"""应用的主要解析器
    峰值内存 ≈ 单个 chunk（默认 2 万行，约 70 MiB）+ pandas 解析开销。
"""

import json
import os
import math
import re
import shutil
import numpy as np
import torch

import pandas as pd

from torch import nn
from tqdm import tqdm
from pathlib import Path
from rdkit import Chem
from ast import literal_eval

from setup import SETTINGS
from setup import FUNCTIONAL_GROUPS
from setup import _COMPILED_PATTERNS

DEFAULT_CHUNK_ROWS = 20_000

_EMPTY_SHIFTS = np.empty(0, dtype=np.float64)
_EMPTY_INDICES = np.empty(0, dtype=np.int64)

def _string_to_Path(*target : str|Path) -> tuple[Path , ...]:
    """部分路径会以 str 格式传入，该函数将其转化为 pathlib.Path"""

    result = []

    for t in target:

        if isinstance(t, str):

            result.append(Path(t))
        else:

            result.append(t)

    return tuple(result)

def parse_parquet_file(
        parquet_file_folder : Path|str = SETTINGS.paths.parquet_dir,
        parquet_names : list = SETTINGS.features.splits,
        parquet_extension: str = ".parquet",
        csv_folder :str|Path = SETTINGS.paths.csv_dir,
        prompt : bool = True,
):
    """载入.parquet文件并解析成.csv文件"""

    parquet_file_folder,csv_folder = _string_to_Path(parquet_file_folder,csv_folder)

    csv_folder.mkdir(parents=True, exist_ok=True)

    target_file = [Path(parquet_file_folder / (name + parquet_extension)) for name in parquet_names]

    pairs = list(zip(target_file, parquet_names))

    if prompt:
        pairs = tqdm(pairs, desc = "正在预处理 .parquet 文件为 .csv 文件")

    for file, name in pairs:

        df = pd.read_parquet(file, columns=["id","smiles","nmr_1h","nmr_13c"])

        df.to_csv(
            Path(csv_folder / (name + ".csv")),
            index = False,
        )

        del df

def _nmr_to_shifts(
        nmr_raw,
) -> np.ndarray:
    """把 nmr_13c 字段解析成 float64 数组"""

    if nmr_raw is None:
        return _EMPTY_SHIFTS

    if isinstance(nmr_raw, str):

        text = nmr_raw.strip()

        if not text:
            return _EMPTY_SHIFTS

        if text[0] == "[" and text[-1] == "]":

            body = text[1:-1].strip()

            if not body:
                return _EMPTY_SHIFTS

            try:
                return np.array(body.split(","), dtype=np.float64)
            except ValueError:
                pass

        try:
            raw = literal_eval(text)
        except (ValueError, SyntaxError):
            return _EMPTY_SHIFTS
    else:
        raw = nmr_raw

    if not isinstance(raw, (list, tuple)):
        return _EMPTY_SHIFTS

    values = []

    for val in raw:
        try:
            values.append(float(val))
        except (ValueError, TypeError):
            continue

    if not values:
        return _EMPTY_SHIFTS

    return np.asarray(values, dtype=np.float64)

def _bucket_indices(
        shifts : np.ndarray,
        n_buckets : int,
        ppm_min : float|int,
        width : float,
) -> np.ndarray:
    """把化学位移换算成桶下标，越界值直接丢弃"""

    if shifts.size == 0:
        return _EMPTY_INDICES

    with np.errstate(invalid="ignore"):
        # astype(np.int64) 对正数向零截断，与旧实现的 int(...) 一致；
        # nan / inf 会得到极值，随后被边界过滤掉
        idx = ((shifts - ppm_min) / width).astype(np.int64)

    return idx[(idx >= 0) & (idx < n_buckets)]

def _encode_nmr_13c_np(
        nmr_raw,
        n_buckets : int = SETTINGS.features.c13.buckets,
        ppm_min : float|int = SETTINGS.features.c13.ppm_min,
        ppm_max : float|int = SETTINGS.features.c13.ppm_max,
) -> np.ndarray:
    """nmr_13c -> [n_buckets] float32 多热向量的 numpy 实现（内存友好版）"""

    vec = np.zeros(n_buckets, dtype=np.float32)

    width = (ppm_max - ppm_min) / n_buckets

    vec[_bucket_indices(_nmr_to_shifts(nmr_raw), n_buckets, ppm_min, width)] = 1.0

    return vec

def encode_nmr_13c(
    nmr_raw,
    n_buckets: int = SETTINGS.features.c13.buckets,
    ppm_min: float|int = SETTINGS.features.c13.ppm_min,
    ppm_max: float|int = SETTINGS.features.c13.ppm_max,
) -> torch.Tensor:
    """
    解析 nmr_13c 字段并分桶编码为固定长度向量（对外接口保持不变）。
    """

    return torch.from_numpy(
        _encode_nmr_13c_np(
            nmr_raw,
            n_buckets = n_buckets,
            ppm_min = ppm_min,
            ppm_max = ppm_max,
        )
    )

def encode_nmr_13c_batch(
        nmr_values,
        n_buckets : int = SETTINGS.features.c13.buckets,
        ppm_min : float|int = SETTINGS.features.c13.ppm_min,
        ppm_max : float|int = SETTINGS.features.c13.ppm_max,
) -> np.ndarray:
    """批量版 encode_nmr_13c，返回 [len(nmr_values), n_buckets] 的 float32 数组"""

    nmr_values = list(nmr_values)

    block = np.zeros((len(nmr_values), n_buckets), dtype=np.float32)

    width = (ppm_max - ppm_min) / n_buckets

    for row, nmr_raw in enumerate(nmr_values):

        block[
            row,
            _bucket_indices(_nmr_to_shifts(nmr_raw), n_buckets, ppm_min, width),
        ] = 1.0

    return block

def encode_1h_integrals(
        peaks_raw,
        n_buckets : int = SETTINGS.features.h1.buckets,
        ppm_min : float|int = SETTINGS.features.h1.ppm_min,
        ppm_max : float|int = SETTINGS.features.h1.ppm_max,
) -> np.ndarray:
    """把 nmr_1h 峰列表编码成 [n_buckets] 的积分直方图。

    nmr_1h 的每个峰形如 [化学位移, 0.0, "m", "2H", [7.4]]，
    第 4 个字段就是积分（氢数）。把每个峰的积分累加到对应化学位移桶，
    最后除以总积分（总氢数），得到"每种化学位移上的氢占比"。
    """

    vec = np.zeros(n_buckets, dtype=np.float32)

    if not isinstance(peaks_raw, str) or (not peaks_raw.strip()):
        return vec

    try:
        peaks = json.loads(peaks_raw)
    except (ValueError, TypeError):
        return vec

    if not isinstance(peaks, (list, tuple)):
        return vec

    width = (ppm_max - ppm_min) / n_buckets

    for peak in peaks:

        try:
            shift = float(peak[0])
            integral = float(re.sub(r"[^0-9.]", "", str(peak[3])) or 0.0)
        except (IndexError, TypeError, ValueError):
            continue

        if integral <= 0:
            continue

        idx = int((shift - ppm_min) / width)

        if 0 <= idx < n_buckets:
            vec[idx] += integral

    total = float(vec.sum())

    if total > 0:
        vec /= total

    return vec

def encode_1h_integrals_batch(
        peaks_values,
        n_buckets : int = SETTINGS.features.h1.buckets,
        ppm_min : float|int = SETTINGS.features.h1.ppm_min,
        ppm_max : float|int = SETTINGS.features.h1.ppm_max,
) -> np.ndarray:
    """批量版 encode_1h_integrals，返回 [len(peaks_values), n_buckets] 的 float32 数组"""

    peaks_values = list(peaks_values)

    block = np.zeros((len(peaks_values), n_buckets), dtype=np.float32)

    for row, peaks_raw in enumerate(peaks_values):
        block[row] = encode_1h_integrals(
            peaks_raw,
            n_buckets = n_buckets,
            ppm_min = ppm_min,
            ppm_max = ppm_max,
        )

    return block


# 1H 通道数 / 全局标量个数由编码实现固定，唯一常量在 setup.py
# （H1_CHANNELS / H1_SCALARS），这里从 SETTINGS 派生，避免同一个数字写两遍。
_H1_CHANNELS = SETTINGS.features.h1.channels
_H1_SCALARS = SETTINGS.features.h1.scalars
_H1_NP_DTYPE = np.float16 if SETTINGS.features.h1.dtype == "float16" else np.float32


def _h1_multiplicity_group(raw) -> str:
    """把多重峰缩写归类成 s / d / t / q / complex / other"""

    text = str(raw or "").strip().lower()

    if text in ("s", "br", "brs", "bs"):
        return "s"
    if text == "d":
        return "d"
    if text == "t":
        return "t"
    if text in ("q", "quin", "quint"):
        return "q"
    if not text:
        return "other"

    return "complex"


def encode_1h_features(
        peaks_raw,
        n_buckets : int = SETTINGS.features.h1.buckets,
        ppm_min : float|int = SETTINGS.features.h1.ppm_min,
        ppm_max : float|int = SETTINGS.features.h1.ppm_max,
) -> np.ndarray:
    """1H 峰列表 -> (3 通道积分直方图 + 12 个全局标量)"""

    channels = np.zeros((_H1_CHANNELS, n_buckets), dtype=np.float32)
    scalars = np.zeros(_H1_SCALARS, dtype=np.float32)

    def _result():
        return np.concatenate([channels.reshape(-1), scalars])

    if not isinstance(peaks_raw, str) or (not peaks_raw.strip()):
        return _result()

    try:
        peaks = json.loads(peaks_raw)
    except (ValueError, TypeError):
        return _result()

    if not isinstance(peaks, (list, tuple)):
        return _result()

    width = (ppm_max - ppm_min) / n_buckets

    total_h = 0.0
    total_jh = 0.0
    h_with_j = 0.0
    n_peaks = 0

    group_h = {"s": 0.0, "d": 0.0, "t": 0.0, "q": 0.0, "complex": 0.0, "other": 0.0}

    j_values = []
    j_weighted_sum = 0.0
    j_counts = 0

    for peak in peaks:

        try:
            shift = float(peak[0])
            integral = float(re.sub(r"[^0-9.]", "", str(peak[3])) or 0.0)
        except (IndexError, TypeError, ValueError):
            continue

        if integral <= 0:
            continue

        n_peaks += 1
        total_h += integral

        group = _h1_multiplicity_group(peak[2] if len(peak) > 2 else "")
        group_h[group] += integral

        js = []

        if (len(peak) > 4) and isinstance(peak[4], (list, tuple)):

            for value in peak[4]:
                try:
                    js.append(float(value))
                except (TypeError, ValueError):
                    continue

        mean_j = float(np.mean(js)) if js else 0.0

        if mean_j > 0:
            h_with_j += integral
            total_jh += integral * mean_j
            j_weighted_sum += mean_j * integral
            j_values.extend(js)
            j_counts += len(js)

        idx = int((shift - ppm_min) / width)

        if 0 <= idx < n_buckets:
            channels[0, idx] += integral

            if group == "complex":
                channels[1, idx] += integral

            channels[2, idx] += integral * mean_j

    if total_h > 0:
        channels[0] /= total_h
        channels[1] /= total_h

        for key in group_h:
            group_h[key] /= total_h

    if total_jh > 0:
        channels[2] /= total_jh

    scalars[0] = min(total_h / 20.0, 1.0)
    scalars[1] = min(n_peaks / 15.0, 1.0)
    scalars[2] = group_h["s"]
    scalars[3] = group_h["d"]
    scalars[4] = group_h["t"]
    scalars[5] = group_h["q"]
    scalars[6] = group_h["complex"]
    scalars[7] = group_h["other"]
    scalars[8] = (h_with_j / total_h) if total_h > 0 else 0.0
    scalars[9] = min((j_weighted_sum / h_with_j) / 12.0, 1.0) if h_with_j > 0 else 0.0
    scalars[10] = min((max(j_values) if j_values else 0.0) / 20.0, 1.0)
    scalars[11] = min((j_counts / n_peaks) / 3.0, 1.0) if n_peaks > 0 else 0.0

    return _result()


def encode_1h_features_batch(
        peaks_values,
        n_buckets : int = SETTINGS.features.h1.buckets,
        ppm_min : float|int = SETTINGS.features.h1.ppm_min,
        ppm_max : float|int = SETTINGS.features.h1.ppm_max,
) -> np.ndarray:
    """批量版 encode_1h_features，返回 [N, 3*n_buckets + 12] 的 float32 数组"""

    peaks_values = list(peaks_values)

    width = _H1_CHANNELS * n_buckets + _H1_SCALARS

    block = np.zeros((len(peaks_values), width), dtype=np.float32)

    for row, peaks_raw in enumerate(peaks_values):
        block[row] = encode_1h_features(
            peaks_raw,
            n_buckets = n_buckets,
            ppm_min = ppm_min,
            ppm_max = ppm_max,
        )

    return block


def _fg_labels_np(
        smiles,
        compiled_patterns : list = _COMPILED_PATTERNS,
) -> np.ndarray | None:
    """单分子官能团多热标签的 numpy 实现；smiles 非法时返回 None"""

    if not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    labels = np.zeros(len(compiled_patterns), dtype=np.float32)

    for ind, (_, pattern) in enumerate(compiled_patterns):

        if (pattern is not None) and mol.HasSubstructMatch(pattern):
            labels[ind] = 1.0

    return labels

def get_FG_labels(
        smiles: str,
        compiled_patterns: list = _COMPILED_PATTERNS,
) -> torch.Tensor | None:
    """传入 smiles 分子式，返回关于官能团的多热向量"""

    labels = _fg_labels_np(smiles, compiled_patterns = compiled_patterns)

    if labels is None:
        return None

    return torch.from_numpy(labels)

class _ChunkedTensorWriter:
    """把逐块算好的二维数组顺序累积到临时二进制文件，最后内存映射并保存为 .pt"""

    def __init__(
            self,
            out_path : Path,
            n_cols : int,
            dtype : np.dtype = np.float32,
    ):
        self.out_path = Path(out_path)
        self.n_cols = int(n_cols)
        self.dtype = np.dtype(dtype)
        self.rows = 0
        self._finished = False

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        self.tmp_path = self.out_path.with_name("." + self.out_path.name + ".tmp")

        self._fh = open(self.tmp_path, "wb")

    def append(self, block : np.ndarray) -> None:
        """追加一块 [k, n_cols] 数据"""

        if self._finished:
            raise RuntimeError("该 writer 已经完成，不能再追加")

        block = np.ascontiguousarray(block, dtype=self.dtype)

        if block.ndim != 2 or block.shape[1] != self.n_cols:
            raise ValueError(
                f"写入块形状应为 [k, {self.n_cols}]，实际为 {block.shape}"
            )

        if block.shape[0] == 0:
            return

        block.tofile(self._fh)

        self.rows += block.shape[0]

    def finalize(self) -> tuple[int, int]:
        """落盘为 .pt 并清理临时文件，返回 (行数, 列数)"""

        self._fh.close()

        tensor = None
        mapped = None

        try:
            if self.rows == 0:
                tensor = torch.empty((0, self.n_cols), dtype=torch.float32)
            else:
                # mode="c"：copy-on-write，得到的数组可写，torch 不会告警，
                # 但即使误写也不会改动临时文件内容
                mapped = np.memmap(
                    self.tmp_path,
                    dtype = self.dtype,
                    mode = "c",
                    shape = (self.rows, self.n_cols),
                )
                tensor = torch.from_numpy(mapped)

            torch.save(tensor, self.out_path)
        finally:
            del tensor
            del mapped
            self._finished = True
            self._cleanup()

        return self.rows, self.n_cols

    def discard(self) -> None:
        """放弃写入（异常/中断时清理临时文件）"""

        self._finished = True
        self._cleanup()

    def _cleanup(self) -> None:
        fh = getattr(self, "_fh", None)

        if (fh is not None) and (not fh.closed):
            fh.close()

        try:
            self.tmp_path.unlink()
        except FileNotFoundError:
            pass

def _count_csv_rows(csv_path : Path) -> int:
    """快速数一下 csv 有多少行（只用于 tqdm 进度条总量，数错也不影响结果）"""

    try:
        newlines = 0

        with open(csv_path, "rb") as fh:

            while True:

                block = fh.read(1 << 22)

                if not block:
                    break

                newlines += block.count(b"\n")

        return max(newlines - 1, 0)  # 去掉表头

    except OSError:
        return 0

def gen_1h_pt(
        name_list : list = SETTINGS.features.splits,
        csv_folder : str|Path = SETTINGS.paths.csv_dir,
        pt_folder : str|Path = SETTINGS.paths.pt_dir,
        n_buckets : int = SETTINGS.features.h1.buckets,
        ppm_min : float|int = SETTINGS.features.h1.ppm_min,
        ppm_max : float|int = SETTINGS.features.h1.ppm_max,
        prompt : bool = True,
        chunk_rows : int = DEFAULT_CHUNK_ROWS,
):
    """从 csv 的 nmr_1h 列生成 1H 特征，保存为 X1H_[name].pt
    """

    csv_folder, pt_folder = _string_to_Path(csv_folder, pt_folder)

    pt_folder.mkdir(parents=True, exist_ok=True)

    for name in name_list:

        csv_path = csv_folder / (name + ".csv")

        if not csv_path.exists():
            print(f"[跳过] 找不到 {csv_path}")
            continue

        if "nmr_1h" not in pd.read_csv(csv_path, nrows=0).columns:
            print(f"[跳过] {csv_path} 没有 nmr_1h 列，请先重新运行 parse_parquet_file()")
            continue

        total_rows = _count_csv_rows(csv_path)
        total_chunks = math.ceil(total_rows / chunk_rows) if total_rows else None

        out_path = Path(pt_folder / ("X1H_" + name + ".pt"))

        n_features = _H1_CHANNELS * n_buckets + _H1_SCALARS

        _warn_if_low_disk(pt_folder, total_rows, n_features, [out_path])

        writer = _ChunkedTensorWriter(out_path, n_features, dtype=_H1_NP_DTYPE)

        kept = 0
        skipped = 0

        try:
            reader = pd.read_csv(
                csv_path,
                usecols = ["smiles","nmr_1h"],
                dtype = {"smiles": object, "nmr_1h": object},
                chunksize = chunk_rows,
            )

            for chunk in tqdm(
                reader,
                total = total_chunks,
                desc = f"生成{name}的1H特征",
                disable = not prompt,
            ):
                block = encode_1h_features_batch(
                    chunk["nmr_1h"].to_numpy(),
                    n_buckets = n_buckets,
                    ppm_min = ppm_min,
                    ppm_max = ppm_max,
                )

                keep = np.zeros(len(chunk), dtype=bool)

                for row, smiles_string in enumerate(chunk["smiles"].to_numpy()):

                    try:
                        if Chem.MolFromSmiles(smiles_string) is not None:
                            keep[row] = True
                    except Exception:
                        continue

                n_keep = int(keep.sum())

                kept += n_keep
                skipped += len(chunk) - n_keep

                if n_keep:
                    writer.append(block[keep])

                del block

        except BaseException:
            writer.discard()
            raise

        writer.finalize()

        x_path = pt_folder / ("X_" + name + ".pt")

        if x_path.exists():

            n_x = int(torch.load(x_path, mmap=True, weights_only=True).shape[0])

            if n_x != kept:
                raise RuntimeError(
                    f"{name}: 1H 特征 {kept} 行，但 X_{name}.pt 有 {n_x} 行，"
                    "特征与标签无法对齐；请用同一份 csv 重新生成 gen_XY_pt 和 gen_1h_pt"
                )

        meta = {
            "bins": int(n_buckets),
            "channels": _H1_CHANNELS,
            "scalars": _H1_SCALARS,
            "ppm_min": float(ppm_min),
            "ppm_max": float(ppm_max),
            "dtype": SETTINGS.features.h1.dtype,
            "total_dim": int(n_features),
        }

        (pt_folder / "X1H_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"{name}: 1H 特征 {kept} 行，维度 {n_features}，跳过 {skipped} 行 -> {out_path.name}")

def _warn_if_low_disk(
        target_dir : Path,
        n_rows : int,
        n_cols : int,
        existing_outputs : list,
) -> None:
    """落盘前粗估磁盘需求并给出提示

    写入过程需要 "临时文件 + (输出文件不存在时的) 最终文件" 两份空间；
    若输出文件已存在，torch.save 会先截断旧文件，因此只需额外留出临时文件的空间。
    """

    if n_rows <= 0:
        return

    estimate = n_rows * n_cols * 4  # float32

    needed = estimate + sum(0 if Path(p).exists() else estimate for p in existing_outputs)

    try:
        free = shutil.disk_usage(target_dir).free
    except OSError:
        return

    if free < needed:
        print(
            f"[警告] {target_dir} 剩余空间 {free / 2**30:.1f} GiB，"
            f"预计需要约 {needed / 2**30:.1f} GiB，可能因磁盘不足中断"
        )

def gen_XY_pt(
        name_list : list = SETTINGS.features.splits,
        csv_folder : str|Path = SETTINGS.paths.csv_dir,
        pt_folder : str|Path = SETTINGS.paths.pt_dir,
        n_buckets: int = SETTINGS.features.c13.buckets,
        ppm_min: float | int = SETTINGS.features.c13.ppm_min,
        ppm_max: float | int = SETTINGS.features.c13.ppm_max,
        compiled_patterns : list = _COMPILED_PATTERNS,
        prompt : bool = True,
        chunk_rows : int = DEFAULT_CHUNK_ROWS,
):
    """从 [csv_folder] 中读取 [name].csv ，分析完成后保存至 [pt_folder] 的 [name]_[X/Y].pt"""

    csv_folder,pt_folder = _string_to_Path(csv_folder,pt_folder)

    pt_folder.mkdir(parents=True, exist_ok=True)

    n_classes = len(compiled_patterns)

    for name in name_list:

        csv_path = csv_folder / (name + ".csv")

        if not csv_path.exists():
            print(f"[跳过] 找不到 {csv_path}")
            continue

        total_rows = _count_csv_rows(csv_path)
        total_chunks = math.ceil(total_rows / chunk_rows) if total_rows else None

        x_out = Path(pt_folder / ("X_" + name + ".pt"))
        y_out = Path(pt_folder / ("Y_" + name + ".pt"))

        _warn_if_low_disk(pt_folder, total_rows, n_buckets + n_classes, [x_out, y_out])

        x_writer = _ChunkedTensorWriter(x_out, n_buckets)
        y_writer = _ChunkedTensorWriter(y_out, n_classes)

        kept = 0
        skipped = 0

        try:
            reader = pd.read_csv(
                csv_path,
                usecols = ["smiles","nmr_13c"],
                dtype = {"smiles": object, "nmr_13c": object},
                chunksize = chunk_rows,
            )

            for chunk in tqdm(
                reader,
                total = total_chunks,
                desc = f"开始分析{name}.csv",
                disable = not prompt,
            ):
                n_rows = len(chunk)

                X_block = encode_nmr_13c_batch(
                    chunk["nmr_13c"].to_numpy(),
                    n_buckets = n_buckets,
                    ppm_min = ppm_min,
                    ppm_max = ppm_max,
                )

                Y_block = np.zeros((n_rows, n_classes), dtype=np.float32)
                keep = np.zeros(n_rows, dtype=bool)

                for row, smiles_string in enumerate(chunk["smiles"].to_numpy()):

                    try:
                        labels = _fg_labels_np(
                            smiles_string,
                            compiled_patterns = compiled_patterns,
                        )
                    except Exception as e:
                        print("加载过程中发现错误:",e,"已经为你跳过该行")
                        continue

                    if labels is None:
                        continue

                    Y_block[row] = labels
                    keep[row] = True

                n_keep = int(keep.sum())

                kept += n_keep
                skipped += n_rows - n_keep

                if n_keep:
                    x_writer.append(X_block[keep])
                    y_writer.append(Y_block[keep])

                del X_block, Y_block, keep

        except BaseException:
            x_writer.discard()
            y_writer.discard()
            raise

        x_shape = x_writer.finalize()
        y_shape = y_writer.finalize()

        print(
            f"{name}: 保留 {kept} 行，跳过 {skipped} 行，"
            f"X {x_shape}，Y {y_shape}"
        )

def main():
    parse_parquet_file()
    gen_XY_pt()

if __name__ == "__main__":
    main()
