"""Preprocessing pipeline for liquid spectrum data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.signal import medfilt, savgol_filter

from liquid_identification.data_loader import find_data_dir


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters used by the preprocessing pipeline."""

    median_window: int = 5
    savgol_window: int = 21
    savgol_polyorder: int = 3
    zscore_threshold: float = 3.5
    iqr_multiplier: float = 1.5


def preprocess_spectrum_data(
    data: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    """Apply denoising, smoothing and z-score normalization per spectrum."""
    active_config = config or PreprocessingConfig()
    required_columns = {
        "sample_id",
        "frequency_hz",
        "frequency_mhz",
        "amplitude_dbm",
        "liquid_type",
        "source_file",
        "range",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        msg = f"Missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    frames: list[pd.DataFrame] = []
    group_columns = ["sample_id", "range", "liquid_type"]
    for _, group in data.groupby(group_columns, sort=False):
        frames.append(_preprocess_one_spectrum(group, active_config))

    return pd.concat(frames, ignore_index=True)


def build_preprocessing_report(
    raw_data: pd.DataFrame,
    processed_data: pd.DataFrame,
) -> str:
    """Build a printable preprocessing summary with the method rationale."""
    outlier_summary = (
        processed_data.groupby(["range", "liquid_type"], observed=True)
        .agg(
            rows=("amplitude_dbm", "size"),
            outliers=("is_outlier", "sum"),
            raw_mean=("amplitude_dbm", "mean"),
            smooth_mean=("amplitude_smoothed_dbm", "mean"),
            normalized_mean=("amplitude_normalized", "mean"),
            normalized_std=("amplitude_normalized", "std"),
        )
        .reset_index()
    )
    outlier_summary["outlier_rate"] = (
        outlier_summary["outliers"] / outlier_summary["rows"] * 100
    )

    frequency_summary = (
        raw_data.groupby("range", observed=True)
        .agg(
            samples=("sample_id", "nunique"),
            rows=("amplitude_dbm", "size"),
            min_mhz=("frequency_mhz", "min"),
            max_mhz=("frequency_mhz", "max"),
        )
        .reset_index()
    )

    lines = [
        "数据预处理完成",
        "",
        "1. 原始数据概览",
        frequency_summary.to_string(index=False),
        "",
        "2. 去噪",
        "方法: 使用局部中值残差检测异常尖峰，异常点需要同时满足 IQR 边界和 Z-score 阈值；异常点不删除，而是按频率顺序线性插值。",
        "理由: 文档要求去除异常尖峰和明显错误点；频谱建模需要每条曲线保留相同频率网格，插值比删除行更适合后续 PCA、SVM 和特征提取。",
        "",
        "3. 平滑处理",
        "方法: 对去噪后的曲线使用 Savitzky-Golay Filter。",
        "理由: 文档推荐 Moving Average 或 Savitzky-Golay；Savitzky-Golay 在降低随机波动的同时更能保留峰值、谷值和曲线形态。",
        "",
        "4. Normalize",
        "方法: 对每条曲线的平滑幅值做 Z-score 标准化。",
        "理由: 文档推荐优先使用 Z-score；后续 SVM、PCA 等方法对尺度敏感，逐条曲线标准化可以减少不同实验之间的幅值偏移。",
        "",
        "5. 分组处理摘要",
        outlier_summary[
            [
                "range",
                "liquid_type",
                "rows",
                "outliers",
                "outlier_rate",
                "raw_mean",
                "smooth_mean",
                "normalized_mean",
                "normalized_std",
            ]
        ].to_string(
            index=False,
            formatters={
                "outlier_rate": "{:.3f}%".format,
                "raw_mean": "{:.3f}".format,
                "smooth_mean": "{:.3f}".format,
                "normalized_mean": "{:.3f}".format,
                "normalized_std": "{:.3f}".format,
            },
        ),
    ]
    return "\n".join(lines)


def save_preprocessed_data(
    processed_data: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    """Save preprocessed long-table data for downstream analysis."""
    if output_path is None:
        output_path = find_data_dir() / "processed" / "preprocessed_spectra.csv"

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(path, index=False)
    return path


def _preprocess_one_spectrum(
    group: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    frame = group.sort_values("frequency_hz").copy()
    amplitude = frame["amplitude_dbm"].astype(float)

    median_baseline = amplitude.rolling(
        window=config.median_window,
        center=True,
        min_periods=1,
    ).median()
    residual = amplitude - median_baseline

    q1 = residual.quantile(0.25)
    q3 = residual.quantile(0.75)
    iqr = q3 - q1
    iqr_lower = q1 - config.iqr_multiplier * iqr
    iqr_upper = q3 + config.iqr_multiplier * iqr
    iqr_outlier = (residual < iqr_lower) | (residual > iqr_upper)

    residual_std = residual.std()
    if residual_std == 0 or pd.isna(residual_std):
        zscore_outlier = pd.Series(False, index=frame.index)
    else:
        residual_zscore = (residual - residual.mean()) / residual_std
        zscore_outlier = residual_zscore.abs() > config.zscore_threshold

    is_outlier = iqr_outlier & zscore_outlier
    denoised = amplitude.mask(is_outlier)
    denoised = denoised.interpolate(method="linear", limit_direction="both")

    if denoised.isna().any():
        denoised = pd.Series(
            medfilt(amplitude.to_numpy(), kernel_size=config.median_window),
            index=frame.index,
        )

    window = _valid_savgol_window(len(frame), config.savgol_window)
    if window <= config.savgol_polyorder:
        smoothed = denoised
    else:
        smoothed = pd.Series(
            savgol_filter(
                denoised.to_numpy(),
                window_length=window,
                polyorder=config.savgol_polyorder,
                mode="interp",
            ),
            index=frame.index,
        )

    smoothed_std = smoothed.std()
    if smoothed_std == 0 or pd.isna(smoothed_std):
        normalized = smoothed - smoothed.mean()
    else:
        normalized = (smoothed - smoothed.mean()) / smoothed_std

    frame["is_outlier"] = is_outlier.to_numpy()
    frame["amplitude_denoised_dbm"] = denoised.to_numpy()
    frame["amplitude_smoothed_dbm"] = smoothed.to_numpy()
    frame["amplitude_normalized"] = normalized.to_numpy()
    return frame


def _valid_savgol_window(length: int, requested_window: int) -> int:
    window = min(length, requested_window)
    if window % 2 == 0:
        window -= 1
    return max(window, 1)
