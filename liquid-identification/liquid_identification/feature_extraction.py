"""Feature extraction for liquid spectrum classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from liquid_identification.data_loader import find_data_dir
from liquid_identification.key_frequency_analysis import KEY_BANDS, KeyBand


@dataclass(frozen=True)
class FeatureSet:
    """A feature matrix and its label vector."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    description: str


def extract_full_spectrum_features(
    processed_data: pd.DataFrame,
    *,
    value_column: str = "amplitude_normalized",
) -> FeatureSet:
    """Use all frequency points as model features."""
    feature_matrix = _pivot_frequency_features(processed_data, value_column)
    return FeatureSet(
        name="full_spectrum",
        X=feature_matrix,
        y=feature_matrix.index.to_series(name="liquid_type"),
        description="全频段特征: 使用全部频率点的预处理后幅值，保留完整频谱信息。",
    )


def extract_key_band_features(
    processed_data: pd.DataFrame,
    *,
    value_column: str = "amplitude_normalized",
    bands: tuple[KeyBand, ...] = KEY_BANDS,
) -> FeatureSet:
    """Use only configured key frequency bands as model features."""
    key_data = _slice_key_bands(processed_data, bands)
    feature_matrix = _pivot_frequency_features(key_data, value_column)
    return FeatureSet(
        name="key_bands",
        X=feature_matrix,
        y=feature_matrix.index.to_series(name="liquid_type"),
        description="关键频段特征: 仅使用 180-220 MHz 与 270-300 MHz，降低冗余并增强可解释性。",
    )


def extract_statistical_features(
    processed_data: pd.DataFrame,
    *,
    value_column: str = "amplitude_normalized",
    bands: tuple[KeyBand, ...] = KEY_BANDS,
) -> FeatureSet:
    """Extract summary statistics from each key frequency band."""
    rows: list[dict[str, float | str]] = []
    for liquid_type, liquid_data in processed_data.groupby("liquid_type", sort=True):
        row: dict[str, float | str] = {"liquid_type": liquid_type}
        for band in bands:
            band_data = _slice_key_bands(liquid_data, (band,)).sort_values(
                "frequency_mhz"
            )
            prefix = _band_feature_prefix(band)
            values = band_data[value_column].astype(float)
            frequencies = band_data["frequency_mhz"].astype(float)

            if band_data.empty:
                msg = f"No data found for {liquid_type} in band {band.name}"
                raise ValueError(msg)

            peak_index = values.idxmax()
            valley_index = values.idxmin()
            slope = np.polyfit(frequencies.to_numpy(), values.to_numpy(), deg=1)[0]

            row[f"{prefix}_mean"] = values.mean()
            row[f"{prefix}_max"] = values.max()
            row[f"{prefix}_min"] = values.min()
            row[f"{prefix}_variance"] = values.var()
            row[f"{prefix}_std"] = values.std()
            row[f"{prefix}_area"] = np.trapezoid(
                values.to_numpy(),
                frequencies.to_numpy(),
            )
            row[f"{prefix}_energy"] = np.trapezoid(
                values.to_numpy() ** 2,
                frequencies.to_numpy(),
            )
            row[f"{prefix}_peak_position_mhz"] = band_data.loc[
                peak_index,
                "frequency_mhz",
            ]
            row[f"{prefix}_valley_position_mhz"] = band_data.loc[
                valley_index,
                "frequency_mhz",
            ]
            row[f"{prefix}_slope"] = slope

        rows.append(row)

    frame = pd.DataFrame(rows).set_index("liquid_type").sort_index()
    return FeatureSet(
        name="statistical",
        X=frame,
        y=frame.index.to_series(name="liquid_type"),
        description="统计特征: 对关键频段提取均值、极值、方差、面积、能量、峰谷位置和斜率。",
    )


def extract_all_feature_sets(processed_data: pd.DataFrame) -> dict[str, FeatureSet]:
    """Build all supported feature sets."""
    feature_sets = [
        extract_full_spectrum_features(processed_data),
        extract_key_band_features(processed_data),
        extract_statistical_features(processed_data),
    ]
    return {feature_set.name: feature_set for feature_set in feature_sets}


def save_feature_sets(
    feature_sets: dict[str, FeatureSet],
    output_dir: str | Path | None = None,
) -> Path:
    """Save feature matrices and labels for downstream modeling."""
    if output_dir is None:
        output_dir = find_data_dir() / "features"

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    for feature_set in feature_sets.values():
        feature_set.X.to_csv(path / f"{feature_set.name}_X.csv")
        feature_set.y.to_csv(path / f"{feature_set.name}_y.csv", index=False)
        with_label = feature_set.X.copy()
        with_label.insert(0, "liquid_type", feature_set.y.to_numpy())
        with_label.to_csv(path / f"{feature_set.name}_features.csv", index=False)

    return path


def build_feature_extraction_report(feature_sets: dict[str, FeatureSet]) -> str:
    """Build a printable feature extraction summary."""
    summary = pd.DataFrame(
        [
            {
                "feature_set": feature_set.name,
                "samples": feature_set.X.shape[0],
                "features": feature_set.X.shape[1],
                "description": feature_set.description,
            }
            for feature_set in feature_sets.values()
        ]
    )
    lines = [
        "特征提取完成",
        "",
        "1. 三种特征入口",
        summary.to_string(index=False),
        "",
        "2. 使用建议",
        "全频段特征适合先作为模型基线，信息保留最完整，但维度最高。",
        "关键频段特征只保留实验步骤指定的 180-220 MHz 与 270-300 MHz，维度更低、解释性更强。",
        "统计特征维度最低，适合小样本场景和可解释建模，但会损失曲线细节。",
        "",
        "3. 输出格式",
        "每种特征都会保存 *_X.csv、*_y.csv 和 *_features.csv。",
        "*_X.csv 是模型输入矩阵，*_y.csv 是标签，*_features.csv 是带标签的便于人工检查的表。",
    ]
    return "\n".join(lines)


def _pivot_frequency_features(
    data: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    required_columns = {"range", "frequency_mhz", "liquid_type", value_column}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        msg = f"Missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    frame = data.copy()
    frame["feature_name"] = frame.apply(
        lambda row: f"{row['range']}__{row['frequency_mhz']:.6f}MHz",
        axis=1,
    )
    return frame.pivot_table(
        index="liquid_type",
        columns="feature_name",
        values=value_column,
        aggfunc="mean",
    ).sort_index(axis=1)


def _slice_key_bands(
    data: pd.DataFrame,
    bands: tuple[KeyBand, ...],
) -> pd.DataFrame:
    masks = [
        (data["frequency_mhz"] >= band.min_mhz)
        & (data["frequency_mhz"] <= band.max_mhz)
        for band in bands
    ]
    if not masks:
        return data.iloc[0:0].copy()

    mask = masks[0]
    for next_mask in masks[1:]:
        mask = mask | next_mask
    return data[mask].copy()


def _band_feature_prefix(band: KeyBand) -> str:
    return band.name.lower().replace("-", "_").replace("mhz", "mhz")
