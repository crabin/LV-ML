"""Key frequency band analysis for preprocessed spectrum data."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from liquid_identification.data_loader import find_data_dir


@dataclass(frozen=True)
class KeyBand:
    """Frequency band definition in MHz."""

    name: str
    min_mhz: float
    max_mhz: float


KEY_BANDS: tuple[KeyBand, ...] = (
    KeyBand("180-220MHz", 180.0, 220.0),
    KeyBand("270-300MHz", 270.0, 300.0),
)


def analyze_key_frequency_bands(
    processed_data: pd.DataFrame,
    *,
    value_column: str = "amplitude_normalized",
    bands: tuple[KeyBand, ...] = KEY_BANDS,
) -> dict[str, pd.DataFrame]:
    """Analyze key frequency bands from preprocessed long-table data."""
    required_columns = {"frequency_mhz", "liquid_type", "sample_id", value_column}
    missing_columns = required_columns.difference(processed_data.columns)
    if missing_columns:
        msg = f"Missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    band_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    distance_frames: list[pd.DataFrame] = []
    separability_frames: list[pd.DataFrame] = []
    anova_rows: list[dict[str, float | str | int]] = []

    for band in bands:
        band_data = _slice_band(processed_data, band).copy()
        if band_data.empty:
            continue

        band_data["band"] = band.name
        band_frames.append(band_data)
        metric_frames.append(_compute_band_metrics(band_data, value_column))
        distance_frames.append(_compute_pairwise_distances(band_data, value_column))
        separability_frames.append(_rank_frequency_separability(band_data, value_column))
        anova_rows.append(_compute_band_anova(band_data, value_column))

    if not band_frames:
        msg = "No data points found in configured key frequency bands."
        raise ValueError(msg)

    return {
        "band_data": pd.concat(band_frames, ignore_index=True),
        "band_metrics": pd.concat(metric_frames, ignore_index=True),
        "pairwise_distances": pd.concat(distance_frames, ignore_index=True),
        "frequency_separability": pd.concat(separability_frames, ignore_index=True),
        "band_anova": pd.DataFrame(anova_rows),
    }


def build_key_frequency_report(analysis: dict[str, pd.DataFrame]) -> str:
    """Build a printable key frequency analysis report."""
    metrics = analysis["band_metrics"]
    distances = analysis["pairwise_distances"]
    separability = analysis["frequency_separability"]
    anova = analysis["band_anova"]

    band_overview = (
        analysis["band_data"]
        .groupby("band", observed=True)
        .agg(
            rows=("amplitude_normalized", "size"),
            liquids=("liquid_type", "nunique"),
            min_mhz=("frequency_mhz", "min"),
            max_mhz=("frequency_mhz", "max"),
        )
        .reset_index()
    )

    top_distances = (
        distances.sort_values(["band", "rms_distance"], ascending=[True, False])
        .groupby("band", observed=True)
        .head(5)
    )
    closest_distances = (
        distances.sort_values(["band", "rms_distance"], ascending=[True, True])
        .groupby("band", observed=True)
        .head(5)
    )
    top_frequencies = (
        separability.sort_values(
            ["band", "between_liquid_std"],
            ascending=[True, False],
        )
        .groupby("band", observed=True)
        .head(8)
    )

    lines = [
        "关键频段分析完成",
        "",
        "1. 分析频段",
        band_overview.to_string(
            index=False,
            formatters={
                "min_mhz": "{:.3f}".format,
                "max_mhz": "{:.3f}".format,
            },
        ),
        "",
        "2. 方法与理由",
        "方法: 在预处理后的 Z-score 曲线上分析 180-220 MHz 与 270-300 MHz。",
        "理由: 实验步骤指出这两个频段中不同液体差异明显；使用标准化后的曲线可以降低整体幅值偏移对频段比较的影响。",
        "指标: 计算均值、峰值位置、谷值位置、线性斜率、频段能量、类间 RMS 距离和频率点区分度。",
        "说明: 当前每种液体每个频段只有一条曲线，ANOVA 和频率点排名用于探索性分析；严格统计检验仍需要后续重复实验样本。",
        "",
        "3. 频段内液体指标",
        metrics.to_string(
            index=False,
            formatters={
                "mean_value": "{:.3f}".format,
                "std_value": "{:.3f}".format,
                "peak_mhz": "{:.3f}".format,
                "peak_value": "{:.3f}".format,
                "valley_mhz": "{:.3f}".format,
                "valley_value": "{:.3f}".format,
                "slope_per_mhz": "{:.5f}".format,
                "energy": "{:.3f}".format,
            },
        ),
        "",
        "4. 类间距离最大的组合",
        top_distances.to_string(
            index=False,
            formatters={"rms_distance": "{:.3f}".format},
        ),
        "",
        "5. 类间距离最近的组合",
        closest_distances.to_string(
            index=False,
            formatters={"rms_distance": "{:.3f}".format},
        ),
        "",
        "6. 区分度最高的频率点",
        top_frequencies.to_string(
            index=False,
            formatters={
                "frequency_mhz": "{:.3f}".format,
                "between_liquid_std": "{:.3f}".format,
                "between_liquid_range": "{:.3f}".format,
            },
        ),
        "",
        "7. 探索性 ANOVA",
        anova.to_string(
            index=False,
            formatters={
                "f_statistic": "{:.3f}".format,
                "p_value": "{:.6f}".format,
            },
        ),
    ]
    return "\n".join(lines)


def save_key_frequency_analysis(
    analysis: dict[str, pd.DataFrame],
    output_dir: str | Path | None = None,
) -> Path:
    """Save key frequency analysis tables for downstream work."""
    if output_dir is None:
        output_dir = find_data_dir() / "analysis"

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    for name, frame in analysis.items():
        frame.to_csv(path / f"key_frequency_{name}.csv", index=False)

    return path


def _slice_band(data: pd.DataFrame, band: KeyBand) -> pd.DataFrame:
    return data[
        (data["frequency_mhz"] >= band.min_mhz)
        & (data["frequency_mhz"] <= band.max_mhz)
    ]


def _compute_band_metrics(
    band_data: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for (band, liquid_type), group in band_data.groupby(
        ["band", "liquid_type"],
        observed=True,
        sort=True,
    ):
        ordered = group.sort_values("frequency_mhz")
        values = ordered[value_column].astype(float)
        frequencies = ordered["frequency_mhz"].astype(float)
        peak_index = values.idxmax()
        valley_index = values.idxmin()
        slope = np.polyfit(frequencies.to_numpy(), values.to_numpy(), deg=1)[0]
        energy = np.trapezoid(values.to_numpy() ** 2, frequencies.to_numpy())

        rows.append(
            {
                "band": band,
                "liquid_type": liquid_type,
                "points": len(ordered),
                "mean_value": values.mean(),
                "std_value": values.std(),
                "peak_mhz": ordered.loc[peak_index, "frequency_mhz"],
                "peak_value": ordered.loc[peak_index, value_column],
                "valley_mhz": ordered.loc[valley_index, "frequency_mhz"],
                "valley_value": ordered.loc[valley_index, value_column],
                "slope_per_mhz": slope,
                "energy": energy,
            }
        )
    return pd.DataFrame(rows)


def _compute_pairwise_distances(
    band_data: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for band, group in band_data.groupby("band", observed=True, sort=True):
        pivot = group.pivot_table(
            index="frequency_mhz",
            columns="liquid_type",
            values=value_column,
            aggfunc="mean",
        ).dropna(axis=1)

        for liquid_a, liquid_b in combinations(pivot.columns, 2):
            diff = pivot[liquid_a] - pivot[liquid_b]
            rows.append(
                {
                    "band": band,
                    "liquid_a": liquid_a,
                    "liquid_b": liquid_b,
                    "points": len(diff),
                    "rms_distance": float(np.sqrt(np.mean(diff.to_numpy() ** 2))),
                }
            )
    return pd.DataFrame(rows)


def _rank_frequency_separability(
    band_data: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for (band, frequency_mhz), group in band_data.groupby(
        ["band", "frequency_mhz"],
        observed=True,
        sort=True,
    ):
        values = group[value_column].astype(float)
        rows.append(
            {
                "band": band,
                "frequency_mhz": frequency_mhz,
                "liquids": group["liquid_type"].nunique(),
                "between_liquid_std": values.std(),
                "between_liquid_range": values.max() - values.min(),
            }
        )
    return pd.DataFrame(rows)


def _compute_band_anova(
    band_data: pd.DataFrame,
    value_column: str,
) -> dict[str, float | str | int]:
    band = str(band_data["band"].iloc[0])
    groups = [
        group[value_column].astype(float).to_numpy()
        for _, group in band_data.groupby("liquid_type", observed=True)
    ]
    f_statistic, p_value = f_oneway(*groups)
    return {
        "band": band,
        "liquids": band_data["liquid_type"].nunique(),
        "points_per_liquid_min": int(
            band_data.groupby("liquid_type", observed=True).size().min()
        ),
        "f_statistic": float(f_statistic),
        "p_value": float(p_value),
    }
