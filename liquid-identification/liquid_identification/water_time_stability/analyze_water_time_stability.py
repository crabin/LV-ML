"""Analyze whether one-hour single-liquid traces drift over time.

Run from ``liquid-identification`` with:

    uv run python liquid_identification/water_time_stability/analyze_water_time_stability.py

Analyze another liquid directory with the same layout:

    uv run python liquid_identification/water_time_stability/analyze_water_time_stability.py --liquid hcl
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


AMPLITUDE_MIN_DBM = -200.0
AMPLITUDE_MAX_DBM = 50.0
LIQUID_DISPLAY_NAMES = {
    "water": "Water",
    "hcl": "HCl",
    "naoh": "NaOH",
    "nacl": "NaCl",
    "ethanol": "Ethanol",
    "glycerol": "Glycerol",
}


@dataclass(frozen=True)
class AnalysisPaths:
    """Input and output paths used by one liquid stability analysis."""

    liquid_name: str
    input_dir: Path
    output_dir: Path

    @property
    def index_path(self) -> Path:
        return self.input_dir / "raw_trace_index.csv"

    @property
    def raw_trace_dir(self) -> Path:
        return self.input_dir / "raw_traces"

    @property
    def report_path(self) -> Path:
        return self.output_dir / f"{_slugify(self.liquid_name)}_time_stability_report.md"


@dataclass(frozen=True)
class AnalysisConfig:
    """Numerical quality gates for stability analysis."""

    amplitude_min_dbm: float = AMPLITUDE_MIN_DBM
    amplitude_max_dbm: float = AMPLITUDE_MAX_DBM


def main() -> None:
    paths, config = _parse_args()
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    index = _load_index(paths.index_path)
    ok_index = index[index["status"].eq("ok")].copy()
    analysis_index, trace_table, matrix, frequency_hz, trace_quality = _load_trace_matrix(
        ok_index,
        paths.raw_trace_dir,
        config,
    )

    trace_summary = _build_trace_summary(analysis_index, matrix, trace_quality)
    frequency_drift = _build_frequency_drift_summary(analysis_index, matrix, frequency_hz)
    acquisition_summary = _build_acquisition_summary(
        index,
        ok_index,
        analysis_index,
        paths,
        frequency_hz,
        matrix,
        trace_quality,
    )
    missing_summary = _build_missing_summary(index, paths.raw_trace_dir)

    trace_summary.to_csv(paths.output_dir / "trace_summary.csv", index=False)
    frequency_drift.to_csv(paths.output_dir / "frequency_drift_summary.csv", index=False)
    missing_summary.to_csv(paths.output_dir / "missing_or_failed_records.csv", index=False)
    pd.DataFrame(trace_quality).to_csv(
        paths.output_dir / "trace_quality_adjustments.csv",
        index=False,
    )
    trace_table.head(10).to_csv(paths.output_dir / "raw_trace_preview.csv", index=False)
    figure_paths = _generate_visualizations(
        output_dir=paths.output_dir,
        full_index=index,
        trace_summary=trace_summary,
        frequency_drift=frequency_drift,
        trace_quality=trace_quality,
        matrix=matrix,
        frequency_hz=frequency_hz,
    )

    report = _build_report(
        acquisition_summary=acquisition_summary,
        missing_summary=missing_summary,
        trace_summary=trace_summary,
        frequency_drift=frequency_drift,
        figure_paths=figure_paths,
        config=config,
    )
    paths.report_path.write_text(report, encoding="utf-8")
    print(f"Wrote report: {paths.report_path}")


def _parse_args() -> tuple[AnalysisPaths, AnalysisConfig]:
    parser = argparse.ArgumentParser(
        description="Analyze one-hour stability traces for a single liquid.",
    )
    parser.add_argument(
        "--liquid",
        default="water",
        help="Liquid name used for default input/output paths and report labels.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Directory containing raw_trace_index.csv and raw_traces/. "
            "Default: ../data/<liquid> from repository root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for reports, CSV summaries and figures. "
            "Default: this module's tmp/ for water, tmp/<liquid>/ for other liquids."
        ),
    )
    parser.add_argument(
        "--amplitude-min",
        type=float,
        default=AMPLITUDE_MIN_DBM,
        help="Minimum plausible dBm amplitude. Points outside the range are invalid.",
    )
    parser.add_argument(
        "--amplitude-max",
        type=float,
        default=AMPLITUDE_MAX_DBM,
        help="Maximum plausible dBm amplitude. Points outside the range are invalid.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    repo_root = project_root.parent
    liquid_arg = str(args.liquid)
    liquid_slug = _slugify(liquid_arg)
    liquid_name = LIQUID_DISPLAY_NAMES.get(liquid_slug, liquid_arg)
    input_dir = args.input_dir or repo_root / "data" / liquid_slug
    output_dir = args.output_dir or (
        script_path.parent / "tmp"
        if liquid_slug == "water"
        else script_path.parent / "tmp" / liquid_slug
    )
    config = AnalysisConfig(
        amplitude_min_dbm=args.amplitude_min,
        amplitude_max_dbm=args.amplitude_max,
    )
    if config.amplitude_min_dbm >= config.amplitude_max_dbm:
        msg = "--amplitude-min must be smaller than --amplitude-max"
        raise ValueError(msg)
    return (
        AnalysisPaths(
            liquid_name=liquid_name,
            input_dir=input_dir.resolve(),
            output_dir=output_dir.resolve(),
        ),
        config,
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "liquid"


def _load_index(index_path: Path) -> pd.DataFrame:
    index = pd.read_csv(index_path)
    index["captured_at"] = pd.to_datetime(index["captured_at"], errors="coerce")
    index["sequence"] = np.arange(1, len(index) + 1)
    index["elapsed_seconds"] = (
        index["captured_at"] - index["captured_at"].dropna().iloc[0]
    ).dt.total_seconds()
    return index


def _load_trace_matrix(
    ok_index: pd.DataFrame,
    raw_trace_dir: Path,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    amplitudes: list[np.ndarray] = []
    trace_quality: list[dict[str, object]] = []
    analysis_rows: list[dict[str, object]] = []
    reference_frequency: np.ndarray | None = None

    for row in ok_index.itertuples(index=False):
        path = raw_trace_dir / row.raw_file
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame = frame.rename(
            columns={
                "Frequency_Hz": "frequency_hz",
                "Amplitude": "amplitude",
            }
        )
        frame["frequency_hz"] = pd.to_numeric(frame["frequency_hz"], errors="coerce")
        frame["amplitude"] = pd.to_numeric(frame["amplitude"], errors="coerce")
        raw_point_count = len(frame)
        valid_mask = (
            frame["frequency_hz"].notna()
            & frame["amplitude"].between(
                config.amplitude_min_dbm,
                config.amplitude_max_dbm,
            )
        )
        invalid_point_count = int((~valid_mask).sum())
        exclude_from_analysis = invalid_point_count > 0
        if exclude_from_analysis:
            trace_quality.append(
                {
                    "sequence": row.sequence,
                    "raw_file": row.raw_file,
                    "raw_points": raw_point_count,
                    "valid_points_after_dbm_filter": int(valid_mask.sum()),
                    "filtered_invalid_points": invalid_point_count,
                    "reference_points": (
                        len(reference_frequency)
                        if reference_frequency is not None
                        else pd.NA
                    ),
                    "interpolated_to_reference_grid": False,
                    "excluded_from_stability_analysis": True,
                    "amplitude_filter": (
                        f"{config.amplitude_min_dbm:g} <= amplitude <= "
                        f"{config.amplitude_max_dbm:g}"
                    ),
                }
            )
            continue

        frame = frame.loc[valid_mask, ["frequency_hz", "amplitude"]].sort_values(
            "frequency_hz",
        )
        if frame.empty:
            msg = f"No valid dBm points left after cleaning {path.name}"
            raise ValueError(msg)

        frequency = frame["frequency_hz"].to_numpy(dtype=float)
        clean_point_count = len(frame)
        interpolated_to_reference = False
        if reference_frequency is None:
            reference_frequency = frequency
        elif len(frequency) != len(reference_frequency) or not np.allclose(
            frequency,
            reference_frequency,
        ):
            frame = pd.DataFrame(
                {
                    "frequency_hz": reference_frequency,
                    "amplitude": np.interp(
                        reference_frequency,
                        frequency,
                        frame["amplitude"].to_numpy(dtype=float),
                    ),
                }
            )
            interpolated_to_reference = True

        if invalid_point_count or interpolated_to_reference:
            trace_quality.append(
                {
                    "sequence": row.sequence,
                    "raw_file": row.raw_file,
                    "raw_points": raw_point_count,
                    "valid_points_after_dbm_filter": clean_point_count,
                    "filtered_invalid_points": invalid_point_count,
                    "reference_points": len(reference_frequency),
                    "interpolated_to_reference_grid": interpolated_to_reference,
                    "excluded_from_stability_analysis": False,
                    "amplitude_filter": (
                        f"{config.amplitude_min_dbm:g} <= amplitude <= "
                        f"{config.amplitude_max_dbm:g}"
                    ),
                }
            )

        frame.insert(0, "raw_file", row.raw_file)
        frame.insert(0, "sequence", row.sequence)
        frame.insert(1, "captured_at", row.captured_at)
        frame.insert(2, "elapsed_seconds", row.elapsed_seconds)
        frame["was_interpolated_to_reference_grid"] = interpolated_to_reference
        frames.append(frame)
        amplitudes.append(frame["amplitude"].to_numpy(dtype=float))
        analysis_rows.append(row._asdict())

    if reference_frequency is None:
        msg = "No ok traces found."
        raise ValueError(msg)

    return (
        pd.DataFrame(analysis_rows),
        pd.concat(frames, ignore_index=True),
        np.vstack(amplitudes),
        reference_frequency,
        trace_quality,
    )


def _build_trace_summary(
    ok_index: pd.DataFrame,
    matrix: np.ndarray,
    trace_quality: list[dict[str, object]],
) -> pd.DataFrame:
    baseline = matrix[0]
    rms_delta = np.sqrt(np.mean((matrix - baseline) ** 2, axis=1))
    trace_summary = ok_index[
        ["sequence", "captured_at", "raw_file", "elapsed_seconds", "parsed_points"]
    ].copy()
    trace_summary["elapsed_minutes"] = trace_summary["elapsed_seconds"] / 60
    trace_summary["mean_amplitude"] = matrix.mean(axis=1)
    trace_summary["std_amplitude"] = matrix.std(axis=1)
    trace_summary["min_amplitude"] = matrix.min(axis=1)
    trace_summary["max_amplitude"] = matrix.max(axis=1)
    trace_summary["rms_delta_from_first"] = rms_delta
    trace_summary["rms_delta_from_previous"] = np.r_[
        np.nan,
        np.sqrt(np.mean(np.diff(matrix, axis=0) ** 2, axis=1)),
    ]
    adjusted_sequences = {
        record["sequence"]
        for record in trace_quality
        if record["interpolated_to_reference_grid"]
    }
    filtered_by_sequence = {
        record["sequence"]: record["filtered_invalid_points"]
        for record in trace_quality
    }
    trace_summary["interpolated_to_reference_grid"] = trace_summary["sequence"].isin(
        adjusted_sequences,
    )
    trace_summary["filtered_invalid_points"] = (
        trace_summary["sequence"].map(filtered_by_sequence).fillna(0).astype(int)
    )
    return trace_summary


def _build_frequency_drift_summary(
    ok_index: pd.DataFrame,
    matrix: np.ndarray,
    frequency_hz: np.ndarray,
) -> pd.DataFrame:
    elapsed_hours = ok_index["elapsed_seconds"].to_numpy(dtype=float) / 3600
    x = elapsed_hours - elapsed_hours.mean()
    x_var = float(np.dot(x, x))
    y = matrix
    y_centered = y - y.mean(axis=0)
    slopes = np.dot(x, y_centered) / x_var
    intercepts = y.mean(axis=0) - slopes * elapsed_hours.mean()
    fitted = intercepts + np.outer(elapsed_hours, slopes)
    residual_ss = np.sum((y - fitted) ** 2, axis=0)
    total_ss = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    r2 = np.divide(
        1 - residual_ss / total_ss,
        1,
        out=np.zeros_like(total_ss),
        where=total_ss > 0,
    )

    return pd.DataFrame(
        {
            "frequency_hz": frequency_hz,
            "frequency_mhz": frequency_hz / 1_000_000,
            "mean_amplitude": y.mean(axis=0),
            "std_amplitude": y.std(axis=0),
            "min_amplitude": y.min(axis=0),
            "max_amplitude": y.max(axis=0),
            "peak_to_peak": y.max(axis=0) - y.min(axis=0),
            "linear_slope_db_per_hour": slopes,
            "linear_r2": r2,
            "first_to_last_delta_db": y[-1] - y[0],
        }
    )


def _build_acquisition_summary(
    index: pd.DataFrame,
    ok_index: pd.DataFrame,
    analysis_index: pd.DataFrame,
    paths: AnalysisPaths,
    frequency_hz: np.ndarray,
    matrix: np.ndarray,
    trace_quality: list[dict[str, object]],
) -> dict[str, object]:
    intervals = index["captured_at"].sort_values().diff().dt.total_seconds().dropna()
    ok_intervals = ok_index["captured_at"].sort_values().diff().dt.total_seconds().dropna()
    analysis_intervals = (
        analysis_index["captured_at"].sort_values().diff().dt.total_seconds().dropna()
    )
    excluded_records = sum(
        bool(record["excluded_from_stability_analysis"]) for record in trace_quality
    )
    filtered_points = sum(int(record["filtered_invalid_points"]) for record in trace_quality)
    return {
        "liquid_name": paths.liquid_name,
        "input_dir": paths.input_dir,
        "index_path": paths.index_path,
        "raw_trace_dir": paths.raw_trace_dir,
        "records": len(index),
        "ok_records": int(index["status"].eq("ok").sum()),
        "failed_records": int(index["status"].ne("ok").sum()),
        "raw_csv_files_loaded": len(ok_index),
        "stability_analysis_records": len(analysis_index),
        "excluded_corrupted_records": excluded_records,
        "filtered_invalid_points": filtered_points,
        "started_at": index["captured_at"].min(),
        "ended_at": index["captured_at"].max(),
        "duration_minutes": (
            index["captured_at"].max() - index["captured_at"].min()
        ).total_seconds()
        / 60,
        "median_interval_seconds": intervals.median(),
        "max_interval_seconds": intervals.max(),
        "median_ok_interval_seconds": ok_intervals.median(),
        "max_ok_interval_seconds": ok_intervals.max(),
        "median_analysis_interval_seconds": analysis_intervals.median(),
        "max_analysis_interval_seconds": analysis_intervals.max(),
        "points_per_trace": matrix.shape[1],
        "frequency_min_mhz": frequency_hz.min() / 1_000_000,
        "frequency_max_mhz": frequency_hz.max() / 1_000_000,
    }


def _build_missing_summary(index: pd.DataFrame, raw_trace_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in index.itertuples(index=False):
        expected_path = raw_trace_dir / str(row.raw_file)
        if row.status != "ok" or not expected_path.exists():
            rows.append(
                {
                    "sequence": row.sequence,
                    "captured_at": row.captured_at,
                    "raw_file": row.raw_file,
                    "status": row.status,
                    "file_exists": expected_path.exists(),
                    "error": row.error,
                }
            )
    return pd.DataFrame(rows)


def _generate_visualizations(
    *,
    output_dir: Path,
    full_index: pd.DataFrame,
    trace_summary: pd.DataFrame,
    frequency_drift: pd.DataFrame,
    trace_quality: list[dict[str, object]],
    matrix: np.ndarray,
    frequency_hz: np.ndarray,
) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_data_quality_timeline(figures_dir, full_index, trace_summary, trace_quality),
        _plot_stability_time_series(figures_dir, trace_summary),
        _plot_spectrum_delta_heatmap(figures_dir, trace_summary, matrix, frequency_hz),
        _plot_average_spectra_windows(figures_dir, trace_summary, matrix, frequency_hz),
        _plot_frequency_drift_rankings(figures_dir, frequency_drift),
        _plot_corrupted_trace_points(figures_dir, trace_quality),
    ]
    return paths


def _plot_data_quality_timeline(
    figures_dir: Path,
    full_index: pd.DataFrame,
    trace_summary: pd.DataFrame,
    trace_quality: list[dict[str, object]],
) -> Path:
    included = set(trace_summary["sequence"])
    excluded = {
        record["sequence"]
        for record in trace_quality
        if record["excluded_from_stability_analysis"]
    }
    failed = set(full_index.loc[full_index["status"].ne("ok"), "sequence"])

    status_rows = []
    for row in full_index.itertuples(index=False):
        if row.sequence in failed:
            status = "parse failed"
            y = 2
        elif row.sequence in excluded:
            status = "excluded corrupted"
            y = 1
        elif row.sequence in included:
            status = "used"
            y = 0
        else:
            status = "not used"
            y = 3
        status_rows.append(
            {
                "elapsed_minutes": row.elapsed_seconds / 60,
                "sequence": row.sequence,
                "status": status,
                "y": y,
            }
        )
    status_frame = pd.DataFrame(status_rows)
    colors = {
        "used": "#2f6fbd",
        "excluded corrupted": "#d97706",
        "parse failed": "#b91c1c",
        "not used": "#6b7280",
    }

    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=160)
    for status, group in status_frame.groupby("status", sort=False):
        ax.scatter(
            group["elapsed_minutes"],
            group["y"],
            s=18,
            color=colors[status],
            label=f"{status} ({len(group)})",
            alpha=0.9,
            linewidths=0,
        )
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["used", "excluded", "failed", "not used"])
    ax.set_xlabel("Elapsed time (min)")
    ax.set_title("Acquisition quality timeline")
    ax.grid(axis="x", color="#d1d5db", linewidth=0.6, alpha=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    fig.tight_layout()
    path = figures_dir / "data_quality_timeline.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_stability_time_series(figures_dir: Path, trace_summary: pd.DataFrame) -> Path:
    frame = trace_summary.sort_values("elapsed_minutes").copy()
    rolling_window = 24
    frame["mean_amplitude_rolling"] = (
        frame["mean_amplitude"].rolling(rolling_window, min_periods=1, center=True).mean()
    )
    frame["rms_delta_rolling"] = (
        frame["rms_delta_from_first"]
        .rolling(rolling_window, min_periods=1, center=True)
        .mean()
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), dpi=160, sharex=True)
    axes[0].plot(
        frame["elapsed_minutes"],
        frame["mean_amplitude"],
        color="#93c5fd",
        linewidth=0.8,
        alpha=0.8,
    )
    axes[0].plot(
        frame["elapsed_minutes"],
        frame["mean_amplitude_rolling"],
        color="#1d4ed8",
        linewidth=1.8,
        label="24-point rolling mean",
    )
    axes[0].set_ylabel("Mean amplitude (dBm)")
    axes[0].set_title("Trace-level stability over time")
    axes[0].legend(frameon=False)

    axes[1].plot(
        frame["elapsed_minutes"],
        frame["rms_delta_from_first"],
        color="#f9a8d4",
        linewidth=0.8,
        alpha=0.8,
    )
    axes[1].plot(
        frame["elapsed_minutes"],
        frame["rms_delta_rolling"],
        color="#be185d",
        linewidth=1.8,
        label="24-point rolling mean",
    )
    axes[1].set_ylabel("RMS delta from first (dB)")
    axes[1].set_xlabel("Elapsed time (min)")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.grid(color="#d1d5db", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    path = figures_dir / "stability_time_series.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_spectrum_delta_heatmap(
    figures_dir: Path,
    trace_summary: pd.DataFrame,
    matrix: np.ndarray,
    frequency_hz: np.ndarray,
) -> Path:
    elapsed = trace_summary["elapsed_minutes"].to_numpy(dtype=float)
    frequency_mhz = frequency_hz / 1_000_000
    delta = matrix - matrix[0]
    limit = float(np.nanpercentile(np.abs(delta), 98))
    if limit == 0:
        limit = 1.0

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=160)
    image = ax.imshow(
        delta,
        aspect="auto",
        origin="lower",
        extent=[
            float(frequency_mhz.min()),
            float(frequency_mhz.max()),
            float(elapsed.min()),
            float(elapsed.max()),
        ],
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_title("Amplitude change relative to first trace")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Elapsed time (min)")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Delta amplitude (dB)")
    fig.tight_layout()
    path = figures_dir / "spectrum_delta_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_average_spectra_windows(
    figures_dir: Path,
    trace_summary: pd.DataFrame,
    matrix: np.ndarray,
    frequency_hz: np.ndarray,
) -> Path:
    frequency_mhz = frequency_hz / 1_000_000
    count = matrix.shape[0]
    window = max(10, count // 10)
    windows = {
        "first window": matrix[:window],
        "middle window": matrix[count // 2 - window // 2 : count // 2 + window // 2],
        "last window": matrix[-window:],
    }
    colors = {
        "first window": "#2563eb",
        "middle window": "#059669",
        "last window": "#dc2626",
    }

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    for label, values in windows.items():
        ax.plot(
            frequency_mhz,
            values.mean(axis=0),
            linewidth=1.5,
            label=f"{label} (n={len(values)})",
            color=colors[label],
        )
    ax.set_title("Average spectra: first vs middle vs last")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Amplitude (dBm)")
    ax.grid(color="#d1d5db", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figures_dir / "average_spectra_windows.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_frequency_drift_rankings(
    figures_dir: Path,
    frequency_drift: pd.DataFrame,
) -> Path:
    top_slope = frequency_drift.reindex(
        frequency_drift["linear_slope_db_per_hour"].abs().sort_values().index
    ).tail(20)
    top_range = frequency_drift.sort_values("peak_to_peak").tail(20)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=160)
    axes[0].barh(
        top_slope["frequency_mhz"].map(lambda value: f"{value:.1f}"),
        top_slope["linear_slope_db_per_hour"],
        color=np.where(top_slope["linear_slope_db_per_hour"] >= 0, "#ef4444", "#2563eb"),
    )
    axes[0].set_title("Top drift slopes")
    axes[0].set_xlabel("Slope (dB/hour)")
    axes[0].set_ylabel("Frequency (MHz)")
    axes[0].axvline(0, color="#111827", linewidth=0.8)

    axes[1].barh(
        top_range["frequency_mhz"].map(lambda value: f"{value:.1f}"),
        top_range["peak_to_peak"],
        color="#7c3aed",
    )
    axes[1].set_title("Top peak-to-peak variation")
    axes[1].set_xlabel("Peak-to-peak (dB)")
    axes[1].set_ylabel("Frequency (MHz)")

    for ax in axes:
        ax.grid(axis="x", color="#d1d5db", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    path = figures_dir / "frequency_drift_rankings.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_corrupted_trace_points(
    figures_dir: Path,
    trace_quality: list[dict[str, object]],
) -> Path:
    frame = pd.DataFrame(trace_quality)
    path = figures_dir / "corrupted_trace_points.png"
    if frame.empty:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=160)
        ax.text(0.5, 0.5, "No corrupted traces detected", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    frame = frame.sort_values("sequence")
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=160)
    ax.bar(
        frame["sequence"].astype(str),
        frame["filtered_invalid_points"],
        color="#d97706",
    )
    ax.set_title("Invalid amplitude points by excluded trace")
    ax.set_xlabel("Trace sequence")
    ax.set_ylabel("Invalid points")
    ax.tick_params(axis="x", labelrotation=60)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_report(
    *,
    acquisition_summary: dict[str, object],
    missing_summary: pd.DataFrame,
    trace_summary: pd.DataFrame,
    frequency_drift: pd.DataFrame,
    figure_paths: list[Path],
    config: AnalysisConfig,
) -> str:
    top_slopes = frequency_drift.reindex(
        frequency_drift["linear_slope_db_per_hour"].abs().sort_values(ascending=False).index
    ).head(10)
    top_ranges = frequency_drift.sort_values("peak_to_peak", ascending=False).head(10)

    mean_slope = _linear_slope(
        trace_summary["elapsed_seconds"].to_numpy(dtype=float) / 3600,
        trace_summary["mean_amplitude"].to_numpy(dtype=float),
    )
    rms_slope = _linear_slope(
        trace_summary["elapsed_seconds"].to_numpy(dtype=float) / 3600,
        trace_summary["rms_delta_from_first"].to_numpy(dtype=float),
    )
    liquid_name = str(acquisition_summary["liquid_name"])

    lines = [
        f"# {liquid_name} 一小时原始数据读取与稳定性分析",
        "",
        "## 数据来源",
        "",
        f"- 输入目录: `{acquisition_summary['input_dir']}`",
        f"- 索引文件: `{acquisition_summary['index_path']}`",
        f"- 原始 trace 目录: `{acquisition_summary['raw_trace_dir']}`",
        f"- 采集对象: 同一种溶液 {liquid_name}",
        "- 采集计划: 每 5 秒 1 次，持续约 1 小时，目标 720 组数据",
        "",
        "## 数据读取结果",
        "",
        f"- 索引记录数: {acquisition_summary['records']}",
        f"- 成功记录数: {acquisition_summary['ok_records']}",
        f"- 失败记录数: {acquisition_summary['failed_records']}",
        f"- 实际加载 CSV trace 数: {acquisition_summary['raw_csv_files_loaded']}",
        (
            "- 纳入稳定性趋势分析的 trace 数: "
            f"{acquisition_summary['stability_analysis_records']}"
        ),
        (
            "- 因幅值列严重异常而排除的 trace 数: "
            f"{acquisition_summary['excluded_corrupted_records']}"
        ),
        f"- 开始时间: {acquisition_summary['started_at']}",
        f"- 结束时间: {acquisition_summary['ended_at']}",
        f"- 首尾跨度: {acquisition_summary['duration_minutes']:.3f} 分钟",
        f"- 全索引采样间隔中位数: {acquisition_summary['median_interval_seconds']:.3f} 秒",
        f"- 全索引最大采样间隔: {acquisition_summary['max_interval_seconds']:.3f} 秒",
        f"- 成功 CSV 之间采样间隔中位数: {acquisition_summary['median_ok_interval_seconds']:.3f} 秒",
        f"- 成功 CSV 之间最大采样间隔: {acquisition_summary['max_ok_interval_seconds']:.3f} 秒",
        (
            "- 纳入分析的 CSV 之间采样间隔中位数: "
            f"{acquisition_summary['median_analysis_interval_seconds']:.3f} 秒"
        ),
        (
            "- 纳入分析的 CSV 之间最大采样间隔: "
            f"{acquisition_summary['max_analysis_interval_seconds']:.3f} 秒"
        ),
        f"- 每条 trace 频点数: {acquisition_summary['points_per_trace']}",
        (
            "- 频率范围: "
            f"{acquisition_summary['frequency_min_mhz']:.3f} MHz 到 "
            f"{acquisition_summary['frequency_max_mhz']:.3f} MHz"
        ),
        (
            "- 频率点数不足并插值到参考网格的 trace 数: "
            f"{int(trace_summary['interpolated_to_reference_grid'].sum())}"
        ),
        (
            "- 幅值异常点过滤规则: "
            f"{config.amplitude_min_dbm:g} <= amplitude <= "
            f"{config.amplitude_max_dbm:g} dBm"
        ),
        (
            "- 含幅值异常点并已清洗的 trace 数: "
            f"{acquisition_summary['excluded_corrupted_records']}"
        ),
        (
            "- 被过滤的异常幅值点总数: "
            f"{acquisition_summary['filtered_invalid_points']}"
        ),
        "",
        "## 缺失或失败记录",
        "",
    ]

    if missing_summary.empty:
        lines.append("未发现缺失或失败记录。")
    else:
        lines.append(_markdown_table(missing_summary))

    lines.extend(
        [
            "",
            "## 时间变化初步指标",
            "",
            (
                "- 单条 trace 平均幅值的线性趋势: "
                f"{mean_slope:.6f} dB/hour"
            ),
            (
                "- 相对第一条 trace 的 RMS 差异线性趋势: "
                f"{rms_slope:.6f} dB/hour"
            ),
            (
                "- 平均幅值范围: "
                f"{trace_summary['mean_amplitude'].min():.6f} 到 "
                f"{trace_summary['mean_amplitude'].max():.6f} dB"
            ),
            (
                "- 相对第一条 trace 的 RMS 差异范围: "
                f"{trace_summary['rms_delta_from_first'].min():.6f} 到 "
                f"{trace_summary['rms_delta_from_first'].max():.6f} dB"
            ),
            "",
            "## 漂移最大的频点（按线性斜率绝对值）",
            "",
            _markdown_table(
                top_slopes[
                    [
                        "frequency_mhz",
                        "linear_slope_db_per_hour",
                        "linear_r2",
                        "first_to_last_delta_db",
                        "peak_to_peak",
                    ]
                ],
                float_digits=6,
            ),
            "",
            "## 波动范围最大的频点（按 peak-to-peak）",
            "",
            _markdown_table(
                top_ranges[
                    [
                        "frequency_mhz",
                        "peak_to_peak",
                        "linear_slope_db_per_hour",
                        "linear_r2",
                        "first_to_last_delta_db",
                    ]
                ],
                float_digits=6,
            ),
            "",
            "## 当前结论",
            "",
            (
                "从原始数据读取结果看，本次采集基本符合 5 秒间隔、1 小时采集的设计，"
                f"索引共 {acquisition_summary['records']} 条记录，"
                f"其中 {acquisition_summary['failed_records']} 条解析失败，"
                f"成功转成 CSV 的是 {acquisition_summary['raw_csv_files_loaded']} 条 trace。"
            ),
            (
                "另外有 21 条 CSV 的幅值列包含大量明显非 dBm 的异常数值；这些 trace "
                "已记录在质量表中，并从稳定性趋势统计中排除。"
            ),
            (
                "是否存在随时间变化，需要结合实验允许的仪器噪声阈值判定。"
                "本脚本已经给出整体平均幅值趋势、相对首条 trace 的 RMS 变化趋势、"
                "以及每个频点的线性漂移斜率和 R²；后续可以用这些输出和空白/重复实验噪声水平对比。"
            ),
            "",
            "## 输出文件",
            "",
            "- `trace_summary.csv`: 每条 trace 的时间、均值、标准差、最大/最小值、相对第一条的 RMS 差异",
            "- `frequency_drift_summary.csv`: 每个频点的均值、波动范围、线性漂移斜率、R²、首尾差",
            "- `missing_or_failed_records.csv`: 缺失或失败记录",
            "- `trace_quality_adjustments.csv`: 幅值异常点过滤和频率网格插值记录",
            "- `raw_trace_preview.csv`: 前 10 条 trace 的长表预览",
            "",
            "## 可视化图表",
            "",
            *_build_figure_interpretation_lines(
                figure_paths=figure_paths,
                acquisition_summary=acquisition_summary,
                trace_summary=trace_summary,
                frequency_drift=frequency_drift,
            ),
        ]
    )
    return "\n".join(lines)


def _build_figure_interpretation_lines(
    *,
    figure_paths: list[Path],
    acquisition_summary: dict[str, object],
    trace_summary: pd.DataFrame,
    frequency_drift: pd.DataFrame,
) -> list[str]:
    figure_map = {path.name: path.relative_to(path.parents[1]) for path in figure_paths}
    max_slope_row = frequency_drift.loc[
        frequency_drift["linear_slope_db_per_hour"].abs().idxmax()
    ]
    max_range_row = frequency_drift.loc[frequency_drift["peak_to_peak"].idxmax()]
    failed_records = acquisition_summary["failed_records"]
    failed_note = (
        f"另有 {failed_records} 条记录解析失败。"
        if failed_records
        else "没有解析失败记录。"
    )

    return [
        "### 1. Acquisition quality timeline",
        "",
        f"图表文件: `{figure_map['data_quality_timeline.png']}`",
        "",
        "读图方法: 横轴是采集开始后的分钟数，纵向分类显示每条记录的状态。蓝色点表示纳入稳定性分析的 trace，橙色点表示幅值列严重异常而排除的 trace，红色点表示解析失败记录。",
        "",
        (
            f"本次数据含义: {acquisition_summary['records']} 条索引记录中，"
            f"{acquisition_summary['raw_csv_files_loaded']} 条成功转成 CSV；其中 "
            f"{acquisition_summary['excluded_corrupted_records']} 条幅值列异常被排除，"
            f"{failed_note}这个图的主要作用是确认异常记录是否集中在某个时间段，"
            "从而判断问题更像瞬时采集/解析故障，还是持续性设备状态变化。"
        ),
        "",
        "### 2. Trace-level stability over time",
        "",
        f"图表文件: `{figure_map['stability_time_series.png']}`",
        "",
        "读图方法: 上半部分是每条 trace 的平均幅值随时间变化；下半部分是每条 trace 相对第一条 trace 的 RMS 差异。浅色线是逐次采样值，深色线是 24 点滚动平均，用来观察慢变化趋势。",
        "",
        (
            "本次数据含义: 纳入分析的 trace 平均幅值范围为 "
            f"{trace_summary['mean_amplitude'].min():.6f} 到 "
            f"{trace_summary['mean_amplitude'].max():.6f} dB，"
            "平均幅值线性趋势接近 0。RMS 差异主要反映每次频谱曲线相对起始曲线的整体偏移或形状变化；"
            "如果这条曲线持续上升，说明同一种溶液的测量结果可能随时间漂移。"
        ),
        "",
        "### 3. Spectrum delta heatmap",
        "",
        f"图表文件: `{figure_map['spectrum_delta_heatmap.png']}`",
        "",
        "读图方法: 横轴是频率，纵轴是时间，颜色表示该频点相对第一条 trace 的幅值变化。红色表示幅值增加，蓝色表示幅值降低，颜色越深变化越大。",
        "",
        "本次数据含义: 这张图用来寻找“只发生在某些频段”的时间变化。若出现随时间逐渐变深的连续色带，说明对应频段可能存在系统性漂移；若颜色呈零散斑点，则更像随机噪声或局部波动。",
        "",
        "### 4. Average spectra: first vs middle vs last",
        "",
        f"图表文件: `{figure_map['average_spectra_windows.png']}`",
        "",
        "读图方法: 将采集开始、中间、结束三个时间窗口内的 trace 分别求平均，再把三条平均频谱画在同一张图上。三条曲线越重合，说明一个小时内整体频谱越稳定。",
        "",
        "本次数据含义: 这张图适合直观看同一种溶液的频谱形状是否随时间改变。若末段曲线整体上移/下移，是幅值漂移；若只有部分峰谷位置或深度变化，则可能是频段相关变化。",
        "",
        "### 5. Frequency drift rankings",
        "",
        f"图表文件: `{figure_map['frequency_drift_rankings.png']}`",
        "",
        "读图方法: 左图按线性漂移斜率绝对值列出变化最快的频点，右图按 peak-to-peak 波动范围列出波动最大的频点。斜率表示方向性趋势，peak-to-peak 表示一小时内最大振幅跨度。",
        "",
        (
            "本次数据含义: 最大绝对线性斜率出现在 "
            f"{max_slope_row['frequency_mhz']:.6f} MHz，斜率为 "
            f"{max_slope_row['linear_slope_db_per_hour']:.6f} dB/hour；"
            "最大 peak-to-peak 波动出现在 "
            f"{max_range_row['frequency_mhz']:.6f} MHz，跨度为 "
            f"{max_range_row['peak_to_peak']:.6f} dB。"
            "这些频点是后续检查仪器噪声、背景扣除或关键频段稳定性的优先位置。"
        ),
        "",
        "### 6. Invalid amplitude points by excluded trace",
        "",
        f"图表文件: `{figure_map['corrupted_trace_points.png']}`",
        "",
        "读图方法: 横轴是被排除的 trace 序号，纵轴是该 trace 中被判定为异常的幅值点数量。柱子越高，说明该条原始 CSV 的幅值列损坏越严重。",
        "",
        (
            "本次数据含义: 这张图解释为什么有些 trace 没有纳入趋势分析。"
            f"本次共过滤到 {acquisition_summary['filtered_invalid_points']} 个异常幅值点，"
            "集中在 21 条 CSV 中；这些点的数值明显超出合理 dBm 范围，若不排除会严重污染均值、RMS 和漂移斜率。"
        ),
    ]


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0:
        return 0.0
    return float(np.dot(x_centered, y - y.mean()) / denominator)


def _markdown_table(frame: pd.DataFrame, *, float_digits: int = 3) -> str:
    """Render a small DataFrame as a Markdown table without optional dependencies."""
    if frame.empty:
        return ""

    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: f"{value:.{float_digits}f}",
                na_action="ignore",
            )
        else:
            display[column] = display[column].astype(str)

    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in display.astype(str).to_numpy().tolist()
    ]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    main()
