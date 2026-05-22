"""Data loading helpers for frequency spectrum experiments."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

LIQUID_LABELS: dict[str, str] = {
    "CSV0.csv": "Air",
    "CSV1.csv": "Water",
    "CSV2.csv": "NaOH",
    "CSV3.csv": "HCl",
    "CSV4.csv": "NaCl",
    "CSV5.csv": "Ethanol",
    "CSV6.csv": "Glycerol",
    "CSV7.csv": "nothing",
}

DEFAULT_RANGES: tuple[str, ...] = ("50-150", "150-250", "350")
TRACE_DATA_MARKER = "Trace Data"


def find_data_dir(start: Path | None = None) -> Path:
    """Find the project data directory from the current file or working tree."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start)

    here = Path(__file__).resolve()
    candidates.extend([Path.cwd(), here.parent, *here.parents])

    for candidate in candidates:
        candidate = candidate.resolve()
        possible_dirs = [
            candidate / "data",
            candidate.parent / "data",
        ]
        for possible_dir in possible_dirs:
            if (possible_dir / "raw").is_dir():
                return possible_dir

    msg = "Could not find data directory containing a raw/ subdirectory."
    raise FileNotFoundError(msg)


def load_raw_trace(
    csv_path: str | Path,
    *,
    liquid_type: str | None = None,
    frequency_range: str | None = None,
) -> pd.DataFrame:
    """Read one instrument-exported raw CSV file into a normalized long table."""
    path = Path(csv_path)
    trace_text = _extract_trace_text(path)

    frame = pd.read_csv(
        StringIO(trace_text),
        header=None,
        usecols=[0, 1],
        names=["frequency_hz", "amplitude_dbm"],
    )
    frame = frame.dropna(subset=["frequency_hz", "amplitude_dbm"]).copy()
    frame["frequency_hz"] = pd.to_numeric(frame["frequency_hz"], errors="coerce")
    frame["amplitude_dbm"] = pd.to_numeric(frame["amplitude_dbm"], errors="coerce")
    frame = frame.dropna(subset=["frequency_hz", "amplitude_dbm"])

    label = liquid_type or LIQUID_LABELS.get(path.name)
    if label is None:
        msg = f"Cannot infer liquid label from file name: {path.name}"
        raise ValueError(msg)

    range_name = frequency_range or path.parent.name
    sample_id = f"{range_name}_{path.stem}"

    frame.insert(0, "sample_id", sample_id)
    frame["frequency_mhz"] = frame["frequency_hz"] / 1_000_000
    frame["liquid_type"] = label
    frame["source_file"] = path.name
    frame["range"] = range_name

    return frame[
        [
            "sample_id",
            "frequency_hz",
            "frequency_mhz",
            "amplitude_dbm",
            "liquid_type",
            "source_file",
            "range",
        ]
    ].reset_index(drop=True)


def load_all_raw_data(
    data_dir: str | Path | None = None,
    *,
    ranges: tuple[str, ...] = DEFAULT_RANGES,
) -> pd.DataFrame:
    """Read all raw CSV traces under data/raw into one long table."""
    root = Path(data_dir) if data_dir is not None else find_data_dir()
    frames: list[pd.DataFrame] = []

    for range_name in ranges:
        range_dir = root / "raw" / range_name
        if not range_dir.is_dir():
            msg = f"Missing raw data directory: {range_dir}"
            raise FileNotFoundError(msg)

        for filename, label in LIQUID_LABELS.items():
            csv_path = range_dir / filename
            if not csv_path.is_file():
                msg = f"Missing raw CSV file: {csv_path}"
                raise FileNotFoundError(msg)
            frames.append(
                load_raw_trace(
                    csv_path,
                    liquid_type=label,
                    frequency_range=range_name,
                )
            )

    return pd.concat(frames, ignore_index=True)


def load_combined_range_data(
    frequency_range: str,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Read an existing all.csv/All.csv wide table and return the same long shape."""
    root = Path(data_dir) if data_dir is not None else find_data_dir()
    range_dir = root / frequency_range
    csv_path = _find_combined_csv(range_dir)

    wide = pd.read_csv(csv_path, encoding="utf-8-sig")
    frequency_column = _find_frequency_column(wide)
    value_columns = [column for column in wide.columns if column != frequency_column]

    long = wide.melt(
        id_vars=[frequency_column],
        value_vars=value_columns,
        var_name="liquid_type",
        value_name="amplitude_dbm",
    )
    long = long.rename(columns={frequency_column: "frequency_hz"})
    long["frequency_hz"] = pd.to_numeric(long["frequency_hz"], errors="coerce")
    long["amplitude_dbm"] = pd.to_numeric(long["amplitude_dbm"], errors="coerce")
    long = long.dropna(subset=["frequency_hz", "amplitude_dbm"]).copy()
    long["frequency_mhz"] = long["frequency_hz"] / 1_000_000
    long["range"] = frequency_range
    long["source_file"] = csv_path.name
    long["sample_id"] = long["liquid_type"].map(
        lambda label: f"{frequency_range}_{_label_to_csv_stem(label)}"
    )

    return long[
        [
            "sample_id",
            "frequency_hz",
            "frequency_mhz",
            "amplitude_dbm",
            "liquid_type",
            "source_file",
            "range",
        ]
    ].reset_index(drop=True)


def _extract_trace_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == TRACE_DATA_MARKER:
            return "\n".join(lines[index + 1 :])

    msg = f"Cannot find '{TRACE_DATA_MARKER}' in raw CSV: {path}"
    raise ValueError(msg)


def _find_combined_csv(range_dir: Path) -> Path:
    for filename in ("all.csv", "All.csv"):
        csv_path = range_dir / filename
        if csv_path.is_file():
            return csv_path

    msg = f"Cannot find all.csv or All.csv in: {range_dir}"
    raise FileNotFoundError(msg)


def _find_frequency_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        normalized = column.strip().lower()
        if normalized in {"frequency(hz)", "frequency_hz", "frequency hz"}:
            return column

    msg = f"Cannot find frequency column in columns: {list(frame.columns)}"
    raise ValueError(msg)


def _label_to_csv_stem(label: str) -> str:
    for filename, known_label in LIQUID_LABELS.items():
        if known_label == label:
            return Path(filename).stem
    return label


if __name__ == "__main__":
    data = load_all_raw_data()
    print(data.head())