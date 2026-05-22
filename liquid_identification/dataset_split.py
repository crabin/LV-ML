"""Dataset splitting utilities for feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from liquid_identification.data_loader import find_data_dir
from liquid_identification.feature_extraction import FeatureSet


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for train/validation/test splitting."""

    train_size: float = 0.60
    validation_size: float = 0.20
    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True


@dataclass(frozen=True)
class DatasetSplit:
    """Split result for one feature set."""

    feature_set_name: str
    status: str
    reason: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def split_feature_set(
    feature_set: FeatureSet,
    config: SplitConfig | None = None,
) -> DatasetSplit:
    """Split one feature set into train, validation and test subsets."""
    active_config = config or SplitConfig()
    _validate_config(active_config)

    X = feature_set.X.copy()
    y = feature_set.y.reindex(X.index)

    class_counts = y.value_counts()
    if active_config.stratify and class_counts.min() < 3:
        reason = (
            "无法执行 stratified train/validation/test split: "
            f"每个类别至少需要 3 个样本，当前最少只有 {class_counts.min()} 个。"
        )
        return _analysis_only_split(feature_set.name, X, y, reason)

    relative_validation_size = active_config.validation_size / (
        active_config.train_size + active_config.validation_size
    )

    try:
        X_train_validation, X_test, y_train_validation, y_test = train_test_split(
            X,
            y,
            test_size=active_config.test_size,
            random_state=active_config.random_state,
            stratify=y if active_config.stratify else None,
        )
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=relative_validation_size,
            random_state=active_config.random_state,
            stratify=y_train_validation if active_config.stratify else None,
        )
    except ValueError as exc:
        reason = f"划分失败: {exc}"
        return _analysis_only_split(feature_set.name, X, y, reason)

    return DatasetSplit(
        feature_set_name=feature_set.name,
        status="split",
        reason="已按 60/20/20 生成 train/validation/test，并使用 stratified split。",
        X_train=X_train.sort_index(),
        y_train=y_train.sort_index(),
        X_validation=X_validation.sort_index(),
        y_validation=y_validation.sort_index(),
        X_test=X_test.sort_index(),
        y_test=y_test.sort_index(),
    )


def split_all_feature_sets(
    feature_sets: dict[str, FeatureSet],
    config: SplitConfig | None = None,
) -> dict[str, DatasetSplit]:
    """Split every feature set."""
    return {
        name: split_feature_set(feature_set, config)
        for name, feature_set in feature_sets.items()
    }


def save_dataset_splits(
    splits: dict[str, DatasetSplit],
    output_dir: str | Path | None = None,
) -> Path:
    """Save split matrices, labels and status files."""
    if output_dir is None:
        output_dir = find_data_dir() / "splits"

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    status_rows: list[dict[str, object]] = []
    for split in splits.values():
        split_dir = path / split.feature_set_name
        split_dir.mkdir(parents=True, exist_ok=True)

        _save_subset(split.X_train, split.y_train, split_dir, "train")
        _save_subset(split.X_validation, split.y_validation, split_dir, "validation")
        _save_subset(split.X_test, split.y_test, split_dir, "test")

        status_rows.append(
            {
                "feature_set": split.feature_set_name,
                "status": split.status,
                "reason": split.reason,
                "train_samples": len(split.y_train),
                "validation_samples": len(split.y_validation),
                "test_samples": len(split.y_test),
            }
        )

    pd.DataFrame(status_rows).to_csv(path / "split_status.csv", index=False)
    return path


def build_dataset_split_report(splits: dict[str, DatasetSplit]) -> str:
    """Build a printable dataset split report."""
    rows: list[dict[str, object]] = []
    for split in splits.values():
        rows.append(
            {
                "feature_set": split.feature_set_name,
                "status": split.status,
                "train": len(split.y_train),
                "validation": len(split.y_validation),
                "test": len(split.y_test),
                "reason": split.reason,
            }
        )

    summary = pd.DataFrame(rows)
    lines = [
        "数据集划分完成",
        "",
        "1. 划分策略",
        "目标策略: 训练集 60%、验证集 20%、测试集 20%，并使用 stratified split 保证每类比例一致。",
        "",
        "2. 当前结果",
        summary.to_string(index=False),
        "",
        "3. 说明",
        "当前每个液体类别只有 1 个样本，不能进行严格的分层训练/验证/测试划分。",
        "因此本轮保存 analysis_only 占位划分: 全部样本放入 train，validation/test 为空。",
        "这样可以保留统一文件结构，同时避免产生不可信的验证集和测试集。",
    ]
    return "\n".join(lines)


def _analysis_only_split(
    feature_set_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    reason: str,
) -> DatasetSplit:
    empty_X = X.iloc[0:0].copy()
    empty_y = y.iloc[0:0].copy()
    return DatasetSplit(
        feature_set_name=feature_set_name,
        status="analysis_only",
        reason=reason,
        X_train=X.sort_index(),
        y_train=y.sort_index(),
        X_validation=empty_X,
        y_validation=empty_y,
        X_test=empty_X,
        y_test=empty_y,
    )


def _save_subset(
    X: pd.DataFrame,
    y: pd.Series,
    split_dir: Path,
    subset_name: str,
) -> None:
    X.to_csv(split_dir / f"{subset_name}_X.csv")
    y.to_csv(split_dir / f"{subset_name}_y.csv", index=False)
    with_label = X.copy()
    with_label.insert(0, "liquid_type", y.to_numpy())
    with_label.to_csv(split_dir / f"{subset_name}_features.csv", index=False)


def _validate_config(config: SplitConfig) -> None:
    total = config.train_size + config.validation_size + config.test_size
    if abs(total - 1.0) > 1e-9:
        msg = f"Split sizes must sum to 1.0, got {total:.4f}"
        raise ValueError(msg)
    if min(config.train_size, config.validation_size, config.test_size) <= 0:
        msg = "Split sizes must all be positive."
        raise ValueError(msg)
