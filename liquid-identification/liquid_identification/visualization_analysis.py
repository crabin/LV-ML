"""Data analysis and visualization for preprocessed spectrum data."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from liquid_identification.data_loader import find_data_dir

matplotlib.use("Agg")

CURVE_COLUMNS: dict[str, str] = {
    "raw": "amplitude_dbm",
    "smoothed": "amplitude_smoothed_dbm",
    "normalized": "amplitude_normalized",
}
CORRELATION_THRESHOLD = 0.98
CORRELATION_HEATMAP_MAX_FEATURES = 180
MIN_PAIR_DISTANCE_MHZ = 5.0
HIGH_CORRELATION_PAIR_LIMIT = 2000


def run_visualization_analysis(
    processed_data: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Create spectrum plots, PCA/t-SNE/UMAP projections and correlation analysis."""
    if output_dir is None:
        output_dir = find_data_dir() / "analysis" / "visualization"

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, Path] = {}
    for name, column in CURVE_COLUMNS.items():
        figure_paths[f"spectrum_{name}"] = _plot_spectrum_curves(
            processed_data,
            column,
            path / f"spectrum_{name}_curves.png",
        )

    feature_matrix, feature_info = build_liquid_feature_matrix(processed_data)
    pca_result = _run_pca(feature_matrix)
    pca_scores = pca_result["scores"]
    explained_variance = pca_result["explained_variance"]

    figure_paths["pca_2d"] = _plot_pca_2d(pca_scores, path / "pca_2d.png")
    figure_paths["pca_3d"] = _plot_pca_3d(pca_scores, path / "pca_3d.png")

    tsne_scores = _run_tsne(feature_matrix)
    figure_paths["tsne_2d"] = _plot_projection_2d(
        tsne_scores,
        "tsne_1",
        "tsne_2",
        "t-SNE 2D projection",
        path / "tsne_2d.png",
    )

    umap_scores = _run_umap(feature_matrix)
    figure_paths["umap_2d"] = _plot_projection_2d(
        umap_scores,
        "umap_1",
        "umap_2",
        "UMAP 2D projection",
        path / "umap_2d.png",
    )

    correlation_features = _select_correlation_heatmap_features(feature_info)
    correlation = feature_matrix.loc[:, correlation_features].corr()
    full_correlation = feature_matrix.corr()
    high_correlation_pairs = _find_high_correlation_pairs(
        full_correlation,
        feature_info,
        threshold=CORRELATION_THRESHOLD,
    )
    selected_features = _select_representative_features(
        feature_matrix,
        feature_info,
        threshold=CORRELATION_THRESHOLD,
    )
    figure_paths["frequency_correlation"] = _plot_correlation_heatmap(
        correlation,
        feature_info,
        path / "frequency_correlation_heatmap.png",
    )

    tables: dict[str, pd.DataFrame] = {
        "feature_matrix": feature_matrix,
        "feature_info": feature_info,
        "pca_scores": pca_scores,
        "pca_explained_variance": explained_variance,
        "tsne_scores": tsne_scores,
        "umap_scores": umap_scores,
        "frequency_correlation": correlation,
        "selected_frequency_features": selected_features,
        "high_correlation_pairs": high_correlation_pairs,
    }
    for name, frame in tables.items():
        frame.to_csv(path / f"{name}.csv")

    return {
        "output_dir": path,
        "figures": figure_paths,
        "tables": tables,
    }


def build_visualization_report(result: dict[str, object]) -> str:
    """Build a concise printable report for visualization analysis."""
    tables = result["tables"]
    if not isinstance(tables, dict):
        msg = "Invalid visualization result: missing tables."
        raise TypeError(msg)

    feature_matrix = tables["feature_matrix"]
    explained_variance = tables["pca_explained_variance"]
    pca_scores = tables["pca_scores"]
    high_correlation_pairs = tables["high_correlation_pairs"]
    selected_features = tables["selected_frequency_features"]
    top_selected_features = selected_features.sort_values(
        "feature_std",
        ascending=False,
    ).head(20)

    top_pca = pca_scores.sort_values("pc1").loc[
        :,
        ["liquid_type", "pc1", "pc2", "pc3"],
    ]

    lines = [
        "数据分析与可视化完成",
        "",
        "1. 曲线可视化",
        "输出: 原始曲线、平滑后曲线、Normalize 后曲线。",
        "理由: 对比三种曲线可以检查去噪和平滑是否改变主要峰谷形态，也能直接观察不同液体的整体差异。",
        "",
        "2. PCA 分析",
        f"特征矩阵: {feature_matrix.shape[0]} 个液体样本 x {feature_matrix.shape[1]} 个频率特征。",
        "理由: PCA 用于观察高维频谱在低维空间中是否可分，并辅助发现离群样本。",
        explained_variance.to_string(
            index=False,
            formatters={
                "explained_variance_ratio": "{:.4f}".format,
                "cumulative_ratio": "{:.4f}".format,
            },
        ),
        "",
        "PCA 坐标摘要",
        top_pca.to_string(
            index=False,
            formatters={
                "pc1": "{:.3f}".format,
                "pc2": "{:.3f}".format,
                "pc3": "{:.3f}".format,
            },
        ),
        "",
        "3. t-SNE / UMAP 分析",
        "输出: t-SNE 2D 分布图。",
        "理由: 当 PCA 线性降维分离不明显时，t-SNE 可用于探索非线性可分性。当前样本数较少，结果只作为可视化参考。",
        "输出: UMAP 2D 分布图。",
        "理由: UMAP 同样用于探索非线性流形结构，通常比 t-SNE 更利于保留局部邻域关系；当前样本数较少，结果只作为可视化参考。",
        "",
        "4. 相关性分析",
        "输出: 降采样后的全频段相关性热力图、高相关特征对、代表频点筛选表。",
        "理由: 原始 2253 个频率点直接画热力图不可读，相邻频点高相关也会淹没有用信息；先按频段均匀抽样绘制热力图，再用相关阈值筛出代表频点，更适合后续特征提取。",
        f"代表频点筛选: 使用 |corr| >= {CORRELATION_THRESHOLD:.2f} 作为冗余阈值，从 {feature_matrix.shape[1]} 个频率特征筛为 {len(selected_features)} 个代表特征。",
        f"高相关特征对: 仅保存绝对相关性最高的前 {HIGH_CORRELATION_PAIR_LIMIT} 对，避免输出被大量重复关系淹没。",
        "代表频点 Top 20，按跨液体标准差排序",
        top_selected_features.to_string(
            index=False,
            formatters={
                "frequency_mhz": "{:.3f}".format,
                "feature_std": "{:.4f}".format,
            },
        ),
        "",
        f"非相邻高相关特征对 Top 10，已过滤小于 {MIN_PAIR_DISTANCE_MHZ:g} MHz 的相邻关系",
        high_correlation_pairs.head(10).to_string(
            index=False,
            formatters={
                "correlation": "{:.4f}".format,
                "frequency_a_mhz": "{:.3f}".format,
                "frequency_b_mhz": "{:.3f}".format,
                "distance_mhz": "{:.3f}".format,
            },
        ),
        "",
        f"结果目录: {result['output_dir']}",
    ]
    return "\n".join(lines)


def build_liquid_feature_matrix(
    processed_data: pd.DataFrame,
    *,
    value_column: str = "amplitude_normalized",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one full-spectrum feature vector per liquid label."""
    required_columns = {"range", "frequency_mhz", "liquid_type", value_column}
    missing_columns = required_columns.difference(processed_data.columns)
    if missing_columns:
        msg = f"Missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    data = processed_data.copy()
    data["feature_name"] = data.apply(
        lambda row: f"{row['range']}__{row['frequency_mhz']:.6f}MHz",
        axis=1,
    )
    feature_matrix = data.pivot_table(
        index="liquid_type",
        columns="feature_name",
        values=value_column,
        aggfunc="mean",
    ).sort_index(axis=1)

    feature_info = (
        data[["feature_name", "range", "frequency_mhz"]]
        .drop_duplicates()
        .sort_values(["range", "frequency_mhz"])
        .reset_index(drop=True)
    )
    return feature_matrix, feature_info


def _plot_spectrum_curves(
    data: pd.DataFrame,
    value_column: str,
    output_path: Path,
) -> Path:
    sns.set_theme(style="whitegrid")
    ranges = list(dict.fromkeys(data["range"].tolist()))
    fig, axes = plt.subplots(
        len(ranges),
        1,
        figsize=(12, 3.6 * len(ranges)),
        sharey=False,
        constrained_layout=True,
    )
    if len(ranges) == 1:
        axes = [axes]

    for ax, range_name in zip(axes, ranges, strict=True):
        range_data = data[data["range"] == range_name]
        sns.lineplot(
            data=range_data,
            x="frequency_mhz",
            y=value_column,
            hue="liquid_type",
            linewidth=1.4,
            ax=ax,
        )
        ax.set_title(f"{range_name} spectrum: {value_column}")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(value_column)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Liquid")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _run_pca(feature_matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scaled = StandardScaler().fit_transform(feature_matrix)
    n_components = min(3, feature_matrix.shape[0], feature_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coordinates = pca.fit_transform(scaled)

    score_columns = [f"pc{index + 1}" for index in range(n_components)]
    scores = pd.DataFrame(coordinates, index=feature_matrix.index, columns=score_columns)
    for column in ("pc1", "pc2", "pc3"):
        if column not in scores:
            scores[column] = 0.0
    scores = scores.reset_index()

    ratios = pca.explained_variance_ratio_
    explained_variance = pd.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(n_components)],
            "explained_variance_ratio": ratios,
            "cumulative_ratio": np.cumsum(ratios),
        }
    )
    return {"scores": scores, "explained_variance": explained_variance}


def _run_tsne(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    scaled = StandardScaler().fit_transform(feature_matrix)
    perplexity = min(3, feature_matrix.shape[0] - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    coordinates = tsne.fit_transform(scaled)
    return pd.DataFrame(
        coordinates,
        index=feature_matrix.index,
        columns=["tsne_1", "tsne_2"],
    ).reset_index()


def _run_umap(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    scaled = StandardScaler().fit_transform(feature_matrix)
    n_neighbors = min(5, feature_matrix.shape[0] - 1)
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    coordinates = reducer.fit_transform(scaled)
    return pd.DataFrame(
        coordinates,
        index=feature_matrix.index,
        columns=["umap_1", "umap_2"],
    ).reset_index()


def _plot_pca_2d(scores: pd.DataFrame, output_path: Path) -> Path:
    return _plot_projection_2d(scores, "pc1", "pc2", "PCA 2D projection", output_path)


def _plot_pca_3d(scores: pd.DataFrame, output_path: Path) -> Path:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for _, row in scores.iterrows():
        ax.scatter(row["pc1"], row["pc2"], row["pc3"], s=70)
        ax.text(row["pc1"], row["pc2"], row["pc3"], row["liquid_type"], fontsize=8)
    ax.set_title("PCA 3D projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_projection_2d(
    scores: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sns.scatterplot(
        data=scores,
        x=x_column,
        y=y_column,
        hue="liquid_type",
        s=90,
        ax=ax,
    )
    for _, row in scores.iterrows():
        ax.text(row[x_column], row[y_column], row["liquid_type"], fontsize=8)
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Liquid")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _select_correlation_heatmap_features(feature_info: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for _, group in feature_info.groupby("range", sort=False):
        ordered = group.sort_values("frequency_mhz")
        keep_count = min(len(ordered), CORRELATION_HEATMAP_MAX_FEATURES // 3)
        positions = np.linspace(0, len(ordered) - 1, keep_count).round().astype(int)
        features.extend(ordered.iloc[positions]["feature_name"].tolist())
    return features


def _find_high_correlation_pairs(
    correlation: pd.DataFrame,
    feature_info: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    feature_lookup = feature_info.set_index("feature_name")
    rows: list[dict[str, float | str]] = []
    columns = list(correlation.columns)
    for left_index, feature_a in enumerate(columns):
        for feature_b in columns[left_index + 1 :]:
            corr = correlation.loc[feature_a, feature_b]
            if abs(corr) < threshold:
                continue

            range_a = feature_lookup.loc[feature_a, "range"]
            range_b = feature_lookup.loc[feature_b, "range"]
            frequency_a = float(feature_lookup.loc[feature_a, "frequency_mhz"])
            frequency_b = float(feature_lookup.loc[feature_b, "frequency_mhz"])
            distance_mhz = abs(frequency_a - frequency_b)
            if range_a == range_b and distance_mhz < MIN_PAIR_DISTANCE_MHZ:
                continue

            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "range_a": range_a,
                    "range_b": range_b,
                    "frequency_a_mhz": frequency_a,
                    "frequency_b_mhz": frequency_b,
                    "distance_mhz": distance_mhz,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "range_a",
                "range_b",
                "frequency_a_mhz",
                "frequency_b_mhz",
                "distance_mhz",
                "correlation",
                "abs_correlation",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_correlation", ascending=False)
        .head(HIGH_CORRELATION_PAIR_LIMIT)
        .reset_index(drop=True)
    )


def _select_representative_features(
    feature_matrix: pd.DataFrame,
    feature_info: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    feature_std = feature_matrix.std(axis=0)
    candidates = feature_info.copy()
    candidates["feature_std"] = candidates["feature_name"].map(feature_std)
    candidates = candidates.sort_values(
        ["feature_std", "range", "frequency_mhz"],
        ascending=[False, True, True],
    )

    selected_rows: list[pd.Series] = []
    selected_features: list[str] = []
    for _, row in candidates.iterrows():
        feature_name = row["feature_name"]
        if not selected_features:
            row = row.copy()
            row["selection_priority"] = len(selected_features) + 1
            selected_rows.append(row)
            selected_features.append(feature_name)
            continue

        correlations = feature_matrix[selected_features].corrwith(
            feature_matrix[feature_name]
        )
        if correlations.abs().max() < threshold:
            row = row.copy()
            row["selection_priority"] = len(selected_features) + 1
            selected_rows.append(row)
            selected_features.append(feature_name)

    selected = pd.DataFrame(selected_rows).sort_values(["range", "frequency_mhz"])
    return selected[
        [
            "selection_priority",
            "feature_name",
            "range",
            "frequency_mhz",
            "feature_std",
        ]
    ].reset_index(drop=True)


def _plot_correlation_heatmap(
    correlation: pd.DataFrame,
    feature_info: pd.DataFrame,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    selected_info = feature_info.set_index("feature_name").loc[correlation.index]
    tick_positions: list[int] = []
    tick_labels: list[str] = []
    for range_name, group in selected_info.reset_index().groupby("range", sort=False):
        positions = [selected_info.index.get_loc(feature) for feature in group["feature_name"]]
        tick_positions.append(int(np.mean(positions)))
        min_mhz = group["frequency_mhz"].min()
        max_mhz = group["frequency_mhz"].max()
        tick_labels.append(f"{range_name}\n{min_mhz:.0f}-{max_mhz:.0f}MHz")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, rotation=0)
    ax.set_title("Frequency feature correlation heatmap, sampled across full spectrum")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
