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

    correlation = _compute_frequency_correlation(feature_matrix)
    high_correlation_pairs = _find_high_correlation_pairs(correlation)
    figure_paths["frequency_correlation"] = _plot_correlation_heatmap(
        correlation,
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
        "输出: 频率相关性热力图和高相关特征对列表。",
        "理由: 识别高度相关的频率点，后续特征提取时可以减少冗余特征。",
        "高相关特征对 Top 10",
        high_correlation_pairs.head(10).to_string(
            index=False,
            formatters={"correlation": "{:.4f}".format},
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


def _compute_frequency_correlation(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    top_features = feature_matrix.std(axis=0).sort_values(ascending=False).head(80).index
    return feature_matrix.loc[:, top_features].corr()


def _find_high_correlation_pairs(correlation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    columns = list(correlation.columns)
    for left_index, feature_a in enumerate(columns):
        for feature_b in columns[left_index + 1 :]:
            corr = correlation.loc[feature_a, feature_b]
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_correlation", ascending=False)
        .reset_index(drop=True)
    )


def _plot_correlation_heatmap(correlation: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title("Frequency feature correlation heatmap")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
