# LV-ML

液体频谱识别实验项目。项目包含原始频谱数据、实验文档、数据分析结果，以及基于 Python 的数据读取、预处理、关键频段分析、可视化、特征提取和数据集划分代码。

## 项目结构

```text
LV-ML/
├── data/
│   ├── raw/                 # 原始仪器 CSV 数据
│   ├── 50-150/              # 50-150 MHz 已整理数据和分析结果
│   ├── 150-250/             # 150-250 MHz 已整理数据和分析结果
│   ├── 350/                 # 250-350 MHz 已整理数据和分析结果
│   ├── analysis/            # 关键频段、PCA、t-SNE、UMAP、相关性等分析输出
│   ├── features/            # 特征提取结果
│   ├── processed/           # 预处理后的长表数据
│   ├── splits/              # 训练/验证/测试划分结果
│   └── README.md            # 数据目录说明
├── docs/
│   ├── 需求分析.md
│   ├── 实验步骤.md
│   └── 数据采集方案.md
└── liquid-identification/
    ├── liquid_identification/
    ├── main.py
    ├── pyproject.toml
    └── README.md
```

## 当前数据状态

当前有效建模样本为 8 类各 1 条完整频谱：

```text
Air, Water, NaOH, HCl, NaCl, Ethanol, Glycerol, nothing
```

当前数据适合用于流程验证、预处理、可视化和探索性分析，不适合训练可靠模型。后续采集方案见：

```text
docs/数据采集方案.md
```

## 实验流程

当前代码已覆盖以下流程：

1. 读取 `data/raw/` 原始频谱数据。
2. 数据预处理：去噪、Savitzky-Golay 平滑、Z-score 标准化。
3. 关键频段分析：重点分析 `180-220 MHz` 和 `270-300 MHz`。
4. 数据分析与可视化：原始/平滑/标准化曲线、PCA、t-SNE、UMAP、相关性分析。
5. 特征提取：全频段特征、关键频段特征、统计特征。
6. 数据集划分：支持 stratified split；当前样本不足时生成 `analysis_only` 占位划分。

## 运行环境

Python 子项目位于 `liquid-identification/`，使用 `uv` 管理依赖。

```bash
cd liquid-identification
uv sync
```

## 运行完整流程

```bash
cd liquid-identification
uv run python main.py
```

运行后会生成或刷新以下结果：

```text
data/processed/preprocessed_spectra.csv
data/analysis/key_frequency_*.csv
data/analysis/visualization/
data/features/
data/splits/
```

## 主要代码入口

```text
liquid_identification/data_loader.py              # 数据读取
liquid_identification/preprocessing.py            # 数据预处理
liquid_identification/key_frequency_analysis.py   # 关键频段分析
liquid_identification/visualization_analysis.py   # 可视化和降维分析
liquid_identification/feature_extraction.py       # 特征提取
liquid_identification/dataset_split.py            # 数据集划分
```

更详细的代码用法见：

```text
liquid-identification/README.md
```

## 输出说明

`data/analysis/visualization/` 中包含：

- 频谱曲线图
- PCA 2D / 3D 图
- t-SNE 2D 图
- UMAP 2D 图
- 频率相关性热力图
- 高相关特征对
- 去冗余后的代表频点

`data/features/` 中包含三类特征：

- `full_spectrum_*`: 全频段特征
- `key_bands_*`: 关键频段特征
- `statistical_*`: 统计特征

`data/splits/` 中包含每类特征的训练/验证/测试文件。由于当前每类只有 1 个样本，现阶段划分状态为 `analysis_only`。

## 数据采集建议

可靠模型训练建议至少采集：

```text
6 个真实液体类别 × 每类 30 条独立样本 = 180 条有效样本
```

每条有效样本建议包含三段频率：

```text
50-150 MHz
150-250 MHz
250-350 MHz
```

详细采集数量、采集流程、保存结构和元数据字段见：

```text
docs/数据采集方案.md
```
