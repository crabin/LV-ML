# liquid-identification

液体频谱识别实验代码。

## 读取数据

默认数据目录位于项目外层的 `../data/`。读取模块会自动查找包含 `raw/` 的 `data` 目录，并默认排除 `data/bg/` 历史备份。

```python
from liquid_identification.data_loader import load_all_raw_data

data = load_all_raw_data()
print(data.head())
```

输出为后续预处理和建模使用的长表：

```text
sample_id, frequency_hz, frequency_mhz, amplitude_dbm, liquid_type, source_file, range
```

也可以读取某个已整理频段目录下的 `all.csv` / `All.csv`：

```python
from liquid_identification.data_loader import load_combined_range_data

data = load_combined_range_data("50-150")
```

命令行快速检查：

```bash
uv run python main.py
```

## 数据预处理

预处理模块按 `docs/实验步骤.md` 中第 4 节执行：

1. 去噪：使用局部中值残差检测异常尖峰，并要求异常点同时满足 IQR 与 Z-score 规则。
2. 平滑：使用 Savitzky-Golay Filter，尽量保留峰值和谷值形态。
3. Normalize：对每条曲线做 Z-score 标准化，减少不同实验之间的幅值偏移。

```python
from liquid_identification.data_loader import load_all_raw_data
from liquid_identification.preprocessing import (
    preprocess_spectrum_data,
    save_preprocessed_data,
)

raw_data = load_all_raw_data()
processed_data = preprocess_spectrum_data(raw_data)
save_preprocessed_data(processed_data)
```

预处理后会新增以下列：

```text
is_outlier, amplitude_denoised_dbm, amplitude_smoothed_dbm, amplitude_normalized
```

默认保存路径：

```text
data/processed/preprocessed_spectra.csv
```

## 关键频段分析

关键频段分析按 `docs/实验步骤.md` 中第 5 节执行，默认关注：

```text
180-220 MHz
270-300 MHz
```

分析内容包括：

- 不同液体在关键频段的均值差异
- 峰值位置和谷值位置
- 曲线斜率
- 频段能量
- 类间 RMS 距离
- 区分度较高的频率点
- 探索性 ANOVA

```python
from liquid_identification.key_frequency_analysis import (
    analyze_key_frequency_bands,
    save_key_frequency_analysis,
)

analysis = analyze_key_frequency_bands(processed_data)
save_key_frequency_analysis(analysis)
```

默认保存路径：

```text
data/analysis/key_frequency_*.csv
```

## 数据分析与可视化

数据分析与可视化按 `docs/实验步骤.md` 中第 6 节执行：

- 曲线可视化：原始曲线、平滑后曲线、Normalize 后曲线
- PCA：2D 分布图、3D 分布图、各主成分解释方差比例
- t-SNE / UMAP：2D 非线性降维分布图
- 相关性分析：频率相关性热力图、高相关特征对

```python
from liquid_identification.visualization_analysis import run_visualization_analysis

result = run_visualization_analysis(processed_data)
```

默认保存路径：

```text
data/analysis/visualization/
```

## 特征提取

特征提取按 `docs/实验步骤.md` 中第 7 节执行，提供三个入口：

```python
from liquid_identification.feature_extraction import (
    extract_full_spectrum_features,
    extract_key_band_features,
    extract_statistical_features,
)

full = extract_full_spectrum_features(processed_data)
key = extract_key_band_features(processed_data)
stats = extract_statistical_features(processed_data)
```

三种特征：

- 全频段特征：使用所有频率点，信息最完整。
- 关键频段特征：只使用 `180-220 MHz` 和 `270-300 MHz`。
- 统计特征：对关键频段提取 `mean/max/min/variance/std/area/energy/peak/valley/slope`。

统一保存：

```python
from liquid_identification.feature_extraction import (
    extract_all_feature_sets,
    save_feature_sets,
)

feature_sets = extract_all_feature_sets(processed_data)
save_feature_sets(feature_sets)
```

默认保存路径：

```text
data/features/
```

每种特征会输出：

```text
*_X.csv
*_y.csv
*_features.csv
```

## 数据集划分

数据集划分按 `docs/实验步骤.md` 中第 8 节执行，目标比例为：

```text
train: 60%
validation: 20%
test: 20%
```

并默认使用 `stratified split`。

```python
from liquid_identification.dataset_split import (
    split_all_feature_sets,
    save_dataset_splits,
)

splits = split_all_feature_sets(feature_sets)
save_dataset_splits(splits)
```

默认保存路径：

```text
data/splits/
```

注意：当前每个液体类别只有 1 个样本，无法进行严格的分层 train/validation/test 划分。因此代码会生成 `analysis_only` 占位划分：全部样本放入 `train`，`validation` 和 `test` 为空，并在 `split_status.csv` 中写明原因。等后续加入重复实验样本后，同一入口会自动执行真正的 stratified split。

### 注意


当前这几个降维方法使用的是同一份特征矩阵：

8 个样本 x 2253 个频率特征

也就是：

- 样本数：8
  - Air
  - Water
  - NaOH
  - HCl
  - NaCl
  - Ethanol
  - Glycerol
  - nothing
- 特征数：2253
  - 50-150 MHz: 751 个频率点
  - 150-250 MHz: 751 个频率点
  - 250-350 MHz: 751 个频率点
  - 总计 751 * 3 = 2253

所以：

│PCA 2D   │ 8 x 2253   │ 8 x 2      │
│ PCA 3D   │ 8 x 2253   │ 8 x 3      │
│ t-SNE 2D │ 8 x 2253   │ 8 x 2      │
│ UMAP 2D  │ 8 x 2253   │ 8 x 2   │

对应文件是：

data/analysis/visualization/feature_matrix.csv

现在每种液体只有一条合并后的频谱曲线，所以降维图只有 8 个点。这个更适合做初步可视化和类别间距离观察，还不适合作为稳定的模型泛化判断。


### 相关性分析


- data/analysis/visualization/frequency_correlation_heatmap.png
- data/analysis/visualization/frequency_correlation.csv
- data/analysis/visualization/high_correlation_pairs.csv
- data/analysis/visualization/selected_frequency_features.csv

这 4 个文件是一组相关性分析结果，核心用途是：看哪些频率点信息重复，哪些频率点可以作为代表特征留下来。

1. frequency_correlation_heatmap.png
   这是频率点之间的相关性热力图。

横轴和纵轴都是频率特征，颜色表示两个频率点在 8 类液体上的变化是否同步：

- 接近 1：两个频率点变化方向几乎一致，信息高度重复。
- 接近 -1：两个频率点变化方向相反，也说明存在强关系。
- 接近 0：两个频率点关系弱，可能提供不同信息。

能获取的信息：

- 哪些频段内部高度相关。
- 哪些频段之间存在强相关或强负相关。
- 是否存在大块冗余区域。

它适合做整体观察，不适合直接拿来选特征。

2. frequency_correlation.csv
   这是热力图对应的数值矩阵。

它保存的是被抽样后的频率点相关性，不是完整 2253 x 2253 矩阵。当前为了可读性，每个频段均匀抽样，总共约 180 个频率点。

用法：

import pandas as pd

corr = pd.read_csv("data/analysis/visualization/frequency_correlation.csv", index_col=0)

能获取的信息：

- 精确查看两个频率点之间的相关系数。
- 复现热力图。
- 找某个频率点和其他频率点的相关关系。

3. high_correlation_pairs.csv
   这是高相关频率点对列表。

它回答的问题是：

哪些频率点之间高度重复？

主要字段含义：

- feature_a, feature_b：两个频率特征。
- range_a, range_b：所属频段。
- frequency_a_mhz, frequency_b_mhz：频率值。
- distance_mhz：两个频率点间隔。
- correlation：相关系数。
- abs_correlation：相关系数绝对值。

当前规则：

- 只保留 |corr| >= 0.98 的强相关关系。
- 过滤掉同频段内距离小于 5 MHz 的相邻点关系。
- 只保存最强的前 2000 对。

能获取的信息：

- 哪些远距离频率点仍然高度相关。
- 哪些特征可能是冗余的。
- 哪些频段之间可能存在同步或反向变化。

例如 correlation = -1.0 表示两个频点在不同液体之间的变化几乎完全相反，这也是强信息关系。

4. selected_frequency_features.csv
   这是最重要的特征筛选结果。

它回答的问题是：

在去掉高度冗余后，哪些频率点应该保留？

当前规则：

- 从所有 2253 个频率特征出发。
- 优先选择跨液体差异大的频率点。
- 如果新频点与已选频点的相关性 |corr| >= 0.98，认为它冗余，不再保留。
- 最终筛到 110 个代表频率点。

主要字段：

- selection_priority：筛选优先级，越小越优先被选中。
- feature_name：特征名。
- range：所属频段。
- frequency_mhz：频率值。
- feature_std：该频率点在不同液体之间的标准差，越大通常区分度越强。

后续建模时，优先用这个文件里的频率点作为特征集合。它比直接使用全部 2253 个频点更稳，因为减少了大量冗余特征。
