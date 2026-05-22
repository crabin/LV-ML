# data 数据说明

本目录保存液体识别实验的频谱数据和已有分析结果。后续读取数据、预处理、特征提取和建模代码应优先使用这里约定的数据结构。

## 目录结构

```text
data/
├── raw/
│   ├── 50-150/
│   ├── 150-250/
│   └── 350/
├── 50-150/
├── 150-250/
├── 350/
├── bg/
└── title.xlsx
```

## 数据分组

当前数据按频率范围分为三组：

| 目录 | 频率范围 | 说明 |
| --- | --- | --- |
| `raw/50-150/` | 50-150 MHz | 原始仪器导出的 CSV |
| `raw/150-250/` | 150-250 MHz | 原始仪器导出的 CSV |
| `raw/350/` | 250-350 MHz | 原始仪器导出的 CSV；目录名沿用已有命名 |
| `50-150/` | 50-150 MHz | 已整理的 CSV、Excel 和图表结果 |
| `150-250/` | 150-250 MHz | 已整理的 CSV、Excel 和图表结果 |
| `350/` | 250-350 MHz | 已整理的 CSV、Excel、PDF、PNG 和压缩结果 |

`bg/` 是历史备份目录，包含旧版本数据、压缩包和中间结果。默认读取和建模流程不应使用 `bg/` 下的数据，除非需要回溯历史结果。

## 样本与标签

每个频率范围下都有 `CSV0.csv` 到 `CSV7.csv`，对应 8 条曲线：

| 文件 | 标签 | 含义 |
| --- | --- | --- |
| `CSV0.csv` | `Air` | 空气背景 |
| `CSV1.csv` | `Water` | 水 |
| `CSV2.csv` | `NaOH` | 氢氧化钠溶液 |
| `CSV3.csv` | `HCl` | 盐酸溶液 |
| `CSV4.csv` | `NaCl` | 氯化钠溶液 |
| `CSV5.csv` | `Ethanol` | 乙醇 |
| `CSV6.csv` | `Glycerol` | 甘油 |
| `CSV7.csv` | `nothing` | 空载或无样品状态 |

其中 `Air` 和 `nothing` 更适合作为背景或对照数据；模型训练时是否作为类别标签，需要在实验方案中明确。

## 原始 CSV 格式

`raw/<range>/CSV*.csv` 是频谱仪导出的原始文件，文件开头包含仪器元数据，例如：

```text
Machine Module,T3SA3200
Y Axis Scale,LOG
Y Axis Unit,dBm
...
Trace Data
50000000.000000,-68.25,,,
```

读取规则：

1. 找到 `Trace Data` 行。
2. 从下一行开始读取数值数据。
3. 第 1 列是频率，单位为 Hz。
4. 第 2 列是幅值，单位为 dBm。
5. 后续空列来自仪器导出格式，读取时应忽略。

整理成建模数据时，建议转换为长表：

| sample_id | frequency_hz | frequency_mhz | amplitude_dbm | liquid_type | source_file | range |
| --- | --- | --- | --- | --- | --- | --- |
| `50-150_CSV0` | `50000000` | `50.0` | `-68.25` | `Air` | `CSV0.csv` | `50-150` |

## 已整理结果文件

各频率范围目录下的 `All.csv` / `all.csv` 和 `All.xlsx` / `all.xlsx` 是已经合并后的宽表数据。典型列如下：

```text
Frequency(Hz), Air, Water, NaOH, HCl, NaCl, Ethanol, Glycerol, nothing
```

读取规则：

1. `Frequency(Hz)` 是频率列，单位为 Hz。
2. 其他列是不同液体或对照状态的幅值，单位为 dBm。
3. 用于机器学习时，建议将宽表转换为长表，每个液体列展开为一组样本。

## 差分分析结果

目录中还包含相对背景的差分表和可视化结果：

| 文件名模式 | 含义 |
| --- | --- |
| `*_minus_Air_frequency_MHz_linear.xlsx` | 各液体幅值减去 `Air` 背景后的结果 |
| `*_minus_Water_frequency_MHz_linear.xlsx` | 各液体幅值减去 `Water` 后的结果 |
| `*_frequency_amplitude_MHz_linear*.xlsx/png/pdf` | 频率-幅值曲线及对应图表 |
| `*_Air_nothing_only*.xlsx/png/pdf` | 仅包含 `Air` 和 `nothing` 的对照结果 |
| `*_curve_color_table.xlsx` | 曲线颜色和线型配置 |

这些文件属于分析结果，不建议作为最原始的训练输入。需要做背景扣除或可视化复现实验时，可以作为参考或中间数据读取。

## 后续读取代码建议

读取代码建议放在项目的数据处理模块中，并遵循以下优先级：

1. 默认读取 `data/raw/` 下的原始 CSV，保证流程可追溯。
2. 提供读取 `data/<range>/all.csv` 的快捷入口，用于快速分析。
3. 默认排除 `data/bg/`。
4. 明确区分 `Air`、`nothing` 与真实液体类别。
5. 统一输出长表格式，便于后续预处理、特征提取和分类建模。
