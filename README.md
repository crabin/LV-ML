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
