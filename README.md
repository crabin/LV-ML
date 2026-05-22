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
