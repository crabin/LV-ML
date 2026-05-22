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
