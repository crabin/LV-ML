# Single-Liquid Time Stability Analysis

这个目录用于存放单一溶液一小时连续采集数据的读取脚本、临时输出和分析文档。

默认分析 `data/water`：

```bash
cd liquid-identification
uv run python liquid_identification/water_time_stability/analyze_water_time_stability.py
```

分析其他溶液时，目录结构需要保持一致：

```text
data/<liquid>/
  raw_trace_index.csv
  raw_traces/
```

例如 HCl：

```bash
uv run python liquid_identification/water_time_stability/analyze_water_time_stability.py --liquid hcl
```

如果数据不在默认目录，可以显式指定：

```bash
uv run python liquid_identification/water_time_stability/analyze_water_time_stability.py \
  --liquid hcl \
  --input-dir ../data/hcl \
  --output-dir liquid_identification/water_time_stability/tmp/hcl
```

脚本输出写入本目录下的 `tmp/`：

- `<liquid>_time_stability_report.md`
- `trace_summary.csv`
- `frequency_drift_summary.csv`
- `missing_or_failed_records.csv`
- `trace_quality_adjustments.csv`
- `raw_trace_preview.csv`

可视化图表写入 `tmp/figures/`：

- `data_quality_timeline.png`: 采集质量时间轴，显示纳入分析、异常排除和解析失败记录。
- `stability_time_series.png`: 每条 trace 的平均幅值和相对首条 trace 的 RMS 差异随时间变化。
- `spectrum_delta_heatmap.png`: 每个频点相对第一条 trace 的幅值变化热图。
- `average_spectra_windows.png`: 开始、中间、结束三个时间窗口的平均频谱对比。
- `frequency_drift_rankings.png`: 线性漂移斜率和 peak-to-peak 波动最大的频点排行。
- `corrupted_trace_points.png`: 被排除 trace 中的异常幅值点数量。
