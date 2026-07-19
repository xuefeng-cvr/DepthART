# NVIDIA RTX A6000 Results

该目录仅保留正式 A6000 实验结果：

```text
a6000/
├── summary.csv                       # 36 行速度结果
├── speed_results/                    # 36 份原始速度 JSON
└── depth_eval/
    ├── depth_metrics.csv             # 18 行基础精度
    ├── backend_depth_metrics.csv     # 54 行后端精度
    ├── metric_results/               # 6 份 Metric 基础结果
    └── backend_results/              # 54 份后端精度结果
```

速度协议为 batch 1、200 次预热、1000 张图片。Relative 精度使用每张图 affine scale+shift 对齐；Metric 精度不进行对齐，NYUD 使用 indoor checkpoint，KITTI 使用 outdoor checkpoint。

平台相关 engine、timing cache、插件二进制和过程日志未包含在发布包中。
