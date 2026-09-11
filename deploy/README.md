# Deployment

部署目录将平台无关实现和平台入口分离：

```text
deploy/
├── shared/
│   ├── onnx/                 # 12 个 SelectiveScan 图 + 6 个 MobileNetV4 标准 ONNX
│   ├── selective_scan/       # PyTorch CUDA op 与 TensorRT plugin 源码
│   ├── tests/                # 整网部署集成测试
│   ├── build_trt_engine.py   # 整网 TensorRT engine 构建器
│   ├── benchmark.py          # 单模型测速
│   ├── run_speed_matrix.py   # 完整 36 组合测速
│   └── install.sh            # 通用编译入口
├── pc/
│   ├── setup.sh
│   ├── benchmark.sh
│   └── results/a6000/        # 正式 A6000 参考结果
└── orin_nx/
    ├── setup.sh
    └── benchmark.sh
```

从 [PC 部署](pc/README.md) 或 [Orin NX 部署](orin_nx/README.md) 开始。一般用户不需要直接调用 `shared/` 内部脚本。

算子单元测试位于 `shared/selective_scan/tests/`，依赖完整模型和 checkpoint
的集成测试位于 `shared/tests/`：

```bash
PYTHONPATH=deploy/shared/selective_scan \
  python -m pytest -q deploy/shared/selective_scan/tests deploy/shared/tests
```

TensorRT engine、timing cache、PyTorch CUDA extension 和 TensorRT plugin 都与设备环境相关，因此发布包只提供源码和 ONNX，所有二进制产物均在目标设备本地生成。

MobileNetV4-S、MobileNetV4-M 和 MobileNetV4-M-slim-SPF 各提供 224 与 448
两种静态 ONNX。这 6 个图仅使用标准 ONNX 算子，不依赖 Selective Scan plugin。
可使用同一导出入口重新生成，例如：

```bash
python deploy/shared/export_model.py \
  --family relative --encoder MNV4-M-SLIM-SPF --resolution 448 \
  --checkpoint checkpoints/relative/depthart_relative_mnv4m_slim_448.pth \
  --device cpu \
  --output deploy/shared/onnx/relative_mnv4m_slim_spf_448_default.onnx
```
