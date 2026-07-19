# Deployment

部署目录将平台无关实现和平台入口分离：

```text
deploy/
├── shared/
│   ├── onnx/                 # 12 个静态 ONNX
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
