# LGMM-Net: Local–Global Mamba-Enhanced Multi-Scale Edge-Aware Segmentation Network

LGMM-Net 是一种专为医疗图像分割任务设计的轻量级深度学习模型。它结合了 Mamba (State Space Models) 的高效长序列建模能力和创新的门控多尺度融合机制，在保持极低参数量的同时，实现了卓越的分割性能。

## 环境要求

建议使用 Python 3.8+ 和 PyTorch 1.13.0+。

```bash
pip install -r requirements.txt
```

主要依赖项包括：
- torch == 1.13.0
- torchvision == 0.14.0
- timm == 0.4.12
- mamba_ssm == 1.0.1
- causal_conv1d == 1.0.0
- einops, scikit-learn, SimpleITK, medpy 等

## 数据准备

项目支持 ISIC2017, ISIC2018 和 PH2 等主流皮肤病变数据集。

1. 下载数据集并将其放置在 `../` 目录下。
2. 运行数据预处理脚本（以 ISIC2017 为例）：
   ```bash
   python dataprepare/Prepare_ISIC2017.py
   ```
   该脚本会将图像缩放至 256x256 并生成 `.npy` 格式文件。

## 模型训练

在 `configs/config_setting.py` 中修改相关配置（如数据集路径、批大小、学习率等），然后运行：

```bash
python train.py
```

## 模型测试

训练完成后，在 `configs/config_setting.py` 中指定 `test_weights` 路径，并运行测试：

```bash
python test.py
```

## 项目结构

```text
LGMM-net/
├── configs/            # 配置文件
├── dataprepare/        # 数据预处理脚本
├── models/             # 模型定义 (LGMM-net.py, UNet.py)
├── checkpoints/        # 存放训练好的模型权重
├── results/            # 训练日志与输出结果
├── loader.py           # 数据加载器
├── train.py            # 训练脚本
├── test.py             # 测试脚本
└── requirements.txt    # 依赖项列表
```

## 许可证

本项目遵循 MIT 许可证。详见 [LICENSE](file:///d:/UltraLight-VM-UNet-main10-2017/LGMM-net/LICENSE) 文件。

## 致谢

感谢 [UltraLight-VM-UNet](https://github.com/hust-vcl/UltraLight-VM-UNet) 等开源项目的启发。
