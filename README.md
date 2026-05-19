# LGMM-Net: Local–Global Mamba-Enhanced Multi-Scale Edge-Aware Segmentation Network

LGMM-Net is a lightweight deep learning model specifically designed for medical image segmentation tasks. It combines the efficient long-sequence modeling capabilities of Mamba (State Space Models) with an innovative gated multi-scale fusion mechanism, achieving superior segmentation performance with a minimal number of parameters.

## Requirements

Python 3.8+ and PyTorch 1.13.0+ are recommended.

```bash
pip install -r requirements.txt
```

Main dependencies:
- torch == 1.13.0
- torchvision == 0.14.0
- timm == 0.4.12
- mamba_ssm == 1.0.1
- causal_conv1d == 1.0.0
- einops, scikit-learn, SimpleITK, medpy, etc.

## Data Preparation

The project supports major skin lesion datasets such as ISIC2017, ISIC2018, and PH2.

1. Download the datasets and place them in the `../` directory.
2. Run the data preprocessing script (using ISIC2017 as an example):
   ```bash
   python dataprepare/Prepare_ISIC2017.py
   ```
   The script will resize images to 256x256 and generate `.npy` format files.

```

## Project Structure

```text
LGMM-net/
├── configs/            # Configuration files
├── dataprepare/        # Data preprocessing scripts
├── models/             # Model definitions (LGMM-net.py, UNet.py)
├── checkpoints/        # Saved model weights
├── results/            # Training logs and outputs
├── loader.py           # Data loader
├── train.py            # Training script
├── test.py             # Testing script
└── requirements.txt    # Dependency list
```

## License

This project is licensed under the MIT License. See the [LICENSE](file:///d:/UltraLight-VM-UNet-main10-2017/LGMM-net/LICENSE) file for details.

## Acknowledgements

Special thanks to open-source projects like [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet?utm_source=chatgpt.com) for their inspiration.
