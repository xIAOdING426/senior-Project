# DiffuSeg - 医学图像分割的域驱动扩散模型

本项目是 DiffuSeg 论文的复现项目，专注于使用扩散模型（Diffusion Model）进行医学图像分割。

## 项目结构

```
seniorProject/
├── diffusion/           # 扩散模型核心代码
│   ├── dataset.py      # DRIVE数据集加载器
│   ├── train_ddpm.py   # DDPM训练脚本
│   ├── sample_ddpm.py  # 图像采样脚本
│   └── utils.py        # 工具函数
├── data/               # 数据目录（需要自行下载DRIVE数据集）
└── README.md
```

## 环境要求

- Python 3.8+
- PyTorch 1.12+
- diffusers
- torchvision
- PIL
- tqdm

## 安装依赖

```bash
pip install torch torchvision diffusers pillow tqdm numpy
```

## 使用方法

### 1. 准备数据

将 DRIVE 数据集的训练图像放在 `data/DRIVE/training/images/` 目录下。

### 2. 训练模型

```bash
cd diffusion
python train_ddpm.py
```

训练配置可以在 `train_ddpm.py` 的 `TrainConfig` 类中修改。

### 3. 生成图像

训练完成后，使用采样脚本生成图像：

```bash
python sample_ddpm.py
```

记得在 `sample_ddpm.py` 中修改 `ckpt_path` 指向你训练好的模型。

## 任务分配

本项目是四人协作项目的一部分，当前代码负责：

- **角色1：扩散引擎工程师**
  - 搭建标准的 Latent Diffusion Model (LDM) / DDPM 模型
  - 实现数据预处理和 Dataset/DataLoader
  - 完成无条件训练
  - 实现 Milestone 1：在 DRIVE 数据集上生成无条件的、FID 合理的真实眼底图像

## 配置说明

### 训练配置（train_ddpm.py）

- `image_root`: DRIVE数据集图像路径
- `image_size`: 图像分辨率（默认256x256）
- `batch_size`: 批次大小（默认8）
- `num_epochs`: 训练轮数（默认100）
- `lr`: 学习率（默认1e-4）
- `num_train_timesteps`: 扩散步数（默认1000）

### 采样配置（sample_ddpm.py）

- `ckpt_path`: 模型检查点路径
- `num_images`: 生成图像数量
- `batch_size`: 采样批次大小

## 注意事项

- 训练过程非常耗时，建议使用GPU
- 如果显存不足，可以减小 `batch_size` 或 `block_out_channels`
- 模型检查点会保存在 `outputs_ddpm/` 目录下

## 许可证

本项目仅用于学术研究目的。

