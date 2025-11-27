# diffusion/utils.py
import os
import random
from typing import Optional

import numpy as np
import torch
from torchvision.utils import save_image


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_images(
    images: torch.Tensor,
    out_dir: str,
    file_name: str,
    nrow: int = 8,
    value_range: Optional[tuple] = (-1, 1)
):
    """
    保存一批图像到磁盘。
    Args:
        images: (B, C, H, W) tensor
        out_dir: 输出目录
        file_name: 文件名，如 "sample_0001.png"
        nrow: 每行多少张图
        value_range: 当前tensor的值域，如果是[-1,1]会自动映射回[0,1]
    """
    os.makedirs(out_dir, exist_ok=True)

    imgs = images.detach().cpu()
    if value_range is not None:
        low, high = value_range
        imgs = (imgs - low) / (high - low)  # -> [0,1]

    save_path = os.path.join(out_dir, file_name)
    save_image(imgs, save_path, nrow=nrow)
    print(f"Saved images to {save_path}")
