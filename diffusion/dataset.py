# diffusion/dataset.py
import os
from glob import glob
from typing import List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DriveImagesDataset(Dataset):
    """
    只负责读DRIVE中的眼底图像，用于无条件Diffusion训练。
    默认读一个目录下所有 jpg/png/tif。
    """

    def __init__(self, image_root: str, image_size: int = 256):
        """
        Args:
            image_root: 存放图像的目录，比如 data/DRIVE/training/images
            image_size: 输出图像的分辨率（方形）
        """
        self.image_paths: List[str] = []
        exts = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]
        for ext in exts:
            self.image_paths.extend(glob(os.path.join(image_root, ext)))
        self.image_paths = sorted(self.image_paths)
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_root}")

        # 变换：resize + center crop + ToTensor + 归一化到[-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),                         # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])          # -> [-1,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "path": path,
        }
