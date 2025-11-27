#!/usr/bin/env python3
"""
检查DRIVE数据集是否准备就绪
"""
import os
from glob import glob

def check_drive_dataset():
    image_root = "data/DRIVE/training/images"
    
    if not os.path.exists(image_root):
        print(f"❌ 目录不存在: {image_root}")
        print("请先创建目录并下载数据")
        return False
    
    # 查找所有图像文件
    exts = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob(os.path.join(image_root, ext)))
    
    image_count = len(image_paths)
    
    if image_count == 0:
        print(f"❌ 在 {image_root} 中未找到任何图像文件")
        print("请确保图像文件已正确放置")
        return False
    
    print(f"✅ 找到 {image_count} 张图像")
    print(f"📁 目录: {image_root}")
    print("\n前5个文件:")
    for i, path in enumerate(sorted(image_paths)[:5], 1):
        print(f"  {i}. {os.path.basename(path)}")
    
    if image_count < 10:
        print(f"\n⚠️  警告: 图像数量较少（{image_count}张），建议至少有10-20张用于训练")
    else:
        print(f"\n✅ 数据准备完成！可以开始训练了")
    
    return True

if __name__ == "__main__":
    check_drive_dataset()

