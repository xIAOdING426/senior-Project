# diffusion/train_ddpm.py
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusers import UNet2DModel, DDPMScheduler

from dataset import DriveImagesDataset
from utils import set_seed, save_images


@dataclass
class TrainConfig:
    image_root: str = "data/DRIVE/training/images"  # TODO: 改成你自己的路径
    image_size: int = 256
    batch_size: int = 8
    num_epochs: int = 100
    lr: float = 1e-4
    num_train_timesteps: int = 1000
    output_dir: str = "outputs_ddpm"
    save_model_every: int = 10        # 每多少个epoch存一次模型
    sample_every: int = 5             # 每多少个epoch采样一次看看效果
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = 4
    grad_accum_steps: int = 1         # 显存不够可以 >1
    mixed_precision: bool = False     # 显卡好可以 True


def create_model(config: TrainConfig) -> UNet2DModel:
    """
    创建一个UNet，用于256x256 RGB图像。
    如果显存比较小，可以把 block_out_channels 调小。
    """
    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


def main():
    config = TrainConfig()
    set_seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    # 1. Dataset & DataLoader
    dataset = DriveImagesDataset(config.image_root, config.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # 2. Model, scheduler, optimizer
    model = create_model(config).to(config.device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

    global_step = 0
    model.train()

    for epoch in range(1, config.num_epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")

        for step, batch in enumerate(pbar):
            clean_images = batch["image"].to(config.device)  # [-1,1]

            # 采样噪声 & 时间步
            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=config.device,
            ).long()

            # 加噪
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                noise_pred = model(noisy_images, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)

            epoch_loss += loss.item()

            scaler.scale(loss / config.grad_accum_steps).backward()

            if (step + 1) % config.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} finished. avg loss = {avg_loss:.4f}")

        # 3. 采样看看效果
        if epoch % config.sample_every == 0:
            model.eval()
            with torch.no_grad():
                # 采 16 张图
                sample = torch.randn(16, 3, config.image_size, config.image_size).to(
                    config.device
                )
                sample = sample_images(model, noise_scheduler, sample)
            save_images(sample, config.output_dir, f"sample_epoch_{epoch:04d}.png")
            model.train()

        # 4. 保存模型
        if epoch % config.save_model_every == 0:
            ckpt_path = os.path.join(
                config.output_dir, f"ddpm_epoch_{epoch:04d}.pt"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.__dict__,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")


def sample_images(model, noise_scheduler, noisy_sample):
    """
    从随机噪声开始，用DDPM反向过程生成图像。
    Args:
        model: UNet2DModel
        noise_scheduler: DDPMScheduler
        noisy_sample: 初始噪声 (B, C, H, W)
    """
    model.eval()
    with torch.no_grad():
        # 设置 scheduler 为 inference 模式
        noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)

        sample = noisy_sample
        for t in tqdm(noise_scheduler.timesteps, desc="Sampling", leave=False):
            # 模型预测当前的noise
            model_output = model(sample, t).sample
            # 根据scheduler公式迈一步
            step_output = noise_scheduler.step(
                model_output, t, sample
            )
            sample = step_output.prev_sample
    return sample


if __name__ == "__main__":
    main()
