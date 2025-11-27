# diffusion/sample_ddpm.py
import os
import torch
from tqdm.auto import tqdm
from dataclasses import dataclass

from diffusers import UNet2DModel, DDPMScheduler

from utils import save_images


@dataclass
class SampleConfig:
    ckpt_path: str = "outputs_ddpm/ddpm_epoch_0100.pt"  # TODO: 改成你的
    num_images: int = 64
    batch_size: int = 16
    image_size: int = 256
    num_train_timesteps: int = 1000
    output_dir: str = "outputs_ddpm/samples"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def create_model_from_ckpt(config: SampleConfig):
    ckpt = torch.load(config.ckpt_path, map_location="cpu")
    model_cfg = ckpt.get("config", None)

    if model_cfg is not None:
        image_size = model_cfg.get("image_size", config.image_size)
    else:
        image_size = config.image_size

    model = UNet2DModel(
        sample_size=image_size,
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
    model.load_state_dict(ckpt["model"])
    model.to(config.device)
    model.eval()
    return model


def sample_images(model, noise_scheduler, sample):
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    with torch.no_grad():
        for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
            model_output = model(sample, t).sample
            step_output = noise_scheduler.step(model_output, t, sample)
            sample = step_output.prev_sample
    return sample


def main():
    config = SampleConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    model = create_model_from_ckpt(config)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
    )

    n_batches = (config.num_images + config.batch_size - 1) // config.batch_size
    all_samples = []

    for i in range(n_batches):
        cur_bs = min(config.batch_size, config.num_images - i * config.batch_size)
        noise = torch.randn(cur_bs, 3, config.image_size, config.image_size).to(
            config.device
        )
        samples = sample_images(model, noise_scheduler, noise)
        all_samples.append(samples)

    all_samples = torch.cat(all_samples, dim=0)
    save_images(all_samples, config.output_dir, "samples_all.png")


if __name__ == "__main__":
    main()
