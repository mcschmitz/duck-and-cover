import argparse
import inspect
import math
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from packaging import version
from torchvision.transforms import CenterCrop, Compose, Normalize, ToTensor
from tqdm.auto import tqdm

from config import DDPMTrainConfig
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, __version__
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import deprecate

logger = get_logger(__name__)
diffusers_version = version.parse(version.parse(__version__).base_version)

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str)
args = parser.parse_args()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def main(config):
    logging_dir = os.path.join(config.output_dir, config.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="no" if config.precision == 32 else "fp16",
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=config.channels,
        out_channels=config.channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    accepts_predict_epsilon = "predict_epsilon" in set(
        inspect.signature(DDPMScheduler.__init__).parameters.keys()
    )

    if accepts_predict_epsilon:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.ddpm_num_steps,
            beta_schedule=config.ddpm_beta_schedule,
            predict_epsilon=config.predict_epsilon,
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.ddpm_num_steps,
            beta_schedule=config.ddpm_beta_schedule,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.gen_lr,
        betas=config.gen_betas,
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    augmentations = Compose(
        [
            CenterCrop(config.image_size),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )

    dataset = load_dataset(
        config.dataset_name, config.dataset_config_name, split="train"
    )

    def transforms(examples):
        images = [
            augmentations(image.convert("RGB")) for image in examples["image"]
        ]
        return {"input": images}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
        // config.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )

    ema_model = EMAModel(
        model,
        inv_gamma=config.ema_inv_gamma,
        power=config.ema_power,
        max_value=config.ema_max_decay,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    global_step = 0
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        for _step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps
            )

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_images, timesteps).sample

                if config.predict_epsilon:
                    loss = F.mse_loss(
                        model_output, noise
                    )  # this could have different weights!
                else:
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod,
                        timesteps,
                        (clean_images.shape[0], 1, 1, 1),
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(
                        model_output, clean_images, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                ema_model.step(model)
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if (
                epoch % config.save_images_epochs == 0
                or epoch == config.num_epochs - 1
            ):
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(ema_model.averaged_model),
                    scheduler=noise_scheduler,
                )

                deprecate(
                    "todo: remove this check",
                    "0.10.0",
                    "when the most used version is >= 0.8.0",
                )
                if diffusers_version < version.parse("0.8.0"):
                    generator = torch.manual_seed(0)
                else:
                    generator = torch.Generator(
                        device=pipeline.device
                    ).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(
                    generator=generator,
                    batch_size=config.batch_size,
                    output_type="numpy",
                ).images

                # denormalize the images and save to tensorboard
                images_processed = (images * 255).round().astype("uint8")
                accelerator.trackers[0].writer.add_images(
                    "test_samples",
                    images_processed.transpose(0, 3, 1, 2),
                    epoch,
                )

            if (
                epoch % config.save_images_epochs == 0
                or epoch == config.num_epochs - 1
            ):
                # save the model
                pipeline.save_pretrained(config.output_dir)
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    config = DDPMTrainConfig(args.config_file)
    main(config)
