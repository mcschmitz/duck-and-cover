import argparse
import os

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import __version__
from packaging import version

from config import DDPMTrainConfig
from networks import DDPM

logger = get_logger(__name__)
diffusers_version = version.parse(version.parse(__version__).base_version)

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str)
args = parser.parse_args()


"""def main(config, train_dataloader):
    for epoch in range(config.num_epochs):
        for _step, batch in enumerate(train_dataloader):



        
        # This needs to be executed before we save a
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

    accelerator.end_training() """


if __name__ == "__main__":
    config = DDPMTrainConfig(args.config_file)

    logging_dir = os.path.join(config.output_dir, config.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="no" if config.precision == 32 else "fp16",
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    dataloader = config.get_dataloader()

    ddpm_network = DDPM(config, accelerator=accelerator)

    train_dataloader = dataloader.train_dataloader()

    ddpm_network.train(trainset=train_dataloader, accelerator=accelerator)
