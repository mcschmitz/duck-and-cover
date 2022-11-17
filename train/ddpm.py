import argparse
import os

from accelerate import Accelerator
from accelerate.logging import get_logger

from config import DDPMTrainConfig
from networks import DDPM

logger = get_logger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str)
args = parser.parse_args()

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
