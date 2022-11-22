import argparse

import pytorch_lightning as pl
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
    dataloader = config.get_dataloader()
    train_dataloader = dataloader.train_dataloader()

    ddpm_network = DDPM(config)

    logger = pl.loggers.WandbLogger(
        project="duck-and-cover",
        entity="mcschmitz",
        tags=["DDPM"] + config.wandb_tags,
        name=config.unique_experiment_name,
        id=ddpm_network.wandb_run_id,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="no" if config.precision == 32 else "fp16",
    )
    ddpm_network.train(
        trainset=train_dataloader, accelerator=accelerator, logger=logger
    )
