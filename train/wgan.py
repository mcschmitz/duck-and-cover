import os
from pathlib import Path

from config import config
from loader import SpotifyDataloader
from networks import WGAN
from utils.image_operations import create_final_gif

IMAGE_SIZE = config.get("image_size")
COVERS_PATH = os.path.join(
    config.get("base_data_path"), f"covers{300 if IMAGE_SIZE > 64 else 64}"
)
image_ratio = config.get("image_ratio")
BATCH_SIZE = 64
LATENT_SIZE = 512
PATH = f"wgan-{LATENT_SIZE}-{IMAGE_SIZE}x{IMAGE_SIZE}"
TRAIN_STEPS = int(2 * 10e5)

warm_start = False

gradient_penalty_weight = 10.0
n_critic = 5

lp_path = os.path.join(config.get("learning_progress_path"), PATH)
Path(lp_path).mkdir(parents=True, exist_ok=True)
model_dump_path = os.path.join(lp_path, "model")
Path(model_dump_path).mkdir(parents=True, exist_ok=True)

data_loader = SpotifyDataloader(
    image_path=COVERS_PATH,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
)

image_width = IMAGE_SIZE * image_ratio[0]
image_height = IMAGE_SIZE * image_ratio[1]

gan = WGAN(
    img_width=image_width,
    img_height=image_height,
    latent_size=LATENT_SIZE,
    use_gpu=True,
    gradient_penalty_weight=gradient_penalty_weight,
)

gan.set_optimizers(
    generator_optimizer={"lr": 0.0001, "betas": (0.5, 0.999)},
    discriminator_optimizer={"lr": 0.0001},
)

if warm_start:
    gan.load(path=model_dump_path)

gan.train(
    data_loader=data_loader,
    global_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    path=lp_path,
    n_critic=n_critic,
    write_model_to=model_dump_path,
)

create_final_gif(path=lp_path)
