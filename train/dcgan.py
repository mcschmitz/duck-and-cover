import os
from pathlib import Path

from tensorflow.keras.optimizers import Adam

from config import config
from constants import BASE_DATA_PATH
from loader import DataLoader
from networks import DCGAN
from utils import init_tf
from utils.image_operations import plot_final_gif

IMAGE_SIZE = config.get("image_size")
COVERS_PATH = os.path.join(
    BASE_DATA_PATH, f"covers{300 if IMAGE_SIZE > 64 else 64}"
)
BATCH_SIZE = 64
LATENT_SIZE = 512
PATH = f"{LATENT_SIZE}_dcgan"
TRAIN_STEPS = int(2 * 10e5)

init_tf()

image_ratio = (1, 1)

lp_path = os.path.join(config.get("learning_progress"), PATH)
Path(lp_path).mkdir(parents=True, exist_ok=True)

data_loader = DataLoader(
    image_path=COVERS_PATH,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
)

image_width = IMAGE_SIZE * image_ratio[0]
image_height = IMAGE_SIZE * image_ratio[1]

gan = DCGAN(
    img_width=image_width,
    img_height=image_height,
    latent_size=LATENT_SIZE,
)
gan.build_models(
    combined_optimizer=Adam(0.0001, beta_1=0.5),
    discriminator_optimizer=Adam(0.000004),
)

gan.train(
    data_loader=data_loader,
    global_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    path=lp_path,
)

plot_final_gif(path=lp_path)
