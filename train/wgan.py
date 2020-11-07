import logging
import os

from tensorflow.keras.optimizers import Adam

from constants import (
    BASE_DATA_PATH,
    LOG_DATETIME_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
)
from loader import DataLoader
from networks import WGAN
from networks.utils import load_progan
from tf_init import init_tf
from utils import create_dir, plot_final_gif

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger(__file__)

IMAGE_SIZE = 64
COVERS_PATH = os.path.join(BASE_DATA_PATH, f"covers{IMAGE_SIZE}")
BATCH_SIZE = 64
PATH = "1_wgan"
TRAIN_STEPS = int(2 * 10e5)

image_size = 64
image_ratio = (1, 1)
images = None
gradient_penalty_weight = 10.0
n_critic = 5
warm_start = False

lp_path = create_dir(f"learning_progress/{PATH}")
model_path = create_dir(os.path.join(lp_path, "model"))

data_loader = DataLoader(
    image_path=COVERS_PATH,
    image_size=IMAGE_SIZE,
)

image_width = image_size * image_ratio[0]

image_height = image_size * image_ratio[1]
minibatch_size = BATCH_SIZE // n_critic

init_tf()
gan = WGAN(
    img_width=image_width,
    img_height=image_height,
    latent_size=1024,
    gradient_penalty_weight=gradient_penalty_weight,
)
gan.build_models(optimizer=Adam(0.0001, beta_1=0.5))

if warm_start:
    gan = load_progan(gan, model_path)

gan.train(
    data_loader=data_loader,
    global_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    path=lp_path,
)

plot_final_gif(path=lp_path)
