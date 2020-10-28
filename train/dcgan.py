import os
import re

import imageio
import numpy as np
from tensorflow.keras.optimizers import Adam
from constants import BASE_DATA_PATH
from loader import DataLoader
from networks import CoverGAN
from networks.utils import load_progan, save_gan
from utils import AnimatedGif, create_dir
from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
import logging

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger(__file__)

IMAGE_SIZE = 64
COVERS_PATH = os.path.join(BASE_DATA_PATH, f"covers{IMAGE_SIZE}")
BATCH_SIZE = 64
PATH = "0_dcgan"
WARM_START = False
DATA_PATH = "data/covers64/all64.npy"
TRAIN_STEPS = int(2 * 10e5)

image_ratio = (1, 1)
images = None

lp_path = create_dir(f"learning_progress/{PATH}")
model_path = create_dir(os.path.join(lp_path, "model"))

data_loader = DataLoader(
    image_path=COVERS_PATH,
    image_size=IMAGE_SIZE,
)

image_width = IMAGE_SIZE * image_ratio[0]
image_height = IMAGE_SIZE * image_ratio[1]

gan = CoverGAN(
    img_width=image_width,
    img_height=image_height,
    latent_size=128,
    batch_size=BATCH_SIZE,
)
gan.build_models(
    combined_optimizer=Adam(0.0001, beta_1=0.5),
    discriminator_optimizer=Adam(0.000004),
)

if WARM_START:
    logger.info(f"Apply warm start. Load models from {model_path}")
    gan = load_progan(gan, model_path)

gan.train(data_loader=data_loader, global_steps=TRAIN_STEPS, batch_size=BATCH_SIZE, path=lp_path)

gif_size = (image_width * 10, image_height + 50)
animated_gif = AnimatedGif(size=gif_size)
images = []
labels = []
for root, dirs, files in os.walk(lp_path):
    for file in files:
        if "fixed_step" in file:
            images.append(imageio.imread(os.path.join(root, file)))
            labels.append(int(re.findall("\d+", file)[0]))

order = np.argsort(labels)
images = [images[i] for i in order]
labels = [labels[i] for i in order]
for img, lab in zip(images, labels):
    animated_gif.add(
        img,
        label="{} Images shown".format(lab),
        label_position=(gif_size[0] * 0.7, gif_size[1] * 0.7),
    )
animated_gif.save(os.path.join(lp_path, "fixed.gif"), fps=len(images) / 30)
