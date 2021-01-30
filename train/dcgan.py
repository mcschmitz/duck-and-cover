import os

from tensorflow.keras.optimizers import Adam

from constants import BASE_DATA_PATH
from loader import DataLoader
from networks import DCGAN
from tf_init import init_tf
from utils import create_dir, plot_final_gif

IMAGE_SIZE = 64
COVERS_PATH = os.path.join(BASE_DATA_PATH, f"covers{IMAGE_SIZE}")
BATCH_SIZE = 64
LATENT_SIZE = 1024
PATH = f"{LATENT_SIZE}_dcgan"
TRAIN_STEPS = int(2 * 10e5)

image_ratio = (1, 1)

lp_path = create_dir(f"learning_progress/{PATH}")

data_loader = DataLoader(
    image_path=COVERS_PATH,
    image_size=IMAGE_SIZE,
)

image_width = IMAGE_SIZE * image_ratio[0]
image_height = IMAGE_SIZE * image_ratio[1]

init_tf()
gan = DCGAN(
    img_width=image_width,
    img_height=image_height,
    latent_size=64,
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
