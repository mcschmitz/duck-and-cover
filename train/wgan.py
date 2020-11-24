import os

from tensorflow.keras.optimizers import Adam

from constants import BASE_DATA_PATH
from loader import DataLoader
from networks import WGAN
from networks.utils import load_progan
from tf_init import init_tf
from utils import create_dir, plot_final_gif

IMAGE_SIZE = 64
COVERS_PATH = os.path.join(BASE_DATA_PATH, f"covers{IMAGE_SIZE}")
BATCH_SIZE = 64
LATENT_SIZE = 64
PATH = f"{LATENT_SIZE}_wgan"
TRAIN_STEPS = int(2 * 10e5)

image_ratio = (1, 1)
gradient_penalty_weight = 10.0
n_critic = 5
train_steps = TRAIN_STEPS * n_critic
warm_start = False

lp_path = create_dir(f"learning_progress/{PATH}")
model_path = create_dir(os.path.join(lp_path, "model"))

data_loader = DataLoader(
    image_path=COVERS_PATH,
    image_size=IMAGE_SIZE,
)

image_width = IMAGE_SIZE * image_ratio[0]
image_height = IMAGE_SIZE * image_ratio[1]
minibatch_size = BATCH_SIZE // n_critic

init_tf()
gan = WGAN(
    img_width=image_width,
    img_height=image_height,
    latent_size=LATENT_SIZE,
    gradient_penalty_weight=gradient_penalty_weight,
)
gan.build_models(optimizer=Adam(0.0001, beta_1=0.5))

if warm_start:
    gan = load_progan(gan, model_path)

gan.train(
    data_loader=data_loader,
    global_steps=train_steps,
    batch_size=BATCH_SIZE,
    path=lp_path,
)

plot_final_gif(path=lp_path)
