"""
Trains the ProGAN.
"""

import os
import itertools
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
from tqdm import tqdm
from keras_gradient_accumulation import GradientAccumulation
from keras.optimizers import Adam
from Loader.cover_loader import load_img, img_to_array, scale_images
from networks import ProGAN
from networks.utils import plot_gan, save_gan, load_gan
from utils import create_dir, generate_images

N_BLOCKS = 6
RESOLUTIONS = 2 ** np.arange(2, N_BLOCKS + 2)
LATENT_SIZE = RESOLUTIONS[-1]

PATH = "celeba/2_progan_1"
FADE = [True, False]
WARM_START = False

BATCH_SIZE = 128
ACCUMULATIVE_UPDATES = {r: i for r, i in zip(RESOLUTIONS, [1, 1, 2, 4, 8, 16, 32])}
MINIBATCH_REPS = 1
GRADIENT_PENALTY_WEIGHT = 10
N_CRITIC = 1
TRAIN_STEPS = int(10e5)


if __name__ == "__main__":
    init_burn_in = False
    lp_path = create_dir("learning_progress/{}".format(PATH))
    base_optimizer = Adam(0.001, beta_1=0.0, beta_2=0.99)
    base_disc_optimizer = Adam(0.001, beta_1=0.0, beta_2=0.99)
    optimizer = GradientAccumulation(base_optimizer, accumulation_steps=1)
    disc_optimizer = GradientAccumulation(base_disc_optimizer, accumulation_steps=1)

    gan = ProGAN(gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT, latent_size=LATENT_SIZE)
    gan.build_models(optimizer=optimizer, discriminator_optimizer=disc_optimizer, image_resolution=RESOLUTIONS[0],
                     channels=3, batch_size=BATCH_SIZE)

    for resolution, fade in itertools.product(RESOLUTIONS, FADE):
        if init_burn_in and resolution == 4:
            continue
        if resolution == 4:
            print("\n\nStarting init burn in with resolution {}\n\n".format(resolution))
            fade = False
            warm_start = False

        if fade:
            print("\n\nStarting fade in with resolution {}\n\n".format(resolution))
        elif resolution > 4:
            print("\n\nStarting burn in with resolution {}\n\n".format(resolution))
        model_dump_path = create_dir(os.path.join(lp_path, "model{}".format(resolution)))

        if WARM_START:
            fade = True
            gan.build_models(optimizer=gan.combined_model.optimizer,
                             discriminator_optimizer=gan.discriminator_model.optimizer,
                             image_resolution=resolution // 2)
            model_dump_path = create_dir(os.path.join(lp_path, "model{}".format(resolution // 2)))
            gan = load_gan(gan, model_dump_path, weights_only=True)

        batch_size = BATCH_SIZE // ACCUMULATIVE_UPDATES[resolution]

        DATA_PATH = "data/celeba/all{}.npy".format(resolution)

        images = None

        if os.path.exists(DATA_PATH) and os.stat(DATA_PATH).st_size < (psutil.virtual_memory().total * .8):
            images = np.load(DATA_PATH)
            img_idx = np.arange(0, images.shape[0])
        elif os.path.exists(DATA_PATH):
            pass
        else:
            try:
                files = [os.path.join("data/celeba", f) for f in os.listdir("data/celeba") if
                         os.path.splitext(os.path.join("data/celeba", f))[1] == ".jpg"]
                images = np.zeros((len(files), resolution, resolution, 3), dtype=K.floatx())

                for i, file_path in tqdm(enumerate(files)):
                    img = load_img(file_path, grayscale=False, target_size=(resolution, resolution, 3))
                    x = img_to_array(img)
                    x = scale_images(x)
                    images[i] = x
                np.save(DATA_PATH, images)
                img_idx = np.arange(0, images.shape[0])
            except MemoryError as ex:
                print("Data does not fit inside Memory. Preallocation is not possible.")

        minibatch_size = batch_size * N_CRITIC
        img_resolution = resolution if not fade else resolution // 2

        batch_idx = 0
        steps = TRAIN_STEPS // batch_size

        initial_iter = 0

        if fade:
            gan.add_fade_in_layers(target_resolution=resolution)
            gan.build_models(optimizer=gan.combined_model.optimizer, batch_size=batch_size,
                             discriminator_optimizer=gan.discriminator_model.optimizer, compile_only=True)
            plot_gan(gan, lp_path, str(resolution) + "_fade_in")
            alphas = np.linspace(0, 1, steps).tolist()

        for step in range(initial_iter // batch_size, steps):
            if fade:
                alpha = alphas.pop(0)
                gan.update_alpha(alpha)
            batch_idx = [i if i < images.shape[0] else i - images.shape[0] for i in
                         np.arange(batch_idx, batch_idx + minibatch_size)]
            if 0 in batch_idx and images.shape[0] in batch_idx:
                np.random.shuffle(img_idx)
                images = images[img_idx]
            batch_images = images[batch_idx]
            batch_idx = batch_idx[-1] + 1
            for _ in range(MINIBATCH_REPS):
                gan.train_on_batch(batch_images, n_critic=N_CRITIC)
            gan.images_shown += batch_size

            if step % (steps//10) == 0:
                print('Images shown {0}: Generator Loss: {1:3,.3f} - Discriminator Loss: {2:3,.3f} - '
                      'Discriminator Loss + : {2:3,.3f} - Discriminator Loss - : {3:3,.3f} -'
                      ' Discriminator Loss Dummies : {4:3,.3f}'.format(
                    gan.images_shown, np.mean(gan.history["G_loss"]), np.mean(gan.history["D_loss"]),
                    np.mean(gan.history["D_loss_positives"]), np.mean(gan.history["D_loss_negatives"]),
                    np.mean(gan.history["D_loss_dummies"])))

                generate_images(gan.generator,
                                os.path.join(lp_path, "step{}.png".format(gan.images_shown)),
                                target_size=(RESOLUTIONS[-1] * 10, RESOLUTIONS[-1]))
                generate_images(gan.generator,
                                os.path.join(lp_path, "fixed_step{}.png".format(gan.images_shown)),
                                target_size=(RESOLUTIONS[-1] * 10, RESOLUTIONS[-1]), seed=101)

                #########################################################
                if fade:
                    alpha = 0
                    gan.update_alpha(alpha)
                    generate_images(gan.generator,
                                    os.path.join(lp_path, "fixed_step{}_alpha0.png".format(gan.images_shown)),
                                    target_size=(RESOLUTIONS[-1] * 10, RESOLUTIONS[-1]), seed=101)
                    alpha = 1
                    gan.update_alpha(alpha)
                    generate_images(gan.generator,
                                    os.path.join(lp_path, "fixed_step{}_alpha1.png".format(gan.images_shown)),
                                    target_size=(RESOLUTIONS[-1] * 10, RESOLUTIONS[-1]), seed=101)
                #########################################################

                x_axis = np.linspace(0, gan.images_shown, len(gan.history["D_loss"]))
                ax = sns.lineplot(x_axis, gan.history["D_loss"])
                plt.ylabel('Discriminator Loss')
                plt.xlabel('Images shown')
                plt.savefig(os.path.join(lp_path, "d_loss.png"))
                plt.close()

                x_axis = np.linspace(0, gan.images_shown, len(gan.history["D_loss_positives"]))
                ax = sns.lineplot(x_axis, gan.history["D_loss_positives"])
                plt.ylabel('Discriminator Loss on Positives')
                plt.xlabel('Images shown')
                plt.savefig(os.path.join(lp_path, "d_lossP.png"))
                plt.close()

                ax = sns.lineplot(x_axis, gan.history["D_loss_negatives"])
                plt.ylabel('Discriminator Loss on Negatives')
                plt.xlabel('Images shown')
                plt.savefig(os.path.join(lp_path, "d_lossN.png"))
                plt.close()

                ax = sns.lineplot(x_axis, gan.history["D_loss_dummies"])
                plt.ylabel('Discriminator Loss on Dummies')
                plt.xlabel('Images shown')
                plt.savefig(os.path.join(lp_path, "d_lossD.png"))
                plt.close()

                ax = sns.lineplot(x_axis, gan.history["G_loss"])
                plt.ylabel('Generator Loss')
                plt.xlabel('Images shown')
                plt.savefig(os.path.join(lp_path, "g_loss.png"))
                plt.close()

        if fade:
            gan.remove_fade_in_layers()
            gan.build_models(optimizer=gan.combined_model.optimizer,
                             discriminator_optimizer=gan.discriminator_model.optimizer,
                             image_resolution=resolution, compile_only=True)
            save_gan(gan, model_dump_path)
            gan.build_models(optimizer=gan.combined_model.optimizer,
                             discriminator_optimizer=gan.discriminator_model.optimizer,
                             image_resolution=resolution)
            gan = load_gan(gan, model_dump_path, weights_only=True)
        plot_gan(gan, lp_path, str(resolution) + "_grown")
        save_gan(gan, model_dump_path)
        if resolution == 4:
            init_burn_in = True

    save_gan(gan, model_dump_path)
