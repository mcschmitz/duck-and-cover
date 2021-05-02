import os
import re

import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from tensorflow.python.keras.preprocessing.image import array_to_img


def plot_final_gif(path: str):
    gif_size = (256 * 10, 256)
    for s in range(25):
        images = []
        labels = []
        for root, _dirs, files in os.walk(path):
            for file in files:
                if file.startswith(f"{s}_fixed_step_gif"):
                    images.append(imageio.imread(os.path.join(root, file)))
                    labels.append(int(re.findall("\d+", file)[1]))
        order = np.argsort(labels)
        images = [images[i] for i in order]
        labels = [labels[i] for i in order]
        animated_gif = AnimatedGif(size=gif_size)
        for img, lab in zip(images, labels):
            animated_gif.add(
                img,
                label=f"{lab} Images shown",
                label_position=(10, gif_size[1] * 0.95),
            )
        animated_gif.save(
            os.path.join(path, f"{s}_fixed.gif"), fps=len(images) / 30
        )


def generate_images(
    generator_model,
    output_dir: str,
    n_imgs: int = 10,
    seed: int = None,
    target_size: tuple = (64, 64),
    **kwargs,
):
    """
    Generates a list of images by predicting with the given generator.

    Feeds normal distributed random numbers into the generator to generate
    `n_imgs`, tiles the image, rescales it and and saves the output to a PNG
    file.

    Args:
        generator_model: Generator used for generating the images
        output_dir: where to save the results
        n_imgs: number of images to generate
        seed: seed to use to generate the numbers of the latent space
        target_size: target size of the image
    """
    use_gpu = kwargs.get("use_gpu", False)
    release_year_scaler = kwargs.get("release_year_scaler")
    if "use_gpu" in kwargs:
        kwargs.pop("use_gpu")
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    generator_model.eval()
    latent_size = generator_model.latent_size
    if release_year_scaler:
        kwargs.pop("release_year_scaler")
        year = np.random.randint(1925, 2025, 1).reshape(-1, 1)
        scaled_year = release_year_scaler.transform(year)
        scaled_year_vec = np.repeat(scaled_year, 10).reshape(-1, 1)
        latent_size -= 1
    idx = 1
    figsize = (np.array(target_size) * [10, n_imgs]).astype(int) / 300
    fig = plt.figure(figsize=figsize, dpi=300)
    for _ in range(n_imgs):
        x0 = np.random.normal(size=latent_size)
        x1 = np.random.normal(size=latent_size)
        x = np.linspace(x0, x1, 10)
        if release_year_scaler:
            x = np.hstack([x, scaled_year_vec])
        x = torch.Tensor(x)
        if use_gpu:
            x = x.cuda()
            generator_model = generator_model.cuda()
        generated_images = generator_model(x, **kwargs).detach().cpu().numpy()
        generated_images = np.moveaxis(generated_images, 1, -1)
        for img in generated_images:
            img = array_to_img(img, scale=True)
            img = img.resize(size=target_size)
            plt.subplot(n_imgs, 10, idx)
            plt.axis("off")
            plt.imshow(img)
            idx += 1
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.1
            )
    if release_year_scaler:
        fig.axes[0].text(
            2,
            15,
            f"Release year: {year[0][0]}",
            color="black",
            fontsize=4,
        )
    plt.savefig(output_dir)
    plt.close()


class AnimatedGif(object):
    def __init__(self, size: tuple = (640, 480)):
        """
        Allows the addition and generation of gifs by adding multiple images.

        Args:
            size: Final size of the gif

        Refs https://tomroelandts.com/articles/how-to-create-animated-gifs-with-python
        """
        self.size = size
        self.fig = plt.figure()
        self.fig.set_size_inches(self.size[0] / 100, self.size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []

    def add(self, image, label="", label_position: tuple = (1, 1)):
        """
        Add an image to thhe gif.

        Args:
            image: Imported image
            label: Label of the image. Will be added to the given position
            label_position: Label position
        """
        plt_im = plt.imshow(image, vmin=0, vmax=1, animated=True)
        plt_txt = plt.text(
            label_position[0],
            label_position[1],
            label,
            color="black",
            fontsize=20,
        )
        self.images.append([plt_im, plt_txt])

    def save(self, filename, fps: float = 10):
        """
        Save Gif.

        Args:
            filename: Filename of the Gif
            fps: Frames per second
        """
        animation = ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer="imagemagick", fps=fps)
