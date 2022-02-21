import os
import re

import imageio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation


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
