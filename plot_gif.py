import os
import re

import imageio
import numpy as np

from utils import create_dir, AnimatedGif

PATH = "2_progan"
lp_path = create_dir("learning_progress/{}".format(PATH))

if __name__ == "__main__":

    gif_size = (256 * 10, 256 + 50)
    animated_gif = AnimatedGif(size=gif_size)
    images = []
    labels = []
    for root, dirs, files in os.walk(lp_path):
        for file in files:
            if "fixed_step" in file:
                images.append(imageio.imread(os.path.join(root, file)))
                labels.append(int(re.findall("\d+", file)[1]))

    order = np.argsort(labels)
    images = [images[i] for i in order]
    labels = [labels[i] for i in order]
    for img, lab in zip(images, labels):
        animated_gif.add(
            img, label="{} Images shown".format(lab), label_position=(gif_size[0] * 0.9, gif_size[1] * 0.9)
        )
    animated_gif.save(os.path.join(lp_path, "fixed.gif"), fps=len(images) / 60)
