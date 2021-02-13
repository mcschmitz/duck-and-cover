import os
from datetime import datetime
from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from networks.utils.cover_gan_utils import load_progan, plot_progan, save_gan
from networks.utils.wgan_utils import gradient_penalty, wasserstein_loss


def plot_metric(path: str, steps: int, metric: List, **kwargs):
    """
    Plots a time series and saves it to the given path.

    Args:
        path: Where to write the plot
        steps: Total number of training steps represented in the time series
        metric: Time series to plot

    Keyword Args:
        x_label (str): Label of the x axis
        y_label (str): Label of the y axis
    """
    x_axis = np.linspace(0, steps, len(metric))
    sns.lineplot(x_axis, metric)
    plt.ylabel(kwargs.get("y_label", ""))
    plt.xlabel(kwargs.get("x_label", "Images shown"))
    plt.savefig(
        os.path.join(path, kwargs.get("file_name", hash(datetime.now())))
    )
    plt.close()
