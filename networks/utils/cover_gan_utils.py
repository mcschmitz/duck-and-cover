import os

from tensorflow.keras.utils import plot_model


def plot_progan(model, path: str, suffix: str = None):
    for fade_in in ["", "_fade_in"]:
        suffix += "_fade_in"
        model_to_plot = getattr(model, f"generator{fade_in}")
        if model_to_plot:
            plot_model(
                model_to_plot,
                to_file=os.path.join(path, f"gen{suffix}.png"),
                show_shapes=True,
                expand_nested=True,
            )
            plot_model(
                getattr(model, f"combined_model{fade_in}"),
                to_file=os.path.join(path, f"comb{suffix}.png"),
                show_shapes=True,
                expand_nested=True,
            )
