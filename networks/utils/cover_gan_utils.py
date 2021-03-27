import numpy as np


def clip_channels(fmaps: int) -> int:
    return int(np.clip(fmaps, 1, 512).item())


def calc_channels_at_stage(stage: int) -> int:
    return clip_channels(512 / (2.0 ** (stage)))
