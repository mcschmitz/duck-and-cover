def calc_channels_at_stage(stage: int) -> int:
    return min(int(8192 / (2.0 ** (stage))), 512)
