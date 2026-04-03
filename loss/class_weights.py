import math


def compute_balanced_class_weights(
    background_count,
    foreground_count,
    *,
    min_fg_weight=1.0,
    max_fg_weight=20.0,
    exponent=0.5,
):
    background_count = float(background_count)
    foreground_count = float(foreground_count)

    if background_count <= 0 or foreground_count <= 0:
        return 1.0, 1.0

    ratio = background_count / foreground_count
    fg_weight = math.pow(ratio, exponent)
    fg_weight = max(min_fg_weight, min(max_fg_weight, fg_weight))
    return 1.0, float(fg_weight)
