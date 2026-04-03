def ratio_or_zero(part, total):
    if total <= 0:
        return 0.0
    return float(part) / float(total)


def make_foreground_stats():
    return {
        "samples": 0,
        "empty": 0,
        "fg_pixels": 0,
        "total_pixels": 0,
        "fg_ratio_sum": 0.0,
    }


def update_foreground_stats(stats, fg_pixels, total_pixels):
    stats["samples"] += 1
    stats["empty"] += int(fg_pixels == 0)
    stats["fg_pixels"] += int(fg_pixels)
    stats["total_pixels"] += int(total_pixels)
    stats["fg_ratio_sum"] += ratio_or_zero(fg_pixels, total_pixels)
    return stats


def format_foreground_stats(name, stats):
    samples = max(int(stats.get("samples", 0)), 1)
    empty = int(stats.get("empty", 0))
    fg_pixels = int(stats.get("fg_pixels", 0))
    total_pixels = int(stats.get("total_pixels", 0))
    mean_fg_ratio = stats.get("fg_ratio_sum", 0.0) / samples
    overall_fg_ratio = ratio_or_zero(fg_pixels, total_pixels)
    return (
        f"{name}: samples={samples} "
        f"empty={empty}/{samples} ({ratio_or_zero(empty, samples):.2%}) "
        f"mean_fg_ratio={mean_fg_ratio:.4%} "
        f"overall_fg_ratio={overall_fg_ratio:.4%}"
    )

