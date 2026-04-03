import unittest

from debug_helpers import (
    format_foreground_stats,
    make_foreground_stats,
    ratio_or_zero,
    update_foreground_stats,
)


class DebugHelpersTests(unittest.TestCase):
    def test_ratio_or_zero_handles_zero_denominator(self):
        self.assertEqual(ratio_or_zero(5, 0), 0.0)

    def test_foreground_stats_track_empty_and_ratios(self):
        stats = make_foreground_stats()
        update_foreground_stats(stats, fg_pixels=0, total_pixels=100)
        update_foreground_stats(stats, fg_pixels=20, total_pixels=100)

        self.assertEqual(stats["samples"], 2)
        self.assertEqual(stats["empty"], 1)
        self.assertEqual(stats["fg_pixels"], 20)
        self.assertEqual(stats["total_pixels"], 200)
        self.assertAlmostEqual(stats["fg_ratio_sum"], 0.2)

    def test_format_foreground_stats_is_human_readable(self):
        stats = make_foreground_stats()
        update_foreground_stats(stats, fg_pixels=10, total_pixels=100)
        formatted = format_foreground_stats("target", stats)

        self.assertIn("target:", formatted)
        self.assertIn("samples=1", formatted)
        self.assertIn("empty=0/1", formatted)
        self.assertIn("mean_fg_ratio=10.0000%", formatted)


if __name__ == "__main__":
    unittest.main()

