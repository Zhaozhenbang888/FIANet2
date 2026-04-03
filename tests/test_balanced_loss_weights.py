import unittest

from loss.class_weights import compute_balanced_class_weights


class BalancedClassWeightsTests(unittest.TestCase):
    def test_balances_extreme_foreground_imbalance(self):
        bg, fg = compute_balanced_class_weights(100000, 100)
        self.assertGreater(fg, bg)
        self.assertGreaterEqual(fg, 10.0)

    def test_clamps_foreground_weight(self):
        bg, fg = compute_balanced_class_weights(1000000, 1, max_fg_weight=20.0)
        self.assertEqual(fg, 20.0)
        self.assertEqual(bg, 1.0)

    def test_handles_missing_foreground(self):
        bg, fg = compute_balanced_class_weights(1000, 0)
        self.assertEqual((bg, fg), (1.0, 1.0))


if __name__ == '__main__':
    unittest.main()

