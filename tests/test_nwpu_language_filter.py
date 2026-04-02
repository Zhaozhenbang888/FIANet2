import unittest

from data.nwpu_text_adapter import classify_text_language, text_matches_language_filter


class NWPULanguageFilterTests(unittest.TestCase):
    def test_classifies_english_text(self):
        self.assertEqual(classify_text_language("airplane on runway"), "english")

    def test_classifies_chinese_text(self):
        self.assertEqual(classify_text_language("\u8dd1\u9053\u4e0a\u7684\u98de\u673a"), "chinese")

    def test_classifies_mixed_text(self):
        self.assertEqual(classify_text_language("airplane \u5728 runway"), "mixed")

    def test_english_filter_is_strict(self):
        self.assertTrue(text_matches_language_filter("airplane on runway", "english"))
        self.assertFalse(text_matches_language_filter("\u8dd1\u9053\u4e0a\u7684\u98de\u673a", "english"))
        self.assertFalse(text_matches_language_filter("airplane \u5728 runway", "english"))

    def test_chinese_filter_is_strict(self):
        self.assertTrue(text_matches_language_filter("\u8dd1\u9053\u4e0a\u7684\u98de\u673a", "chinese"))
        self.assertFalse(text_matches_language_filter("airplane on runway", "chinese"))
        self.assertFalse(text_matches_language_filter("airplane \u5728 runway", "chinese"))

    def test_all_filter_keeps_everything(self):
        self.assertTrue(text_matches_language_filter("airplane on runway", "all"))
        self.assertTrue(text_matches_language_filter("\u8dd1\u9053\u4e0a\u7684\u98de\u673a", "all"))
        self.assertTrue(text_matches_language_filter("airplane \u5728 runway", "all"))


if __name__ == "__main__":
    unittest.main()
