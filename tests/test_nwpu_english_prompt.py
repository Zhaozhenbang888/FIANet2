import unittest

from data.nwpu_text_adapter import build_prompt_spec


class NWPUEnglishPromptTests(unittest.TestCase):
    def test_plural_target_phrase_for_car(self):
        spec = build_prompt_spec('all cars', 'car')
        self.assertIn('cars', spec.target_phrases)

    def test_alias_target_phrase_for_soccer_field(self):
        spec = build_prompt_spec('soccer field on the right side', 'football court')
        self.assertIn('soccer field', spec.target_phrases)
        self.assertIn('right', spec.position_phrases)

    def test_alias_target_phrase_for_excavator(self):
        spec = build_prompt_spec('two excavators on top', 'digger')
        self.assertIn('excavators', spec.target_phrases)
        self.assertIn('top', spec.position_phrases)

    def test_alias_target_phrase_for_photovoltaic_panel(self):
        spec = build_prompt_spec('all photovoltaic panels', 'photovolatic')
        self.assertIn('photovoltaic panels', spec.target_phrases)


if __name__ == '__main__':
    unittest.main()
