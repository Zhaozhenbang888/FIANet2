import unittest

import torch

from lib.graph_reasoning_chain_v2 import GraphReasoningChain


def _make_inputs(batch_size=2, channels=4, height=3, width=3, text_dim=8, seq_len=5):
    x = torch.randn(batch_size, channels, height, width)
    l = torch.randn(batch_size, text_dim, seq_len)
    l_mask = torch.ones(batch_size, seq_len, 1)
    text_ids = torch.randint(0, 10, (batch_size, seq_len))
    return x, l, l_mask, text_ids


def _minimal_parse_batch(batch_size, valid_len):
    return [
        {
            "target": {"word": None, "token_span": None},
            "entities": [],
            "attributes": [],
            "relations": [],
            "tokens": [],
            "entity_attributes": {},
            "spatial_triplets": [],
            "entity_tokens_map": {},
            "attr_tokens_map": {},
            "rel_tokens_map": {},
            "reasoning_units": [(0, valid_len)],
            "valid_len": valid_len,
        }
        for _ in range(batch_size)
    ]


class GraphReasoningModeTests(unittest.TestCase):
    def test_off_mode_returns_input_without_running_grl(self):
        x, l, l_mask, text_ids = _make_inputs()
        grl = GraphReasoningChain(in_channels=4, text_dim=8, hidden_dim=4, num_visual_nodes=3, grl_mode="off")

        output = grl(x, l, l_mask, text_ids=text_ids)

        torch.testing.assert_close(output, x)

    def test_full_mode_uses_text_parser(self):
        x, l, l_mask, text_ids = _make_inputs(batch_size=1)
        grl = GraphReasoningChain(in_channels=4, text_dim=8, hidden_dim=4, num_visual_nodes=3, grl_mode="full")
        calls = {"count": 0}

        def fake_parse(batch_text_ids, batch_mask):
            calls["count"] += 1
            return _minimal_parse_batch(batch_size=1, valid_len=batch_text_ids.shape[1])

        grl.text_parser.parse_batch_detailed = fake_parse

        output = grl(x, l, l_mask, text_ids=text_ids)

        self.assertEqual(calls["count"], 1)
        self.assertEqual(output.shape, x.shape)

    def test_no_parser_mode_skips_text_parser(self):
        x, l, l_mask, text_ids = _make_inputs(batch_size=1)
        grl = GraphReasoningChain(in_channels=4, text_dim=8, hidden_dim=4, num_visual_nodes=3, grl_mode="no_parser")

        def fail_parse(*args, **kwargs):
            raise AssertionError("text parser should not be used in no_parser mode")

        grl.text_parser.parse_batch_detailed = fail_parse

        output = grl(x, l, l_mask, text_ids=text_ids)

        self.assertEqual(output.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
