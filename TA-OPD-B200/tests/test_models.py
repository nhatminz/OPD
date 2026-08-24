from __future__ import annotations

import unittest
from types import SimpleNamespace

from b200_experiment.models import (
    assert_tokenizer_compatibility,
    validate_shared_tokenizer_protocol,
)


class _FakeTokenizer:
    def __init__(self, vocab, added_vocab, special_tokens_map, length=None):
        self._vocab = dict(vocab)
        self._added_vocab = dict(added_vocab)
        self.special_tokens_map = dict(special_tokens_map)
        self._length = len(self._vocab) if length is None else int(length)

    def get_vocab(self):
        return dict(self._vocab)

    def get_added_vocab(self):
        return dict(self._added_vocab)

    def __len__(self):
        return self._length


class ModelCompatibilityTests(unittest.TestCase):
    def test_special_token_roles_may_differ_when_integer_vocab_is_exact(self):
        vocab = {"a": 0, "<eos>": 1, "<pad>": 2}
        student = _FakeTokenizer(vocab, {"<pad>": 2}, {"eos_token": "<eos>"})
        teacher = _FakeTokenizer(
            vocab,
            {"<pad>": 2},
            {"eos_token": "<eos>", "pad_token": "<pad>"},
        )

        result = assert_tokenizer_compatibility(
            student,
            teacher,
            SimpleNamespace(vocab_size=3),
            SimpleNamespace(vocab_size=3),
        )

        self.assertTrue(result["vocab_mapping"])
        self.assertTrue(result["added_vocab"])
        self.assertFalse(result["special_tokens_map_equal"])

    def test_different_token_id_mapping_remains_fatal(self):
        student = _FakeTokenizer({"a": 0, "b": 1}, {}, {})
        teacher = _FakeTokenizer({"a": 1, "b": 0}, {}, {})

        with self.assertRaisesRegex(ValueError, "vocab_mapping"):
            assert_tokenizer_compatibility(
                student,
                teacher,
                SimpleNamespace(vocab_size=2),
                SimpleNamespace(vocab_size=2),
            )

    def test_all_other_requested_compatibility_checks_remain_fatal(self):
        vocab = {"a": 0, "b": 1}
        student = _FakeTokenizer(vocab, {"b": 1}, {})
        cases = (
            (
                "added_vocab",
                _FakeTokenizer(vocab, {}, {}),
                SimpleNamespace(vocab_size=2),
            ),
            (
                "model_vocab_size",
                _FakeTokenizer(vocab, {"b": 1}, {}),
                SimpleNamespace(vocab_size=3),
            ),
            (
                "tokenizer_length",
                _FakeTokenizer(vocab, {"b": 1}, {}, length=3),
                SimpleNamespace(vocab_size=2),
            ),
        )
        for expected_check, teacher, teacher_config in cases:
            with self.subTest(check=expected_check):
                with self.assertRaisesRegex(ValueError, expected_check):
                    assert_tokenizer_compatibility(
                        student,
                        teacher,
                        SimpleNamespace(vocab_size=2),
                        teacher_config,
                    )

    def test_teacher_protocol_requires_shared_no_think_prompt(self):
        config = {
            "models": {"teacher_no_think": True},
            "data": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        protocol = validate_shared_tokenizer_protocol(config)
        self.assertEqual(protocol["tokenizer_source"], "student")
        self.assertFalse(protocol["teacher_retokenization"])
        self.assertTrue(protocol["teacher_no_think"])

        config["data"]["chat_template_kwargs"]["enable_thinking"] = True
        with self.assertRaisesRegex(ValueError, "enable_thinking must be false"):
            validate_shared_tokenizer_protocol(config)


if __name__ == "__main__":
    unittest.main()
