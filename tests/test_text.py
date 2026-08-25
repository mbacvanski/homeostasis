"""Tests for the character stream used by the language experiments."""

import numpy as np
import pytest

from homeostasis.text import CharStream, char_class


class TestCharStream:
    def test_vocab_and_ids(self):
        s = CharStream("abcab")
        assert s.vocab == ["a", "b", "c"]
        assert list(s.ids) == [0, 1, 2, 0, 1]
        assert s.n_tokens == 3

    def test_one_hot(self):
        s = CharStream("ab")
        assert list(s.one_hot(1)) == [0.0, 1.0]

    def test_bigram_surprisal_orders_by_probability(self):
        # In "ababac", a->b occurs twice and a->c once, so b-after-a must be
        # less surprising than c-after-a.
        s = CharStream("ababac")
        surp = s.bigram_surprisal()
        i_b = 3  # position of second 'b' (follows 'a')
        i_c = 5  # position of 'c' (follows 'a')
        assert surp[i_b] < surp[i_c]

    def test_deterministic(self):
        a = CharStream("hello world")
        b = CharStream("hello world")
        assert np.array_equal(a.bigram_surprisal(), b.bigram_surprisal())

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            CharStream("")


def test_char_classes():
    assert char_class("e") == "vowel"
    assert char_class("t") == "consonant"
    assert char_class("T") == "uppercase"
    assert char_class(" ") == "whitespace"
    assert char_class(",") == "punctuation"
