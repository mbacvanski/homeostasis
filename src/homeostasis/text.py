"""Character streams for the language experiments (stage 1: passive listening).

A :class:`CharStream` turns a text corpus into the one-active-input-per-step
format the reservoir consumes: each character is one timestep, presented as a
one-hot vector over the corpus vocabulary — the direct analog of the 2021
paper's word tokens, at character granularity.

Also provides the corpus-level statistics the stage-1 analyses need: bigram
surprisal (for the activation-tracks-surprisal test) and a coarse character
classification (for the embedding-clustering test).
"""

from __future__ import annotations

import pathlib

import numpy as np

__all__ = ["CharStream", "char_class"]


class CharStream:
    """A deterministic character-by-character one-hot stream over a corpus."""

    def __init__(self, text: str):
        if not text:
            raise ValueError("empty corpus")
        self.text = text
        self.vocab = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.ids = np.array([self.char_to_id[ch] for ch in text], dtype=np.int64)
        self._eye = np.eye(len(self.vocab))

    @classmethod
    def from_file(cls, path: str | pathlib.Path, limit: int | None = None) -> "CharStream":
        text = pathlib.Path(path).read_text()
        return cls(text[:limit] if limit else text)

    @property
    def n_tokens(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.ids)

    def one_hot(self, token_id: int) -> np.ndarray:
        return self._eye[token_id]

    def bigram_surprisal(self, alpha: float = 0.5) -> np.ndarray:
        """Per-position surprisal -log2 P(c_t | c_{t-1}) under an add-alpha
        bigram model estimated from this stream (position 0 gets the unigram
        surprisal)."""
        v = self.n_tokens
        counts = np.full((v, v), alpha)
        np.add.at(counts, (self.ids[:-1], self.ids[1:]), 1.0)
        cond = counts / counts.sum(axis=1, keepdims=True)
        uni = np.bincount(self.ids, minlength=v).astype(float)
        uni /= uni.sum()
        s = np.empty(len(self.ids))
        s[0] = -np.log2(uni[self.ids[0]])
        s[1:] = -np.log2(cond[self.ids[:-1], self.ids[1:]])
        return s


def char_class(ch: str) -> str:
    """Coarse class labels used by the embedding-clustering analysis."""
    if ch in "aeiou":
        return "vowel"
    if ch.islower():
        return "consonant"
    if ch.isupper():
        return "uppercase"
    if ch in " \n\t":
        return "whitespace"
    return "punctuation"
