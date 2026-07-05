from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from ..formats import FormatConfig
from .data import CorpusDeck, Vocab


@dataclass
class PMIModel:
    """Smoothed PPMI over deck co-occurrence (word2vec-style alpha smoothing on
    the unigram distribution, minimum pair count to kill rare-pair noise)."""

    n_decks: int
    doc_freq: np.ndarray  # (vocab,) int64
    pairs: dict[tuple[int, int], float]  # (i, j) with i < j -> ppmi

    def ppmi(self, a: int, b: int) -> float:
        if a == b:
            return 0.0
        return self.pairs.get((min(a, b), max(a, b)), 0.0)

    def synergy(self, target: int, partial_idxs: np.ndarray) -> float:
        """Mean PPMI between the candidate and the distinct cards already in the deck."""
        distinct = np.unique(partial_idxs)
        if distinct.size == 0:
            return 0.0
        return float(np.mean([self.ppmi(target, int(c)) for c in distinct]))

    def top_contributors(
        self, target: int, partial_idxs: np.ndarray, k: int = 3
    ) -> list[tuple[int, float]]:
        scored = [(int(c), self.ppmi(target, int(c))) for c in np.unique(partial_idxs)]
        scored = [(c, s) for c, s in scored if s > 0]
        return sorted(scored, key=lambda cs: -cs[1])[:k]

    def save(self, path: Path) -> None:
        keys = np.array(sorted(self.pairs), dtype=np.int64)
        vals = np.array([self.pairs[tuple(k)] for k in keys], dtype=np.float32)
        np.savez_compressed(
            path,
            n_decks=self.n_decks,
            doc_freq=self.doc_freq,
            pair_keys=keys,
            pair_vals=vals,
        )

    @classmethod
    def load(cls, path: Path) -> "PMIModel":
        data = np.load(path)
        pairs = {
            (int(i), int(j)): float(v)
            for (i, j), v in zip(data["pair_keys"], data["pair_vals"])
        }
        return cls(n_decks=int(data["n_decks"]), doc_freq=data["doc_freq"], pairs=pairs)


def build_pmi(
    deck_card_sets: list[np.ndarray],
    vocab_size: int,
    min_count: int = 20,
    alpha: float = 0.75,
) -> PMIModel:
    """deck_card_sets: one array of distinct card indices per deck."""
    n_decks = len(deck_card_sets)
    doc_freq = np.zeros(vocab_size, dtype=np.int64)
    pair_counts: dict[tuple[int, int], int] = {}
    for cards in deck_card_sets:
        distinct = np.unique(cards)
        doc_freq[distinct] += 1
        for a, b in combinations(sorted(int(c) for c in distinct), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    p_alpha = doc_freq.astype(np.float64) ** alpha
    p_alpha /= p_alpha.sum() or 1.0
    pairs = {}
    for (a, b), count in pair_counts.items():
        if count < min_count:
            continue
        p_ab = count / n_decks
        value = np.log(p_ab) - np.log(p_alpha[a]) - np.log(p_alpha[b])
        if value > 0:
            pairs[(a, b)] = float(value)
    return PMIModel(n_decks=n_decks, doc_freq=doc_freq, pairs=pairs)


def corpus_card_sets(decks: list[CorpusDeck]) -> list[np.ndarray]:
    sets = []
    for deck in decks:
        cards = deck.main_idxs
        if deck.commander_idx is not None:
            cards = np.append(cards, deck.commander_idx)
        sets.append(np.unique(cards))
    return sets


def structure_reward(vocab: Vocab, fmt: FormatConfig, deck_idxs: np.ndarray) -> float:
    """Terminal structural score in [-1, 0]: distance of the land fraction from
    the format target. (Color consistency is enforced by the action mask.)"""
    if deck_idxs.size == 0:
        return -1.0
    land_frac = vocab.land[deck_idxs].sum() / deck_idxs.size
    return -abs(float(land_frac) - fmt.land_fraction_target)


def step_reward(
    pmi: PMIModel,
    vocab: Vocab,
    fmt: FormatConfig,
    partial_idxs: np.ndarray,
    action: int,
    done: bool,
) -> float:
    reward = fmt.synergy_weight * pmi.synergy(action, partial_idxs)
    if done:
        deck = np.append(partial_idxs, action)
        reward += fmt.structure_weight * structure_reward(vocab, fmt, deck)
    return reward
