import sqlite3

import numpy as np

from ..formats import FormatConfig
from .data import Vocab, load_corpus


def neighbor_frequencies(
    conn: sqlite3.Connection,
    vocab: Vocab,
    fmt: FormatConfig,
    partial_idxs: np.ndarray,
    n_neighbors: int = 50,
    near_dup: float = 0.95,
) -> np.ndarray | None:
    """Per-card frequency among the corpus decks most similar to the query deck
    (Jaccard over distinct cards). Counters the global popularity skew: cards
    common in decks *like this one* score high even if globally niche.

    Decks with Jaccard >= near_dup are excluded (the query itself, or straight
    netdeck copies, would dominate). Returns None when there is nothing to
    match against."""
    if partial_idxs.size == 0:
        return None
    decks = load_corpus(conn, vocab, fmt)
    if not decks:
        return None
    query = np.unique(partial_idxs)
    card_sets = []
    sims = np.empty(len(decks))
    for i, deck in enumerate(decks):
        cards = deck.main_idxs
        if deck.commander_idx is not None:
            cards = np.append(cards, deck.commander_idx)
        cards = np.unique(cards)
        inter = np.intersect1d(query, cards, assume_unique=True).size
        sims[i] = inter / (query.size + cards.size - inter)
        card_sets.append(cards)

    candidates = np.flatnonzero(sims < near_dup)
    if candidates.size == 0:
        return None
    top = candidates[np.argsort(-sims[candidates])][:n_neighbors]
    freq = np.zeros(len(vocab), dtype=np.float32)
    for i in top:
        freq[card_sets[i]] += 1.0
    return freq / top.size


def blend(scores: np.ndarray, freqs: np.ndarray, weight: float) -> np.ndarray:
    """Mix model scores with neighbor frequencies. Model scores are
    rank-normalized to [0, 1] over the legal (finite) pool so the two signals
    are on the same scale; masked (-inf) entries stay masked."""
    pool = np.flatnonzero(np.isfinite(scores))
    blended = np.full_like(scores, -np.inf)
    ranks = scores[pool].argsort().argsort()
    norm = ranks / max(pool.size - 1, 1)
    blended[pool] = (1.0 - weight) * norm + weight * freqs[pool]
    return blended
