"""Swap recommendations: what to cut, what to add in its place, and why
(external review 2026-07, item #3 — the last of its top three).

Cut ranking blends three orthogonal signals, each already validated in
this codebase: the model's own score for the card given the rest of the
deck (would it even suggest this card today?), total PMI synergy with the
deck (the signal that identified a zero-synergy card for the first manual
swap), and role-quota surplus (cutting from an overfull role costs less).
Adds come from the same model+personalize blend as recommend/suggest.

ponytail: pairs are greedy rank-order matches, not a learned pair value —
a swap-delta model is the marked upgrade path (rl-strategy-research.md)."""

import sqlite3
from dataclasses import dataclass

import numpy as np

from .formats import FormatConfig
from .ml.data import ROLE_ORDER, Vocab

_QUOTAS = {"ramp": 11, "draw": 10, "removal": 11, "board_wipe": 3}
_PROTECT_ROLES = ("wincon", "evasive")  # never rank scarce win conditions as cuts


@dataclass
class Cut:
    idx: int
    name: str
    badness: float  # 0-1, higher = better cut
    reason: str


def rank_cuts(
    conn: sqlite3.Connection,
    model,
    vocab: Vocab,
    fmt: FormatConfig,
    partial: np.ndarray,
    commander_idx: int | None,
    partner_idx: int | None = None,
    pmi=None,
    k: int = 10,
) -> list[Cut]:
    """Rank the deck's nonland cards as cut candidates, worst-fitting first.
    Lands are the manabase engine's job; the commander is never a cut."""
    from .ml.eval import score_state

    def card_name(i):
        row = conn.execute(
            "SELECT name FROM cards WHERE oracle_id = ?", (vocab.oracle_ids[int(i)],)
        ).fetchone()
        return row[0] if row else vocab.oracle_ids[int(i)]

    uniq = [int(i) for i in np.unique(partial) if not vocab.land[int(i)]]
    if not uniq:
        return []

    role_counts = vocab.roles[np.array(uniq)].sum(axis=0)
    wincon_total = sum(int(role_counts[ROLE_ORDER.index(r)]) for r in _PROTECT_ROLES)

    model_scores, synergy = {}, {}
    for i in uniq:
        rest = partial[partial != i]
        if partial[partial == i].size > 1:  # multi-copy: remove one copy only
            drop = np.flatnonzero(partial == i)[0]
            rest = np.delete(partial, drop)
        scores = score_state(model, vocab, fmt, rest, commander_idx, partner_idx)
        model_scores[i] = float(scores[i]) if np.isfinite(scores[i]) else -1e9
        if pmi is not None:
            synergy[i] = sum(v for _c, v in pmi.top_contributors(i, rest, k=99))

    def norm(d):
        vals = np.array([d[i] for i in uniq], dtype=np.float64)
        span = vals.max() - vals.min()
        return {i: (d[i] - vals.min()) / span if span else 0.5 for i in uniq}

    m_norm = norm(model_scores)
    s_norm = norm(synergy) if synergy else {i: 0.5 for i in uniq}

    cuts = []
    for i in uniq:
        roles = [r for r, flag in zip(ROLE_ORDER, vocab.roles[i]) if flag]
        if wincon_total <= 2 and any(r in _PROTECT_ROLES for r in roles):
            continue  # don't cut your last ways to win
        surplus = any(
            _QUOTAS.get(r) is not None and role_counts[ROLE_ORDER.index(r)] > _QUOTAS[r]
            for r in roles
        )
        badness = (
            0.5 * (1 - m_norm[i]) + 0.35 * (1 - s_norm[i]) + (0.15 if surplus else 0.0)
        )
        reasons = []
        if m_norm[i] < 0.25:
            reasons.append("model would not pick it today")
        if s_norm[i] < 0.25:
            reasons.append("little synergy with the deck")
        if surplus:
            over = [
                r
                for r in roles
                if _QUOTAS.get(r) and role_counts[ROLE_ORDER.index(r)] > _QUOTAS[r]
            ]
            reasons.append(f"{'/'.join(over)} quota already exceeded")
        cuts.append(
            Cut(
                i,
                card_name(i),
                round(badness, 3),
                "; ".join(reasons) or "weakest overall fit",
            )
        )
    return sorted(cuts, key=lambda c: -c.badness)[:k]


def recommend_swaps(
    conn: sqlite3.Connection,
    model,
    vocab: Vocab,
    fmt: FormatConfig,
    partial: np.ndarray,
    commander_idx: int | None,
    partner_idx: int | None = None,
    pmi=None,
    k: int = 5,
    extra_mask=None,
) -> list[dict]:
    """Top-k (cut, add) pairs: worst cuts matched to best adds by rank.
    Each pair carries the cut's reason and the add's synergy partners."""
    from .ml.eval import score_state
    from .ml.neighbors import blend, neighbor_frequencies

    cuts = rank_cuts(
        conn, model, vocab, fmt, partial, commander_idx, partner_idx, pmi, k=k
    )
    scores = score_state(
        model, vocab, fmt, partial, commander_idx, partner_idx, extra_mask
    )
    freqs = neighbor_frequencies(conn, vocab, fmt, partial)
    if freqs is not None:
        scores = blend(scores, freqs, 0.3)
    add_order = [i for i in np.argsort(-scores) if np.isfinite(scores[i])][:k]

    def card_name(i):
        row = conn.execute(
            "SELECT name FROM cards WHERE oracle_id = ?", (vocab.oracle_ids[int(i)],)
        ).fetchone()
        return row[0] if row else vocab.oracle_ids[int(i)]

    swaps = []
    for cut, add_idx in zip(cuts, add_order):
        entry = {
            "cut": cut.name,
            "add": card_name(add_idx),
            "reason": cut.reason,
            "add_score": round(float(scores[add_idx]), 3),
        }
        if pmi is not None:
            entry["add_synergy"] = [
                card_name(c) for c, _v in pmi.top_contributors(int(add_idx), partial)
            ]
        swaps.append(entry)
    return swaps
