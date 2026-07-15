"""Backend-agnostic inference: masking, greedy completion, and evaluation.

This module never imports torch. It drives any model exposing
`score(partial_idxs, commander_idx, state_feats, pool) -> np.ndarray` —
the torch TwoTowerQ for training-time eval, or the numpy NpTwoTowerQ that
recommend/complete use at runtime."""

import numpy as np

from ..analysis import SOURCES_NEEDED
from ..formats import FormatConfig
from .data import ROLE_ORDER, CorpusDeck, Vocab, action_mask, state_features

_WINCON_IDXS = (ROLE_ORDER.index("wincon"), ROLE_ORDER.index("evasive"))
_QUALITY_QUOTAS = (
    (ROLE_ORDER.index("ramp"), 11),
    (ROLE_ORDER.index("draw"), 10),
    (ROLE_ORDER.index("removal"), 11),
    (ROLE_ORDER.index("board_wipe"), 3),
)


def score_state(
    model,
    vocab: Vocab,
    fmt: FormatConfig,
    partial_idxs: np.ndarray,
    commander_idx: int | None,
    partner_idx: int | None = None,
    extra_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Scores over the full vocab for one state; illegal actions are -inf.
    `extra_mask` (bool, vocab-length) further restricts the pool — used for
    budget caps."""
    mask = action_mask(vocab, fmt, partial_idxs, commander_idx, partner_idx)
    if extra_mask is not None:
        mask = mask & extra_mask
    pool = np.flatnonzero(mask)
    feats = state_features(vocab, fmt, partial_idxs, commander_idx, partner_idx)
    scores = np.full(len(vocab), -np.inf, dtype=np.float32)
    scores[pool] = model.score(partial_idxs, commander_idx, feats, pool)
    return scores


def complete_deck(
    model,
    vocab: Vocab,
    fmt: FormatConfig,
    partial_idxs: np.ndarray,
    commander_idx: int | None,
    partner_idx: int | None = None,
    extra_mask: np.ndarray | None = None,
    capped_idxs: np.ndarray | None = None,
    cap: int = 0,
) -> tuple[list[int], np.ndarray]:
    """Greedily fill the deck's nonland slots (deck size minus the land-target
    slots), re-scoring after each add. Lands stay the user's job. Returns
    (added indices in order, final partial).

    `capped_idxs` cards may be added at most `cap` more times in total
    (bracket-limited Game Changers); once the budget is spent they are masked.
    Stops early if every remaining action is masked (tiny legal pools)."""
    target_nonland = fmt.deck_size - round(fmt.land_fraction_target * fmt.deck_size)
    if commander_idx is not None:
        target_nonland -= 1  # the commander occupies a nonland slot
    if partner_idx is not None:
        target_nonland -= 1  # partner also occupies a nonland slot
    partial = partial_idxs.copy()
    added: list[int] = []
    committed = sum(1 for x in (commander_idx, partner_idx) if x is not None)
    capped_set = set(map(int, capped_idxs)) if capped_idxs is not None else None
    remaining_cap = cap
    while (
        int((~vocab.land[partial]).sum()) < target_nonland
        and partial.size + committed < fmt.deck_size
    ):
        mask = extra_mask
        if capped_set is not None and remaining_cap <= 0:
            mask = (
                extra_mask.copy()
                if extra_mask is not None
                else np.ones(len(vocab), dtype=bool)
            )
            mask[list(capped_set)] = False
        scores = score_state(
            model, vocab, fmt, partial, commander_idx, partner_idx, mask
        )
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            break
        if capped_set is not None and best in capped_set:
            remaining_cap -= 1
        added.append(best)
        partial = np.append(partial, best)
    return added, partial


def structural_quality(
    model,
    decks: list[CorpusDeck],
    vocab: Vocab,
    fmt: FormatConfig,
    mask_frac: float = 0.5,
    rng: np.random.Generator | None = None,
    max_decks: int = 50,
) -> dict:
    """How well the model's greedy completions are *built*, not just how well
    they imitate. Mask mask_frac of each holdout deck's nonland cards, run
    complete_deck, and score the result on color sufficiency, role quotas,
    and win-condition presence. Composite in [0, 1], higher is better
    (weights fixed and documented in docs/rl-strategy-research.md)."""
    rng = rng or np.random.default_rng(0)
    needed_table = SOURCES_NEEDED.get(fmt.name, SOURCES_NEEDED["commander"])
    color_pens, quota_pens, win_oks = [], [], []
    for deck in decks[:max_decks]:
        nonland = deck.nonland_positions
        if nonland.size < 10:
            continue
        hidden = rng.choice(nonland, size=int(nonland.size * mask_frac), replace=False)
        keep = np.ones(deck.main_idxs.size, dtype=bool)
        keep[hidden] = False
        partial = deck.main_idxs[keep]
        _, final = complete_deck(
            model, vocab, fmt, partial, deck.commander_idx, deck.partner_idx
        )

        max_pips = vocab.pips[final].max(axis=0)
        eff_sources = vocab.src_w[final].sum(axis=0)
        color_pen = 0.0
        for c in range(5):
            if max_pips[c] > 0:
                needed = needed_table[min(int(max_pips[c]), 3)]
                color_pen = max(
                    color_pen, max(0.0, needed - float(eff_sources[c])) / needed
                )
        color_pens.append(color_pen)

        role_counts = vocab.roles[final].sum(axis=0)
        quota_pens.append(
            float(
                np.mean([max(0.0, q - role_counts[i]) / q for i, q in _QUALITY_QUOTAS])
            )
        )
        win_oks.append(float(any(role_counts[i] > 0 for i in _WINCON_IDXS)))

    if not color_pens:
        return {"decks": 0, "composite": 0.0}
    color = float(np.mean(color_pens))
    quota = float(np.mean(quota_pens))
    win = float(np.mean(win_oks))
    composite = 1.0 - (0.5 * color + 0.4 * quota + 0.1 * (1.0 - win))
    return {
        "decks": len(color_pens),
        "composite": round(composite, 4),
        "color_shortfall": round(color, 4),
        "quota_deficit": round(quota, 4),
        "wincon_rate": round(win, 4),
    }


def goldfish_quality(
    model,
    decks: list[CorpusDeck],
    vocab: Vocab,
    fmt: FormatConfig,
    statics,
    mask_frac: float = 0.5,
    rng: np.random.Generator | None = None,
    max_decks: int = 50,
    games: int = 100,
) -> dict:
    """Mean goldfish score of the model's greedy completions — the
    simulator-based counterpart of structural_quality (statics from
    goldfish.vocab_statics)."""
    from .goldfish import simulate, slice_deck

    rng = rng or np.random.default_rng(0)
    scores = []
    for deck in decks[:max_decks]:
        nonland = deck.nonland_positions
        if nonland.size < 10:
            continue
        hidden = rng.choice(nonland, size=int(nonland.size * mask_frac), replace=False)
        keep = np.ones(deck.main_idxs.size, dtype=bool)
        keep[hidden] = False
        partial = deck.main_idxs[keep]
        _, final = complete_deck(
            model, vocab, fmt, partial, deck.commander_idx, deck.partner_idx
        )
        ds = slice_deck(statics, final, deck.commander_idx)
        scores.append(simulate(ds, games=games, turns=10, seed=0)["score"])
    return {
        "decks": len(scores),
        "goldfish_score": round(float(np.mean(scores)), 4) if scores else 0.0,
    }


def recovery_at_k(
    model,
    decks: list[CorpusDeck],
    vocab: Vocab,
    fmt: FormatConfig,
    n_hide: int = 10,
    ks: tuple[int, ...] = (10, 50, 100),
    rng: np.random.Generator | None = None,
    max_decks: int = 200,
) -> dict:
    """Hide n_hide random nonland cards per held-out deck; measure how many of
    the hidden (distinct) cards appear in the model's top-k suggestions."""
    rng = rng or np.random.default_rng(0)
    totals = {k: [] for k in ks}
    structural = []
    for deck in decks[:max_decks]:
        nonland = deck.nonland_positions
        if nonland.size <= n_hide:
            continue
        hidden_pos = rng.choice(nonland, size=n_hide, replace=False)
        keep = np.ones(deck.main_idxs.size, dtype=bool)
        keep[hidden_pos] = False
        partial = deck.main_idxs[keep]
        hidden = np.unique(deck.main_idxs[hidden_pos])

        scores = score_state(
            model, vocab, fmt, partial, deck.commander_idx, deck.partner_idx
        )
        order = np.argsort(-scores)
        for k in ks:
            top = order[:k]
            totals[k].append(len(np.intersect1d(top, hidden)) / hidden.size)
        structural.append(float(np.mean(vocab.cmc[order[: ks[0]]])))
    return {
        "decks": len(totals[ks[0]]),
        "recovery": {k: round(float(np.mean(v)) * 100, 2) for k, v in totals.items()},
        "mean_top_cmc": round(float(np.mean(structural)), 2) if structural else None,
    }
