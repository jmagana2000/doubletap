import numpy as np
import torch

from ..formats import FormatConfig
from .data import CorpusDeck, Vocab, action_mask, state_features
from .model import TwoTowerQ


def score_state(
    model: TwoTowerQ,
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
    with torch.no_grad():
        bag = torch.from_numpy(partial_idxs)
        offsets = torch.zeros(1, dtype=torch.int64)
        commander = torch.tensor([commander_idx if commander_idx is not None else -1])
        feats = torch.from_numpy(
            state_features(vocab, fmt, partial_idxs, commander_idx, partner_idx)
        ).unsqueeze(0)
        state = model.state_repr(bag, offsets, commander, feats)[0]
        mask = action_mask(vocab, fmt, partial_idxs, commander_idx, partner_idx)
        if extra_mask is not None:
            mask = mask & extra_mask
        pool = np.flatnonzero(mask)
        scores = np.full(len(vocab), -np.inf, dtype=np.float32)
        scores[pool] = model.score_pool(state, torch.from_numpy(pool)).numpy()
    return scores


def complete_deck(
    model: TwoTowerQ,
    vocab: Vocab,
    fmt: FormatConfig,
    partial_idxs: np.ndarray,
    commander_idx: int | None,
    partner_idx: int | None = None,
    extra_mask: np.ndarray | None = None,
) -> tuple[list[int], np.ndarray]:
    """Greedily fill the deck's nonland slots (deck size minus the land-target
    slots), re-scoring after each add. Lands stay the user's job. Returns
    (added indices in order, final partial).

    Stops early if every remaining action is masked (tiny legal pools)."""
    target_nonland = fmt.deck_size - round(fmt.land_fraction_target * fmt.deck_size)
    if commander_idx is not None:
        target_nonland -= 1  # the commander occupies a nonland slot
    if partner_idx is not None:
        target_nonland -= 1  # partner also occupies a nonland slot
    partial = partial_idxs.copy()
    added: list[int] = []
    committed = sum(1 for x in (commander_idx, partner_idx) if x is not None)
    while (
        int((~vocab.land[partial]).sum()) < target_nonland
        and partial.size + committed < fmt.deck_size
    ):
        scores = score_state(
            model, vocab, fmt, partial, commander_idx, partner_idx, extra_mask
        )
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            break
        added.append(best)
        partial = np.append(partial, best)
    return added, partial


def recovery_at_k(
    model: TwoTowerQ,
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
    model.eval()
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
