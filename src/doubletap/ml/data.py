import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from ..analysis import (
    classify,
    color_pip_counts,
    effective_lands,
    is_cheap_draw_ramp,
    source_weights,
)
from ..formats import FormatConfig, is_basic_land, is_land, named_copy_cap

COLOR_ORDER = "WUBRG"
# functional roles carried as model features and used by the quota reward
ROLE_ORDER = (
    "ramp",
    "draw",
    "removal",
    "removal_instant",
    "board_wipe",
    "tutor",
    "wincon",
    "evasive",
)
TYPE_ORDER = [
    "Creature",
    "Instant",
    "Sorcery",
    "Artifact",
    "Enchantment",
    "Planeswalker",
    "Land",
    "Battle",
]
RARITY_ORDER = ["common", "uncommon", "rare", "mythic"]
BASE_FEATURE_DIM = 1 + len(TYPE_ORDER) + 5 + 5 + 1 + len(RARITY_ORDER) + 1 + 1  # 26
BASE_STATE_DIM = 9 + 1 + 1 + 5  # curve histogram, land fraction, progress, identity


def feature_dim(fmt: FormatConfig) -> int:
    """Card-feature width; +5 WUBRG fractional-source dims (Karsten weights)
    where the format's keep-bar accepted mana-math features."""
    return BASE_FEATURE_DIM + (5 if fmt.pip_state else 0)


def state_dim(fmt: FormatConfig) -> int:
    """State-feature width; +5 WUBRG pip-demand dims where the format's
    keep-bar accepted them (commander yes, modern no — see
    docs/rl-strategy-research.md)."""
    return BASE_STATE_DIM + (5 if fmt.pip_state else 0)


def _identity_bits(color_identity: list[str]) -> int:
    return sum(1 << COLOR_ORDER.index(c) for c in color_identity)


def card_features(card: dict, fmt: FormatConfig) -> np.ndarray:
    feats = np.zeros(feature_dim(fmt), dtype=np.float32)
    feats[0] = min(float(card.get("cmc", 0.0)), 8.0) / 8.0
    type_line = card["type_line"]
    for i, t in enumerate(TYPE_ORDER):
        feats[1 + i] = float(t in type_line)
    for i, c in enumerate(COLOR_ORDER):
        feats[9 + i] = float(c in (card.get("colors") or []))
        feats[14 + i] = float(c in (card.get("color_identity") or []))
    feats[19] = float("Legendary" in type_line)
    rarity = card.get("rarity", "common")
    rarity_idx = RARITY_ORDER.index(rarity) if rarity in RARITY_ORDER else 2
    feats[20 + rarity_idx] = 1.0
    rank = card.get("edhrec_rank")
    feats[24] = 1.0 - min(np.log1p(rank) / np.log1p(50_000), 1.0) if rank else 0.0
    feats[25] = float(is_basic_land(card))
    if fmt.pip_state:
        weights = source_weights(card)
        feats[26:31] = [weights.get(c, 0.0) for c in COLOR_ORDER]
    return feats


@dataclass
class Vocab:
    """All format-legal cards, indexed. Everything the model and masks need,
    as parallel arrays."""

    oracle_ids: list[str]
    index: dict[str, int]
    features: np.ndarray  # (n, feature_dim(fmt)) float32
    cmc: np.ndarray  # (n,) float32
    identity_bits: np.ndarray  # (n,) uint8, WUBRG bitmask
    land: np.ndarray  # (n,) bool
    basic: np.ndarray  # (n,) bool
    copy_cap: np.ndarray  # (n,) int32 — card-text copy cap; 0 = format limit
    # strategy arrays (docs/rl-strategy-research.md): functional roles,
    # Karsten effective-land and fractional-source weights, colored pips
    roles: np.ndarray  # (n, 8) bool, ROLE_ORDER
    eff_land: np.ndarray  # (n,) float32 — lands 1.0, spell//land MDFCs 0.38/0.74
    cheap_dr: np.ndarray  # (n,) bool — MV<=2 draw/ramp (Karsten regression term)
    src_w: np.ndarray  # (n, 5) float32 — fractional colored sources, WUBRG
    pips: np.ndarray  # (n, 5) int8 — colored pips in cost, WUBRG

    def __len__(self) -> int:
        return len(self.oracle_ids)


def build_vocab(conn: sqlite3.Connection, fmt: FormatConfig) -> Vocab:
    oracle_ids, feats, cmc, bits, land, basic, any_num = [], [], [], [], [], [], []
    roles, eff_land, cheap_dr, src_w, pips = [], [], [], [], []
    for (raw,) in conn.execute("SELECT json FROM cards ORDER BY oracle_id"):
        card = json.loads(raw)
        if card["legalities"].get(fmt.legality_key) != "legal":
            continue
        oracle_ids.append(card["oracle_id"])
        feats.append(card_features(card, fmt))
        cmc.append(min(float(card.get("cmc", 0.0)), 8.0))
        bits.append(_identity_bits(card.get("color_identity") or []))
        land.append(is_land(card))
        basic.append(is_basic_land(card))
        any_num.append(named_copy_cap(card) or 0)
        card_roles = classify(card)
        roles.append([r in card_roles for r in ROLE_ORDER])
        eff_land.append(effective_lands(card))
        cheap_dr.append(is_cheap_draw_ramp(card))
        weights = source_weights(card)
        src_w.append([weights.get(c, 0.0) for c in COLOR_ORDER])
        pip_counts = color_pip_counts(card)
        pips.append([pip_counts.get(c, 0) for c in COLOR_ORDER])
    return Vocab(
        oracle_ids=oracle_ids,
        index={oid: i for i, oid in enumerate(oracle_ids)},
        features=np.array(feats, dtype=np.float32),
        cmc=np.array(cmc, dtype=np.float32),
        identity_bits=np.array(bits, dtype=np.uint8),
        land=np.array(land, dtype=bool),
        basic=np.array(basic, dtype=bool),
        copy_cap=np.array(any_num, dtype=np.int32),
        roles=np.array(roles, dtype=bool),
        eff_land=np.array(eff_land, dtype=np.float32),
        cheap_dr=np.array(cheap_dr, dtype=bool),
        src_w=np.array(src_w, dtype=np.float32),
        pips=np.array(pips, dtype=np.int8),
    )


def action_mask(
    vocab: Vocab,
    fmt: FormatConfig,
    partial_idxs: np.ndarray,
    commander_idx: int | None,
    partner_idx: int | None = None,
) -> np.ndarray:
    """Legal next-card mask over the vocab: nonland only (mana bases are handled
    by the gap report, not the model), copy limits, commander color identity.
    For partner commanders, the combined identity of both is used."""
    mask = ~vocab.land
    if partial_idxs.size:
        idxs, counts = np.unique(partial_idxs, return_counts=True)
        caps = np.where(vocab.copy_cap[idxs] > 0, vocab.copy_cap[idxs], fmt.copy_limit)
        mask[idxs[counts >= caps]] = False
    if commander_idx is not None or partner_idx is not None:
        combined_bits = np.uint8(0)
        if commander_idx is not None:
            combined_bits = combined_bits | vocab.identity_bits[commander_idx]
        if partner_idx is not None:
            combined_bits = combined_bits | vocab.identity_bits[partner_idx]
        mask &= (vocab.identity_bits & ~combined_bits) == 0
        if commander_idx is not None:
            mask[commander_idx] = False
        if partner_idx is not None:
            mask[partner_idx] = False
    return mask


def state_features(
    vocab: Vocab,
    fmt: FormatConfig,
    partial_idxs: np.ndarray,
    commander_idx: int | None,
    partner_idx: int | None = None,
) -> np.ndarray:
    feats = np.zeros(state_dim(fmt), dtype=np.float32)
    if partial_idxs.size:
        nonland = partial_idxs[~vocab.land[partial_idxs]]
        hist, _ = np.histogram(vocab.cmc[nonland], bins=np.arange(10) - 0.5)
        feats[0:9] = hist / fmt.deck_size
        feats[9] = vocab.land[partial_idxs].sum() / fmt.deck_size
        feats[10] = partial_idxs.size / fmt.deck_size
        if fmt.pip_state:
            # WUBRG pip demand of the partial: what the mana base must support
            feats[16:21] = vocab.pips[nonland].sum(axis=0) / fmt.deck_size
    if commander_idx is not None or partner_idx is not None:
        bits = 0
        if commander_idx is not None:
            bits |= int(vocab.identity_bits[commander_idx])
        if partner_idx is not None:
            bits |= int(vocab.identity_bits[partner_idx])
        feats[11:16] = [(bits >> i) & 1 for i in range(5)]
    return feats


@dataclass
class CorpusDeck:
    deck_id: int
    commander_idx: int | None
    main_idxs: np.ndarray  # all main-deck cards expanded by qty (commanders excluded)
    nonland_positions: np.ndarray  # positions in main_idxs holding nonland cards
    partner_idx: int | None = None


def load_corpus(
    conn: sqlite3.Connection, vocab: Vocab, fmt: FormatConfig
) -> list[CorpusDeck]:
    decks = []
    rows = conn.execute(
        "SELECT deck_id, commander_oracle_id, partner_oracle_id FROM decks WHERE status = 'parsed' AND format = ?",
        (fmt.name,),
    ).fetchall()
    for deck_id, commander_oid, partner_oid in rows:
        commander_idx = vocab.index.get(commander_oid) if commander_oid else None
        partner_idx = vocab.index.get(partner_oid) if partner_oid else None
        commander_oids = {o for o in (commander_oid, partner_oid) if o}
        main = []
        ok = True
        for oid, qty in conn.execute(
            "SELECT oracle_id, qty FROM deck_cards WHERE deck_id = ?", (deck_id,)
        ):
            if oid in commander_oids:
                continue
            idx = vocab.index.get(oid)
            if idx is None:  # legality changed since the crawl
                ok = False
                break
            main.extend([idx] * qty)
        if not ok or not main:
            continue
        main_idxs = np.array(main, dtype=np.int64)
        decks.append(
            CorpusDeck(
                deck_id=deck_id,
                commander_idx=commander_idx,
                main_idxs=main_idxs,
                nonland_positions=np.flatnonzero(~vocab.land[main_idxs]),
                partner_idx=partner_idx,
            )
        )
    return decks


@dataclass
class Batch:
    """A batch of (partial deck -> next nonland card) transitions.

    Partial decks are variable-length, stored flat with offsets (EmbeddingBag
    layout). `done` marks transitions whose next state completes the deck."""

    bag: np.ndarray  # (sum of partial sizes,) int64
    offsets: np.ndarray  # (batch,) int64
    state_feats: np.ndarray  # (batch, STATE_DIM) float32
    next_state_feats: np.ndarray  # (batch, STATE_DIM) float32
    action: np.ndarray  # (batch,) int64
    commander: np.ndarray  # (batch,) int64, -1 when absent
    done: np.ndarray  # (batch,) float32


def sample_batch(
    decks: list[CorpusDeck],
    vocab: Vocab,
    fmt: FormatConfig,
    batch_size: int,
    rng: np.random.Generator,
    with_next: bool = False,  # next-state features are only needed for TD (CQL)
) -> Batch:
    bags, offsets, sf, nsf, actions, commanders, dones = [], [], [], [], [], [], []
    offset = 0
    while len(actions) < batch_size:
        deck = decks[rng.integers(len(decks))]
        n = deck.main_idxs.size
        perm = rng.permutation(n)
        # prefix length k, then the target is a uniformly chosen remaining nonland
        k = int(rng.integers(0, n))
        remaining_nonland = perm[k:][~vocab.land[deck.main_idxs[perm[k:]]]]
        if remaining_nonland.size == 0:
            continue
        target_pos = remaining_nonland[rng.integers(remaining_nonland.size)]
        partial = deck.main_idxs[perm[:k]]
        target = int(deck.main_idxs[target_pos])

        bags.append(partial)
        offsets.append(offset)
        offset += partial.size
        sf.append(
            state_features(vocab, fmt, partial, deck.commander_idx, deck.partner_idx)
        )
        if with_next:
            next_partial = np.append(partial, target)
            nsf.append(
                state_features(
                    vocab, fmt, next_partial, deck.commander_idx, deck.partner_idx
                )
            )
        actions.append(target)
        commanders.append(deck.commander_idx if deck.commander_idx is not None else -1)
        dones.append(1.0 if k + 1 == n else 0.0)
    return Batch(
        bag=np.concatenate(bags) if bags else np.empty(0, dtype=np.int64),
        offsets=np.array(offsets, dtype=np.int64),
        state_feats=np.stack(sf).astype(np.float32),
        next_state_feats=(
            np.stack(nsf).astype(np.float32)
            if with_next
            else np.zeros((batch_size, state_dim(fmt)), dtype=np.float32)
        ),
        action=np.array(actions, dtype=np.int64),
        commander=np.array(commanders, dtype=np.int64),
        done=np.array(dones, dtype=np.float32),
    )
