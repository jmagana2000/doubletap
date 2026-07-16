"""Goldfish simulator tests, including calibration against Frank Karsten's
published land-drop tables (docs/rl-strategy-research.md §A2) — the sim
must reproduce the primary literature or it doesn't graduate to reward
duty (docs/goldfish-sim-design.md)."""

import numpy as np
import pytest

from doubletap.analysis import etb_tapped, is_land_ramp, mana_production
from doubletap.ml.goldfish import _mulligan, compile_deck, simulate

SWAMP = {
    "type_line": "Basic Land — Swamp",
    "oracle_text": "({T}: Add {B}.)",
    "cmc": 0,
    "produced_mana": ["B"],
}


def spell(mv, pips_b=1):
    cost = (
        "{" + str(mv - pips_b) + "}" + "{B}" * pips_b if mv > pips_b else "{B}" * pips_b
    )
    return {"type_line": "Sorcery", "cmc": mv, "mana_cost": cost, "oracle_text": ""}


# --- card statics -------------------------------------------------------------


def test_mana_production_parses_amounts():
    sol_ring = {"type_line": "Artifact", "oracle_text": "{T}: Add {C}{C}.", "cmc": 1}
    assert mana_production(sol_ring) == ([], 2)
    signet = {
        "type_line": "Artifact",
        "oracle_text": "{1}, {T}: Add {W}{U}.",
        "produced_mana": ["W", "U"],
        "cmc": 2,
    }
    assert mana_production(signet) == (["W", "U"], 2)
    any_color = {
        "type_line": "Creature — Bird",
        "oracle_text": "{T}: Add one mana of any color.",
        "produced_mana": ["B", "G", "R", "U", "W"],
        "cmc": 1,
    }
    colors, amount = mana_production(any_color)
    assert len(colors) == 5 and amount == 1
    assert mana_production({"type_line": "Instant", "oracle_text": "", "cmc": 1}) == (
        [],
        0,
    )


def test_etb_tapped_detection():
    guildgate = {
        "type_line": "Land — Gate",
        "oracle_text": "This land enters the battlefield tapped.\n{T}: Add {W} or {U}.",
    }
    assert etb_tapped(guildgate)
    checkland = {
        "type_line": "Land",
        "oracle_text": "This land enters tapped unless you control a Plains.",
    }
    assert not etb_tapped(checkland)  # conditional -> optimistic untapped (v1)
    assert not etb_tapped(SWAMP)


def test_is_land_ramp():
    cultivate = {
        "type_line": "Sorcery",
        "oracle_text": "Search your library for up to two basic land cards, reveal"
        " them, put one onto the battlefield tapped and the other into your hand.",
    }
    assert is_land_ramp(cultivate)
    tutor = {
        "type_line": "Sorcery",
        "oracle_text": "Search your library for a card and put it into your hand.",
    }
    assert not is_land_ramp(tutor)


# --- mulligan model -----------------------------------------------------------


def test_mulligan_rejects_bad_hands():
    # 60-card deck stacked so the top 7 have zero lands: must mulligan
    ds = compile_deck([(spell(2), 36), (SWAMP, 24)])
    order = np.concatenate([np.arange(36), np.arange(36, 60)])  # spells first
    rng = np.random.default_rng(0)
    hand, library, mulls = _mulligan(ds, order, rng, "karsten2017")
    assert mulls >= 1
    assert len(hand) + len(library) == 60
    assert 2 <= int(ds.land[hand].sum()) <= 5 or len(hand) <= 5


def test_london_bottoms_to_size():
    ds = compile_deck([(spell(2), 36), (SWAMP, 24)])
    rng = np.random.default_rng(1)
    for _ in range(20):
        order = rng.permutation(60)
        hand, library, mulls = _mulligan(ds, order, rng, "london")
        assert len(hand) == 7 - mulls
        assert len(hand) + len(library) == 60


# --- simulation sanity ---------------------------------------------------------


def test_simulate_deterministic_and_bounded():
    ds = compile_deck([(spell(2), 20), (spell(3), 16), (SWAMP, 24)])
    a = simulate(ds, games=50, seed=7)
    b = simulate(ds, games=50, seed=7)
    assert a == b  # same seed, same result
    assert 0.0 <= a["score"] <= 1.0
    assert a["mana_efficiency"] > 0.3  # a sane deck spends mana


def test_commander_gets_cast():
    commander = {
        "type_line": "Legendary Creature — Human",
        "cmc": 3,
        "mana_cost": "{2}{B}",
        "oracle_text": "",
    }
    ds = compile_deck([(spell(2), 30), (SWAMP, 30)], commander)
    r = simulate(ds, games=100, seed=0)
    assert r["commander_on_curve"] is not None
    assert r["commander_on_curve"] > 0.3  # 30 lands: usually castable turn 3


def test_ramp_spell_accelerates():
    cultivate = {
        "type_line": "Sorcery",
        "cmc": 3,
        "mana_cost": "{2}{B}",
        "oracle_text": "Search your library for up to two basic land cards,"
        " put one onto the battlefield tapped.",
    }
    base = compile_deck([(spell(3), 36), (SWAMP, 24)])
    ramped = compile_deck([(spell(3), 28), (cultivate, 8), (SWAMP, 24)])
    r_base = simulate(base, games=300, seed=3)
    r_ramped = simulate(ramped, games=300, seed=3)
    # ramp raises land counts over the game -> more turn-4 land drops
    assert r_ramped["land4_rate"] > r_base["land4_rate"]


# --- calibration against Karsten's published tables ---------------------------
# 2017 article, karsten2017 mulligan mode. Tolerance ±3 points at 3000
# games (binomial SE ~0.6pt) — slack for modeling differences in the
# scry/bottom heuristics, tight enough to catch real bugs.


def _land_deck(n_lands, n_cards):
    return compile_deck([(SWAMP, n_lands), (spell(3), n_cards - n_lands)])


@pytest.mark.parametrize(
    "lands,size,on_play,published",
    [
        (25, 60, True, 0.904),  # 25/60: P(3 lands by T3) play
        (25, 60, False, 0.946),  # and on the draw
        (17, 40, True, 0.916),  # Limited 17/40 play
        (24, 60, True, 0.887),  # 24/60 play
    ],
)
def test_calibration_land_drops_vs_karsten(lands, size, on_play, published):
    ds = _land_deck(lands, size)
    r = simulate(ds, games=3000, turns=4, on_play=on_play, mode="karsten2017", seed=42)
    assert r["land3_rate"] == pytest.approx(published, abs=0.03), (
        f"{lands}/{size} on_play={on_play}: sim {r['land3_rate']} vs Karsten {published}"
    )


# --- Stage 2: reward shaping ---------------------------------------------------


def test_shaper_delta_and_vocab_statics(loaded_conn):
    from doubletap.formats import COMMANDER
    from doubletap.ml.data import build_vocab
    from doubletap.ml.goldfish import Shaper, slice_deck, vocab_statics
    from doubletap.names import lookup

    vocab = build_vocab(loaded_conn, COMMANDER)
    vs = vocab_statics(loaded_conn, vocab)
    assert vs.n == len(vocab)

    def idx(name):
        return vocab.index[lookup(loaded_conn, name)[0].oracle_id]

    # statics line up with the vocab
    assert vs.land[idx("Swamp")] and not vs.land[idx("Sol Ring")]
    assert vs.prod_amount[idx("Sol Ring")] == 2

    shaper = Shaper(vs, weight=1.0, games=4, turns=6)
    swamps = np.array([idx("Swamp")] * 20 + [idx("Juzám Djinn")] * 5, dtype=np.int64)
    assert shaper.phi(np.empty(0, dtype=np.int64), None) == 0.0  # tiny partial
    phi = shaper.phi(swamps, None)
    assert 0.0 < phi <= 1.0
    delta = shaper.delta(swamps, idx("Sol Ring"), None)
    assert isinstance(delta, float)

    # deterministic: same inputs, same phi
    assert shaper.phi(swamps, None) == phi
    ds = slice_deck(vs, swamps, idx("Juzám Djinn"))
    assert ds.commander_mv == 4


def test_goldfish_reranker_promotes_castable(loaded_conn):
    """With a mono-black mana base, the reranker should prefer a castable
    black spell over an equally-scored card the mana can't cast."""
    from doubletap.formats import COMMANDER
    from doubletap.ml.data import build_vocab
    from doubletap.ml.goldfish import vocab_statics
    from doubletap.ml.policy import make_goldfish_reranker
    from doubletap.names import lookup

    vocab = build_vocab(loaded_conn, COMMANDER)
    vs = vocab_statics(loaded_conn, vocab)

    def idx(name):
        return vocab.index[lookup(loaded_conn, name)[0].oracle_id]

    partial = np.array([idx("Swamp")] * 24 + [idx("Juzám Djinn")] * 6, dtype=np.int64)
    scores = np.full(len(vocab), -np.inf, dtype=np.float32)
    rats, once = idx("Relentless Rats"), idx("Once Upon a Time")
    scores[once] = 1.001  # model narrowly prefers the green (uncastable) card
    scores[rats] = 1.000

    rerank = make_goldfish_reranker(vs, top_m=10, weight=0.5, games=8, turns=8)
    new = rerank(scores, partial, None)
    assert new[rats] > new[once]  # goldfish flips it: Rats are castable

    # land-less partial: reranker must no-op
    spells_only = np.array([idx("Juzám Djinn")] * 30, dtype=np.int64)
    same = rerank(scores, spells_only, None)
    assert np.array_equal(same, scores)
