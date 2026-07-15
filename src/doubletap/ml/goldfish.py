"""Goldfish (solitaire) simulator: how a deck functions under real draws.

No opponent, no combat — shuffle, mulligan, play lands, cast what the mana
allows, measure. The signal comes from the game's mathematics rather than
the training corpus, which is what makes it usable as a NON-circular
reward (docs/goldfish-sim-design.md, docs/rl-strategy-research.md
§Results). Torch-free.

Calibration: with Karsten's 2017 mulligan model ("karsten2017" mode, scry
after mulligans) this simulator must reproduce his published land-drop
tables within tolerance — pinned by tests."""

from dataclasses import dataclass

import numpy as np

from ..analysis import (
    _is_land,
    color_pip_counts,
    etb_tapped,
    is_land_ramp,
    mana_production,
)

COLOR_ORDER = "WUBRG"


@dataclass
class DeckStatic:
    """Per-slot parallel arrays (deck expanded by quantity), plus the
    commander's cost. Compile once, simulate many."""

    n: int
    land: np.ndarray  # bool
    tapped: np.ndarray  # bool — unconditionally ETB tapped
    prod_colors: np.ndarray  # (n,5) bool — colors the card can produce
    prod_amount: np.ndarray  # int8 — mana per activation (0 = not a producer)
    is_creature: np.ndarray  # bool — producers get summoning sickness
    mv: np.ndarray  # int16
    generic: np.ndarray  # int16 — generic portion of cost
    pips: np.ndarray  # (n,5) int8
    ramp_spell: np.ndarray  # bool — puts a land onto the battlefield
    commander_mv: int = 0
    commander_generic: int = 0
    commander_pips: np.ndarray | None = None  # (5,)


def _card_statics(card: dict) -> dict:
    pips = color_pip_counts(card)
    pip_arr = [pips.get(c, 0) for c in COLOR_ORDER]
    colors, amount = mana_production(card)
    mv = int(card.get("cmc") or 0)
    return {
        "land": _is_land(card),
        "tapped": etb_tapped(card),
        "prod_colors": [c in colors for c in COLOR_ORDER],
        "prod_amount": amount,
        "is_creature": "Creature" in card.get("type_line", ""),
        "mv": mv,
        "generic": max(0, mv - sum(pip_arr)),
        "pips": pip_arr,
        "ramp_spell": is_land_ramp(card),
    }


def compile_deck(
    cards_with_qty: list[tuple[dict, int]], commander: dict | None = None
) -> DeckStatic:
    """cards_with_qty: (raw Scryfall card dict, quantity) for the 99/60."""
    rows = []
    for card, qty in cards_with_qty:
        rows.extend([_card_statics(card)] * qty)
    ds = DeckStatic(
        n=len(rows),
        land=np.array([r["land"] for r in rows], dtype=bool),
        tapped=np.array([r["tapped"] for r in rows], dtype=bool),
        prod_colors=np.array([r["prod_colors"] for r in rows], dtype=bool),
        prod_amount=np.array([r["prod_amount"] for r in rows], dtype=np.int8),
        is_creature=np.array([r["is_creature"] for r in rows], dtype=bool),
        mv=np.array([r["mv"] for r in rows], dtype=np.int16),
        generic=np.array([r["generic"] for r in rows], dtype=np.int16),
        pips=np.array([r["pips"] for r in rows], dtype=np.int8),
        ramp_spell=np.array([r["ramp_spell"] for r in rows], dtype=bool),
    )
    if commander is not None:
        s = _card_statics(commander)
        ds.commander_mv = s["mv"]
        ds.commander_generic = s["generic"]
        ds.commander_pips = np.array(s["pips"], dtype=np.int8)
    return ds


def _mulligan(ds: DeckStatic, order: np.ndarray, rng, mode: str):
    """Karsten's canonical keep/mull rule (docs/rl-strategy-research.md §A2).
    Returns (hand indices, remaining library order, mulligan count).

    - "karsten2017": Vancouver — draw (7 − mulls) cards; mull any 7-carder
      with 0/1/6/7 lands, 6-carder with 0/1/5/6, 5-carder with 0/5, keep
      any 4; after any mulligan, scry (land stays, spell bottoms). This is
      the exact model behind his published tables — calibration mode.
    - "london": always draw 7 and judge with the 7-card rule, then bottom
      the excess (extra lands above 3 first, then the priciest spells)."""
    bad = {7: {0, 1, 6, 7}, 6: {0, 1, 5, 6}, 5: {0, 5}, 4: set()}
    for mull in range(4):  # sizes 7, 6, 5, 4 (always keep 4)
        size = 7 - mull
        if mode == "london":
            seven, rest = order[:7], order[7:]
            if size > 4 and int(ds.land[seven].sum()) in bad[7]:
                order = rng.permutation(order)
                continue
            hand, bottomed = _london_bottom(ds, seven, size)
            return hand, np.concatenate([rest, bottomed]), mull
        hand, rest = order[:size], order[size:]
        if int(ds.land[hand].sum()) in bad[size]:
            order = rng.permutation(order)
            continue
        if mull > 0:
            rest = _karsten_scry(ds, hand, rest)
        return hand, rest, mull
    return order[:4], order[4:], 3


def _london_bottom(ds: DeckStatic, seven: np.ndarray, size: int):
    """Bottom 7-size cards: excess lands above 3 first, then priciest spells."""
    hand = list(seven)
    bottomed = []
    while len(hand) > size:
        lands = [i for i in hand if ds.land[i]]
        if len(lands) > 3:
            pick = lands[-1]
        else:
            spells = [i for i in hand if not ds.land[i]]
            pick = max(spells, key=lambda i: ds.mv[i]) if spells else hand[-1]
        hand.remove(pick)
        bottomed.append(pick)
    return np.array(hand, dtype=seven.dtype), np.array(bottomed, dtype=seven.dtype)


def _karsten_scry(ds: DeckStatic, hand: np.ndarray, rest: np.ndarray):
    """His 2017 modeling: after a mulligan, 'scry a land to the top and a
    spell to the bottom' — keep the top card if it's a land, else bottom it."""
    if rest.size and not ds.land[rest[0]]:
        rest = np.concatenate([rest[1:], rest[:1]])
    return rest


def simulate(
    ds: DeckStatic,
    games: int = 200,
    turns: int = 10,
    on_play: bool = True,
    mode: str = "london",
    seed: int = 0,
) -> dict:
    """Run goldfish games; return per-deck metrics.

    Composite score weights (documented, fixed): 0.4 mana efficiency +
    0.3 curve-out rate + 0.2 (1 - dead-turn rate) + 0.1 commander on curve.
    """
    rng = np.random.default_rng(seed)
    m = {
        "land3": 0,
        "land4": 0,
        "spent": 0.0,
        "avail": 0.0,
        "curve_turns": 0,
        "curve_hits": 0,
        "dead": 0,
        "cmd_on_curve": 0,
        "mulligans": 0,
    }
    for _ in range(games):
        order = rng.permutation(ds.n)
        hand, library, mulls = _mulligan(ds, order, rng, mode)
        m["mulligans"] += mulls
        hand = list(hand)
        library = list(library)
        lands_played = 0
        battlefield: list[int] = []  # producer slot indices on the battlefield
        entered_turn: dict[int, int] = {}
        commander_cast_turn = None

        for turn in range(1, turns + 1):
            if not (turn == 1 and on_play) and library:
                hand.append(library.pop(0))

            # play a land: untapped preferred, then highest production
            lands_in_hand = [i for i in hand if ds.land[i]]
            if lands_in_hand:
                pick = min(
                    lands_in_hand,
                    key=lambda i: (bool(ds.tapped[i]), -int(ds.prod_amount[i])),
                )
                hand.remove(pick)
                battlefield.append(pick)
                entered_turn[pick] = turn
                lands_played += 1
            if turn == 3:
                m["land3"] += lands_played >= 3
            if turn == 4:
                m["land4"] += lands_played >= 4

            # available mana this turn
            usable = [
                i
                for i in battlefield
                if ds.prod_amount[i] > 0
                and not (ds.land[i] and ds.tapped[i] and entered_turn[i] == turn)
                and not (ds.is_creature[i] and entered_turn[i] == turn)
            ]
            total = int(sum(ds.prod_amount[i] for i in usable))
            color_cap = np.zeros(5, dtype=np.int32)
            for i in usable:
                color_cap += ds.prod_colors[i] * int(ds.prod_amount[i])
            avail = total
            m["avail"] += avail
            spent = 0

            def castable(mv, pips):
                if mv > total - spent:
                    return False
                # ponytail: per-color capacity check, not full matching —
                # flexible sources can be double-counted on 4+ color costs
                return all(pips[c] <= color_cap[c] for c in range(5))

            # commander first among affordable big plays, then greedy MV desc
            if (
                commander_cast_turn is None
                and ds.commander_pips is not None
                and castable(ds.commander_mv, ds.commander_pips)
            ):
                spent += ds.commander_mv
                commander_cast_turn = turn
                if turn <= max(ds.commander_mv, 1):
                    m["cmd_on_curve"] += 1

            for i in sorted(
                [i for i in hand if not ds.land[i]],
                key=lambda i: -int(ds.mv[i]),
            ):
                if ds.mv[i] == 0 or not castable(ds.mv[i], ds.pips[i]):
                    continue
                spent += int(ds.mv[i])
                hand.remove(i)
                if ds.prod_amount[i] > 0:
                    battlefield.append(i)
                    entered_turn[i] = turn
                if ds.ramp_spell[i]:
                    land_pos = next(
                        (
                            k
                            for k in range(len(library) - 1, -1, -1)
                            if ds.land[library[k]]
                        ),
                        None,
                    )
                    if land_pos is not None:
                        land = library.pop(land_pos)
                        battlefield.append(land)
                        entered_turn[land] = turn  # enters tapped (Cultivate)
                        lands_played += 1
                if 2 <= turn <= 6 and ds.mv[i] == turn:
                    m["curve_hits"] += 1

            if 2 <= turn <= 6:
                m["curve_turns"] += 1
            m["spent"] += spent
            if spent == 0 and avail > 0 and any(not ds.land[i] for i in hand):
                m["dead"] += 1

    g = games
    efficiency = m["spent"] / m["avail"] if m["avail"] else 0.0
    curve_rate = m["curve_hits"] / m["curve_turns"] if m["curve_turns"] else 0.0
    dead_rate = m["dead"] / (g * turns)
    cmd_rate = m["cmd_on_curve"] / g if ds.commander_pips is not None else None
    score = (
        0.4 * efficiency
        + 0.3 * min(curve_rate, 1.0)
        + 0.2 * (1.0 - dead_rate)
        + 0.1 * (cmd_rate if cmd_rate is not None else 1.0)
    )
    return {
        "games": g,
        "score": round(float(score), 4),
        "mana_efficiency": round(float(efficiency), 4),
        "curve_out_rate": round(float(curve_rate), 4),
        "dead_turn_rate": round(float(dead_rate), 4),
        "commander_on_curve": round(float(cmd_rate), 4)
        if cmd_rate is not None
        else None,
        "land3_rate": round(m["land3"] / g, 4),
        "land4_rate": round(m["land4"] / g, 4),
        "avg_mulligans": round(m["mulligans"] / g, 3),
    }


# --- Stage 2: reward shaping (docs/goldfish-sim-design.md) ---------------------


def vocab_statics(conn, vocab) -> DeckStatic:
    """Card statics aligned to a Vocab's indices, compiled once per training
    run. Slicing these arrays by deck indices yields a DeckStatic without
    touching card dicts again."""
    import json as _json

    rows: list[dict | None] = [None] * len(vocab)
    for oid, raw in conn.execute("SELECT oracle_id, json FROM cards"):
        i = vocab.index.get(oid)
        if i is not None:
            rows[i] = _card_statics(_json.loads(raw))
    assert all(r is not None for r in rows)
    return DeckStatic(
        n=len(rows),
        land=np.array([r["land"] for r in rows], dtype=bool),
        tapped=np.array([r["tapped"] for r in rows], dtype=bool),
        prod_colors=np.array([r["prod_colors"] for r in rows], dtype=bool),
        prod_amount=np.array([r["prod_amount"] for r in rows], dtype=np.int8),
        is_creature=np.array([r["is_creature"] for r in rows], dtype=bool),
        mv=np.array([r["mv"] for r in rows], dtype=np.int16),
        generic=np.array([r["generic"] for r in rows], dtype=np.int16),
        pips=np.array([r["pips"] for r in rows], dtype=np.int8),
        ramp_spell=np.array([r["ramp_spell"] for r in rows], dtype=bool),
    )


def slice_deck(
    vs: DeckStatic, idxs: np.ndarray, commander_idx: int | None = None
) -> DeckStatic:
    """A concrete deck as a view of vocab-level statics."""
    ds = DeckStatic(
        n=int(idxs.size),
        land=vs.land[idxs],
        tapped=vs.tapped[idxs],
        prod_colors=vs.prod_colors[idxs],
        prod_amount=vs.prod_amount[idxs],
        is_creature=vs.is_creature[idxs],
        mv=vs.mv[idxs],
        generic=vs.generic[idxs],
        pips=vs.pips[idxs],
        ramp_spell=vs.ramp_spell[idxs],
    )
    if commander_idx is not None:
        ds.commander_mv = int(vs.mv[commander_idx])
        ds.commander_generic = int(vs.generic[commander_idx])
        ds.commander_pips = vs.pips[commander_idx]
    return ds


class Shaper:
    """Potential-based reward shaping: r += weight * (gamma*phi(s') - phi(s)),
    phi = goldfish score of the partial deck. Dense credit at every step —
    the fix for the terminal-only failure mode of the 2026-07-15 structural
    reward experiment. Fixed seed keeps phi deterministic (common random
    numbers across the s/s' pair)."""

    def __init__(self, vs: DeckStatic, gamma: float = 0.99, weight: float = 1.0,
                 games: int = 4, turns: int = 8, seed: int = 0):
        self.vs = vs
        self.gamma = gamma
        self.weight = weight
        self.games = games
        self.turns = turns
        self.seed = seed

    def phi(self, idxs: np.ndarray, commander_idx: int | None) -> float:
        if idxs.size < 8:
            return 0.0  # too small to goldfish meaningfully
        ds = slice_deck(self.vs, idxs, commander_idx)
        if not ds.land.any():
            return 0.0
        return simulate(
            ds, games=self.games, turns=self.turns, seed=self.seed
        )["score"]

    def delta(
        self, partial: np.ndarray, action: int, commander_idx: int | None
    ) -> float:
        before = self.phi(partial, commander_idx)
        after = self.phi(np.append(partial, action), commander_idx)
        return self.weight * (self.gamma * after - before)
