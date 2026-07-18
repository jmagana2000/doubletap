"""Mana-base recommendation: pure Karsten math and greedy optimization, no
ML (external review 2026-07, item #2 — "do not begin with RL").

Land count from the curve regression, per-color source requirements from
the hypergeometric tables, real nonbasic lands chosen greedily to cover
deficits (untapped preferred, budget and bracket respected), basics fill
the rest pip-proportionally, and the goldfish simulator scores the result
against an all-basics baseline so the value of the nonbasics is measured,
not asserted.

ponytail: conditional producers (Ancient Ziggurat, filter lands) are taken
at face value by source_weights and the sim — restriction-aware weighting
is the v2 upgrade path."""

import json
import sqlite3
from dataclasses import dataclass, field

from .analysis import (
    SOURCES_NEEDED,
    card_price,
    color_pip_counts,
    deck_report,
    etb_tapped,
    source_weights,
)
from .formats import FormatConfig, is_basic_land, is_game_changer, is_land

BASICS = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}


def basic_land_split(pips: dict[str, int], n_lands: int) -> dict[str, int]:
    """Split n_lands basics proportionally to WUBRG pip counts (largest-
    remainder rounding); colorless decks get an even split."""
    order = "WUBRG"
    total = sum(pips.get(c, 0) for c in order)
    shares = [
        (pips.get(c, 0) / total * n_lands) if total else n_lands / 5 for c in order
    ]
    counts = [int(s) for s in shares]
    remainders = sorted(range(5), key=lambda i: shares[i] - counts[i], reverse=True)
    for i in range(n_lands - sum(counts)):
        counts[remainders[i % 5]] += 1
    return {c: n for c, n in zip(order, counts) if n}


@dataclass
class ManabaseResult:
    land_count: int
    karsten_target: int
    lands: list = field(default_factory=list)  # (name, qty, colors, tapped, price)
    basics: dict = field(default_factory=dict)  # basic name -> qty
    achieved: dict = field(default_factory=dict)  # color -> effective sources
    needed: dict = field(default_factory=dict)  # color -> Karsten requirement
    notes: list = field(default_factory=list)


def _land_candidates(
    conn: sqlite3.Connection,
    fmt: FormatConfig,
    identity: set[str],
    budget: float | None,
    bracket: int,
) -> list[dict]:
    """Format-legal nonbasic lands inside the color identity; strict on
    price under a budget; Game Changer lands excluded at bracket <= 3
    (they would spend the deck's whole allowance on mana)."""
    out = []
    for (raw,) in conn.execute(
        "SELECT json FROM cards WHERE json_extract(json, '$.legalities.' || ?)"
        " = 'legal' AND json_extract(json, '$.type_line') LIKE '%Land%'",
        (fmt.legality_key,),
    ):
        card = json.loads(raw)
        if not is_land(card) or is_basic_land(card):
            continue
        if (
            identity is not None
            and not set(card.get("color_identity") or []) <= identity
        ):
            continue
        if bracket <= 3 and is_game_changer(card):
            continue
        if budget is not None:
            price = card_price(card)
            if price is None or price > budget:
                continue
        if not any(c in "WUBRG" for c in card.get("produced_mana") or []):
            continue  # ponytail: pure-utility/colorless lands are out of scope v1
        out.append(card)
    return out


def recommend_manabase(
    conn: sqlite3.Connection,
    entries: dict[str, int],
    fmt: FormatConfig,
    commander_oid: str | None = None,
    land_count: int | None = None,
    budget: float | None = None,
    bracket: int = 3,
) -> ManabaseResult:
    """Recommend a full mana base for the deck's nonland cards.
    `entries` maps oracle_id -> qty (lands inside are ignored — the
    recommendation replaces the mana base wholesale)."""
    from .formats import get_card

    nonland = {}
    for oid, qty in entries.items():
        card = get_card(conn, oid)
        if not is_land(card):
            nonland[oid] = qty
    if commander_oid:
        nonland.setdefault(commander_oid, 1)
    report = deck_report(conn, nonland, fmt.name)

    identity = None
    if commander_oid:
        identity = set(get_card(conn, commander_oid).get("color_identity") or [])

    n = land_count if land_count is not None else report.karsten_lands
    needed_table = SOURCES_NEEDED.get(fmt.name, SOURCES_NEEDED["commander"])
    needed = {c: needed_table[min(p, 3)] for c, p in report.max_pips.items() if p > 0}
    achieved: dict[str, float] = {c: 0.0 for c in needed}
    result = ManabaseResult(
        land_count=n, karsten_target=report.karsten_lands, needed=dict(needed)
    )

    max_copies = 1 if fmt.copy_limit == 1 else fmt.copy_limit
    chosen: dict[str, int] = {}
    cards_by_oid: dict[str, dict] = {}
    candidates = _land_candidates(conn, fmt, identity, budget, bracket)
    picked = 0
    while picked < n:
        deficits = {c: max(0.0, needed[c] - achieved[c]) for c in needed}
        if sum(deficits.values()) <= 0 or not candidates:
            break
        best, best_score = None, 0.0
        for card in candidates:
            oid = card["oracle_id"]
            if chosen.get(oid, 0) >= max_copies:
                continue
            weights = source_weights(card)
            score = sum(min(w, deficits.get(c, 0.0)) for c, w in weights.items())
            if score <= 0:
                continue
            if etb_tapped(card):
                score -= 0.3  # a turn of tempo is worth ~a third of a source
            if score > best_score:
                best, best_score = card, score
        if best is None:
            break
        oid = best["oracle_id"]
        chosen[oid] = chosen.get(oid, 0) + 1
        cards_by_oid[oid] = best
        for c, w in source_weights(best).items():
            if c in achieved:
                achieved[c] += w
        picked += 1

    remaining = n - picked
    if remaining > 0:
        basic_pips = {
            c: p for c, p in report.pips.items() if identity is None or c in identity
        }
        if not basic_pips and (identity is not None and not identity):
            # colorless identity (Kozilek-style): every colored basic is an
            # identity violation — Wastes is the only legal basic
            result.basics = {"Wastes": remaining}
        else:
            split = basic_land_split(basic_pips, remaining)
            result.basics = {BASICS[c]: q for c, q in split.items()}
            for c, q in split.items():
                if c in achieved:
                    achieved[c] += q  # a basic is a full source of its color

    result.lands = sorted(
        (
            (
                card["name"],
                qty,
                "".join(c for c in "WUBRG" if c in (card.get("produced_mana") or [])),
                etb_tapped(card),
                card_price(card),
            )
            for oid, qty in chosen.items()
            for card in [cards_by_oid[oid]]
        ),
        key=lambda t: t[0],
    )
    result.achieved = {c: round(v, 1) for c, v in achieved.items()}
    for c in sorted(needed):
        if achieved.get(c, 0.0) < needed[c]:
            result.notes.append(
                f"{c}: {achieved[c]:.1f} of {needed[c]} sources — consider more"
                f" {BASICS[c]}s or duals"
            )
    if land_count is None and fmt.exact_size:
        result.notes.append(
            "land count is Karsten's target; complete/exact-size decks may"
            " need a different slot count"
        )
    return result


def goldfish_compare(
    conn: sqlite3.Connection,
    entries: dict[str, int],
    result: ManabaseResult,
    commander_oid: str | None,
    games: int = 200,
) -> dict:
    """Goldfish the nonlands with the recommended base vs an all-basics
    base of the same size — the measured value of the nonbasic lands."""
    from .formats import get_card
    from .ml.goldfish import compile_deck, simulate
    from .names import lookup

    nonland_cards = []
    pips: dict[str, int] = {}
    for oid, qty in entries.items():
        card = get_card(conn, oid)
        if is_land(card):
            continue
        nonland_cards.append((card, qty))
        for c, p in color_pip_counts(card).items():
            pips[c] = pips.get(c, 0) + p * qty
    commander = get_card(conn, commander_oid) if commander_oid else None

    def basic_cards(split: dict[str, int]):
        return [
            (get_card(conn, lookup(conn, name)[0].oracle_id), qty)
            for name, qty in split.items()
        ]

    rec_lands = [
        (get_card(conn, lookup(conn, name)[0].oracle_id), qty)
        for name, qty, *_ in result.lands
    ] + basic_cards(result.basics)
    rec = simulate(
        compile_deck(nonland_cards + rec_lands, commander), games=games, seed=0
    )
    baseline_split = {
        BASICS[c]: q for c, q in basic_land_split(pips, result.land_count).items()
    }
    base = simulate(
        compile_deck(nonland_cards + basic_cards(baseline_split), commander),
        games=games,
        seed=0,
    )
    return {"recommended": rec, "all_basics": base}
