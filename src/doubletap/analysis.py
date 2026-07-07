"""Deck-level analysis: functional card roles (the "how does this deck win"
breakdown) and market prices (budget constraints). Heuristics run on Scryfall
oracle text — approximate by design, good enough to spot structural gaps."""

import re
import sqlite3

# --- Market prices -----------------------------------------------------------


def card_price(card: dict) -> float | None:
    """Cheapest available USD finish from Scryfall, None when unpriced
    (digital-only or brand-new cards)."""
    prices = card.get("prices") or {}
    values = [
        float(p)
        for p in (prices.get("usd"), prices.get("usd_foil"), prices.get("usd_etched"))
        if p
    ]
    return min(values) if values else None


# --- Functional roles --------------------------------------------------------

# Each pattern runs case-insensitively against the card's combined oracle text.
_ROLE_PATTERNS = {
    "ramp": re.compile(
        r"add \{|search your library for (?:a|up to \w+) (?:basic )?land", re.I
    ),
    "draw": re.compile(r"draws? (?:a|two|three|four|x) cards?", re.I),
    "removal": re.compile(
        r"destroy target|exile target|counter target"
        r"|deals? \d+ damage to (?:any target|target creature|target planeswalker)"
        r"|target creature gets? [-—]\d+/[-—]\d+",
        re.I,
    ),
    "board_wipe": re.compile(
        r"destroy all|exile all|deals? \d+ damage to each creature"
        r"|all creatures get [-—]\d+/[-—]\d+",
        re.I,
    ),
    "wincon": re.compile(r"you win the game|each opponent loses the game", re.I),
}

BIG_THREAT_POWER = 5

# Community consensus quotas for a functional Commander deck. Not rules —
# a starting point for spotting gaps.
COMMANDER_TARGETS = {
    "lands": 36,
    "ramp": 10,
    "draw": 10,
    "removal": 10,
    "board_wipe": 3,
}


def _oracle_text(card: dict) -> str:
    if "oracle_text" in card:
        return card["oracle_text"]
    return "\n".join(f.get("oracle_text", "") for f in card.get("card_faces", []))


def _power(card: dict) -> int:
    for source in (card, *card.get("card_faces", [])):
        raw = source.get("power")
        if raw and raw.isdigit():
            return int(raw)
    return 0


def classify(card: dict) -> set[str]:
    """Which functional roles a card fills. A card can fill several
    (e.g. a creature that ramps); lands are counted separately."""
    if "Land" in card["type_line"].split(" // ")[0]:
        return {"land"}
    text = _oracle_text(card)
    roles = {role for role, pat in _ROLE_PATTERNS.items() if pat.search(text)}
    # mana-producing lands are the mana base, not ramp; big creatures are a
    # path to winning through combat even without explicit wincon text
    if "Creature" in card["type_line"] and _power(card) >= BIG_THREAT_POWER:
        roles.add("threat")
    return roles


def analyze_deck(
    conn: sqlite3.Connection, entries: dict[str, int]
) -> tuple[dict[str, list[tuple[str, int]]], float, int]:
    """Classify every card in a deck (oracle_id -> qty). Returns
    (role -> [(name, qty)], total_price_usd, n_unpriced)."""
    import json

    by_role: dict[str, list[tuple[str, int]]] = {}
    total = 0.0
    unpriced = 0
    for oid, qty in entries.items():
        row = conn.execute(
            "SELECT name, json FROM cards WHERE oracle_id = ?", (oid,)
        ).fetchone()
        if row is None:
            continue
        name, raw = row
        card = json.loads(raw)
        for role in classify(card):
            by_role.setdefault(role, []).append((name, qty))
        price = card_price(card)
        if price is None:
            unpriced += qty
        else:
            total += price * qty
    return by_role, total, unpriced
