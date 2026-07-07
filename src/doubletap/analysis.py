"""Deck-level analysis: functional card roles (the "how does this deck win"
breakdown), mana curve and color balance, and market prices (budget
constraints). Heuristics run on Scryfall oracle text — approximate by design,
good enough to spot structural gaps. The gap list this implements is
documented in docs/gameplay-blindspots.md."""

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field

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
    "mill": re.compile(
        r"\bmills?\b"
        r"|puts? the top .{0,30} cards? of (?:their|that player's|each player's)"
        r" library into (?:their|its owner's) graveyard",
        re.I,
    ),
}

# "search your library for <what>" is a tutor unless <what> is a land (that's
# ramp / mana fixing, already counted)
_TUTOR_RE = re.compile(r"search your library for ([^.\n]*)", re.I)

EVASION_KEYWORDS = frozenset(
    {"Flying", "Trample", "Menace", "Fear", "Intimidate", "Shadow", "Skulk"}
)
POISON_KEYWORDS = frozenset({"Infect", "Toxic"})

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


def _mana_cost(card: dict) -> str:
    if card.get("mana_cost"):
        return card["mana_cost"]
    faces = card.get("card_faces") or []
    return faces[0].get("mana_cost", "") if faces else ""


def _power(card: dict) -> int:
    for source in (card, *card.get("card_faces", [])):
        raw = source.get("power")
        if raw and raw.isdigit():
            return int(raw)
    return 0


def _is_land(card: dict) -> bool:
    return "Land" in card["type_line"].split(" // ")[0]


def is_instant_speed(card: dict) -> bool:
    """Castable on other players' turns: an instant, or anything with flash."""
    return "Instant" in card["type_line"].split("//")[0] or "Flash" in card.get(
        "keywords", []
    )


def classify(card: dict) -> set[str]:
    """Which functional roles a card fills. A card can fill several
    (e.g. a creature that ramps); lands are counted separately."""
    if _is_land(card):
        return {"land"}
    text = _oracle_text(card)
    keywords = set(card.get("keywords", []))
    roles = {role for role, pat in _ROLE_PATTERNS.items() if pat.search(text)}

    tutor_match = _TUTOR_RE.search(text)
    if tutor_match and "land" not in tutor_match.group(1).casefold():
        roles.add("tutor")

    is_creature = "Creature" in card["type_line"]
    # big creatures are a path to winning through combat even without
    # explicit wincon text
    if is_creature and _power(card) >= BIG_THREAT_POWER:
        roles.add("threat")
    if is_creature and (
        keywords & EVASION_KEYWORDS or "can't be blocked" in text.casefold()
    ):
        roles.add("evasive")
    if keywords & POISON_KEYWORDS:
        roles.add("poison")
    # removal you can cast on opponents' turns is worth more than sorceries
    if "removal" in roles and is_instant_speed(card):
        roles.add("removal_instant")
    return roles


# --- Curve and color balance ---------------------------------------------------

CURVE_TOP_BUCKET = 7  # mana values 7+ share one histogram bucket


@dataclass
class DeckReport:
    by_role: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    total_price: float = 0.0
    unpriced: int = 0
    curve: Counter = field(default_factory=Counter)  # mv bucket -> nonland count
    avg_mv: float = 0.0
    early_plays: int = 0  # nonland cards with mv <= 2
    pips: Counter = field(default_factory=Counter)  # color -> symbols in costs
    sources: Counter = field(default_factory=Counter)  # color -> lands making it


def deck_report(conn: sqlite3.Connection, entries: dict[str, int]) -> DeckReport:
    """Full structural report for a deck (oracle_id -> qty): roles, price,
    mana curve, and colored-cost vs land-color balance."""
    report = DeckReport()
    total_nonland = 0
    total_mv = 0.0
    for oid, qty in entries.items():
        row = conn.execute(
            "SELECT name, json FROM cards WHERE oracle_id = ?", (oid,)
        ).fetchone()
        if row is None:
            continue
        name, raw = row
        card = json.loads(raw)
        for role in classify(card):
            report.by_role.setdefault(role, []).append((name, qty))

        price = card_price(card)
        if price is None:
            report.unpriced += qty
        else:
            report.total_price += price * qty

        if _is_land(card):
            for color in card.get("produced_mana") or []:
                if color in "WUBRG":
                    report.sources[color] += qty
        else:
            mv = card.get("cmc") or 0
            report.curve[min(int(mv), CURVE_TOP_BUCKET)] += qty
            total_nonland += qty
            total_mv += mv * qty
            if mv <= 2:
                report.early_plays += qty
            for symbol in re.findall(r"\{([^}]+)\}", _mana_cost(card)):
                for color in "WUBRG":
                    if color in symbol:
                        report.pips[color] += qty
    if total_nonland:
        report.avg_mv = total_mv / total_nonland
    return report


def short_colors(report: DeckReport) -> list[str]:
    """Colors whose share of land sources falls well below their share of
    mana symbols — hands with those spells won't be castable on time."""
    pip_total = sum(report.pips.values())
    source_total = sum(report.sources.values())
    if not pip_total or not source_total:
        return []
    short = []
    for color, n in report.pips.items():
        need = n / pip_total
        have = report.sources.get(color, 0) / source_total
        if need > 0.1 and have < need * 0.6:
            short.append(color)
    return short


def analyze_deck(
    conn: sqlite3.Connection, entries: dict[str, int]
) -> tuple[dict[str, list[tuple[str, int]]], float, int]:
    """Classify every card in a deck (oracle_id -> qty). Returns
    (role -> [(name, qty)], total_price_usd, n_unpriced)."""
    report = deck_report(conn, entries)
    return report.by_role, report.total_price, report.unpriced
