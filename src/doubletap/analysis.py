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


# --- Karsten mana-base mathematics ---------------------------------------------
# Sources: Frank Karsten's hypergeometric research (ChannelFireball colored-
# sources tables; TCGplayer land-count regression). Verified claims and full
# citations in docs/rl-strategy-research.md.

# colored sources needed for a card with N pips of one color to be
# "consistently castable" ((89+MV)% on curve)
SOURCES_NEEDED = {
    "standard": {1: 14, 2: 21, 3: 23},
    "commander": {1: 19, 2: 30, 3: 36},
    "modern": {1: 14, 2: 21, 3: 23},
}


def karsten_land_target(avg_mv: float, cheap_draw_ramp: int, fmt_name: str) -> int:
    """Optimal land count as a function of the deck's own curve and its cheap
    (MV<=2) draw/ramp density — Karsten's fitted regressions, clamped to sane
    format ranges."""
    if fmt_name == "commander":
        raw = 31.42 + 3.13 * avg_mv - 0.28 * cheap_draw_ramp
        lo, hi = 34, 45
    else:
        raw = 19.59 + 1.90 * avg_mv - 0.28 * cheap_draw_ramp
        lo, hi = 18, 28
    return int(min(max(round(raw), lo), hi))


def _is_mdfc_land(card: dict) -> tuple[bool, bool]:
    """(is a spell//land MDFC, is mythic). Front-face lands count as real
    lands elsewhere; this catches the spell-front variants only."""
    tl = card.get("type_line", "")
    if "//" not in tl or "Land" not in tl:
        return False, False
    if _is_land(card):  # front face is the land — treated as a land proper
        return False, False
    return True, card.get("rarity") == "mythic"


def effective_lands(card: dict) -> float:
    """How much 'land' a card is for land-count purposes: real lands 1.0,
    spell//land MDFCs 0.38 (0.74 mythic), everything else 0."""
    if _is_land(card):
        return 1.0
    mdfc, mythic = _is_mdfc_land(card)
    if mdfc:
        return 0.74 if mythic else 0.38
    return 0.0


def source_weights(card: dict) -> dict[str, float]:
    """Fractional colored-source contribution per color (Karsten weights):
    lands 1.0, mana creatures 0.5, mana artifacts 0.75, other producers 0.5,
    spell//land MDFCs 0.8 (1.0 mythic)."""
    produced = [c for c in (card.get("produced_mana") or []) if c in "WUBRG"]
    if not produced:
        return {}
    if _is_land(card):
        weight = 1.0
    else:
        mdfc, mythic = _is_mdfc_land(card)
        if mdfc:
            weight = 1.0 if mythic else 0.8
        elif "Creature" in card.get("type_line", ""):
            weight = 0.5
        elif "Artifact" in card.get("type_line", ""):
            weight = 0.75
        else:
            weight = 0.5
    return {c: weight for c in produced}


def color_pip_counts(card: dict) -> dict[str, int]:
    """Colored pips per color in the card's mana cost (hybrid counts toward
    each half is overkill — count the first color of a hybrid symbol)."""
    pips: dict[str, int] = {}
    for symbol in re.findall(r"\{([^}]+)\}", _mana_cost(card)):
        for c in "WUBRG":
            if c in symbol:
                pips[c] = pips.get(c, 0) + 1
                break
    return pips


def is_cheap_draw_ramp(card: dict) -> bool:
    """Karsten's 'cheap card draw or mana ramp' regression term: MV<=2 cards
    that draw or make mana."""
    if _is_land(card):
        return False
    if (card.get("cmc") or 0) > 2:
        return False
    roles = classify(card)
    return "ramp" in roles or "draw" in roles


# --- Goldfish-simulation card statics (docs/goldfish-sim-design.md) -----------

_ETB_TAPPED_RE = re.compile(r"enters (?:the battlefield )?tapped", re.I)
_ADD_CLAUSE_RE = re.compile(r"\badd ((?:\{[^}]+\})+|one mana of any color)", re.I)
_LAND_TO_BATTLEFIELD_RE = re.compile(
    r"search your library for .{0,40}land .{0,60}onto the battlefield", re.I
)


def etb_tapped(card: dict) -> bool:
    """Unconditionally enters tapped. Conditional lands (check/fast/reveal)
    are treated as untapped — ponytail: optimistic v1, model conditions if
    calibration demands it. Judged per face: a spell//land MDFC's land face
    is read on its own text, so the spell face's "you may" can't mask an
    unconditional tapped clause."""
    faces = card.get("card_faces") or [card]
    for face in faces:
        if "Land" not in (face.get("type_line") or card.get("type_line", "")):
            continue
        text = face.get("oracle_text") or ""
        if "unless" in text.casefold() or "you may" in text.casefold():
            continue
        if _ETB_TAPPED_RE.search(text):
            return True
    return False


def mana_production(card: dict) -> tuple[list[str], int]:
    """(colors producible, mana amount per activation) parsed from the
    card's 'Add ...' clause. Sol Ring -> ([], 2) with colorless counted via
    empty color list + amount; 'any color' -> all five. Returns ([], 0) for
    non-producers."""
    produced = [c for c in (card.get("produced_mana") or []) if c in "WUBRG"]
    best = 0
    for m in _ADD_CLAUSE_RE.finditer(_oracle_text(card)):
        clause = m.group(1)
        if clause.casefold().startswith("one mana"):
            best = max(best, 1)
        else:
            best = max(best, len(re.findall(r"\{[^}]+\}", clause)))
    if best == 0 and (card.get("produced_mana") or []):
        best = 1  # produced_mana without a parseable clause: assume 1
    return produced, best


def is_land_ramp(card: dict) -> bool:
    """Spell that puts a land onto the battlefield when cast (Cultivate,
    Rampant Growth)."""
    if _is_land(card):
        return False
    return bool(_LAND_TO_BATTLEFIELD_RE.search(_oracle_text(card)))


# --- Curve and color balance ---------------------------------------------------

CURVE_TOP_BUCKET = 7  # mana values 7+ share one histogram bucket


@dataclass
class DeckReport:
    fmt_name: str = "commander"
    by_role: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    total_price: float = 0.0
    unpriced: int = 0
    curve: Counter = field(default_factory=Counter)  # mv bucket -> nonland count
    avg_mv: float = 0.0
    early_plays: int = 0  # nonland cards with mv <= 2
    pips: Counter = field(default_factory=Counter)  # color -> symbols in costs
    sources: Counter = field(default_factory=Counter)  # color -> lands making it
    eff_lands: float = 0.0  # lands + fractional MDFC credit
    cheap_draw_ramp: int = 0  # MV<=2 draw/ramp (Karsten regression term)
    karsten_lands: int = 0  # regression-recommended land count
    eff_sources: dict = field(default_factory=dict)  # color -> fractional sources
    max_pips: dict = field(default_factory=dict)  # color -> most pips on one card


def deck_report(
    conn: sqlite3.Connection, entries: dict[str, int], fmt_name: str = "commander"
) -> DeckReport:
    """Full structural report for a deck (oracle_id -> qty): roles, price,
    mana curve, Karsten land target, and colored-cost vs source balance."""
    report = DeckReport(fmt_name=fmt_name)
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

        report.eff_lands += effective_lands(card) * qty
        if is_cheap_draw_ramp(card):
            report.cheap_draw_ramp += qty
        for color, w in source_weights(card).items():
            report.eff_sources[color] = report.eff_sources.get(color, 0.0) + w * qty

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
            for color, n in color_pip_counts(card).items():
                report.pips[color] += n * qty
                report.max_pips[color] = max(report.max_pips.get(color, 0), n)
    if total_nonland:
        report.avg_mv = total_mv / total_nonland
    report.karsten_lands = karsten_land_target(
        report.avg_mv, report.cheap_draw_ramp, fmt_name
    )
    return report


def short_colors(report: DeckReport) -> list[str]:
    """Colors whose fractional sources fall short of Karsten's requirement
    for the deck's most pip-demanding card of that color — those spells
    won't be reliably castable on curve."""
    needed_table = SOURCES_NEEDED.get(report.fmt_name, SOURCES_NEEDED["commander"])
    short = []
    for color, pips in report.max_pips.items():
        needed = needed_table[min(pips, 3)]
        if report.eff_sources.get(color, 0.0) < needed:
            short.append(color)
    return [c for c in "WUBRG" if c in short]


def analyze_deck(
    conn: sqlite3.Connection, entries: dict[str, int]
) -> tuple[dict[str, list[tuple[str, int]]], float, int]:
    """Classify every card in a deck (oracle_id -> qty). Returns
    (role -> [(name, qty)], total_price_usd, n_unpriced)."""
    report = deck_report(conn, entries)
    return report.by_role, report.total_price, report.unpriced
