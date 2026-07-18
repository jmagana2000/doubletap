import json
from collections import Counter
import re
import sqlite3
from dataclasses import dataclass

from .decks import Deck


@dataclass(frozen=True)
class FormatConfig:
    name: str
    legality_key: str  # key into Scryfall's legalities map
    deck_size: int
    exact_size: bool  # commander is exactly 100; modern is a 60-card minimum
    copy_limit: int
    requires_commander: bool
    # structural targets, used by the reward (Phase 5) and gap reports (Phase 8)
    land_fraction_target: float
    synergy_weight: float
    structure_weight: float
    # pip-demand state features: passed commander's keep-bar, failed modern's
    # (3-seed sweep, rl-strategy-research.md) — hence per-format
    pip_state: bool = False


COMMANDER = FormatConfig(
    name="commander",
    legality_key="commander",
    deck_size=100,
    exact_size=True,
    copy_limit=1,
    requires_commander=True,
    land_fraction_target=0.37,
    synergy_weight=1.0,
    structure_weight=1.0,
    pip_state=True,
)

MODERN = FormatConfig(
    name="modern",
    legality_key="modern",
    deck_size=60,
    exact_size=False,
    copy_limit=4,
    requires_commander=False,
    land_fraction_target=0.40,
    synergy_weight=1.0,
    structure_weight=1.0,
)

STANDARD = FormatConfig(
    name="standard",
    legality_key="standard",
    deck_size=60,
    exact_size=False,
    copy_limit=4,
    requires_commander=False,
    land_fraction_target=0.40,
    synergy_weight=1.0,
    structure_weight=1.0,
)

FORMATS = {f.name: f for f in (COMMANDER, MODERN, STANDARD)}

# WotC Commander Brackets Beta — cards that count toward the bracket threshold.
# 0 in a deck → Bracket 1/2; 1-3 → Bracket 3; 4+ → Bracket 4/5.
# Source: https://magic.wizards.com/en/news/announcements/introducing-commander-brackets-beta
BRACKETS = {
    1: "Exhibition  — ultra-casual, no Game Changers, no combos, no land denial",
    2: "Core        — precon power, no Game Changers, no combos",
    3: "Upgraded    — optimized, up to 3 Game Changers, no early infinite combos",
    4: "Optimized   — high-power, unrestricted (banned list only)",
    5: "cEDH        — competitive tournament play",
}


def is_game_changer(card: dict) -> bool:
    """Scryfall tracks the official Game Changers list per card — prefer
    this over the hardcoded snapshot; it updates with every cards sync."""
    return bool(card.get("game_changer"))


def compute_bracket(cards: list[dict]) -> tuple[int, list[str]]:
    """Return (bracket_number, names_of_game_changers_present) for card JSONs."""
    found = [c["name"] for c in cards if is_game_changer(c)]
    count = len(found)
    if count == 0:
        bracket = (
            2  # assume Core rather than Exhibition; Exhibition requires manual check
        )
    elif count <= 3:
        bracket = 3
    else:
        bracket = 4
    return bracket, found


def get_format(name: str) -> FormatConfig:
    try:
        return FORMATS[name.casefold()]
    except KeyError:
        raise ValueError(f"Unknown format {name!r}; supported: {', '.join(FORMATS)}")


@dataclass
class Violation:
    code: str
    message: str
    oracle_id: str | None = None


def get_card(conn: sqlite3.Connection, oracle_id: str) -> dict:
    row = conn.execute(
        "SELECT json FROM cards WHERE oracle_id = ?", (oracle_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"oracle_id {oracle_id} not in card cache")
    return json.loads(row[0])


def is_basic_land(card: dict) -> bool:
    return "Basic" in card["type_line"].split("—")[0]


def is_land(card: dict) -> bool:
    return "Land" in card["type_line"].split("//")[0]


ANY_NUMBER = 1_000_000  # effectively unlimited, fits an int32 vocab array
_NAMED_CAP_RE = re.compile(r"up to (\w+) cards named", re.IGNORECASE)
_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}  # fmt: skip


def named_copy_cap(card: dict) -> int | None:
    """Card-text override of the format copy limit (rule 903.5b exceptions):
    "any number of cards named" (Relentless Rats) -> ANY_NUMBER; "up to
    seven/nine cards named" (Seven Dwarves 7, Nazgûl 9) -> that number;
    None -> the format's normal limit applies."""
    text = card.get("oracle_text", "")
    if "any number of cards named" in text:
        return ANY_NUMBER
    m = _NAMED_CAP_RE.search(text)
    if m:
        word = m.group(1).casefold()
        return _NUMBER_WORDS.get(word) or (int(word) if word.isdigit() else None)
    return None


def can_be_commander(card: dict) -> bool:
    type_line = card["type_line"].split("//")[0]
    if "Legendary" in type_line and "Creature" in type_line:
        return True
    return "can be your commander" in card.get("oracle_text", "")


def has_partner_keyword(card: dict) -> bool:
    return any(k.startswith("Partner") for k in card.get("keywords", []))


def has_companion_keyword(card: dict) -> bool:
    return "Companion" in card.get("keywords", [])


# --- Companion deckbuilding restrictions --------------------------------------
# Each rule receives every starting-deck card as (card, qty) — commander
# included, companion excluded — and returns a violation message or None.

_PERMANENT_TYPES = (
    "Creature",
    "Artifact",
    "Enchantment",
    "Land",
    "Planeswalker",
    "Battle",
)
_KAHEERA_TYPES = ("Cat", "Elemental", "Nightmare", "Dinosaur", "Beast")
_CARD_TYPES = {
    "Creature",
    "Artifact",
    "Enchantment",
    "Instant",
    "Sorcery",
    "Planeswalker",
    "Battle",
}


def _mv(card: dict) -> int:
    return int(card.get("cmc") or 0)


def _is_permanent(card: dict) -> bool:
    front = card["type_line"].split("//")[0]
    return any(t in front for t in _PERMANENT_TYPES)


def _rule_lurrus(entries, fmt):
    for card, _ in entries:
        if _is_permanent(card) and _mv(card) > 2:
            return f"{card['name']} is a permanent with mana value over 2"


def _rule_keruga(entries, fmt):
    for card, _ in entries:
        if not is_land(card) and _mv(card) < 3:
            return f"{card['name']} has mana value under 3"


def _rule_gyruda(entries, fmt):
    for card, _ in entries:
        if _mv(card) % 2 != 0:
            return f"{card['name']} has an odd mana value"


def _rule_obosh(entries, fmt):
    for card, _ in entries:
        if not is_land(card) and _mv(card) % 2 == 0:
            return f"{card['name']} has an even mana value"


def _rule_kaheera(entries, fmt):
    for card, _ in entries:
        front = card["type_line"].split("//")[0]
        if "Creature" in front and not any(t in front for t in _KAHEERA_TYPES):
            return (
                f"{card['name']} is not a Cat, Elemental, Nightmare, Dinosaur, or Beast"
            )


def _rule_umori(entries, fmt):
    shared = None
    for card, _ in entries:
        if is_land(card):
            continue
        types = _CARD_TYPES & set(card["type_line"].split("//")[0].split())
        shared = types if shared is None else shared & types
        if not shared:
            return (
                f"nonland cards do not all share a card type ({card['name']} differs)"
            )


def _faces(card: dict) -> list[dict]:
    """The card itself, or each face of a multiface card — companion rules
    must read per-face costs/text (top-level fields are absent on MDFCs)."""
    return card.get("card_faces") or [card]


def _rule_jegantha(entries, fmt):
    for card, _ in entries:
        for face in _faces(card):
            symbols = [
                s
                for s in re.findall(r"\{([^}]+)\}", face.get("mana_cost") or "")
                if not s.isdigit()
            ]
            for s in set(symbols):
                if symbols.count(s) > 1:
                    return f"{card['name']} has more than one {{{s}}} in its mana cost"


def _rule_lutri(entries, fmt):
    for card, qty in entries:
        if qty > 1 and not is_land(card):
            return f"{qty}x {card['name']} — nonland cards must be singleton"


# keyword abilities that are activated abilities without a ":" in oracle text
_ACTIVATION_KEYWORDS = {"Equip", "Crew", "Cycling", "Reconfigure", "Fortify"}


def _rule_zirda(entries, fmt):
    for card, _ in entries:
        has_activated = any(
            ":" in (face.get("oracle_text") or "") for face in _faces(card)
        ) or bool(_ACTIVATION_KEYWORDS & set(card.get("keywords", [])))
        if _is_permanent(card) and not is_land(card) and not has_activated:
            return f"{card['name']} is a permanent with no activated ability"


def _rule_yorion(entries, fmt):
    if fmt.exact_size:
        return f"Yorion cannot be a companion in {fmt.name} (deck size is fixed)"
    total = sum(qty for _, qty in entries)
    if total < fmt.deck_size + 20:
        return f"deck must be at least {fmt.deck_size + 20} cards, got {total}"


COMPANION_RULES = {
    "Gyruda, Doom of Depths": _rule_gyruda,
    "Jegantha, the Wellspring": _rule_jegantha,
    "Kaheera, the Orphanguard": _rule_kaheera,
    "Keruga, the Macrosage": _rule_keruga,
    "Lurrus of the Dream-Den": _rule_lurrus,
    "Lutri, the Spellchaser": _rule_lutri,
    "Obosh, the Preypiercer": _rule_obosh,
    "Umori, the Collector": _rule_umori,
    "Yorion, Sky Nomad": _rule_yorion,
    "Zirda, the Dawnwaker": _rule_zirda,
}


def validate(conn: sqlite3.Connection, deck: Deck) -> list[Violation]:
    fmt = get_format(deck.format)
    violations = []
    cards = {oid: get_card(conn, oid) for oid in deck.entries}

    size = deck.size()
    if fmt.exact_size and size != fmt.deck_size:
        violations.append(
            Violation(
                "wrong_size",
                f"{fmt.name} decks are exactly {fmt.deck_size} cards, got {size}",
            )
        )
    elif not fmt.exact_size and size < fmt.deck_size:
        violations.append(
            Violation(
                "wrong_size",
                f"{fmt.name} decks are at least {fmt.deck_size} cards, got {size}",
            )
        )

    identity = None
    commander_oids: set[str] = set()
    if fmt.requires_commander:
        if deck.commander is None:
            violations.append(
                Violation("missing_commander", f"{fmt.name} decks need a commander")
            )
        else:
            commander = get_card(conn, deck.commander)
            cards[deck.commander] = commander
            commander_oids.add(deck.commander)
            if not can_be_commander(commander):
                violations.append(
                    Violation(
                        "invalid_commander",
                        f"{commander['name']} cannot be a commander",
                        deck.commander,
                    )
                )
            identity = set(commander.get("color_identity", []))

            if deck.partner is not None:
                partner = get_card(conn, deck.partner)
                cards[deck.partner] = partner
                commander_oids.add(deck.partner)
                if not can_be_commander(partner):
                    violations.append(
                        Violation(
                            "invalid_commander",
                            f"{partner['name']} cannot be a commander",
                            deck.partner,
                        )
                    )
                if not has_partner_keyword(commander) or not has_partner_keyword(
                    partner
                ):
                    violations.append(
                        Violation(
                            "invalid_partner",
                            "Both commanders must have the Partner keyword to use partner commanders",
                        )
                    )
                identity |= set(partner.get("color_identity", []))

    if deck.companion is not None:
        comp = get_card(conn, deck.companion)
        # legality and color identity apply to the companion like any card,
        # but it sits outside the starting deck (size, copies, restrictions)
        cards[deck.companion] = comp
        if not has_companion_keyword(comp):
            violations.append(
                Violation(
                    "invalid_companion",
                    f"{comp['name']} does not have the Companion ability",
                    deck.companion,
                )
            )
        else:
            rule = COMPANION_RULES.get(comp["name"])
            if rule is not None:
                starting = [
                    (cards[oid], qty)
                    for oid, qty in deck.entries.items()
                    # mainboard copies of the companion count; only the
                    # companion-zone copy itself sits outside
                ] + [(cards[oid], 1) for oid in commander_oids]
                problem = rule(starting, fmt)
                if problem:
                    violations.append(
                        Violation(
                            "companion_restriction",
                            f"{comp['name']}: {problem}",
                            deck.companion,
                        )
                    )

    for oid, card in cards.items():
        status = card["legalities"].get(fmt.legality_key, "not_legal")
        if status == "banned":
            violations.append(
                Violation("banned", f"{card['name']} is banned in {fmt.name}", oid)
            )
        elif status != "legal":
            violations.append(
                Violation(
                    "not_legal", f"{card['name']} is not legal in {fmt.name}", oid
                )
            )

    # singleton/copy limits count the WHOLE deck by oracle_id — command-zone
    # copies included (Atraxa as commander + Atraxa in the 99 is illegal)
    zone_oids = [o for o in (deck.commander, deck.partner, deck.companion) if o]
    all_copies = Counter(deck.entries)
    for oid in zone_oids:
        all_copies[oid] += 1
    for oid, qty in all_copies.items():
        card = cards.get(oid) or get_card(conn, oid)
        cap = named_copy_cap(card) or fmt.copy_limit
        if qty > cap and not is_basic_land(card):
            violations.append(
                Violation(
                    "too_many_copies",
                    f"{qty}x {card['name']} exceeds the {cap}-copy limit",
                    oid,
                )
            )

    if identity is not None:
        for oid, card in cards.items():
            if oid in commander_oids:
                continue
            if not set(card.get("color_identity", [])) <= identity:
                violations.append(
                    Violation(
                        "color_identity",
                        f"{card['name']} is outside the commander's color identity",
                        oid,
                    )
                )

    return violations
