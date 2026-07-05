import json
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

FORMATS = {f.name: f for f in (COMMANDER, MODERN)}


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


def allows_any_number(card: dict) -> bool:
    return "any number of cards named" in card.get("oracle_text", "")


def can_be_commander(card: dict) -> bool:
    type_line = card["type_line"].split("//")[0]
    if "Legendary" in type_line and "Creature" in type_line:
        return True
    return "can be your commander" in card.get("oracle_text", "")


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
    if fmt.requires_commander:
        if deck.commander is None:
            violations.append(
                Violation("missing_commander", f"{fmt.name} decks need a commander")
            )
        else:
            commander = get_card(conn, deck.commander)
            cards[deck.commander] = commander
            if not can_be_commander(commander):
                violations.append(
                    Violation(
                        "invalid_commander",
                        f"{commander['name']} cannot be a commander",
                        deck.commander,
                    )
                )
            identity = set(commander.get("color_identity", []))

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

    for oid, qty in deck.entries.items():
        card = cards[oid]
        if (
            qty > fmt.copy_limit
            and not is_basic_land(card)
            and not allows_any_number(card)
        ):
            violations.append(
                Violation(
                    "too_many_copies",
                    f"{qty}x {card['name']} exceeds the {fmt.copy_limit}-copy limit",
                    oid,
                )
            )

    if identity is not None:
        for oid, card in cards.items():
            if oid == deck.commander:
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
