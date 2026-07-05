from collections import Counter

import pytest

from doubletap.decks import Deck
from doubletap.formats import get_format, validate
from doubletap.names import lookup


def oid(conn, name):
    return lookup(conn, name)[0].oracle_id


def codes(violations):
    return {v.code for v in violations}


def test_get_format_unknown():
    with pytest.raises(ValueError):
        get_format("pauper")


def test_valid_modern_deck(loaded_conn):
    deck = Deck(
        format="modern",
        entries=Counter(
            {
                oid(loaded_conn, "Lightning Bolt"): 4,
                oid(loaded_conn, "Fire // Ice"): 4,
                oid(loaded_conn, "Swamp"): 52,
            }
        ),
    )
    assert validate(loaded_conn, deck) == []


def test_modern_banned_and_fifth_copy(loaded_conn):
    deck = Deck(
        format="modern",
        entries=Counter(
            {
                oid(loaded_conn, "Once Upon a Time"): 4,  # banned in modern
                oid(loaded_conn, "Lightning Bolt"): 5,
                oid(loaded_conn, "Swamp"): 51,
            }
        ),
    )
    assert codes(validate(loaded_conn, deck)) == {"banned", "too_many_copies"}


def test_modern_too_small_and_digital_only(loaded_conn):
    deck = Deck(
        format="modern",
        entries=Counter({oid(loaded_conn, "Ace Flockbringer"): 4}),
    )
    assert codes(validate(loaded_conn, deck)) == {"wrong_size", "not_legal"}


def _commander_deck(conn, filler=97):
    return Deck(
        format="commander",
        commander=oid(conn, "Atraxa, Praetors' Voice"),
        entries=Counter(
            {
                oid(conn, "Sol Ring"): 1,
                oid(conn, "Relentless Rats"): 1,
                oid(conn, "Swamp"): filler,
            }
        ),
    )


def test_valid_commander_deck(loaded_conn):
    assert validate(loaded_conn, _commander_deck(loaded_conn)) == []


def test_commander_exact_size(loaded_conn):
    violations = validate(loaded_conn, _commander_deck(loaded_conn, filler=90))
    assert codes(violations) == {"wrong_size"}


def test_commander_singleton_with_exemptions(loaded_conn):
    deck = _commander_deck(loaded_conn, filler=68)
    deck.entries[oid(loaded_conn, "Relentless Rats")] += 20  # any-number text
    deck.entries[oid(loaded_conn, "Swamp")] += 8  # basic land
    deck.entries[oid(loaded_conn, "Sol Ring")] += 1  # 2x Sol Ring: violation
    assert codes(validate(loaded_conn, deck)) == {"too_many_copies"}


def test_commander_color_identity(loaded_conn):
    deck = _commander_deck(loaded_conn, filler=96)
    deck.entries[oid(loaded_conn, "Lightning Bolt")] = 1  # red, outside GWUB
    violations = validate(loaded_conn, deck)
    assert codes(violations) == {"color_identity"}


def test_missing_and_invalid_commander(loaded_conn):
    deck = Deck(format="commander", entries=Counter({oid(loaded_conn, "Swamp"): 100}))
    assert "missing_commander" in codes(validate(loaded_conn, deck))

    deck.commander = oid(loaded_conn, "Sol Ring")
    deck.entries[oid(loaded_conn, "Swamp")] = 99
    assert "invalid_commander" in codes(validate(loaded_conn, deck))
