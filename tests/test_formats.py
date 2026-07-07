from collections import Counter

import pytest

from doubletap.decks import Deck
from doubletap.formats import (
    COMMANDER,
    COMPANION_RULES,
    MODERN,
    get_format,
    validate,
)
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


def test_valid_partner_commander_deck(loaded_conn):
    # Thrasios (UG) + Tymna (WB) → combined identity WUBG
    # Juzám Djinn (B) and Sol Ring (colorless) are within WUBG
    deck = Deck(
        format="commander",
        commander=oid(loaded_conn, "Thrasios, Triton Hero"),
        partner=oid(loaded_conn, "Tymna the Weaver"),
        entries=Counter(
            {
                oid(loaded_conn, "Sol Ring"): 1,
                oid(loaded_conn, "Juzám Djinn"): 1,
                oid(loaded_conn, "Swamp"): 96,
            }
        ),
    )
    assert validate(loaded_conn, deck) == []


def test_partner_rejects_card_outside_combined_identity(loaded_conn):
    # Lightning Bolt is Red — outside Thrasios+Tymna's WUBG identity
    deck = Deck(
        format="commander",
        commander=oid(loaded_conn, "Thrasios, Triton Hero"),
        partner=oid(loaded_conn, "Tymna the Weaver"),
        entries=Counter(
            {
                oid(loaded_conn, "Lightning Bolt"): 1,
                oid(loaded_conn, "Swamp"): 96,
            }
        ),
    )
    assert "color_identity" in codes(validate(loaded_conn, deck))


def test_partner_requires_partner_keyword(loaded_conn):
    # Atraxa has no Partner keyword — pairing with Thrasios is invalid
    deck = Deck(
        format="commander",
        commander=oid(loaded_conn, "Thrasios, Triton Hero"),
        partner=oid(loaded_conn, "Atraxa, Praetors' Voice"),
        entries=Counter({oid(loaded_conn, "Swamp"): 97}),
    )
    assert "invalid_partner" in codes(validate(loaded_conn, deck))


def test_valid_modern_companion_deck(loaded_conn):
    # Lurrus: every permanent has mana value <= 2; Bolt and Fire//Ice are
    # spells, Swamps are mv 0 — all fine
    deck = Deck(
        format="modern",
        companion=oid(loaded_conn, "Lurrus of the Dream-Den"),
        entries=Counter(
            {
                oid(loaded_conn, "Lightning Bolt"): 4,
                oid(loaded_conn, "Fire // Ice"): 4,
                oid(loaded_conn, "Swamp"): 52,
            }
        ),
    )
    assert validate(loaded_conn, deck) == []


def _keruga_commander_deck(conn):
    # Keruga: only cards with mana value >= 3 and lands. Atraxa (4) and
    # Juzám Djinn (4) qualify; 98 Swamps are lands.
    return Deck(
        format="commander",
        commander=oid(conn, "Atraxa, Praetors' Voice"),
        companion=oid(conn, "Keruga, the Macrosage"),
        entries=Counter(
            {
                oid(conn, "Juzám Djinn"): 1,
                oid(conn, "Swamp"): 98,
            }
        ),
    )


def test_valid_commander_companion_deck(loaded_conn):
    assert validate(loaded_conn, _keruga_commander_deck(loaded_conn)) == []


def test_companion_restriction_violated(loaded_conn):
    deck = _keruga_commander_deck(loaded_conn)
    deck.entries[oid(loaded_conn, "Swamp")] -= 1
    deck.entries[oid(loaded_conn, "Sol Ring")] = 1  # mv 1 breaks Keruga
    violations = validate(loaded_conn, deck)
    assert codes(violations) == {"companion_restriction"}
    assert "Sol Ring" in violations[0].message


def test_companion_outside_commander_identity(loaded_conn):
    # Keruga is GU; Tymna's identity is WB
    deck = Deck(
        format="commander",
        commander=oid(loaded_conn, "Tymna the Weaver"),
        companion=oid(loaded_conn, "Keruga, the Macrosage"),
        entries=Counter({oid(loaded_conn, "Swamp"): 99}),
    )
    assert codes(validate(loaded_conn, deck)) == {"color_identity"}


def test_companion_requires_companion_keyword(loaded_conn):
    deck = _keruga_commander_deck(loaded_conn)
    deck.companion = oid(loaded_conn, "Sol Ring")
    assert "invalid_companion" in codes(validate(loaded_conn, deck))


def _c(name, type_line, cmc=0, mana_cost="", text=""):
    return {
        "name": name,
        "type_line": type_line,
        "cmc": cmc,
        "mana_cost": mana_cost,
        "oracle_text": text,
    }


def test_companion_rule_gyruda_even_only():
    rule = COMPANION_RULES["Gyruda, Doom of Depths"]
    assert rule([(_c("Even", "Sorcery", 2), 1)], MODERN) is None
    assert rule([(_c("Odd", "Sorcery", 3), 1)], MODERN)


def test_companion_rule_obosh_odd_or_land():
    rule = COMPANION_RULES["Obosh, the Preypiercer"]
    ok = [(_c("Odd", "Sorcery", 3), 1), (_c("Swamp", "Basic Land — Swamp"), 20)]
    assert rule(ok, MODERN) is None
    assert rule([(_c("Even", "Sorcery", 2), 1)], MODERN)


def test_companion_rule_kaheera_creature_types():
    rule = COMPANION_RULES["Kaheera, the Orphanguard"]
    assert rule([(_c("Kitty", "Creature — Cat", 2), 1)], MODERN) is None
    assert rule([(_c("Bird", "Creature — Bird", 2), 1)], MODERN)
    assert rule([(_c("Spell", "Instant", 2), 1)], MODERN) is None


def test_companion_rule_umori_shared_type():
    rule = COMPANION_RULES["Umori, the Collector"]
    ok = [
        (_c("A", "Creature — Ooze", 2), 1),
        (_c("B", "Legendary Creature — Human", 3), 1),
        (_c("Swamp", "Basic Land — Swamp"), 20),  # lands exempt
    ]
    assert rule(ok, MODERN) is None
    mixed = [(_c("A", "Creature — Ooze", 2), 1), (_c("B", "Sorcery", 3), 1)]
    assert rule(mixed, MODERN)


def test_companion_rule_jegantha_no_repeated_symbols():
    rule = COMPANION_RULES["Jegantha, the Wellspring"]
    assert rule([(_c("A", "Sorcery", 5, "{3}{R}{G}"), 1)], MODERN) is None
    assert rule([(_c("B", "Creature — Rat", 3, "{1}{B}{B}"), 1)], MODERN)
    assert rule([(_c("C", "Creature — Hippo", 5, "{3}{G/U}{G/U}"), 1)], MODERN)


def test_companion_rule_lutri_singleton_nonland():
    rule = COMPANION_RULES["Lutri, the Spellchaser"]
    ok = [(_c("A", "Instant", 1), 1), (_c("Swamp", "Basic Land — Swamp"), 20)]
    assert rule(ok, MODERN) is None
    assert rule([(_c("A", "Instant", 1), 2)], MODERN)


def test_companion_rule_zirda_activated_abilities():
    rule = COMPANION_RULES["Zirda, the Dawnwaker"]
    ok = [
        (_c("Mine", "Artifact", 2, text="{T}: Add {C}."), 1),
        (_c("Spell", "Instant", 2), 1),  # nonpermanents exempt
        (_c("Swamp", "Basic Land — Swamp"), 20),  # lands exempt
    ]
    assert rule(ok, MODERN) is None
    assert rule([(_c("Vanilla", "Creature — Bear", 2, text=""), 1)], MODERN)


def test_companion_rule_yorion_bigger_deck():
    rule = COMPANION_RULES["Yorion, Sky Nomad"]
    assert rule([(_c("Swamp", "Basic Land — Swamp"), 80)], MODERN) is None
    assert rule([(_c("Swamp", "Basic Land — Swamp"), 60)], MODERN)
    assert rule([(_c("Swamp", "Basic Land — Swamp"), 120)], COMMANDER)
