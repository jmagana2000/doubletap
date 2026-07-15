from doubletap.analysis import (
    analyze_deck,
    card_price,
    classify,
    deck_report,
    is_instant_speed,
    short_colors,
)
from doubletap.names import lookup


def test_card_price_cheapest_finish():
    assert card_price({"prices": {"usd": "1.82", "usd_foil": "9.99"}}) == 1.82
    assert card_price({"prices": {"usd": None, "usd_foil": "9.99"}}) == 9.99
    assert card_price({"prices": {"usd": None}}) is None
    assert card_price({}) is None


def test_classify_removal():
    bolt = {
        "type_line": "Instant",
        "oracle_text": "Lightning Bolt deals 3 damage to any target.",
    }
    # an instant answer counts as removal you can hold up on opponents' turns
    assert classify(bolt) == {"removal", "removal_instant"}
    sorcery = {
        "type_line": "Sorcery",
        "oracle_text": "Destroy target creature.",
    }
    assert classify(sorcery) == {"removal"}


def test_classify_ramp():
    sol_ring = {"type_line": "Artifact", "oracle_text": "{T}: Add {C}{C}."}
    assert classify(sol_ring) == {"ramp"}


def test_classify_draw_and_wipe():
    assert classify({"type_line": "Sorcery", "oracle_text": "Draw two cards."}) == {
        "draw"
    }
    assert classify(
        {"type_line": "Sorcery", "oracle_text": "Destroy all creatures."}
    ) == {"board_wipe"}


def test_classify_wincon_and_threat():
    assert classify(
        {"type_line": "Creature — Merfolk Wizard", "oracle_text": "You win the game."}
    ) == {"wincon"}
    assert classify(
        {"type_line": "Creature — Djinn", "oracle_text": "", "power": "5"}
    ) == {"threat"}


def test_classify_land_not_ramp():
    swamp = {"type_line": "Basic Land — Swamp", "oracle_text": "{T}: Add {B}."}
    assert classify(swamp) == {"land"}


def test_classify_mdfc_uses_faces():
    mdfc = {
        "type_line": "Instant // Land",
        "card_faces": [
            {"oracle_text": "Return target creature card to the battlefield."},
            {"oracle_text": "{T}: Add {B}."},
        ],
    }
    # front face is an Instant, so it's not counted as a land
    assert "land" not in classify(mdfc)


def test_analyze_deck_counts_and_prices(loaded_conn):
    def oid(name):
        return lookup(loaded_conn, name)[0].oracle_id

    entries = {oid("Lightning Bolt"): 4, oid("Swamp"): 20, oid("Juzám Djinn"): 2}
    by_role, total, unpriced = analyze_deck(loaded_conn, entries)
    assert ("Lightning Bolt", 4) in by_role["removal"]
    assert ("Swamp", 20) in by_role["land"]
    # fixture cards carry no price data
    assert total == 0.0 and unpriced == 26


def test_is_instant_speed():
    assert is_instant_speed({"type_line": "Instant", "keywords": []})
    assert is_instant_speed({"type_line": "Creature — Cat", "keywords": ["Flash"]})
    assert not is_instant_speed({"type_line": "Sorcery", "keywords": []})


def test_classify_flash_removal_is_instant_speed():
    ambusher = {
        "type_line": "Creature — Elf",
        "oracle_text": "When this creature enters, destroy target artifact.",
        "keywords": ["Flash"],
    }
    assert "removal_instant" in classify(ambusher)


def test_classify_evasion():
    flier = {"type_line": "Creature — Bird", "oracle_text": "", "keywords": ["Flying"]}
    assert classify(flier) == {"evasive"}
    sneaky = {
        "type_line": "Creature — Rogue",
        "oracle_text": "This creature can't be blocked.",
        "keywords": [],
    }
    assert classify(sneaky) == {"evasive"}
    ground = {"type_line": "Creature — Bear", "oracle_text": "", "keywords": []}
    assert classify(ground) == set()
    # evasion is a creature property; a flying-granting enchantment is not evasive
    anthem = {"type_line": "Enchantment", "oracle_text": "", "keywords": ["Flying"]}
    assert classify(anthem) == set()


def test_classify_poison_and_mill():
    infect = {
        "type_line": "Creature — Phyrexian",
        "oracle_text": "Infect",
        "keywords": ["Infect"],
    }
    assert classify(infect) == {"poison"}
    mill = {
        "type_line": "Sorcery",
        "oracle_text": "Each opponent mills eight cards.",
        "keywords": [],
    }
    assert classify(mill) == {"mill"}
    old_mill = {
        "type_line": "Sorcery",
        "oracle_text": "That player puts the top five cards of their library into their graveyard.",
        "keywords": [],
    }
    assert classify(old_mill) == {"mill"}


def test_classify_tutor_but_not_land_search():
    demonic = {
        "type_line": "Sorcery",
        "oracle_text": "Search your library for a card, put it into your hand, then shuffle.",
        "keywords": [],
    }
    assert "tutor" in classify(demonic)
    cultivate = {
        "type_line": "Sorcery",
        "oracle_text": "Search your library for up to two basic land cards.",
        "keywords": [],
    }
    roles = classify(cultivate)
    assert "tutor" not in roles and "ramp" in roles


def test_deck_report_curve_and_colors(loaded_conn):
    def oid(name):
        return lookup(loaded_conn, name)[0].oracle_id

    # Bolt {R} mv1 x4, Juzám {2}{B}{B} mv4 x2, Swamp x20 (produces B only)
    entries = {oid("Lightning Bolt"): 4, oid("Juzám Djinn"): 2, oid("Swamp"): 20}
    report = deck_report(loaded_conn, entries)
    assert report.curve == {1: 4, 4: 2}
    assert report.early_plays == 4
    assert abs(report.avg_mv - (4 * 1 + 2 * 4) / 6) < 1e-9
    assert report.pips == {"R": 4, "B": 4}
    assert report.sources == {"B": 20}
    # Karsten: Bolt's single R pip needs 19 sources (have 0) and Juzám's
    # double B pip needs 30 (have 20) — both colors are honestly short
    assert short_colors(report) == ["B", "R"]


def test_short_colors_empty_when_balanced():
    from collections import Counter

    from doubletap.analysis import DeckReport

    balanced = DeckReport(
        pips=Counter({"B": 10, "R": 10}), sources=Counter({"B": 10, "R": 10})
    )
    assert short_colors(balanced) == []
    no_lands = DeckReport(pips=Counter({"B": 10}))
    assert short_colors(no_lands) == []


# --- Karsten mana-base math (docs/rl-strategy-research.md) -------------------


def test_karsten_land_target():
    from doubletap.analysis import karsten_land_target

    # commander regression: 31.42 + 3.13*avgMV - 0.28*cheap
    assert karsten_land_target(3.0, 10, "commander") == 38
    assert karsten_land_target(2.0, 15, "commander") == 34  # clamped floor
    assert karsten_land_target(5.0, 0, "commander") == 45  # clamped ceiling
    # 60-card regression: 19.59 + 1.90*avgMV - 0.28*cheap
    assert karsten_land_target(3.0, 4, "modern") == 24


def test_effective_lands_and_mdfc_fractions():
    from doubletap.analysis import effective_lands

    swamp = {"type_line": "Basic Land — Swamp"}
    assert effective_lands(swamp) == 1.0
    mdfc = {"type_line": "Instant // Land", "rarity": "rare"}
    assert effective_lands(mdfc) == 0.38
    mythic_mdfc = {"type_line": "Sorcery // Land", "rarity": "mythic"}
    assert effective_lands(mythic_mdfc) == 0.74
    spell = {"type_line": "Instant"}
    assert effective_lands(spell) == 0.0
    # front-face land MDFCs are just lands
    front_land = {"type_line": "Land // Instant", "rarity": "mythic"}
    assert effective_lands(front_land) == 1.0


def test_source_weights_by_producer_type():
    from doubletap.analysis import source_weights

    land = {"type_line": "Land", "produced_mana": ["W", "U"]}
    assert source_weights(land) == {"W": 1.0, "U": 1.0}
    dork = {"type_line": "Creature — Elf Druid", "produced_mana": ["G"]}
    assert source_weights(dork) == {"G": 0.5}
    rock = {"type_line": "Artifact", "produced_mana": ["B", "G", "R", "U", "W"]}
    assert source_weights(rock)["W"] == 0.75
    colorless_rock = {"type_line": "Artifact", "produced_mana": ["C"]}
    assert source_weights(colorless_rock) == {}  # colorless isn't a colored source
    nothing = {"type_line": "Instant"}
    assert source_weights(nothing) == {}


def test_is_cheap_draw_ramp():
    from doubletap.analysis import is_cheap_draw_ramp

    sol_ring = {"type_line": "Artifact", "cmc": 1, "oracle_text": "{T}: Add {C}{C}."}
    assert is_cheap_draw_ramp(sol_ring)
    cantrip = {"type_line": "Instant", "cmc": 1, "oracle_text": "Draw a card."}
    assert is_cheap_draw_ramp(cantrip)
    big_draw = {"type_line": "Sorcery", "cmc": 4, "oracle_text": "Draw three cards."}
    assert not is_cheap_draw_ramp(big_draw)  # not cheap
    bolt = {
        "type_line": "Instant",
        "cmc": 1,
        "oracle_text": "Deals 3 damage to any target.",
    }
    assert not is_cheap_draw_ramp(bolt)  # neither draw nor ramp


def test_deck_report_karsten_fields(loaded_conn):
    def oid(name):
        return lookup(loaded_conn, name)[0].oracle_id

    # Juzám {2}{B}{B} = double black pip -> needs 30 B sources in commander;
    # 20 swamps fall short. Bolt needs 19 R sources; zero -> short.
    entries = {oid("Lightning Bolt"): 4, oid("Juzám Djinn"): 2, oid("Swamp"): 20}
    report = deck_report(loaded_conn, entries, "commander")
    assert report.eff_lands == 20.0
    assert report.max_pips == {"R": 1, "B": 2}
    assert report.eff_sources == {"B": 20.0}
    assert short_colors(report) == ["B", "R"]
    assert report.karsten_lands == 38  # avgMV 2.0, 4 cheap (bolt isn't) -> clamp
