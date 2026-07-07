from doubletap.analysis import analyze_deck, card_price, classify
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
    assert classify(bolt) == {"removal"}


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
