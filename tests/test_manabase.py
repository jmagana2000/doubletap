"""Mana-base engine: Karsten count, deficit-driven land picks, budget and
identity constraints, basics fill, goldfish comparison."""

import pytest

from doubletap import manabase
from doubletap.formats import COMMANDER
from doubletap.names import lookup


def oid(conn, name):
    return lookup(conn, name)[0].oracle_id


@pytest.fixture
def wb_deck(loaded_conn):
    """A W/B commander deck: Juzám Djinn (BB pips) under a WUBG commander,
    heavy enough black demand that duals matter."""
    return {oid(loaded_conn, "Juzám Djinn"): 1, oid(loaded_conn, "Standard Strike"): 0}


def test_recommend_covers_deficits_prefers_untapped(loaded_conn):
    entries = {oid(loaded_conn, "Juzám Djinn"): 1}
    result = manabase.recommend_manabase(
        loaded_conn,
        entries,
        COMMANDER,
        commander_oid=oid(loaded_conn, "Atraxa, Praetors' Voice"),
        land_count=37,
    )
    assert result.land_count == 37
    total = sum(q for _, q, *_ in result.lands) + sum(result.basics.values())
    assert total == 37
    names = {n for n, *_ in result.lands}
    # with a deep deficit both duals are correct picks (tapped included)
    assert {"Testing Grounds", "Sleepy Gate"} <= names
    # B needs Karsten's double-pip requirement; achieved must be reported
    assert result.needed["B"] == 30
    assert result.achieved["B"] > 0

    # untapped preference shows in pick order: with one slot, the untapped
    # dual wins over its functionally identical tapped twin
    one = manabase.recommend_manabase(
        loaded_conn,
        entries,
        COMMANDER,
        commander_oid=oid(loaded_conn, "Atraxa, Praetors' Voice"),
        land_count=1,
    )
    assert [n for n, *_ in one.lands] == ["Testing Grounds"]


def test_budget_excludes_pricy_lands(loaded_conn):
    entries = {oid(loaded_conn, "Juzám Djinn"): 1}
    result = manabase.recommend_manabase(
        loaded_conn,
        entries,
        COMMANDER,
        commander_oid=oid(loaded_conn, "Atraxa, Praetors' Voice"),
        land_count=37,
        budget=5.00,
    )
    assert all(n != "Pricy Cavern" for n, *_ in result.lands)


def test_identity_restricts_candidates(loaded_conn):
    # mono-black commander: the W/B dual is off-identity
    entries = {oid(loaded_conn, "Juzám Djinn"): 1}
    result = manabase.recommend_manabase(
        loaded_conn,
        entries,
        COMMANDER,
        commander_oid=oid(loaded_conn, "Juzám Djinn"),
        land_count=37,
    )
    names = {n for n, *_ in result.lands}
    assert "Testing Grounds" not in names
    assert result.basics.get("Swamp", 0) > 30  # mono-black: basics carry it


def test_goldfish_compare_runs(loaded_conn):
    entries = {oid(loaded_conn, "Juzám Djinn"): 1}
    result = manabase.recommend_manabase(
        loaded_conn,
        entries,
        COMMANDER,
        commander_oid=oid(loaded_conn, "Juzám Djinn"),
        land_count=20,
    )
    cmp = manabase.goldfish_compare(
        loaded_conn, entries, result, oid(loaded_conn, "Juzám Djinn"), games=20
    )
    assert 0.0 <= cmp["recommended"]["score"] <= 1.0
    assert 0.0 <= cmp["all_basics"]["score"] <= 1.0


def test_basic_land_split_proportional():
    split = manabase.basic_land_split({"W": 10, "B": 20}, 30)
    assert split == {"W": 10, "B": 20}
    assert sum(manabase.basic_land_split({}, 7).values()) == 7


def test_colorless_commander_gets_wastes(loaded_conn):
    entries = {oid(loaded_conn, "Sol Ring"): 1}
    result = manabase.recommend_manabase(
        loaded_conn,
        entries,
        COMMANDER,
        commander_oid=oid(loaded_conn, "Test Titan of the Wastes"),
        land_count=20,
    )
    assert result.basics == {"Wastes": 20}
    assert not result.lands or all(
        colors == "" for _n, _q, colors, *_ in result.lands
    )
