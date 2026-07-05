import gzip
import json

import httpx
import pytest
import respx

from doubletap import archidekt
from doubletap.archidekt import (
    DECK_URL,
    SEARCH_URL,
    RateLimiter,
    crawl,
    get_json,
    parse_trimmed,
)
from doubletap.names import lookup


def oid(conn, name):
    return lookup(conn, name)[0].oracle_id


def api_card(conn, name, qty, categories=()):
    return {
        "quantity": qty,
        "categories": list(categories),
        "card": {
            "uid": "printing-uid",
            "oracleCard": {"uid": oid(conn, name), "name": name},
        },
    }


def commander_api_deck(conn, deck_id=101):
    return {
        "id": deck_id,
        "name": "fixture commander deck",
        "deckFormat": 3,
        "cards": [
            api_card(conn, "Atraxa, Praetors' Voice", 1, ["Commander"]),
            api_card(conn, "Sol Ring", 1, ["Artifact"]),
            api_card(conn, "Relentless Rats", 1, ["Creature"]),
            api_card(conn, "Swamp", 97, ["Land"]),
            api_card(conn, "Lightning Bolt", 1, ["Maybeboard"]),  # dropped
        ],
    }


def modern_api_deck(conn, deck_id=201):
    return {
        "id": deck_id,
        "name": "fixture modern deck",
        "deckFormat": 2,
        "cards": [
            api_card(conn, "Lightning Bolt", 4, ["Instant"]),
            api_card(conn, "Fire // Ice", 4, ["Instant"]),
            api_card(conn, "Swamp", 52, ["Land"]),
            api_card(conn, "Relentless Rats", 2, ["Sideboard"]),  # dropped
        ],
    }


def _trim(conn, raw, fmt):
    return archidekt.trim_deck(raw, fmt)


def test_parse_commander_deck(loaded_conn):
    deck, reason = parse_trimmed(
        loaded_conn, _trim(loaded_conn, commander_api_deck(loaded_conn), "commander")
    )
    assert reason == "ok"
    assert deck.commander == oid(loaded_conn, "Atraxa, Praetors' Voice")
    assert deck.size() == 100
    assert oid(loaded_conn, "Lightning Bolt") not in deck.entries  # maybeboard dropped


def test_parse_modern_deck_drops_sideboard(loaded_conn):
    deck, reason = parse_trimmed(
        loaded_conn, _trim(loaded_conn, modern_api_deck(loaded_conn), "modern")
    )
    assert reason == "ok"
    assert deck.size() == 60
    assert oid(loaded_conn, "Relentless Rats") not in deck.entries


def test_parse_rejects_partner_commanders(loaded_conn):
    raw = commander_api_deck(loaded_conn)
    raw["cards"].append(api_card(loaded_conn, "Juzám Djinn", 1, ["Commander"]))
    deck, reason = parse_trimmed(loaded_conn, _trim(loaded_conn, raw, "commander"))
    assert deck is None and reason == "commander_count"


def test_parse_rejects_unresolvable_cards(loaded_conn):
    raw = modern_api_deck(loaded_conn)
    raw["cards"].append(
        {
            "quantity": 4,
            "categories": [],
            "card": {"oracleCard": {"uid": "not-a-real-oracle-id", "name": "Mystery"}},
        }
    )
    deck, reason = parse_trimmed(loaded_conn, _trim(loaded_conn, raw, "modern"))
    assert deck is None and reason == "unresolved_cards"


def test_parse_rejects_invalid_deck(loaded_conn):
    raw = commander_api_deck(loaded_conn)
    raw["cards"][3]["quantity"] = 90  # 93 swamps -> size != 100
    deck, reason = parse_trimmed(loaded_conn, _trim(loaded_conn, raw, "commander"))
    assert deck is None and reason == "invalid"


def _search_page(entries):
    # `next` is deliberately absent: discovery paginates by explicit page
    # number and must stop on the first empty page, not on a null next link
    return {"count": len(entries), "results": [{"id": e} for e in entries]}


@respx.mock
def test_crawl_end_to_end_and_resumability(loaded_conn, data_home):
    limiter = RateLimiter(interval=0, jitter=0, sleep=lambda s: None)
    # all pages share the URL path, so one route serves them in order
    # (twice: crawl() is called again below to prove resumability)
    respx.get(SEARCH_URL).mock(
        side_effect=[
            httpx.Response(200, json=_search_page([101])),
            httpx.Response(200, json=_search_page([102])),
            httpx.Response(200, json=_search_page([])),
        ]
        * 2
    )
    deck_ok = respx.get(DECK_URL.format(deck_id=101)).mock(
        return_value=httpx.Response(200, json=commander_api_deck(loaded_conn, 101))
    )
    bad = commander_api_deck(loaded_conn, 102)
    bad["cards"][3]["quantity"] = 90
    respx.get(DECK_URL.format(deck_id=102)).mock(
        return_value=httpx.Response(200, json=bad)
    )

    client = httpx.Client()
    outcomes = crawl(loaded_conn, "commander", 10, client=client, limiter=limiter)
    assert outcomes == {"ok": 1, "invalid": 1}
    statuses = dict(loaded_conn.execute("SELECT deck_id, status FROM decks").fetchall())
    assert statuses == {101: "parsed", 102: "rejected"}
    n_cards = loaded_conn.execute(
        "SELECT COUNT(*) FROM deck_cards WHERE deck_id = 101"
    ).fetchone()[0]
    assert n_cards == 4  # commander + 3 distinct entries

    # resumable: nothing queued, so no deck is fetched twice
    crawl(loaded_conn, "commander", 10, client=client, limiter=limiter)
    assert deck_ok.call_count == 1

    shard = archidekt.shard_path("commander")
    with gzip.open(shard, "rt") as f:
        assert len(f.readlines()) == 2


@respx.mock
def test_get_json_backs_off_on_429():
    sleeps = []
    limiter = RateLimiter(interval=0, jitter=0, sleep=sleeps.append)
    route = respx.get("https://archidekt.com/api/thing")
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(500),
        httpx.Response(200, json={"ok": True}),
    ]
    assert get_json(httpx.Client(), limiter, "https://archidekt.com/api/thing") == {
        "ok": True
    }
    assert sleeps == [1, 2]  # exponential backoff 2^0, 2^1


@respx.mock
def test_get_json_hard_stops():
    limiter = RateLimiter(interval=0, jitter=0, sleep=lambda s: None)
    respx.get("https://archidekt.com/api/thing").mock(return_value=httpx.Response(429))
    with pytest.raises(RuntimeError):
        get_json(
            httpx.Client(), limiter, "https://archidekt.com/api/thing", max_retries=3
        )


def test_rate_limiter_spacing():
    clock = iter([0.0, 0.2, 0.2, 1.5, 1.5]).__next__
    sleeps = []
    limiter = RateLimiter(interval=1.0, jitter=0, sleep=sleeps.append, clock=clock)
    limiter.wait()  # first call: no sleep
    limiter.wait()  # 0.2s elapsed -> sleep 0.8
    limiter.wait()  # 1.3s elapsed -> no sleep
    assert sleeps == [pytest.approx(0.8)]
