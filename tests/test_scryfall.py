import gzip
import json
from pathlib import Path

import httpx
import respx

from doubletap import scryfall

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_COUNT = len(
    (FIXTURES / "oracle-cards.jsonl").read_text(encoding="utf-8").splitlines()
)

DOWNLOAD_URL = "https://data.scryfall.io/oracle-cards/test.jsonl.gz"
CATALOG = {
    "data": [
        {"type": "rulings", "updated_at": "x", "jsonl_download_uri": "https://x/r.gz"},
        {
            "type": "oracle_cards",
            "updated_at": "2026-07-04T09:00:00+00:00",
            "jsonl_download_uri": DOWNLOAD_URL,
        },
    ]
}


def _mock_routes():
    bulk_bytes = gzip.compress((FIXTURES / "oracle-cards.jsonl").read_bytes())
    respx.get(scryfall.BULK_DATA_URL).mock(
        return_value=httpx.Response(200, json=CATALOG)
    )
    return respx.get(DOWNLOAD_URL).mock(
        return_value=httpx.Response(200, content=bulk_bytes)
    )


@respx.mock
def test_sync_loads_skips_and_forces(conn):
    download = _mock_routes()
    client = httpx.Client()

    result = scryfall.sync(conn, client=client)
    assert result.updated
    assert result.card_count == FIXTURE_COUNT
    assert download.call_count == 1

    result = scryfall.sync(conn, client=client)
    assert not result.updated
    assert result.card_count == FIXTURE_COUNT
    assert download.call_count == 1  # no re-download when updated_at matches

    result = scryfall.sync(conn, client=client, force=True)
    assert result.updated
    assert download.call_count == 2


def test_load_cards_is_a_full_rebuild(conn, fixture_bulk_gz):
    assert scryfall.load_cards(conn, fixture_bulk_gz) == FIXTURE_COUNT
    assert scryfall.load_cards(conn, fixture_bulk_gz) == FIXTURE_COUNT
    assert conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == FIXTURE_COUNT


def test_multiface_names_indexed(loaded_conn):
    rows = loaded_conn.execute(
        "SELECT n.name_norm FROM card_names n JOIN cards c ON c.oracle_id = n.oracle_id"
        " WHERE c.name = 'Malakir Rebirth // Malakir Mire'"
    ).fetchall()
    norms = {r[0] for r in rows}
    assert norms == {"malakir rebirth malakir mire", "malakir rebirth", "malakir mire"}


def test_full_card_json_round_trips(loaded_conn):
    (raw,) = loaded_conn.execute(
        "SELECT json FROM cards WHERE name = 'Sol Ring'"
    ).fetchone()
    card = json.loads(raw)
    assert card["legalities"]["commander"] == "legal"
    assert card["legalities"]["modern"] == "not_legal"
