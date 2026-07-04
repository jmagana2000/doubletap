import gzip
from pathlib import Path

import pytest

from doubletap import db, scryfall

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DOUBLETAP_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def conn(data_home):
    conn = db.connect()
    yield conn
    conn.close()


@pytest.fixture
def fixture_bulk_gz(tmp_path):
    """The card fixture JSONL, gzipped the way Scryfall serves it."""
    dst = tmp_path / "oracle-cards.jsonl.gz"
    dst.write_bytes(gzip.compress((FIXTURES / "oracle-cards.jsonl").read_bytes()))
    return dst


@pytest.fixture
def loaded_conn(conn, fixture_bulk_gz):
    scryfall.load_cards(conn, fixture_bulk_gz)
    return conn
