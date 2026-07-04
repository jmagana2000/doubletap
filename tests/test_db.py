from doubletap import db


def test_connect_creates_schema_idempotently(data_home):
    conn = db.connect()
    conn.close()
    conn = db.connect()
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"cards", "card_names", "meta", "decks", "deck_cards"} <= tables
    conn.close()


def test_data_home_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DOUBLETAP_HOME", str(tmp_path / "custom"))
    assert db.data_home() == tmp_path / "custom"
