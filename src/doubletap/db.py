import os
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    oracle_id TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    name_norm TEXT NOT NULL,
    json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cards_name_norm ON cards(name_norm);

-- one row per searchable name: full card name plus each face name for
-- multi-faced cards, so "malakir rebirth" resolves the MDFC
CREATE TABLE IF NOT EXISTS card_names (
    name_norm TEXT NOT NULL,
    oracle_id TEXT NOT NULL,
    PRIMARY KEY (name_norm, oracle_id)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decks (
    deck_id             INTEGER PRIMARY KEY,
    source              TEXT NOT NULL,
    format              TEXT NOT NULL,
    commander_oracle_id TEXT,
    url                 TEXT,
    fetched_at          TEXT,
    status              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS deck_cards (
    deck_id   INTEGER NOT NULL,
    oracle_id TEXT NOT NULL,
    qty       INTEGER NOT NULL,
    PRIMARY KEY (deck_id, oracle_id)
);
"""


def data_home() -> Path:
    return Path(os.environ.get("DOUBLETAP_HOME", str(Path.home() / ".doubletap")))


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or data_home() / "doubletap.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn
