import gzip
import json
import sqlite3
from dataclasses import dataclass

import httpx

from .db import data_home
from .names import normalize

BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
USER_AGENT = "DoubleTap/0.1 (joshua.magana@gmail.com)"


@dataclass
class SyncResult:
    updated: bool
    updated_at: str
    card_count: int


def _bulk_info(client: httpx.Client) -> dict:
    resp = client.get(BULK_DATA_URL)
    resp.raise_for_status()
    for entry in resp.json()["data"]:
        if entry["type"] == "oracle_cards":
            return entry
    raise RuntimeError("oracle_cards bulk entry not found in Scryfall catalog")


def searchable_names(card: dict) -> set[str]:
    names = {card["name"]}
    for face in card.get("card_faces", []):
        names.add(face["name"])
    return names


def sync(
    conn: sqlite3.Connection, client: httpx.Client | None = None, force: bool = False
) -> SyncResult:
    """Refresh the local card cache from Scryfall's oracle_cards bulk file.

    Skips the download when the bulk file's updated_at matches the cached one.
    The gzipped JSONL is kept on disk so tables can be rebuilt without
    re-downloading.
    """
    own_client = client is None
    if own_client:
        client = httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True
        )
    try:
        info = _bulk_info(client)
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'bulk_updated_at'"
        ).fetchone()
        if not force and row and row[0] == info["updated_at"]:
            count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
            return SyncResult(
                updated=False, updated_at=info["updated_at"], card_count=count
            )

        cache_path = data_home() / "cache" / "oracle-cards.jsonl.gz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with client.stream("GET", info["jsonl_download_uri"]) as resp:
            resp.raise_for_status()
            with open(cache_path, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)

        count = load_cards(conn, cache_path)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('bulk_updated_at', ?)",
            (info["updated_at"],),
        )
        conn.commit()
        return SyncResult(updated=True, updated_at=info["updated_at"], card_count=count)
    finally:
        if own_client:
            client.close()


def load_cards(conn: sqlite3.Connection, jsonl_gz_path) -> int:
    """Full rebuild of cards/card_names from a gzipped JSONL bulk file."""
    conn.execute("DELETE FROM cards")
    conn.execute("DELETE FROM card_names")
    count = 0
    with gzip.open(jsonl_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            if card.get("object") != "card":
                continue
            conn.execute(
                "INSERT OR REPLACE INTO cards (oracle_id, name, name_norm, json) VALUES (?, ?, ?, ?)",
                (
                    card["oracle_id"],
                    card["name"],
                    normalize(card["name"]),
                    json.dumps(card),
                ),
            )
            for name in searchable_names(card):
                conn.execute(
                    "INSERT OR IGNORE INTO card_names (name_norm, oracle_id) VALUES (?, ?)",
                    (normalize(name), card["oracle_id"]),
                )
            count += 1
    conn.commit()
    return count
