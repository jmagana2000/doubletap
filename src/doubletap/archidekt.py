import gzip
import json
import random
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx

from .db import data_home
from .decks import Deck
from .formats import get_format, validate

SEARCH_URL = "https://archidekt.com/api/decks/v3/"
DECK_URL = "https://archidekt.com/api/decks/{deck_id}/"
USER_AGENT = "DoubleTap/0.1 (joshua.magana@gmail.com)"

# verified empirically 2026-07-04: deckFormat=2 returns Modern decks,
# deckFormat=3 returns Commander decks (7 is Custom — do not trust folklore ids)
FORMAT_IDS = {"commander": 3, "modern": 2}

_DROPPED_CATEGORIES = {"sideboard", "maybeboard", "considering", "wishlist", "tokens"}
MAX_UNRESOLVED_FRACTION = 0.02


class RateLimiter:
    def __init__(
        self,
        interval: float = 1.0,
        jitter: float = 0.3,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.interval = interval
        self.jitter = jitter
        self.sleep = sleep
        self.clock = clock
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            delay = (
                self.interval
                + random.uniform(0, self.jitter)
                - (self.clock() - self._last)
            )
            if delay > 0:
                self.sleep(delay)
        self._last = self.clock()


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=60,
        follow_redirects=True,
    )


def get_json(
    client: httpx.Client,
    limiter: RateLimiter,
    url: str,
    params: dict | None = None,
    max_retries: int = 5,
):
    """GET with rate limiting and exponential backoff on 429/5xx; hard stop after retries."""
    for attempt in range(max_retries):
        limiter.wait()
        resp = client.get(url, params=params)
        if resp.status_code == 429 or resp.status_code >= 500:
            limiter.sleep(2**attempt)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(
        f"giving up on {url} after {max_retries} attempts (rate limited?)"
    )


def discover(
    conn: sqlite3.Connection,
    client: httpx.Client,
    limiter: RateLimiter,
    format_name: str,
    max_decks: int,
) -> int:
    """Queue deck ids from the search API (most-viewed first). Idempotent."""
    fmt_id = FORMAT_IDS[format_name]
    queued = 0
    url, params = SEARCH_URL, {"deckFormat": fmt_id, "orderBy": "-viewCount"}
    while queued < max_decks and url:
        data = get_json(client, limiter, url, params=params)
        for entry in data.get("results", []):
            cur = conn.execute(
                "INSERT OR IGNORE INTO decks (deck_id, source, format, url, status) VALUES (?, 'archidekt', ?, ?, 'queued')",
                (
                    entry["id"],
                    format_name,
                    f"https://archidekt.com/decks/{entry['id']}/",
                ),
            )
            queued += cur.rowcount
            if queued >= max_decks:
                break
        url, params = data.get("next"), None
        conn.commit()
    return queued


def trim_deck(raw: dict, format_name: str) -> dict:
    """Reduce a deck API response to the fields the corpus needs."""
    cards = []
    for entry in raw.get("cards", []):
        oracle = entry["card"].get("oracleCard") or {}
        cards.append(
            {
                "qty": entry.get("quantity", 1),
                "oracle_id": oracle.get("uid"),
                "name": oracle.get("name"),
                "categories": entry.get("categories") or [],
            }
        )
    return {
        "id": raw["id"],
        "name": raw.get("name"),
        "format": format_name,
        "cards": cards,
    }


def shard_path(format_name: str) -> Path:
    path = data_home() / "corpus" / "raw" / f"{format_name}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fetch_queued(
    conn: sqlite3.Connection,
    client: httpx.Client,
    limiter: RateLimiter,
    format_name: str,
    progress=None,
) -> int:
    """Fetch queued decks, appending trimmed records to the raw shard. Resumable:
    already-fetched decks are never requested again."""
    rows = conn.execute(
        "SELECT deck_id FROM decks WHERE source = 'archidekt' AND format = ? AND status = 'queued'",
        (format_name,),
    ).fetchall()
    fetched = 0
    path = shard_path(format_name)
    for (deck_id,) in rows:
        raw = get_json(client, limiter, DECK_URL.format(deck_id=deck_id))
        with gzip.open(path, "at", encoding="utf-8") as f:
            f.write(json.dumps(trim_deck(raw, format_name), ensure_ascii=False) + "\n")
        conn.execute(
            "UPDATE decks SET status = 'fetched', fetched_at = datetime('now') WHERE deck_id = ?",
            (deck_id,),
        )
        conn.commit()
        fetched += 1
        if progress:
            progress(fetched, len(rows))
    return fetched


def parse_trimmed(conn: sqlite3.Connection, trimmed: dict) -> tuple[Deck | None, str]:
    """Build a validated Deck from a trimmed record; (None, reason) on rejection."""
    fmt = get_format(trimmed["format"])
    deck = Deck(format=fmt.name)
    commanders = []
    unresolved = 0
    total = 0
    known = set()
    for card in trimmed["cards"]:
        categories = {c.casefold() for c in card["categories"]}
        if categories & _DROPPED_CATEGORIES:
            continue
        total += card["qty"]
        oid = card["oracle_id"]
        if oid not in known:
            exists = conn.execute(
                "SELECT 1 FROM cards WHERE oracle_id = ?", (oid,)
            ).fetchone()
            if not exists:
                unresolved += card["qty"]
                continue
            known.add(oid)
        if "commander" in categories:
            commanders.append(oid)
        else:
            deck.entries[oid] += card["qty"]
    if total == 0:
        return None, "empty"
    if unresolved / total > MAX_UNRESOLVED_FRACTION:
        return None, "unresolved_cards"
    if fmt.requires_commander:
        if len(commanders) != 1:  # partners/companions are out of scope in v1
            return None, "commander_count"
        deck.commander = commanders[0]
    elif commanders:
        return None, "unexpected_commander"
    if validate(conn, deck):
        return None, "invalid"
    return deck, "ok"


def parse_shards(conn: sqlite3.Connection, format_name: str) -> Counter:
    """(Re)parse the raw shard into decks/deck_cards. Only fetched/parsed/rejected
    rows are touched, so this can rebuild tables without re-crawling."""
    outcomes = Counter()
    path = shard_path(format_name)
    if not path.exists():
        return outcomes
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            trimmed = json.loads(line)
            deck, reason = parse_trimmed(conn, trimmed)
            outcomes[reason] += 1
            deck_id = trimmed["id"]
            conn.execute("DELETE FROM deck_cards WHERE deck_id = ?", (deck_id,))
            if deck is None:
                conn.execute(
                    "UPDATE decks SET status = 'rejected' WHERE deck_id = ?", (deck_id,)
                )
                continue
            conn.execute(
                "UPDATE decks SET status = 'parsed', commander_oracle_id = ? WHERE deck_id = ?",
                (deck.commander, deck_id),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, ?)",
                [(deck_id, oid, qty) for oid, qty in deck.entries.items()],
            )
            if deck.commander:
                conn.execute(
                    "INSERT OR REPLACE INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, 1)",
                    (deck_id, deck.commander),
                )
    conn.commit()
    return outcomes


def crawl(
    conn: sqlite3.Connection,
    format_name: str,
    max_decks: int,
    client: httpx.Client | None = None,
    limiter: RateLimiter | None = None,
    progress=None,
) -> Counter:
    get_format(format_name)  # fail fast on unknown formats
    own_client = client is None
    client = client or make_client()
    limiter = limiter or RateLimiter()
    try:
        discover(conn, client, limiter, format_name, max_decks)
        fetch_queued(conn, client, limiter, format_name, progress=progress)
        return parse_shards(conn, format_name)
    finally:
        if own_client:
            client.close()
