import gzip
import json
import random
import sqlite3
import time
from collections import Counter
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

_DROPPED_CATEGORIES = {
    "sideboard",
    "maybeboard",
    "considering",
    "wishlist",
    "tokens",
    # the companion sits outside the starting deck; v1 corpus ignores it
    "companion",
}
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
    """GET with rate limiting and exponential backoff on 429/5xx and transport
    failures (dropped connections, timeouts); hard stop after retries."""
    for attempt in range(max_retries):
        limiter.wait()
        try:
            resp = client.get(url, params=params)
        except httpx.TransportError:
            limiter.sleep(2**attempt)
            continue
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
    order_by: str = "-viewCount",
) -> int:
    """Queue deck ids from the search API. Idempotent.

    Pages are requested explicitly: the API's `next` link goes null around
    page 100 (~6k decks) but direct page access keeps returning results far
    deeper (verified to page 700), so `next` cannot be trusted for large
    crawls. Stops on the first empty page.

    The max_decks bound counts entries *seen*, not newly inserted — a resumed
    crawl re-walks the same pages, and counting inserts would keep it paging
    until it found max_decks decks it had never seen. max_decks=0 skips
    discovery entirely (fetch the existing queue only)."""
    fmt_id = FORMAT_IDS[format_name]
    queued = 0
    seen = 0
    page = 1
    while seen < max_decks:
        data = get_json(
            client,
            limiter,
            SEARCH_URL,
            params={"deckFormat": fmt_id, "orderBy": order_by, "page": page},
        )
        results = data.get("results", [])
        if not results:
            break
        for entry in results:
            cur = conn.execute(
                "INSERT OR IGNORE INTO decks (deck_id, source, format, url, status) VALUES (?, 'archidekt', ?, ?, 'queued')",
                (
                    entry["id"],
                    format_name,
                    f"https://archidekt.com/decks/{entry['id']}/",
                ),
            )
            queued += cur.rowcount
            seen += 1
            if seen >= max_decks:
                break
        page += 1
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
    already-fetched decks are never requested again. Decks the API refuses with
    a 4xx (deleted/private) are marked 'gone' and skipped — one dead deck must
    not kill a large crawl."""
    rows = conn.execute(
        "SELECT deck_id FROM decks WHERE source = 'archidekt' AND format = ? AND status = 'queued'",
        (format_name,),
    ).fetchall()
    fetched = 0
    path = shard_path(format_name)
    for (deck_id,) in rows:
        try:
            raw = get_json(client, limiter, DECK_URL.format(deck_id=deck_id))
        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500:
                conn.execute(
                    "UPDATE decks SET status = 'gone', fetched_at = datetime('now') WHERE deck_id = ?",
                    (deck_id,),
                )
                conn.commit()
                continue
            raise
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
        if len(commanders) == 1:
            deck.commander = commanders[0]
        elif len(commanders) == 2:
            # Accept partner commanders: both must carry a Partner keyword
            def _has_partner(oid):
                row = conn.execute(
                    "SELECT json FROM cards WHERE oracle_id = ?", (oid,)
                ).fetchone()
                if not row:
                    return False
                card = json.loads(row[0])
                return any(k.startswith("Partner") for k in card.get("keywords", []))

            if _has_partner(commanders[0]) and _has_partner(commanders[1]):
                deck.commander = commanders[0]
                deck.partner = commanders[1]
            else:
                return None, "commander_count"
        else:
            return None, "commander_count"
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
                "UPDATE decks SET status = 'parsed', commander_oracle_id = ?, partner_oracle_id = ? WHERE deck_id = ?",
                (deck.commander, deck.partner, deck_id),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, ?)",
                [(deck_id, oid, qty) for oid, qty in deck.entries.items()],
            )
            for cmd_oid in (deck.commander, deck.partner):
                if cmd_oid:
                    conn.execute(
                        "INSERT OR REPLACE INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, 1)",
                        (deck_id, cmd_oid),
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
    order_by: str = "-viewCount",
) -> Counter:
    get_format(format_name)  # fail fast on unknown formats
    own_client = client is None
    client = client or make_client()
    limiter = limiter or RateLimiter()
    try:
        discover(conn, client, limiter, format_name, max_decks, order_by=order_by)
        fetch_queued(conn, client, limiter, format_name, progress=progress)
        return parse_shards(conn, format_name)
    finally:
        if own_client:
            client.close()
