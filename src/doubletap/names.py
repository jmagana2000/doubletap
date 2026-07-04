import sqlite3
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz, process


def normalize(name: str) -> str:
    """Casefold, strip diacritics, collapse punctuation and whitespace."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    return " ".join(s.split())


@dataclass
class Match:
    oracle_id: str
    name: str
    score: float


def _cards_for_norm(
    conn: sqlite3.Connection, name_norm: str, score: float
) -> list[Match]:
    # a face name can collide with another card's full name; return every card
    # that carries this name, full-name matches before face matches
    rows = conn.execute(
        "SELECT c.oracle_id, c.name FROM card_names n"
        " JOIN cards c ON c.oracle_id = n.oracle_id"
        " WHERE n.name_norm = ? ORDER BY c.name_norm != ?, c.name",
        (name_norm, name_norm),
    ).fetchall()
    return [Match(oracle_id, name, score) for oracle_id, name in rows]


def lookup(conn: sqlite3.Connection, query: str, limit: int = 3) -> list[Match]:
    """Resolve a card name. Exact normalized match wins; otherwise return the
    top fuzzy candidates (never a silent best guess — the caller decides)."""
    q = normalize(query)
    if not q:
        return []

    matches = _cards_for_norm(conn, q, 100.0)
    if matches:
        return matches

    norms = [
        row[0] for row in conn.execute("SELECT DISTINCT name_norm FROM card_names")
    ]
    if not norms:
        return []
    hits = process.extract(q, norms, scorer=fuzz.WRatio, limit=limit, score_cutoff=60)
    matches = []
    for name_norm, score, _ in hits:
        matches.extend(_cards_for_norm(conn, name_norm, float(score)))
    return matches[:limit]
