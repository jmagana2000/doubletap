import csv
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .names import Match, lookup

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".bmp"}

# "4 Lightning Bolt", "4x Lightning Bolt", "Lightning Bolt"
_QTY_RE = re.compile(r"^(\d+)\s*[xX]?\s+(.+)$")
# trailing Moxfield/Arena-style "(2XM) 190" set + collector number
_SET_TAIL_RE = re.compile(r"\s+\([A-Za-z0-9]{2,6}\)\s+[\w★-]+\s*$")
_MARKER_RE = re.compile(r"\s*\*(CMDR|F|E)\*\s*", re.IGNORECASE)

_COMMANDER_SECTIONS = {"commander", "commanders"}
_SKIPPED_SECTIONS = {"sideboard", "side", "maybeboard", "considering", "tokens"}


@dataclass
class Deck:
    format: str
    entries: Counter = field(default_factory=Counter)  # oracle_id -> qty
    commander: str | None = None  # oracle_id
    partner: str | None = None  # oracle_id; set only for partner-commander pairs

    def size(self) -> int:
        return (
            sum(self.entries.values())
            + (1 if self.commander else 0)
            + (1 if self.partner else 0)
        )

    def save(self, conn: sqlite3.Connection, path: Path) -> None:
        def named(oracle_id):
            (name,) = conn.execute(
                "SELECT name FROM cards WHERE oracle_id = ?", (oracle_id,)
            ).fetchone()
            return name

        doc = {
            "format": self.format,
            "commander": (
                {"oracle_id": self.commander, "name": named(self.commander)}
                if self.commander
                else None
            ),
            "partner": (
                {"oracle_id": self.partner, "name": named(self.partner)}
                if self.partner
                else None
            ),
            "cards": [
                {"oracle_id": oid, "name": named(oid), "qty": qty}
                for oid, qty in sorted(
                    self.entries.items(), key=lambda kv: named(kv[0])
                )
            ],
        }
        Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Deck":
        doc = json.loads(Path(path).read_text())
        deck = cls(format=doc["format"])
        if doc.get("commander"):
            deck.commander = doc["commander"]["oracle_id"]
        if doc.get("partner"):
            deck.partner = doc["partner"]["oracle_id"]
        for card in doc["cards"]:
            deck.entries[card["oracle_id"]] += card["qty"]
        return deck


@dataclass
class ParsedLine:
    raw: str
    qty: int
    name: str
    is_commander: bool = False


@dataclass
class LineResult:
    line: ParsedLine
    status: str  # resolved | assumed | ambiguous | unmatched
    matches: list[Match] = field(default_factory=list)


@dataclass
class ImportResult:
    deck: Deck
    resolved: list[LineResult] = field(default_factory=list)
    assumed: list[LineResult] = field(default_factory=list)
    ambiguous: list[LineResult] = field(default_factory=list)
    unmatched: list[LineResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.ambiguous and not self.unmatched


def parse_text_lines(lines: list[str]) -> list[ParsedLine]:
    """Parse a plain-text decklist: quantities, section headers, Moxfield markers.

    Sideboard/maybeboard sections are dropped (out of scope in v1)."""
    parsed = []
    section = "main"
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(("#", "//")):
            continue
        header = line.rstrip(":").casefold()
        if header in _COMMANDER_SECTIONS | _SKIPPED_SECTIONS | {
            "deck",
            "mainboard",
            "main",
        }:
            section = header
            continue
        if section in _SKIPPED_SECTIONS:
            continue
        is_commander = section in _COMMANDER_SECTIONS
        if (
            _MARKER_RE.search(line)
            and "cmdr" in _MARKER_RE.search(line).group(1).casefold()
        ):
            is_commander = True
        line = _MARKER_RE.sub(" ", line).strip()
        line = _SET_TAIL_RE.sub("", line)
        m = _QTY_RE.match(line)
        qty, name = (int(m.group(1)), m.group(2)) if m else (1, line)
        parsed.append(
            ParsedLine(raw=raw.strip(), qty=qty, name=name, is_commander=is_commander)
        )
    return parsed


def parse_csv(path: Path) -> list[ParsedLine]:
    """Parse a Moxfield/Archidekt-style CSV export (Count/Quantity + Name columns)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.casefold(): c for c in reader.fieldnames or []}
        qty_col = next(
            (cols[k] for k in ("count", "quantity", "qty") if k in cols), None
        )
        name_col = cols.get("name")
        if not qty_col or not name_col:
            raise ValueError(
                f"CSV needs a count/quantity column and a name column, got: {reader.fieldnames}"
            )
        return [
            ParsedLine(
                raw=row[name_col], qty=int(row[qty_col] or 1), name=row[name_col]
            )
            for row in reader
            if row.get(name_col, "").strip()
        ]


def resolve(
    conn: sqlite3.Connection,
    parsed: list[ParsedLine],
    deck_format: str,
    threshold: float = 90.0,
    gap: float = 5.0,
    chooser: Callable[[ParsedLine, list[Match]], Match | None] | None = None,
) -> ImportResult:
    """Resolve parsed lines against the card cache. Exact matches resolve; fuzzy
    matches above `threshold` with a clear `gap` to the next distinct card are
    accepted but reported as assumed; everything else is ambiguous/unmatched
    (optionally settled by `chooser` — e.g. an interactive prompt)."""
    result = ImportResult(deck=Deck(format=deck_format))
    buckets = {
        "resolved": result.resolved,
        "assumed": result.assumed,
        "ambiguous": result.ambiguous,
        "unmatched": result.unmatched,
    }
    for line in parsed:
        matches = lookup(conn, line.name)
        top = matches[0] if matches else None
        if top and top.score == 100.0:
            status = "resolved"
        elif top and top.score >= threshold:
            others = [m for m in matches if m.name != top.name]
            status = (
                "assumed"
                if not others or top.score - others[0].score >= gap
                else "ambiguous"
            )
        else:
            status = "unmatched"

        if status in ("ambiguous", "unmatched") and chooser and matches:
            chosen = chooser(line, matches)
            if chosen is not None:
                matches, status = [chosen], "resolved"

        buckets[status].append(LineResult(line, status, matches))
        if status in ("resolved", "assumed"):
            if line.is_commander:
                if result.deck.commander is None:
                    result.deck.commander = matches[0].oracle_id
                else:
                    result.deck.partner = matches[0].oracle_id
            else:
                result.deck.entries[matches[0].oracle_id] += line.qty
    return result


def load_lines(path: Path) -> list[ParsedLine]:
    """Route a file to the right parser: CSV, image (OCR), or plain text."""
    path = Path(path)
    if path.suffix.casefold() == ".csv":
        return parse_csv(path)
    if path.suffix.casefold() in IMAGE_SUFFIXES:
        from .ocr import recognize_text

        lines = [text for text, conf in recognize_text(path) if conf >= 0.3]
        return parse_text_lines(lines)
    return parse_text_lines(path.read_text(encoding="utf-8").splitlines())
