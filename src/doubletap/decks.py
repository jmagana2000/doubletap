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

# Patterns that disqualify a line from being a card name in physical card OCR
_NON_NAME_TOKENS = re.compile(
    r"\b(?:Creature|Instant|Sorcery|Artifact|Enchantment|Land|Planeswalker|Battle)\b"
)
_COLLECTOR_OR_PT = re.compile(r"^\d+$|^\d+[/\\]\d+|^[A-Z]\s+\d+$")

_COMMANDER_SECTIONS = {"commander", "commanders"}
_COMPANION_SECTIONS = {"companion"}
_SKIPPED_SECTIONS = {"sideboard", "side", "maybeboard", "considering", "tokens"}


@dataclass
class Deck:
    format: str
    entries: Counter = field(default_factory=Counter)  # oracle_id -> qty
    commander: str | None = None  # oracle_id
    partner: str | None = None  # oracle_id; set only for partner-commander pairs
    companion: str | None = None  # oracle_id; sits outside the starting deck

    def size(self) -> int:
        # the companion is not part of the starting deck, so it never counts
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
            "companion": (
                {"oracle_id": self.companion, "name": named(self.companion)}
                if self.companion
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
        if doc.get("companion"):
            deck.companion = doc["companion"]["oracle_id"]
        for card in doc["cards"]:
            deck.entries[card["oracle_id"]] += card["qty"]
        return deck


@dataclass
class ParsedLine:
    raw: str
    qty: int
    name: str
    is_commander: bool = False
    is_companion: bool = False


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
        if header in _COMMANDER_SECTIONS | _COMPANION_SECTIONS | _SKIPPED_SECTIONS | {
            "deck",
            "mainboard",
            "main",
        }:
            section = header
            continue
        if section in _SKIPPED_SECTIONS:
            continue
        is_commander = section in _COMMANDER_SECTIONS
        is_companion = section in _COMPANION_SECTIONS
        # a line can carry several markers (foil commanders: "*F* *CMDR*")
        if any("cmdr" in m.group(1).casefold() for m in _MARKER_RE.finditer(line)):
            is_commander = True
        line = _MARKER_RE.sub(" ", line).strip()
        line = _SET_TAIL_RE.sub("", line)
        m = _QTY_RE.match(line)
        qty, name = (int(m.group(1)), m.group(2)) if m else (1, line)
        parsed.append(
            ParsedLine(
                raw=raw.strip(),
                qty=qty,
                name=name,
                is_commander=is_commander,
                is_companion=is_companion,
            )
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
        category_col = next(
            (cols[k] for k in ("category", "board", "section") if k in cols), None
        )
        out = []
        for row in reader:
            name = (row.get(name_col) or "").strip()  # short rows fill None
            if not name:
                continue
            category = (
                (row.get(category_col) or "").strip().casefold() if category_col else ""
            )
            if category in _SKIPPED_SECTIONS:
                continue
            out.append(
                ParsedLine(
                    raw=name,
                    qty=int(row.get(qty_col) or 1),
                    name=name,
                    is_commander=category in _COMMANDER_SECTIONS,
                    is_companion=category in _COMPANION_SECTIONS,
                )
            )
        return out


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
            # command-zone slots only exist in commander; elsewhere a *CMDR*
            # marker is just a main-deck card (and gets legality-checked).
            # local import: formats imports Deck from this module
            from .formats import get_format

            uses_zone = get_format(deck_format).requires_commander
            if line.is_commander and uses_zone:
                if result.deck.commander is None:
                    result.deck.commander = matches[0].oracle_id
                elif result.deck.partner is None:
                    result.deck.partner = matches[0].oracle_id
                else:
                    # a third commander line is a malformed list, not a silent drop
                    buckets["ambiguous"].append(LineResult(line, "ambiguous", matches))
            elif line.is_companion:  # companions exist in every format
                if result.deck.companion is None:
                    result.deck.companion = matches[0].oracle_id
                else:
                    buckets["ambiguous"].append(LineResult(line, "ambiguous", matches))
            else:
                result.deck.entries[matches[0].oracle_id] += line.qty
    return result


def _looks_like_card_name(text: str) -> bool:
    """Heuristic: could this OCR line be an MTG card name?
    Rejects type lines, P/T, collector numbers, and pure non-alpha strings."""
    if len(text) < 2 or len(text) > 60:
        return False
    if not any(c.isalpha() for c in text):
        return False
    if _COLLECTOR_OR_PT.match(text):
        return False
    if _NON_NAME_TOKENS.search(text):
        return False
    return True


def load_lines(path: Path) -> list[ParsedLine]:
    """Route a file to the right parser: CSV, image (OCR), or plain text."""
    path = Path(path)
    if path.suffix.casefold() == ".csv":
        return parse_csv(path)
    if path.suffix.casefold() in IMAGE_SUFFIXES:
        from .ocr import recognize_text

        try:
            raw = recognize_text(path)
        except RuntimeError as e:
            raise ValueError(f"Could not read image {path.name}: {e}") from e

        lines_with_conf = [(text, conf) for text, conf in raw if conf >= 0.3]
        texts = [text for text, _ in lines_with_conf]

        # Decklist screenshots have quantity-prefixed lines ("4 Lightning Bolt").
        # Physical card photos don't — but a decklist without quantities is a
        # list of MANY name-like lines; only collapse to a single card when the
        # image really looks like one card, else parse every line as qty 1.
        if not any(_QTY_RE.match(t) for t in texts):
            name_lines = [
                text for text, _conf in lines_with_conf if _looks_like_card_name(text)
            ]
            if len(name_lines) <= 3:  # physical card photo: title + noise
                return [ParsedLine(raw=t, qty=1, name=t) for t in name_lines[:1]]
            return [ParsedLine(raw=t, qty=1, name=t) for t in name_lines]

        return parse_text_lines(texts)
    return parse_text_lines(path.read_text(encoding="utf-8").splitlines())
