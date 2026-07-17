"""Local web UI: a thin HTTP layer over the CLI.

Every action runs the real Typer app in-process, so the web UI has CLI parity
by construction — it cannot drift from the command line. Stdlib server only;
no new dependencies."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "static"

# first CLI token a browser is allowed to run; "web" itself is excluded
ALLOWED_COMMANDS = {
    "cards",
    "deck",
    "corpus",
    "train",
    "eval",
    "recommend",
    "complete",
}

# CliRunner patches sys.stdout — one command at a time
_run_lock = threading.Lock()


def run_cli(args: list[str]) -> tuple[str, int]:
    """Run the real CLI in-process and return (output, exit_code)."""
    # ponytail: click's test runner is the in-process CLI entry point;
    # it ships with click, so this is not a test-only import in practice
    from typer.testing import CliRunner

    from .cli import app

    with _run_lock:
        result = CliRunner().invoke(app, args)
    output = result.output
    if result.exception and not isinstance(result.exception, SystemExit):
        output += f"\nerror: {result.exception}"
        return output, 1
    return output, result.exit_code


def list_decks() -> list[dict]:
    """Structured deck list for the UI's deck browser."""
    from . import db, decks

    conn = db.connect()

    def named(oid):
        row = conn.execute(
            "SELECT name FROM cards WHERE oracle_id = ?", (oid,)
        ).fetchone()
        return row[0] if row else oid

    def art_and_colors(deck):
        """Commander art (or first card's art) plus identity for gallery cards."""
        from . import formats

        oid = deck.commander or next(iter(deck.entries), None)
        if not oid:
            return "", []
        card = formats.get_card(conn, oid)
        images = card.get("image_uris") or (
            (card.get("card_faces") or [{}])[0].get("image_uris") or {}
        )
        colors = set(card.get("color_identity") or [])
        if deck.partner:
            partner = formats.get_card(conn, deck.partner)
            colors |= set(partner.get("color_identity") or [])
        return images.get("art_crop", ""), [c for c in "WUBRG" if c in colors]

    out = []
    for f in sorted(db.decks_dir().glob("*.json")):
        try:
            deck = decks.Deck.load(f)
        except Exception:
            continue
        commander = named(deck.commander) if deck.commander else None
        if deck.partner:
            commander += f" + {named(deck.partner)}"
        art, colors = art_and_colors(deck)
        out.append(
            {
                "name": f.stem,
                "path": str(f),
                "format": deck.format,
                "cards": deck.size(),
                "commander": commander,
                "art": art,
                "colors": colors,
                "mtime": f.stat().st_mtime,
            }
        )
    return out


def _card_view(card: dict) -> dict:
    """The card fields the builder UI renders, from the raw Scryfall JSON."""
    from .analysis import card_price

    images = card.get("image_uris") or (
        (card.get("card_faces") or [{}])[0].get("image_uris") or {}
    )
    cost = card.get("mana_cost")
    if not cost and card.get("card_faces"):
        cost = card["card_faces"][0].get("mana_cost", "")
    text = card.get("oracle_text")
    if text is None:
        text = "\n—\n".join(
            f.get("oracle_text", "") for f in card.get("card_faces", [])
        )
    from .analysis import source_weights

    return {
        "oracle_id": card["oracle_id"],
        "name": card["name"],
        "mana_cost": cost or "",
        "cmc": card.get("cmc", 0),
        "type_line": card.get("type_line", ""),
        "oracle_text": text,
        "colors": card.get("color_identity") or [],
        "art": images.get("art_crop", ""),
        "image": images.get("normal", ""),
        "price": card_price(card),
        "sources": source_weights(card),  # Karsten fractional colored sources
    }


def search_cards(
    q: str = "",
    colors: str = "",
    type_: str = "",
    max_mv: str = "",
    fmt: str = "commander",
    limit: int = 60,
) -> list[dict]:
    """Filterable card browse for the builder grid, popularity-ordered.
    Filters run in SQL over the JSON blobs; only matching rows are parsed."""
    from . import db
    from .names import normalize

    conn = db.connect()
    sql = (
        "SELECT json FROM cards WHERE"
        " json_extract(json, '$.legalities.' || ?) = 'legal'"
    )
    params: list = [fmt]
    if q:
        sql += " AND name_norm LIKE ?"
        params.append(f"%{normalize(q)}%")
    if type_:
        sql += " AND json_extract(json, '$.type_line') LIKE ?"
        params.append(f"%{type_}%")
    if max_mv:
        sql += " AND json_extract(json, '$.cmc') <= ?"
        params.append(float(max_mv))
    if colors:  # cards castable within this identity: no off-color symbols
        for c in set("WUBRG") - set(colors.upper()):
            # cards.json must be table-qualified or SQLite won't correlate it
            sql += (
                " AND NOT EXISTS (SELECT 1 FROM"
                " json_each(cards.json, '$.color_identity') je WHERE je.value = ?)"
            )
            params.append(c)
    sql += " ORDER BY COALESCE(json_extract(json, '$.edhrec_rank'), 99999) LIMIT ?"
    params.append(limit)
    return [_card_view(json.loads(raw)) for (raw,) in conn.execute(sql, params)]


def suggest_cards(
    path: str,
    k: int = 12,
    type_: str = "",
    personalize: float = 0.3,
    max_price: float | None = None,
) -> dict:
    """Structured recommendations for the builder and Suggestions pages:
    top-k additions with scores and synergy rationale, optionally filtered
    by card type and price — the interactive counterpart of the recommend
    CLI (same model, same blend)."""
    import numpy as np

    from . import db, decks, formats
    from .cli import _budget_mask, _deck_to_idxs, _load_model

    conn = db.connect()
    deck = decks.Deck.load(Path(path))
    fmt = formats.get_format(deck.format)
    vocab, model, ckpt, _model_path = _load_model(conn, fmt, None)

    from .ml.eval import score_state
    from .ml.neighbors import blend, neighbor_frequencies
    from .ml.reward import PMIModel

    partial, commander_idx, partner_idx = _deck_to_idxs(conn, deck, vocab, fmt)
    extra_mask = _budget_mask(conn, vocab, max_price) if max_price else None
    scores = score_state(
        model, vocab, fmt, partial, commander_idx, partner_idx, extra_mask
    )
    freqs = neighbor_frequencies(conn, vocab, fmt, partial)
    if freqs is not None and personalize > 0:
        scores = blend(scores, freqs, personalize)
    pmi_path = db.data_home() / "models" / f"pmi_{fmt.name}.npz"
    pmi = PMIModel.load(pmi_path) if pmi_path.exists() else None

    suggestions = []
    for idx in np.argsort(-scores):
        if not np.isfinite(scores[idx]) or len(suggestions) == k:
            break
        card = formats.get_card(conn, vocab.oracle_ids[int(idx)])
        if type_ and type_ not in card.get("type_line", ""):
            continue
        view = _card_view(card)
        view["score"] = round(float(scores[idx]), 3)
        if pmi is not None:
            view["synergy"] = [
                formats.get_card(conn, vocab.oracle_ids[c])["name"]
                for c, _v in pmi.top_contributors(int(idx), partial)
            ]
        suggestions.append(view)
    return {
        "model": ckpt["algo"],
        "suggestions": suggestions,
        # so the UI can steer full decks toward Swap instead of Add
        "deck_size": deck.size(),
        "target_size": fmt.deck_size,
        "exact_size": fmt.exact_size,
    }


def deck_detail(path: str) -> dict:
    """One deck, structured for the builder: slots, per-card data, violations."""
    from . import db, decks, formats

    conn = db.connect()
    deck = decks.Deck.load(Path(path))

    def view(oid, qty=1):
        card = formats.get_card(conn, oid)
        return {**_card_view(card), "qty": qty}

    return {
        "path": path,
        "name": Path(path).stem,
        "format": deck.format,
        "size": deck.size(),
        "commander": view(deck.commander) if deck.commander else None,
        "partner": view(deck.partner) if deck.partner else None,
        "companion": view(deck.companion) if deck.companion else None,
        "cards": [view(oid, qty) for oid, qty in deck.entries.items()],
        "violations": [v.message for v in formats.validate(conn, deck)],
    }


def deck_analysis(path: str) -> dict:
    """Structured analytics for one deck: roles, curve, color balance,
    bracket, price — the data behind the Analytics screen."""
    from . import analysis, db, decks, formats

    conn = db.connect()
    deck = decks.Deck.load(Path(path))
    entries = dict(deck.entries)
    for oid in (deck.commander, deck.partner, deck.companion):
        if oid:
            entries[oid] = entries.get(oid, 0) + 1
    report = analysis.deck_report(conn, entries, deck.format)
    bracket, game_changers = formats.compute_bracket(
        [formats.get_card(conn, oid) for oid in entries]
    )
    return {
        "roles": {r: sorted(cards) for r, cards in report.by_role.items()},
        "targets": analysis.COMMANDER_TARGETS if deck.format == "commander" else {},
        "karsten_lands": report.karsten_lands,
        "eff_lands": round(report.eff_lands, 1),
        "cheap_draw_ramp": report.cheap_draw_ramp,
        "eff_sources": {c: round(v, 1) for c, v in report.eff_sources.items()},
        "max_pips": dict(report.max_pips),
        "curve": {str(mv): n for mv, n in sorted(report.curve.items())},
        "avg_mv": round(report.avg_mv, 2),
        "early_plays": report.early_plays,
        "pips": dict(report.pips),
        "sources": dict(report.sources),
        "short_colors": analysis.short_colors(report),
        "price": round(report.total_price, 2),
        "unpriced": report.unpriced,
        "bracket": bracket,
        "game_changers": sorted(game_changers),
        "violations": [v.message for v in formats.validate(conn, deck)],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet server
        pass

    def _send(self, code, body, content_type="application/json", no_cache=False):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if no_cache:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse

        url = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path in ("/", "/index.html"):
            # no-cache: stale browser copies of the single-page UI have
            # repeatedly hidden new features behind a silent cache hit
            self._send(
                200,
                (STATIC / "index.html").read_bytes(),
                "text/html",
                no_cache=True,
            )
        elif url.path == "/api/decks":
            self._send(200, list_decks())
        elif url.path == "/api/cards":
            self._send(
                200,
                search_cards(
                    q=qs.get("q", ""),
                    colors=qs.get("colors", ""),
                    type_=qs.get("type", ""),
                    max_mv=qs.get("max_mv", ""),
                    fmt=qs.get("format", "commander"),
                ),
            )
        elif url.path == "/api/deck":
            try:
                self._send(200, deck_detail(qs["path"]))
            except Exception as e:
                self._send(404, {"error": str(e)})
        elif url.path == "/api/analysis":
            try:
                self._send(200, deck_analysis(qs["path"]))
            except Exception as e:
                self._send(404, {"error": str(e)})
        elif url.path == "/api/suggest":
            try:
                self._send(
                    200,
                    suggest_cards(
                        qs["path"],
                        k=int(qs.get("k", "12")),
                        type_=qs.get("type", ""),
                        personalize=float(qs.get("personalize", "0.3")),
                        max_price=float(qs["max_price"])
                        if qs.get("max_price")
                        else None,
                    ),
                )
            except Exception as e:
                self._send(404, {"error": str(e)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        # custom-header requirement: cross-site pages can't send this without
        # a CORS preflight, which we never grant — blocks CSRF to localhost
        if self.headers.get("X-DoubleTap") != "1":
            self._send(403, {"error": "missing X-DoubleTap header"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            args = body["args"]
            assert isinstance(args, list) and all(isinstance(a, str) for a in args)
        except Exception:
            self._send(400, {"error": 'body must be {"args": [str, ...]}'})
            return

        if self.path == "/api/import":
            # pasted decklist: write it to a temp file the CLI can import
            import tempfile

            from . import db

            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
                tmp.write(body.get("text", ""))
            args = [
                a.replace("@TEXT@", tmp.name).replace("@DECKS@", str(db.decks_dir()))
                for a in args
            ]
        elif self.path != "/api/run":
            self._send(404, {"error": "not found"})
            return

        if not args or args[0] not in ALLOWED_COMMANDS:
            self._send(400, {"error": f"command not allowed: {args[:1]}"})
            return
        output, code = run_cli(args)
        self._send(200, {"output": output, "exit_code": code})


def serve(port: int = 8787) -> ThreadingHTTPServer:
    """Bind localhost only — this is a single-user local tool."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server
