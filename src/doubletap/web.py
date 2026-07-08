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

    out = []
    for f in sorted(db.decks_dir().glob("*.json")):
        try:
            deck = decks.Deck.load(f)
        except Exception:
            continue
        commander = named(deck.commander) if deck.commander else None
        if deck.partner:
            commander += f" + {named(deck.partner)}"
        out.append(
            {
                "name": f.stem,
                "path": str(f),
                "format": deck.format,
                "cards": deck.size(),
                "commander": commander,
            }
        )
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet server
        pass

    def _send(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html")
        elif self.path == "/api/decks":
            self._send(200, list_decks())
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
