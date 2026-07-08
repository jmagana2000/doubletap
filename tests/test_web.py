"""Web UI tests: a real server on an ephemeral port, exercised over HTTP.

Parity is the point — /api/run executes the actual Typer app, so these tests
prove every CLI command is reachable from the browser, the UI has a control
for each, and the security gates hold."""

import threading

import httpx
import pytest

from doubletap import web
from doubletap.cli import app

HEADERS = {"X-DoubleTap": "1"}

# every user-facing CLI command (group, name); "web" itself is excluded
ALL_COMMANDS = [
    ("cards", "sync"),
    ("cards", "lookup"),
    ("deck", "import"),
    ("deck", "list"),
    ("deck", "show"),
    ("deck", "add"),
    ("deck", "remove"),
    ("deck", "commander"),
    ("deck", "merge"),
    ("deck", "bracket"),
    ("deck", "analyze"),
    ("deck", "price"),
    ("deck", "validate"),
    ("corpus", "crawl"),
    ("corpus", "stats"),
    ("corpus", "pmi"),
    ("train", "bc"),
    ("train", "cql"),
    ("eval", None),
    ("recommend", None),
    ("complete", None),
]


@pytest.fixture
def client(loaded_conn):
    server = web.serve(port=0)  # ephemeral port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    with httpx.Client(base_url=base, timeout=30) as c:
        yield c
    server.shutdown()


def run(client, args):
    r = client.post("/api/run", json={"args": args}, headers=HEADERS)
    assert r.status_code == 200, r.text
    return r.json()


def import_deck(client, text, name="webdeck"):
    r = client.post(
        "/api/import",
        json={
            "text": text,
            "args": [
                "deck",
                "import",
                "@TEXT@",
                "-f",
                "commander",
                "--no-interactive",
                "-o",
                f"@DECKS@/{name}.json",
            ],
        },
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text
    return r.json()


# --- parity: every CLI command is reachable through the web API -------------


def test_every_cli_command_reachable(client):
    for group, sub in ALL_COMMANDS:
        args = [group, sub, "--help"] if sub else [group, "--help"]
        res = run(client, args)
        assert res["exit_code"] == 0, f"{args}: {res['output']}"


def test_every_command_has_a_ui_control():
    html = (web.STATIC / "index.html").read_text()
    # commands driven by dedicated controls carry a data-cmd marker; the rest
    # appear in the JS arg builders — either way the name must be in the page
    for group, sub in ALL_COMMANDS:
        needle = f"{group} {sub}" if sub else group
        assert needle in html or f'"{group}", "{sub}"' in html.replace("'", '"'), (
            f"no UI control for: {needle}"
        )


def test_cli_registers_web_command():
    commands = {c.name for c in app.registered_commands} | {
        g.name for g in app.registered_groups
    }
    assert "web" in commands


# --- real executions through HTTP -------------------------------------------


def test_index_and_decks_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200 and "DoubleTap" in r.text
    assert client.get("/api/decks").json() == []


def test_import_show_modify_analyze_flow(client):
    res = import_deck(
        client, "Commander\n1 Atraxa, Praetors' Voice\nDeck\n1 Sol Ring\n30 Swamp\n"
    )
    assert res["exit_code"] == 0, res["output"]

    decks = client.get("/api/decks").json()
    assert [d["name"] for d in decks] == ["webdeck"]
    assert decks[0]["commander"] == "Atraxa, Praetors' Voice"
    path = decks[0]["path"]

    assert "Sol Ring" in run(client, ["deck", "show", path])["output"]
    assert run(client, ["deck", "add", path, "Juzám Djinn"])["exit_code"] == 0
    assert run(client, ["deck", "remove", path, "Sol Ring"])["exit_code"] == 0

    res = run(client, ["deck", "validate", path])
    assert "Commander: Atraxa, Praetors' Voice" in res["output"]
    for cmd in ("analyze", "bracket", "price"):
        assert run(client, ["deck", cmd, path])["exit_code"] == 0

    res = run(client, ["deck", "commander", path])
    assert "Atraxa" in res["output"]


def test_lookup_and_stats(client):
    res = run(client, ["cards", "lookup", "lightning blot"])
    assert res["exit_code"] == 0 and "Lightning Bolt" in res["output"]
    assert run(client, ["corpus", "stats"])["exit_code"] == 0


def test_merge_through_api(client):
    import_deck(client, "1 Sol Ring\n", "a")
    import_deck(client, "4 Lightning Bolt\n", "b")
    paths = {d["name"]: d["path"] for d in client.get("/api/decks").json()}
    res = run(client, ["deck", "merge", paths["a"], paths["b"], "--format", "modern"])
    assert res["exit_code"] == 0 and "5 cards" in res["output"]


def test_failures_surface_as_nonzero_exit(client):
    res = run(client, ["cards", "lookup", "Zzyzx Quuxblade"])
    assert res["exit_code"] == 1


# --- security gates ----------------------------------------------------------


def test_rejects_missing_header(client):
    r = client.post("/api/run", json={"args": ["deck", "list"]})
    assert r.status_code == 403


def test_rejects_unlisted_command(client):
    r = client.post("/api/run", json={"args": ["web"]}, headers=HEADERS)
    assert r.status_code == 400
    r = client.post("/api/run", json={"args": ["rm", "-rf", "/"]}, headers=HEADERS)
    assert r.status_code == 400


def test_rejects_malformed_body(client):
    r = client.post("/api/run", json={"args": "deck list"}, headers=HEADERS)
    assert r.status_code == 400
    r = client.post("/api/run", content=b"not json", headers=HEADERS)
    assert r.status_code == 400


# --- builder endpoints --------------------------------------------------------


def test_cards_search_endpoint(client):
    cards = client.get("/api/cards", params={"q": "bolt"}).json()
    assert any(c["name"] == "Lightning Bolt" for c in cards)
    bolt = next(c for c in cards if c["name"] == "Lightning Bolt")
    assert bolt["mana_cost"] == "{R}" and bolt["type_line"] == "Instant"
    assert "3 damage" in bolt["oracle_text"]

    # identity filter: mono-white browse must exclude the red Bolt
    white = client.get("/api/cards", params={"colors": "W"}).json()
    assert all("R" not in c["colors"] for c in white)

    # type + mana-value filters combine
    cheap_creatures = client.get(
        "/api/cards", params={"type": "Creature", "max_mv": "3"}
    ).json()
    assert cheap_creatures and all(
        "Creature" in c["type_line"] and c["cmc"] <= 3 for c in cheap_creatures
    )


def test_deck_detail_endpoint(client):
    import_deck(
        client, "Commander\n1 Atraxa, Praetors' Voice\nDeck\n1 Sol Ring\n30 Swamp\n"
    )
    path = client.get("/api/decks").json()[0]["path"]
    detail = client.get("/api/deck", params={"path": path}).json()
    assert detail["commander"]["name"] == "Atraxa, Praetors' Voice"
    assert detail["size"] == 32
    by_name = {c["name"]: c for c in detail["cards"]}
    assert by_name["Swamp"]["qty"] == 30
    assert by_name["Sol Ring"]["mana_cost"] == "{1}"
    assert any("exactly 100" in v for v in detail["violations"])

    r = client.get("/api/deck", params={"path": "/nope/missing.json"})
    assert r.status_code == 404
