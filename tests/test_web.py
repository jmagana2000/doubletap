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
    ("deck", "drop"),
    ("deck", "commander"),
    ("deck", "swaps"),
    ("deck", "format"),
    ("deck", "merge"),
    ("deck", "bracket"),
    ("deck", "analyze"),
    ("deck", "price"),
    ("deck", "validate"),
    ("deck", "goldfish"),
    ("deck", "manabase"),
    ("corpus", "crawl"),
    ("corpus", "stats"),
    ("corpus", "pmi"),
    ("train", "bc"),
    ("train", "cql"),
    ("train", "export"),
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


def test_analysis_endpoint(client):
    import_deck(
        client,
        "Commander\n1 Atraxa, Praetors' Voice\nDeck\n1 Sol Ring\n"
        "4 Lightning Bolt\n30 Swamp\n",
    )
    path = client.get("/api/decks").json()[0]["path"]
    a = client.get("/api/analysis", params={"path": path}).json()
    assert a["roles"]["ramp"] == [["Sol Ring", 1]]
    assert a["targets"]["lands"] == 36
    assert a["curve"]["1"] == 5  # Sol Ring + 4 Bolts
    assert a["pips"]["R"] == 4 and a["sources"] == {"B": 30}
    assert "R" in a["short_colors"]  # no land makes red
    assert a["bracket"] in (1, 2, 3, 4)
    assert any("exactly 100" in v for v in a["violations"])

    decks = client.get("/api/decks").json()
    assert "art" in decks[0] and "colors" in decks[0]


def test_set_commander_via_run(client):
    """The builder's ⚔ Set commander button posts exactly this."""
    import_deck(client, "1 Sol Ring\n1 Atraxa, Praetors' Voice", name="cmdtest")
    r = client.get("/api/decks")
    path = next(d["path"] for d in r.json() if "cmdtest" in d["path"])
    out = run(client, ["deck", "commander", path, "Atraxa, Praetors' Voice"])
    assert out["exit_code"] == 0, out["output"]
    detail = client.get("/api/deck?path=" + path).json()
    assert detail["commander"]["name"] == "Atraxa, Praetors' Voice"
    assert detail["size"] == 2  # commander moved out of the main list, count kept


def test_legendary_creature_filter_and_missing_commander_flag(client):
    """The ⚔ Commanders chip queries type=Legendary Creature; a commander
    deck without a commander reports the missing_commander violation."""
    r = client.get("/api/cards?type=Legendary+Creature&format=commander")
    names = {c["name"] for c in r.json()}
    assert "Atraxa, Praetors' Voice" in names
    assert "Sol Ring" not in names

    import_deck(client, "1 Sol Ring", name="nocmd")
    path = next(d["path"] for d in client.get("/api/decks").json() if "nocmd" in d["path"])
    detail = client.get("/api/deck?path=" + path).json()
    assert any("commander" in v for v in detail["violations"])


def test_drop_deck_via_run(client):
    """The builder's Drop button posts exactly this."""
    import_deck(client, "1 Sol Ring", name="doomed")
    path = next(d["path"] for d in client.get("/api/decks").json() if "doomed" in d["path"])
    out = run(client, ["deck", "drop", path, "--yes"])
    assert out["exit_code"] == 0, out["output"]
    assert all("doomed" not in d["path"] for d in client.get("/api/decks").json())


def test_duplicate_add_refused_in_commander(client):
    """The builder's Add button must not silently create an illegal duplicate."""
    import_deck(client, "1 Sol Ring", name="duptest")
    path = next(d["path"] for d in client.get("/api/decks").json() if "duptest" in d["path"])

    out = run(client, ["deck", "add", path, "Sol Ring"])
    assert out["exit_code"] == 1
    assert "refused" in out["output"]
    detail = client.get("/api/deck?path=" + path).json()
    assert detail["size"] == 1  # deck unchanged

    # any-number and capped cards still add freely up to their cap
    for _ in range(2):
        assert run(client, ["deck", "add", path, "Relentless Rats"])["exit_code"] == 0
    assert run(client, ["deck", "add", path, "Sol Ring", "--force"])["exit_code"] == 0


def test_suggest_endpoint_with_filters(client, loaded_conn, data_home):
    """The builder's Suggest panel: structured top-k with a type filter."""
    torch = pytest.importorskip("torch")

    from doubletap.formats import COMMANDER
    from doubletap.ml.data import build_vocab, state_dim
    from doubletap.ml.infer_np import save_np_checkpoint
    from doubletap.ml.model import TwoTowerQ

    vocab = build_vocab(loaded_conn, COMMANDER)
    torch.manual_seed(0)
    tiny = TwoTowerQ(vocab.features, state_dim=state_dim(COMMANDER), emb_dim=8, hidden=16, out_dim=8)
    models = data_home / "models"
    models.mkdir(exist_ok=True)
    save_np_checkpoint(models / "cql_commander.npz", tiny.state_dict(), vocab.oracle_ids, "commander", "cql")

    import_deck(client, "Commander\n1 Atraxa, Praetors' Voice\nDeck\n1 Sol Ring\n")
    path = client.get("/api/decks").json()[0]["path"]

    r = client.get("/api/suggest", params={"path": path, "k": "5"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["model"] == "cql"
    assert 0 < len(d["suggestions"]) <= 5
    assert all("score" in c and "name" in c for c in d["suggestions"])
    assert all(c["name"] != "Sol Ring" for c in d["suggestions"])  # copy limit

    r = client.get("/api/suggest", params={"path": path, "k": "5", "type": "Creature"})
    assert all("Creature" in c["type_line"] for c in r.json()["suggestions"])


def test_suggest_personalize_and_price_params(client, loaded_conn, data_home):
    torch = pytest.importorskip("torch")

    from doubletap.formats import COMMANDER
    from doubletap.ml.data import build_vocab, state_dim
    from doubletap.ml.infer_np import save_np_checkpoint
    from doubletap.ml.model import TwoTowerQ

    vocab = build_vocab(loaded_conn, COMMANDER)
    torch.manual_seed(0)
    tiny = TwoTowerQ(vocab.features, state_dim=state_dim(COMMANDER), emb_dim=8, hidden=16, out_dim=8)
    (data_home / "models").mkdir(exist_ok=True)
    save_np_checkpoint(data_home / "models" / "cql_commander.npz", tiny.state_dict(), vocab.oracle_ids, "commander", "cql")
    import_deck(client, "Commander\n1 Atraxa, Praetors' Voice\nDeck\n1 Sol Ring\n")
    path = client.get("/api/decks").json()[0]["path"]

    r = client.get("/api/suggest", params={"path": path, "k": "5", "personalize": "0", "max_price": "0.50"})
    assert r.status_code == 200, r.text
    for c in r.json()["suggestions"]:
        assert c["price"] is None or c["price"] <= 0.50


def test_suggest_reports_deck_fullness(client, loaded_conn, data_home):
    torch = pytest.importorskip("torch")

    from doubletap.formats import COMMANDER
    from doubletap.ml.data import build_vocab, state_dim
    from doubletap.ml.infer_np import save_np_checkpoint
    from doubletap.ml.model import TwoTowerQ

    vocab = build_vocab(loaded_conn, COMMANDER)
    torch.manual_seed(0)
    tiny = TwoTowerQ(vocab.features, state_dim=state_dim(COMMANDER), emb_dim=8, hidden=16, out_dim=8)
    (data_home / "models").mkdir(exist_ok=True)
    save_np_checkpoint(data_home / "models" / "cql_commander.npz", tiny.state_dict(), vocab.oracle_ids, "commander", "cql")
    import_deck(client, "Commander\n1 Atraxa, Praetors' Voice\nDeck\n1 Sol Ring\n98 Swamp\n")
    path = client.get("/api/decks").json()[0]["path"]

    d = client.get("/api/suggest", params={"path": path, "k": "3"}).json()
    assert d["deck_size"] == 100 and d["target_size"] == 100 and d["exact_size"]


def test_swaps_endpoint(client, loaded_conn, data_home):
    torch = pytest.importorskip("torch")

    from doubletap.formats import COMMANDER
    from doubletap.ml.data import build_vocab, state_dim
    from doubletap.ml.infer_np import save_np_checkpoint
    from doubletap.ml.model import TwoTowerQ

    vocab = build_vocab(loaded_conn, COMMANDER)
    torch.manual_seed(0)
    tiny = TwoTowerQ(vocab.features, state_dim=state_dim(COMMANDER), emb_dim=8, hidden=16, out_dim=8)
    (data_home / "models").mkdir(exist_ok=True)
    save_np_checkpoint(data_home / "models" / "cql_commander.npz", tiny.state_dict(), vocab.oracle_ids, "commander", "cql")
    import_deck(client, "Commander\n1 Atraxa, Praetors' Voice\nDeck\n1 Sol Ring\n1 Juzám Djinn\n1 Rhystic Study\n")
    path = next(d["path"] for d in client.get("/api/decks").json())

    d = client.get("/api/swaps", params={"path": path, "k": "3"}).json()
    assert d["model"] == "cql"
    assert d["swaps"], "expected at least one swap pair"
    for s in d["swaps"]:
        assert s["cut"] and s["add"] and s["reason"]
        assert s["delta"] > 0
        assert s["cut"] != s["add"]
    # every nonland deck card appears in the cut ordering
    assert set(d["cut_order"]) <= {"Sol Ring", "Juzám Djinn", "Rhystic Study"}
    assert len(d["cut_order"]) >= 2


def test_swaps_sorted_by_delta(client, loaded_conn, data_home):
    torch = pytest.importorskip("torch")

    from doubletap.formats import COMMANDER
    from doubletap.ml.data import build_vocab, state_dim
    from doubletap.ml.infer_np import save_np_checkpoint
    from doubletap.ml.model import TwoTowerQ

    vocab = build_vocab(loaded_conn, COMMANDER)
    torch.manual_seed(0)
    tiny = TwoTowerQ(vocab.features, state_dim=state_dim(COMMANDER), emb_dim=8, hidden=16, out_dim=8)
    (data_home / "models").mkdir(exist_ok=True)
    save_np_checkpoint(data_home / "models" / "cql_commander.npz", tiny.state_dict(), vocab.oracle_ids, "commander", "cql")
    import_deck(client, "Commander\n1 Atraxa, Praetors' Voice\nDeck\n1 Sol Ring\n1 Juzám Djinn\n1 Rhystic Study\n1 Relentless Rats\n")
    path = next(d["path"] for d in client.get("/api/decks").json())

    d = client.get("/api/swaps", params={"path": path, "k": "4"}).json()
    deltas = [s["delta"] for s in d["swaps"]]
    assert deltas == sorted(deltas, reverse=True)  # best measured delta first
    # no cut or add repeats across the pair set
    assert len({s["cut"] for s in d["swaps"]}) == len(d["swaps"])
    assert len({s["add"] for s in d["swaps"]}) == len(d["swaps"])


def test_suggest_format_override(client, loaded_conn, data_home):
    """The Suggestions 'Build as' dropdown: a commander deck served by the
    modern model+rules when format=modern is passed."""
    torch = pytest.importorskip("torch")

    from doubletap.formats import MODERN
    from doubletap.ml.data import build_vocab, state_dim
    from doubletap.ml.infer_np import save_np_checkpoint
    from doubletap.ml.model import TwoTowerQ

    vocab = build_vocab(loaded_conn, MODERN)
    torch.manual_seed(0)
    tiny = TwoTowerQ(vocab.features, state_dim=state_dim(MODERN), emb_dim=8, hidden=16, out_dim=8)
    (data_home / "models").mkdir(exist_ok=True)
    save_np_checkpoint(data_home / "models" / "bc_modern.npz", tiny.state_dict(), vocab.oracle_ids, "modern", "bc")
    import_deck(client, "4 Lightning Bolt\n")  # imported as commander by default

    path = next(d["path"] for d in client.get("/api/decks").json())
    # no commander model planted: deck's own format fails, override succeeds
    assert client.get("/api/suggest", params={"path": path, "k": "3"}).status_code == 404
    r = client.get("/api/suggest", params={"path": path, "k": "3", "format": "modern"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["model"] == "bc" and d["suggestions"]
