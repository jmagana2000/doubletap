"""Command-level tests: every CLI command invoked through Typer's runner,
against the fixture card cache in a temporary DOUBLETAP_HOME."""

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from doubletap import cli
from doubletap.cli import app
from doubletap.names import lookup

from test_ml_data import _insert_corpus_deck

runner = CliRunner()


def oid(conn, name):
    return lookup(conn, name)[0].oracle_id


def write_decklist(tmp_path, name="decklist.txt", body=None):
    body = body or (
        "Commander\n"
        "1 Atraxa, Praetors' Voice\n"
        "Deck\n"
        "1 Sol Ring\n"
        "1 Juzám Djinn\n"
        "30 Swamp\n"
    )
    path = tmp_path / name
    path.write_text(body)
    return path


def import_deck(tmp_path, out_name="deck.json", **kw):
    src = write_decklist(tmp_path, **kw)
    out = tmp_path / out_name
    result = runner.invoke(app, ["deck", "import", str(src), "-o", str(out)])
    assert result.exit_code == 0, result.output
    return out


# --- cards ---------------------------------------------------------------


def test_cards_sync_reports_result(loaded_conn, monkeypatch):
    monkeypatch.setattr(
        cli.scryfall,
        "sync",
        lambda conn, force=False: SimpleNamespace(
            updated=True, card_count=42, updated_at="2026-07-06"
        ),
    )
    result = runner.invoke(app, ["cards", "sync"])
    assert result.exit_code == 0
    assert "Synced 42 cards" in result.output


def test_cards_lookup_found_and_missing(loaded_conn):
    result = runner.invoke(app, ["cards", "lookup", "lightning blot"])
    assert result.exit_code == 0
    assert "Lightning Bolt" in result.output
    assert "[R (red)]" in result.output  # color identity shown per match

    result = runner.invoke(app, ["cards", "lookup", "Zzyzx Quuxblade"])
    assert result.exit_code == 1


# --- deck import / list / merge -------------------------------------------


def test_deck_import_and_validate(loaded_conn, tmp_path):
    out = import_deck(tmp_path)
    assert out.exists()
    result = runner.invoke(app, ["deck", "validate", str(out)])
    assert result.exit_code == 1
    assert "wrong_size" in result.output
    # deck identity is shown even when the size is wrong
    assert "Commander: Atraxa, Praetors' Voice" in result.output
    assert "Color identity: WUBG (white, blue, black, green)" in result.output
    assert "needs exactly 100" in result.output


def test_deck_import_unmatched_exits_nonzero(loaded_conn, tmp_path):
    src = write_decklist(tmp_path, body="1 Zzyzx Quuxblade\n")
    result = runner.invoke(app, ["deck", "import", str(src)])
    assert result.exit_code == 1
    assert "unmatched" in result.output


def test_deck_import_saves_single_card_under_card_name(loaded_conn, tmp_path):
    src = write_decklist(tmp_path, body="1 Juzám Djinn\n")
    result = runner.invoke(app, ["deck", "import", str(src)])
    assert result.exit_code == 0
    from doubletap import db

    assert (db.decks_dir() / "juzam-djinn.json").exists()


def test_deck_list_empty_and_populated(loaded_conn, tmp_path):
    result = runner.invoke(app, ["deck", "list"])
    assert result.exit_code == 0
    assert "No decks saved" in result.output

    src = write_decklist(tmp_path)
    result = runner.invoke(app, ["deck", "import", str(src)])
    assert result.exit_code == 0
    result = runner.invoke(app, ["deck", "list"])
    assert "Commander / contents" in result.output
    assert "Atraxa" in result.output


def test_deck_merge_combines_and_overrides_format(loaded_conn, tmp_path):
    a = import_deck(tmp_path, "a.json", body="1 Sol Ring\n")
    b = import_deck(tmp_path, "b.json", body="4 Lightning Bolt\n")
    out = tmp_path / "merged.json"
    result = runner.invoke(
        app,
        ["deck", "merge", str(a), str(b), "-o", str(out), "--format", "modern"],
    )
    assert result.exit_code == 0, result.output
    assert "5 cards (2 distinct)" in result.output

    result = runner.invoke(app, ["deck", "merge", str(a)])
    assert result.exit_code == 1  # needs at least two files


def test_deck_commander_set_and_swap(loaded_conn, tmp_path):
    # commander-less import: promote a card from the main deck
    out = import_deck(tmp_path, body="1 Atraxa, Praetors' Voice\n1 Sol Ring\n")
    result = runner.invoke(
        app, ["deck", "commander", str(out), "Atraxa, Praetors' Voice"]
    )
    assert result.exit_code == 0, result.output
    from doubletap.decks import Deck

    deck = Deck.load(out)
    assert deck.commander == oid(loaded_conn, "Atraxa, Praetors' Voice")
    assert deck.commander not in deck.entries  # moved out of the main deck
    assert deck.size() == 2

    # changing commanders: the old one moves back into the main deck
    result = runner.invoke(
        app,
        [
            "deck",
            "commander",
            str(out),
            "Thrasios, Triton Hero",
            "--partner",
            "Tymna the Weaver",
        ],
    )
    assert result.exit_code == 0, result.output
    deck = Deck.load(out)
    assert deck.partner == oid(loaded_conn, "Tymna the Weaver")
    assert deck.entries[oid(loaded_conn, "Atraxa, Praetors' Voice")] == 1
    assert deck.size() == 4

    # a name that doesn't resolve exactly fails with suggestions
    result = runner.invoke(app, ["deck", "commander", str(out), "Atraxxa"])
    assert result.exit_code == 1

    # no name argument: show the current commander without changing anything
    result = runner.invoke(app, ["deck", "commander", str(out)])
    assert result.exit_code == 0
    assert "Thrasios, Triton Hero + Tymna the Weaver" in result.output
    assert "Color identity: WUBG" in result.output  # UG + WB combined
    assert Deck.load(out).size() == 4  # unchanged


def test_deck_add_and_remove(loaded_conn, tmp_path):
    from doubletap.decks import Deck

    out = import_deck(tmp_path, body="1 Sol Ring\n")

    result = runner.invoke(app, ["deck", "add", str(out), "Lightning Bolt", "-n", "4"])
    assert result.exit_code == 0, result.output
    assert Deck.load(out).entries[oid(loaded_conn, "Lightning Bolt")] == 4

    result = runner.invoke(
        app, ["deck", "remove", str(out), "Lightning Bolt", "-n", "3"]
    )
    assert result.exit_code == 0, result.output
    deck = Deck.load(out)
    assert deck.entries[oid(loaded_conn, "Lightning Bolt")] == 1

    # removing the last copy deletes the entry entirely
    result = runner.invoke(app, ["deck", "remove", str(out), "Lightning Bolt"])
    assert result.exit_code == 0
    assert oid(loaded_conn, "Lightning Bolt") not in Deck.load(out).entries

    # removing a card that isn't there fails cleanly
    result = runner.invoke(app, ["deck", "remove", str(out), "Lightning Bolt"])
    assert result.exit_code == 1

    # fuzzy/unknown names are rejected with suggestions, nothing changes
    result = runner.invoke(app, ["deck", "add", str(out), "Lightning Blot"])
    assert result.exit_code == 1
    assert Deck.load(out).size() == 1  # just Sol Ring


def test_deck_add_warns_on_violation(loaded_conn, tmp_path):
    # Juzám Djinn is black; Atraxa deck is WUBG, so no warning there —
    # use a commander deck and add a second copy to trip the copy limit
    out = import_deck(tmp_path)
    result = runner.invoke(app, ["deck", "add", str(out), "Sol Ring"])
    assert result.exit_code == 0
    assert "exceeds the 1-copy limit" in result.output


def test_deck_remove_clears_commander_slot(loaded_conn, tmp_path):
    from doubletap.decks import Deck

    out = import_deck(tmp_path)  # commander: Atraxa
    result = runner.invoke(app, ["deck", "remove", str(out), "Atraxa, Praetors' Voice"])
    assert result.exit_code == 0
    assert "Removed commander" in result.output
    assert Deck.load(out).commander is None


def test_deck_show_lists_cards(loaded_conn, tmp_path):
    out = import_deck(tmp_path)
    result = runner.invoke(app, ["deck", "show", str(out)])
    assert result.exit_code == 0
    assert "Atraxa, Praetors' Voice" in result.output
    assert " 30  Swamp" in result.output
    assert "  1  Sol Ring" in result.output
    # every row shows mana cost and type line
    assert "{1}" in result.output and "Artifact" in result.output
    assert "Basic Land — Swamp" in result.output
    assert "{G}{W}{U}{B}" in result.output  # commander's cost shown too


# --- deck bracket / analyze / price / validate -----------------------------


def test_deck_bracket(loaded_conn, tmp_path):
    out = import_deck(tmp_path)
    result = runner.invoke(app, ["deck", "bracket", str(out)])
    assert result.exit_code == 0
    assert "Bracket 2" in result.output  # no Game Changers in the fixture set


def test_deck_analyze(loaded_conn, tmp_path):
    out = import_deck(tmp_path)
    result = runner.invoke(app, ["deck", "analyze", str(out)])
    assert result.exit_code == 0
    assert "Functional roles" in result.output
    assert "Mana curve" in result.output
    assert "Color balance" in result.output
    assert "Ways to win" in result.output
    assert "Market price" in result.output


def test_deck_analyze_flags_missing_color(loaded_conn, tmp_path):
    # Bolt needs R but the only lands are Swamps (B sources)
    out = import_deck(tmp_path, body="4 Lightning Bolt\n20 Swamp\n")
    result = runner.invoke(app, ["deck", "analyze", str(out)])
    assert result.exit_code == 0
    assert "not enough lands make this color" in result.output
    assert "Interaction speed: 4 of 4 removal spells" in result.output


def test_deck_price(loaded_conn, tmp_path):
    out = import_deck(tmp_path)
    result = runner.invoke(app, ["deck", "price", str(out)])
    assert result.exit_code == 0
    assert "Total market price" in result.output


def test_deck_validate_valid_deck(loaded_conn, tmp_path):
    out = import_deck(
        tmp_path,
        body="Commander\n1 Atraxa, Praetors' Voice\nDeck\n1 Sol Ring\n98 Swamp\n",
    )
    result = runner.invoke(app, ["deck", "validate", str(out)])
    assert result.exit_code == 0
    assert "commander deck, 100 cards" in result.output
    assert "Valid — no violations." in result.output


# --- corpus ----------------------------------------------------------------


def _rigged_corpus(conn):
    # train_bc refuses to run on fewer than 20 parsed decks
    for deck_id in range(1, 21):
        _insert_corpus_deck(
            conn,
            deck_id,
            "commander",
            {
                oid(conn, "Sol Ring"): 1,
                oid(conn, "Relentless Rats"): 20,
                oid(conn, "Juzám Djinn"): 1,
                oid(conn, "Swamp"): 77,
            },
            commander_oid=oid(conn, "Atraxa, Praetors' Voice"),
        )
    conn.commit()


def test_corpus_crawl_reports_outcomes(loaded_conn, monkeypatch):
    monkeypatch.setattr(
        cli.archidekt,
        "crawl",
        lambda conn, fmt, max_decks, progress=None, order_by=None: {"ok": 3},
    )
    result = runner.invoke(app, ["corpus", "crawl", "-f", "commander", "--max", "3"])
    assert result.exit_code == 0
    assert "parse outcomes" in result.output


def test_corpus_stats(loaded_conn):
    _rigged_corpus(loaded_conn)
    result = runner.invoke(app, ["corpus", "stats"])
    assert result.exit_code == 0
    assert "parsed" in result.output


def test_corpus_pmi_requires_corpus(loaded_conn):
    result = runner.invoke(app, ["corpus", "pmi", "-f", "commander"])
    assert result.exit_code == 1


# --- ML pipeline: pmi -> train -> eval -> recommend -> complete -------------


def test_ml_pipeline_end_to_end(loaded_conn, tmp_path):
    pytest.importorskip("torch")
    _rigged_corpus(loaded_conn)

    result = runner.invoke(
        app, ["corpus", "pmi", "-f", "commander", "--min-count", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "PPMI pairs" in result.output

    result = runner.invoke(app, ["train", "bc", "-f", "commander", "--steps", "3"])
    assert result.exit_code == 0, result.output
    assert "Wrote" in result.output

    result = runner.invoke(app, ["train", "cql", "-f", "commander", "--steps", "2"])
    assert result.exit_code == 0, result.output

    from doubletap import db

    bc_ckpt = db.data_home() / "models" / "bc_commander.pt"
    assert bc_ckpt.exists()
    # training writes torch-free numpy weights alongside the torch checkpoint
    assert bc_ckpt.with_suffix(".npz").exists()
    result = runner.invoke(app, ["eval", "--model", str(bc_ckpt), "--n-hide", "2"])
    assert result.exit_code == 0, result.output
    assert "recovery" in result.output

    deck = import_deck(tmp_path)
    result = runner.invoke(app, ["recommend", "--deck", str(deck), "-k", "3"])
    assert result.exit_code == 0, result.output
    assert "Top 3 additions" in result.output
    # default resolution: CQL first (cleared the keep-bar), torch-free weights
    assert "cql_commander.npz" in result.output

    # explicit torch checkpoint still works on training machines
    result = runner.invoke(
        app, ["recommend", "--deck", str(deck), "-k", "3", "--model", str(bc_ckpt)]
    )
    assert result.exit_code == 0, result.output

    # budget cap: fixture cards are unpriced, so everything stays eligible
    result = runner.invoke(
        app,
        ["recommend", "--deck", str(deck), "-k", "3", "--max-card-price", "1.00"],
    )
    assert result.exit_code == 0, result.output
    assert "max $1.00/card" in result.output

    out = tmp_path / "completed.json"
    result = runner.invoke(app, ["complete", "--deck", str(deck), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "bracket ≤3" in result.output  # default bracket applies

    result = runner.invoke(
        app, ["complete", "--deck", str(deck), "-o", str(out), "--bracket", "5"]
    )
    assert result.exit_code == 0, result.output
    assert "bracket" not in result.output  # 4-5: unrestricted, no cap in play


def test_recommend_without_model_exits_cleanly(loaded_conn, tmp_path):
    deck = import_deck(tmp_path)
    result = runner.invoke(app, ["recommend", "--deck", str(deck)])
    assert result.exit_code == 1
    assert "No trained model" in result.output
