import re
import sys
from pathlib import Path

import typer

from . import analysis, archidekt, db, decks, formats, names, scryfall

app = typer.Typer(no_args_is_help=True)
cards_app = typer.Typer(no_args_is_help=True, help="Local Scryfall card cache")
app.add_typer(cards_app, name="cards")
deck_app = typer.Typer(no_args_is_help=True, help="Import and manage decks")
app.add_typer(deck_app, name="deck")
corpus_app = typer.Typer(
    no_args_is_help=True, help="Training corpus of public decklists"
)
app.add_typer(corpus_app, name="corpus")
train_app = typer.Typer(no_args_is_help=True, help="Train recommendation models")
app.add_typer(train_app, name="train")


@app.command("web")
def web_cmd(
    port: int = typer.Option(8787, "--port", help="Port to serve the web UI on"),
):
    """Serve the local web UI at http://127.0.0.1:<port> (Ctrl-C to stop)."""
    from . import web

    server = web.serve(port)
    typer.echo(f"DoubleTap UI: http://127.0.0.1:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


@cards_app.command("sync")
def cards_sync(
    force: bool = typer.Option(False, help="Re-download even if the cache is current"),
):
    """Download/refresh the Scryfall oracle_cards bulk data into the local cache."""
    conn = db.connect()
    result = scryfall.sync(conn, force=force)
    if result.updated:
        typer.echo(
            f"Synced {result.card_count} cards (bulk updated_at {result.updated_at})"
        )
    else:
        typer.echo(
            f"Cache current: {result.card_count} cards (bulk updated_at {result.updated_at})"
        )


_COLOR_NAMES = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}


def _colors_label(colors) -> str:
    """Spell out a set of color letters: 'WUR (white, blue, red)'."""
    letters = [c for c in "WUBRG" if c in colors]
    if not letters:
        return "colorless"
    return "".join(letters) + " (" + ", ".join(_COLOR_NAMES[c] for c in letters) + ")"


def _identity_label(conn, oid: str) -> str:
    """Human-readable color identity of one card."""
    card = formats.get_card(conn, oid)
    return _colors_label(card.get("color_identity") or [])


@cards_app.command("lookup")
def cards_lookup(name: str):
    """Resolve a card name (exact or fuzzy) against the local cache."""
    conn = db.connect()
    matches = names.lookup(conn, name)
    if not matches:
        typer.echo("No match found. Is the cache synced? (doubletap cards sync)")
        raise typer.Exit(code=1)
    for m in matches:
        typer.echo(
            f"{m.score:5.1f}  {m.name}  [{_identity_label(conn, m.oracle_id)}]"
            f"  ({m.oracle_id})"
        )


def _prompt_chooser(line, matches):
    typer.echo(f"\nAmbiguous: {line.raw!r}")
    for i, m in enumerate(matches, 1):
        typer.echo(f"  {i}. {m.name}  ({m.score:.1f})")
    choice = typer.prompt("Pick a number, or s to skip", default="s")
    if choice.isdigit() and 1 <= int(choice) <= len(matches):
        return matches[int(choice) - 1]
    return None


@deck_app.command("import")
def deck_import(
    path: Path = typer.Argument(..., exists=True, readable=True),
    deck_format: str = typer.Option(
        "commander",
        "--format",
        "-f",
        help="Deck format: commander (default) or modern. Can be changed at merge time.",
    ),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Write the deck as JSON (default: ~/.doubletap/decks/<name>.json)",
    ),
    commander: str = typer.Option(None, help="Commander card name (Commander format)"),
    companion: str = typer.Option(
        None, help="Companion card name (sits outside the starting deck)"
    ),
    threshold: float = typer.Option(90.0, help="Fuzzy auto-accept score threshold"),
    interactive: bool = typer.Option(True, help="Prompt on ambiguous/unmatched names"),
):
    """Import a deck from a CSV, plain-text decklist, or decklist photo.
    Saved to ~/.doubletap/decks/ by default; use -o to override.
    Format defaults to commander and can be overridden at merge time with
    doubletap deck merge --format <format>."""
    conn = db.connect()
    parsed = decks.load_lines(path)
    if commander:
        parsed.append(
            decks.ParsedLine(raw=commander, qty=1, name=commander, is_commander=True)
        )
    if companion:
        parsed.append(
            decks.ParsedLine(raw=companion, qty=1, name=companion, is_companion=True)
        )
    chooser = _prompt_chooser if interactive and sys.stdin.isatty() else None
    result = decks.resolve(
        conn, parsed, deck_format, threshold=threshold, chooser=chooser
    )

    for res in result.assumed:
        typer.echo(
            f"assumed   {res.matches[0].name}  <- {res.line.raw!r} ({res.matches[0].score:.1f})"
        )
    for res in result.ambiguous:
        options = ", ".join(m.name for m in res.matches)
        typer.echo(f"ambiguous {res.line.raw!r}: {options}")
    for res in result.unmatched:
        typer.echo(f"unmatched {res.line.raw!r}")

    deck = result.deck
    typer.echo(
        f"\n{deck.size()} cards ({len(deck.entries)} distinct), format {deck.format}"
        + (", commander set" if deck.commander else "")
    )
    if not result.ok:
        typer.echo("Import incomplete: fix ambiguous/unmatched lines above.", err=True)
        raise typer.Exit(code=1)
    save_path = out or db.decks_dir() / f"{_default_stem(conn, deck, path)}.json"
    deck.save(conn, save_path)
    typer.echo(f"Wrote {save_path}")


def _default_stem(conn, deck, source_path: Path) -> str:
    """Filename for an auto-saved import. Single-card imports (physical card
    photos) are named after the card — IMG_3793.json says nothing about what
    is inside."""
    oids = list(deck.entries) + [o for o in (deck.commander, deck.partner) if o]
    if len(oids) == 1 and sum(deck.entries.values()) <= 1:
        (name,) = conn.execute(
            "SELECT name FROM cards WHERE oracle_id = ?", (oids[0],)
        ).fetchone()
        slug = re.sub(r"[^a-z0-9]+", "-", names.normalize(name)).strip("-")
        # same card photographed again must not overwrite the first import
        stem, n = slug, 2
        while (db.decks_dir() / f"{stem}.json").exists():
            stem = f"{slug}-{n}"
            n += 1
        return stem
    return Path(source_path).stem


@deck_app.command("list")
def deck_list():
    """List saved decks in ~/.doubletap/decks/."""
    conn = db.connect()
    deck_files = sorted(db.decks_dir().glob("*.json"))
    if not deck_files:
        typer.echo("No decks saved yet. Import one with: doubletap deck import <file>")
        return

    def named(oid):
        row = conn.execute(
            "SELECT name FROM cards WHERE oracle_id = ?", (oid,)
        ).fetchone()
        return row[0] if row else oid

    typer.echo(f"{'File':<32} {'Format':<10} {'Cards':>5}  Commander / contents")
    typer.echo("-" * 70)
    for f in deck_files:
        try:
            deck = decks.Deck.load(f)
        except Exception:
            typer.echo(f"{f.stem:<32} {'':10} {'':>5}  (unreadable)")
            continue
        if deck.commander:
            detail = named(deck.commander)
            if deck.partner:
                detail += f" + {named(deck.partner)}"
        elif len(deck.entries) <= 3:
            # tiny commander-less decks (single-card photo imports): show
            # what's inside rather than a blank column
            detail = ", ".join(named(oid) for oid in deck.entries)
        else:
            detail = ""
        typer.echo(f"{f.stem:<32} {deck.format:<10} {deck.size():>5}  {detail}")


def _card_name_by_oid(conn, oid: str) -> str:
    row = conn.execute("SELECT name FROM cards WHERE oracle_id = ?", (oid,)).fetchone()
    return row[0] if row else oid


def _resolve_one(conn, name: str) -> tuple[str, str]:
    """Resolve a single card name to (oracle_id, name) or exit with candidates."""
    matches = names.lookup(conn, name)
    if matches and matches[0].score == 100.0:
        return matches[0].oracle_id, matches[0].name
    if matches:
        options = ", ".join(m.name for m in matches[:3])
        typer.echo(f"No exact match for {name!r}. Did you mean: {options}?", err=True)
    else:
        typer.echo(f"No match for {name!r}.", err=True)
    raise typer.Exit(code=1)


@deck_app.command("commander")
def deck_commander(
    path: Path = typer.Argument(..., exists=True, readable=True),
    name: str = typer.Argument(
        None, help="Card to make the commander; omit to show the current one"
    ),
    partner: str = typer.Option(
        None, help="Second commander (both cards need the Partner ability)"
    ),
):
    """Show, set, or change a saved deck's commander (and optionally its
    partner). When changing, the previous commander moves into the main deck;
    if the new commander was in the main deck, it moves out — the card count
    stays the same."""
    conn = db.connect()
    deck = decks.Deck.load(path)

    if name is None:
        if deck.commander is None:
            typer.echo("No commander set.")
        else:
            current = _card_name_by_oid(conn, deck.commander)
            identity = set(formats.get_card(conn, deck.commander)["color_identity"])
            if deck.partner:
                current += f" + {_card_name_by_oid(conn, deck.partner)}"
                identity |= set(formats.get_card(conn, deck.partner)["color_identity"])
            typer.echo(f"Commander: {current}")
            typer.echo(f"Color identity: {_colors_label(identity)}")
        return

    def promote(card_name):
        oid, resolved = _resolve_one(conn, card_name)
        if deck.entries.get(oid):
            deck.entries[oid] -= 1
            if not deck.entries[oid]:
                del deck.entries[oid]
        return oid, resolved

    for old in (deck.commander, deck.partner):
        if old:
            deck.entries[old] += 1
    deck.commander, commander_name = promote(name)
    deck.partner = None
    if partner:
        deck.partner, partner_name = promote(partner)

    deck.save(conn, path)
    label = commander_name + (f" + {partner_name}" if partner else "")
    typer.echo(f"Commander set: {label} ({deck.size()} cards) → {path}")
    for v in formats.validate(conn, deck):
        if v.code != "wrong_size":  # partial decks are normal here
            typer.echo(f"note: {v.message}")


@deck_app.command("add")
def deck_add(
    path: Path = typer.Argument(..., exists=True, readable=True),
    name: str = typer.Argument(..., help="Card to add"),
    qty: int = typer.Option(1, "--qty", "-n", help="How many copies"),
):
    """Add a card to a saved deck. Warns if it breaks format rules
    (copy limit, color identity) but saves anyway — fix at your leisure."""
    conn = db.connect()
    deck = decks.Deck.load(path)
    oid, resolved = _resolve_one(conn, name)
    deck.entries[oid] += qty
    deck.save(conn, path)
    typer.echo(f"+{qty} {resolved} ({deck.size()} cards) → {path}")
    for v in formats.validate(conn, deck):
        if v.code != "wrong_size" and v.oracle_id == oid:
            typer.echo(f"note: {v.message}")


@deck_app.command("remove")
def deck_remove(
    path: Path = typer.Argument(..., exists=True, readable=True),
    name: str = typer.Argument(..., help="Card to remove"),
    qty: int = typer.Option(1, "--qty", "-n", help="How many copies"),
):
    """Remove a card from a saved deck. Removing the commander, partner,
    or companion clears that slot."""
    conn = db.connect()
    deck = decks.Deck.load(path)
    oid, resolved = _resolve_one(conn, name)

    if oid == deck.commander:
        deck.commander, deck.partner = deck.partner, None
        typer.echo(f"Removed commander {resolved}")
    elif oid == deck.partner:
        deck.partner = None
        typer.echo(f"Removed partner {resolved}")
    elif oid == deck.companion:
        deck.companion = None
        typer.echo(f"Removed companion {resolved}")
    elif deck.entries.get(oid):
        removed = min(qty, deck.entries[oid])
        deck.entries[oid] -= removed
        if not deck.entries[oid]:
            del deck.entries[oid]
        typer.echo(f"-{removed} {resolved}")
    else:
        typer.echo(f"{resolved} is not in this deck.", err=True)
        raise typer.Exit(code=1)

    deck.save(conn, path)
    typer.echo(f"{deck.size()} cards → {path}")


@deck_app.command("show")
def deck_show(path: Path = typer.Argument(..., exists=True, readable=True)):
    """List every card in a deck with its mana cost and type, grouped by slot
    (commander, companion, deck)."""
    import json

    conn = db.connect()
    deck = decks.Deck.load(path)

    def card_row(oid):
        """(name, mana_cost, type_line); falls back to front face for MDFCs."""
        row = conn.execute(
            "SELECT name, json FROM cards WHERE oracle_id = ?", (oid,)
        ).fetchone()
        if row is None:
            return oid, "", ""
        card = json.loads(row[1])
        cost = card.get("mana_cost") or ""
        if not cost and card.get("card_faces"):
            cost = card["card_faces"][0].get("mana_cost", "")
        return row[0], cost, card.get("type_line", "")

    def line(qty, oid):
        name, cost, type_line = card_row(oid)
        return f"{qty:3}  {name:<40} {cost:<14} {type_line}"

    typer.echo(f"{deck.format} deck, {deck.size()} cards")
    if deck.commander:
        typer.echo(f"\nCommander:\n{line(1, deck.commander)}")
        if deck.partner:
            typer.echo(f"Partner:\n{line(1, deck.partner)}")
    if deck.companion:
        typer.echo(f"Companion:\n{line(1, deck.companion)}")
    if deck.entries:
        typer.echo("")
        for oid, qty in sorted(deck.entries.items(), key=lambda kv: card_row(kv[0])[0]):
            typer.echo(line(qty, oid))


@corpus_app.command("crawl")
def corpus_crawl(
    deck_format: str = typer.Option(..., "--format", "-f"),
    max_decks: int = typer.Option(1000, "--max", help="Deck ids to queue this run"),
    order_by: str = typer.Option(
        "-viewCount", help="Search ordering; different orderings reach different decks"
    ),
):
    """Crawl public Archidekt decks (rate-limited, resumable) into the corpus."""
    conn = db.connect()

    def progress(done, total):
        if done % 25 == 0 or done == total:
            typer.echo(f"fetched {done}/{total}")

    outcomes = archidekt.crawl(
        conn, deck_format, max_decks, progress=progress, order_by=order_by
    )
    typer.echo(f"parse outcomes: {dict(outcomes)}")


@corpus_app.command("pmi")
def corpus_pmi(
    deck_format: str = typer.Option(..., "--format", "-f"),
    min_count: int = typer.Option(20, help="Minimum decks a pair must share"),
    top: int = typer.Option(20, help="Top synergy pairs to display"),
):
    """Build the smoothed PPMI synergy table for a format and show top pairs."""
    from .ml import data as ml_data
    from .ml import reward as ml_reward

    conn = db.connect()
    fmt = formats.get_format(deck_format)
    vocab = ml_data.build_vocab(conn, fmt)
    corpus = ml_data.load_corpus(conn, vocab, fmt)
    if not corpus:
        typer.echo("No parsed decks for this format; run corpus crawl first.", err=True)
        raise typer.Exit(code=1)
    pmi = ml_reward.build_pmi(
        ml_reward.corpus_card_sets(corpus), len(vocab), min_count=min_count
    )
    out = db.data_home() / "models"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"pmi_{fmt.name}.npz"
    pmi.save(path)
    typer.echo(
        f"{len(corpus)} decks, {len(pmi.pairs)} PPMI pairs (min_count={min_count}) -> {path}"
    )

    def card_name(idx):
        (name,) = conn.execute(
            "SELECT name FROM cards WHERE oracle_id = ?", (vocab.oracle_ids[idx],)
        ).fetchone()
        return name

    best = sorted(pmi.pairs.items(), key=lambda kv: -kv[1])[:top]
    for (a, b), value in best:
        typer.echo(f"{value:5.2f}  {card_name(a)}  +  {card_name(b)}")


@corpus_app.command("stats")
def corpus_stats():
    """Corpus size and coverage by format."""
    conn = db.connect()
    rows = conn.execute(
        "SELECT format, status, COUNT(*) FROM decks GROUP BY format, status ORDER BY format, status"
    ).fetchall()
    for fmt, status, n in rows:
        typer.echo(f"{fmt:12} {status:9} {n}")
    for fmt, n, cards, avg in conn.execute(
        "SELECT d.format, COUNT(DISTINCT d.deck_id), COUNT(DISTINCT dc.oracle_id),"
        " ROUND(AVG(sz), 1) FROM decks d JOIN deck_cards dc ON dc.deck_id = d.deck_id"
        " JOIN (SELECT deck_id, SUM(qty) sz FROM deck_cards GROUP BY deck_id) s"
        " ON s.deck_id = d.deck_id WHERE d.status = 'parsed' GROUP BY d.format"
    ).fetchall():
        typer.echo(f"{fmt}: {n} parsed decks, {cards} distinct cards, avg size {avg}")


@train_app.command("bc")
def train_bc_cmd(
    deck_format: str = typer.Option(..., "--format", "-f"),
    steps: int = typer.Option(1500),
    seed: int = typer.Option(0),
):
    """Train the behavior-cloning baseline."""
    from .ml.train_bc import train_bc

    conn = db.connect()
    path = train_bc(
        conn, formats.get_format(deck_format), steps=steps, seed=seed, log=typer.echo
    )
    typer.echo(f"Wrote {path}")


@train_app.command("cql")
def train_cql_cmd(
    deck_format: str = typer.Option(..., "--format", "-f"),
    steps: int = typer.Option(1500),
    alpha: float = typer.Option(1.0, help="Conservative penalty weight"),
    seed: int = typer.Option(0),
    init_from_bc: bool = typer.Option(True, help="Initialize from the BC checkpoint"),
):
    """Train CQL on PPMI+structure rewards (A/B against BC on the same eval)."""
    from .ml.reward import PMIModel
    from .ml.train_cql import train_cql

    conn = db.connect()
    fmt = formats.get_format(deck_format)
    pmi_path = db.data_home() / "models" / f"pmi_{fmt.name}.npz"
    if not pmi_path.exists():
        typer.echo(f"Missing {pmi_path}; run corpus pmi first.", err=True)
        raise typer.Exit(code=1)
    bc_path = db.data_home() / "models" / f"bc_{fmt.name}.pt"
    init = bc_path if init_from_bc and bc_path.exists() else None
    path = train_cql(
        conn,
        fmt,
        PMIModel.load(pmi_path),
        steps=steps,
        alpha=alpha,
        seed=seed,
        init_from=init,
        log=typer.echo,
    )
    typer.echo(f"Wrote {path}")


@train_app.command("export")
def train_export():
    """Convert existing .pt checkpoints to torch-free .npz weights (needs
    torch; new training runs write both automatically)."""
    import torch

    from .ml.infer_np import save_np_checkpoint

    models_dir = db.data_home() / "models"
    converted = 0
    for pt in sorted(models_dir.glob("*.pt")):
        ckpt = torch.load(pt, map_location="cpu")
        save_np_checkpoint(
            pt.with_suffix(".npz"),
            ckpt["state_dict"],
            ckpt["oracle_ids"],
            ckpt["format"],
            ckpt["algo"],
        )
        typer.echo(f"{pt.name} -> {pt.with_suffix('.npz').name}")
        converted += 1
    if not converted:
        typer.echo("No .pt checkpoints found in ~/.doubletap/models/")


@app.command("eval")
def eval_cmd(
    model_path: Path = typer.Option(..., "--model", exists=True),
    n_hide: int = typer.Option(10),
    seed: int = typer.Option(0),
):
    """Recovery@k of a checkpoint on the holdout split it was trained against."""
    import numpy as np
    import torch

    from .ml.data import build_vocab, load_corpus
    from .ml.eval import recovery_at_k
    from .ml.model import load_checkpoint
    from .ml.train_bc import split_corpus

    conn = db.connect()
    ckpt_meta = torch.load(model_path, map_location="cpu")
    fmt = formats.get_format(ckpt_meta["format"])
    vocab = build_vocab(conn, fmt)
    model, ckpt = load_checkpoint(model_path, vocab)
    _, holdout = split_corpus(load_corpus(conn, vocab, fmt), seed=seed)
    metrics = recovery_at_k(
        model, holdout, vocab, fmt, n_hide=n_hide, rng=np.random.default_rng(seed)
    )
    typer.echo(f"{ckpt['algo']} {fmt.name}: {metrics}")


def _load_model(conn, fmt, model_path: Path | None):
    """Resolve and load the checkpoint plus its matching vocab. BC ships as
    default: CQL missed the agreed keep-bar (+2 recovery@50) on the reliable
    200-deck eval. Prefers the torch-free .npz weights; .pt needs torch
    (run `doubletap train export` once to convert old checkpoints)."""
    if model_path is None:
        models_dir = db.data_home() / "models"
        for candidate in (
            models_dir / f"bc_{fmt.name}.npz",
            models_dir / f"cql_{fmt.name}.npz",
            models_dir / f"bc_{fmt.name}.pt",
            models_dir / f"cql_{fmt.name}.pt",
        ):
            if candidate.exists():
                model_path = candidate
                break
        if model_path is None:
            typer.echo("No trained model found; run doubletap train first.", err=True)
            raise typer.Exit(code=1)

    from .ml.data import build_vocab

    vocab = build_vocab(conn, fmt)
    if model_path.suffix == ".npz":
        from .ml.infer_np import load_np_checkpoint

        model, ckpt = load_np_checkpoint(model_path, vocab)
    else:
        # torch path; kept for explicit --model foo.pt on training machines
        from .ml.model import load_checkpoint

        model, ckpt = load_checkpoint(model_path, vocab)
    return vocab, model, ckpt, model_path


def _deck_to_idxs(conn, deck, vocab, fmt):
    """Map a Deck to vocab indices (expanded by qty), reporting problems."""
    import numpy as np

    for violation in formats.validate(conn, deck):
        if violation.code != "wrong_size":  # partial decks are the normal input
            typer.echo(f"note: {violation.message}")
    partial = []
    for oid, qty in deck.entries.items():
        idx = vocab.index.get(oid)
        if idx is None:
            typer.echo(f"note: skipping card not legal in {fmt.name} (oracle {oid})")
            continue
        partial.extend([idx] * qty)
    commander_idx = vocab.index.get(deck.commander) if deck.commander else None
    partner_idx = vocab.index.get(deck.partner) if deck.partner else None
    return np.array(partial, dtype=np.int64), commander_idx, partner_idx


def _card_name(conn, vocab, idx):
    (name,) = conn.execute(
        "SELECT name FROM cards WHERE oracle_id = ?", (vocab.oracle_ids[idx],)
    ).fetchone()
    return name


def _budget_mask(conn, vocab, max_card_price):
    """Vocab-length bool mask: True where the card's market price is at or
    under the cap. Unpriced cards stay eligible (paper cards almost always
    have a price; unpriced ones are mostly digital-only)."""
    import json

    import numpy as np

    mask = np.ones(len(vocab), dtype=bool)
    for oid, raw in conn.execute("SELECT oracle_id, json FROM cards"):
        i = vocab.index.get(oid)
        if i is None:
            continue
        price = analysis.card_price(json.loads(raw))
        if price is not None and price > max_card_price:
            mask[i] = False
    return mask


def _game_changer_idxs(conn, vocab) -> set[int]:
    """Vocab indices of every Game Changer present in the card cache."""
    idxs = set()
    for name in formats.GAME_CHANGERS:
        row = conn.execute(
            "SELECT oracle_id FROM cards WHERE name = ?", (name,)
        ).fetchone()
        if row and row[0] in vocab.index:
            idxs.add(vocab.index[row[0]])
    return idxs


def _structure_report(deck, vocab, fmt, partial):
    size = deck.size()
    n_lands = int(vocab.land[partial].sum())
    land_frac = n_lands / max(size, 1)
    target_lands = round(fmt.land_fraction_target * fmt.deck_size)
    typer.echo(
        f"\nStructure: {size}/{fmt.deck_size} cards; {n_lands} lands ({land_frac:.0%})"
        f" vs ~{target_lands} target — mana base is yours to tune."
    )


@app.command("recommend")
def recommend(
    deck_path: Path = typer.Option(..., "--deck", exists=True, readable=True),
    k: int = typer.Option(20, "-k", help="Number of suggestions"),
    model_path: Path = typer.Option(
        None,
        "--model",
        help="Checkpoint; defaults to bc_<format>.pt, falling back to cql_<format>.pt",
    ),
    personalize: float = typer.Option(
        0.3,
        help="Blend weight for neighbor-deck frequencies (0 = model only): "
        "cards common in corpus decks similar to yours rank higher",
    ),
    max_card_price: float = typer.Option(
        None,
        "--max-card-price",
        help="Budget cap in USD: only suggest cards at or under this market price",
    ),
):
    """Suggest the top-k additions for a (partial) deck, with synergy rationale
    and a structural gap report. Lands are handled by the gap report, not the
    model."""
    import numpy as np

    conn = db.connect()
    deck = decks.Deck.load(deck_path)
    fmt = formats.get_format(deck.format)
    vocab, model, ckpt, model_path = _load_model(conn, fmt, model_path)

    # torch-backed imports after model resolution: a torch-less install gets
    # _load_model's friendly "no trained model" message, not a traceback
    from .ml.eval import score_state
    from .ml.neighbors import blend, neighbor_frequencies
    from .ml.reward import PMIModel

    pmi_path = db.data_home() / "models" / f"pmi_{fmt.name}.npz"
    pmi = PMIModel.load(pmi_path) if pmi_path.exists() else None
    partial, commander_idx, partner_idx = _deck_to_idxs(conn, deck, vocab, fmt)

    extra_mask = (
        _budget_mask(conn, vocab, max_card_price)
        if max_card_price is not None
        else None
    )
    scores = score_state(
        model, vocab, fmt, partial, commander_idx, partner_idx, extra_mask
    )
    label = f"{ckpt['algo']} model, {model_path.name}"
    if max_card_price is not None:
        label += f", max ${max_card_price:.2f}/card"
    if personalize > 0:
        freqs = neighbor_frequencies(conn, vocab, fmt, partial)
        if freqs is not None:
            scores = blend(scores, freqs, personalize)
            label += f", personalize={personalize}"

    top = np.argsort(-scores)[:k]
    typer.echo(f"Top {k} additions ({label}):")
    for rank, idx in enumerate(top, 1):
        line = f"{rank:3}. {_card_name(conn, vocab, idx):42} {scores[idx]:7.3f}"
        if pmi is not None:
            contributors = pmi.top_contributors(int(idx), partial)
            if contributors:
                line += "  with " + ", ".join(
                    f"{_card_name(conn, vocab, c)} ({v:.1f})" for c, v in contributors
                )
        typer.echo(line)
    _structure_report(deck, vocab, fmt, partial)


@app.command("complete")
def complete(
    deck_path: Path = typer.Option(..., "--deck", exists=True, readable=True),
    out: Path = typer.Option(
        None, "--out", "-o", help="Write the completed deck as JSON"
    ),
    model_path: Path = typer.Option(None, "--model"),
    max_card_price: float = typer.Option(
        None,
        "--max-card-price",
        help="Budget cap in USD: only add cards at or under this market price",
    ),
    bracket: int = typer.Option(
        3,
        "--bracket",
        min=1,
        max=5,
        help="Target Commander Bracket: 1-2 add no Game Changers, 3 keeps the"
        " deck at three or fewer, 4-5 unrestricted",
    ),
):
    """Fill a partial deck's nonland slots with the model's top picks (greedy,
    re-scored after each add). Lands are reported as a gap, not added.
    Stays within the target Commander Bracket (default 3) unless raised."""
    import numpy as np

    conn = db.connect()
    deck = decks.Deck.load(deck_path)
    fmt = formats.get_format(deck.format)
    vocab, model, ckpt, model_path = _load_model(conn, fmt, model_path)

    from .ml.eval import complete_deck

    partial, commander_idx, partner_idx = _deck_to_idxs(conn, deck, vocab, fmt)

    extra_mask = (
        _budget_mask(conn, vocab, max_card_price)
        if max_card_price is not None
        else None
    )
    capped_idxs, cap = None, 0
    if fmt.name == "commander" and bracket <= 3:
        gc_idxs = _game_changer_idxs(conn, vocab)
        already = sum(1 for i in (*partial, commander_idx, partner_idx) if i in gc_idxs)
        cap = max(0, 3 - already) if bracket == 3 else 0
        capped_idxs = np.array(sorted(gc_idxs), dtype=np.int64)

    added, final = complete_deck(
        model,
        vocab,
        fmt,
        partial,
        commander_idx,
        partner_idx,
        extra_mask,
        capped_idxs=capped_idxs,
        cap=cap,
    )
    for idx in added:
        deck.entries[vocab.oracle_ids[idx]] += 1
        typer.echo(f"+ {_card_name(conn, vocab, idx)}")
    lands_needed = fmt.deck_size - deck.size()
    bracket_note = f", bracket ≤{bracket}" if capped_idxs is not None else ""
    typer.echo(
        f"\nAdded {len(added)} cards ({ckpt['algo']} model{bracket_note}); "
        f"add {lands_needed} lands to reach {fmt.deck_size}."
    )
    _structure_report(deck, vocab, fmt, final)
    if out:
        deck.save(conn, out)
        typer.echo(f"Wrote {out}")


@deck_app.command("merge")
def deck_merge(
    paths: list[Path] = typer.Argument(..., help="Deck JSON files to merge"),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Output file (default: ~/.doubletap/decks/merged.json)",
    ),
    deck_format: str = typer.Option(
        None,
        "--format",
        "-f",
        help="Override the format of the merged deck (e.g. commander, modern)",
    ),
):
    """Combine multiple deck JSON files into one deck.
    Use --format to set or change the format of the result."""
    if len(paths) < 2:
        typer.echo("Provide at least two deck files to merge.", err=True)
        raise typer.Exit(code=1)

    loaded = []
    for p in paths:
        if not p.exists():
            typer.echo(f"File not found: {p}", err=True)
            raise typer.Exit(code=1)
        loaded.append(decks.Deck.load(p))

    fmt_names = {d.format for d in loaded}
    if deck_format:
        # explicit override: validate it's a known format, then apply
        formats.get_format(deck_format)
        final_format = deck_format
    elif len(fmt_names) > 1:
        typer.echo(
            f"Decks have different formats {fmt_names}; use --format to set one.",
            err=True,
        )
        raise typer.Exit(code=1)
    else:
        final_format = loaded[0].format

    merged = decks.Deck(format=final_format)
    commanders_seen = []
    partners_seen = []
    for d in loaded:
        merged.entries.update(d.entries)
        if d.commander:
            commanders_seen.append(d.commander)
        if d.partner:
            partners_seen.append(d.partner)
        if d.companion and merged.companion is None:
            merged.companion = d.companion

    if commanders_seen:
        merged.commander = commanders_seen[0]
        if len(commanders_seen) > 1:
            typer.echo(
                f"note: multiple commanders found; keeping first. "
                f"Others: {commanders_seen[1:]}"
            )
    if partners_seen:
        merged.partner = partners_seen[0]

    conn = db.connect()
    save_path = out or db.decks_dir() / "merged.json"
    merged.save(conn, save_path)
    typer.echo(f"{merged.size()} cards ({len(merged.entries)} distinct) → {save_path}")


@deck_app.command("bracket")
def deck_bracket(path: Path = typer.Argument(..., exists=True, readable=True)):
    """Show the Commander Bracket for a deck based on its Game Changers content."""
    conn = db.connect()
    deck = decks.Deck.load(path)
    all_oids = list(deck.entries)
    if deck.commander:
        all_oids.append(deck.commander)
    if deck.partner:
        all_oids.append(deck.partner)
    card_names = []
    for oid in all_oids:
        row = conn.execute(
            "SELECT name FROM cards WHERE oracle_id = ?", (oid,)
        ).fetchone()
        if row:
            card_names.append(row[0])
    bracket, found = formats.compute_bracket(card_names)
    typer.echo(f"Bracket {bracket}: {formats.BRACKETS[bracket]}")
    if found:
        typer.echo(f"\nGame Changers present ({len(found)}):")
        for name in sorted(found):
            typer.echo(f"  • {name}")
    else:
        typer.echo("\nNo Game Changers — eligible for Bracket 1 or 2.")
        typer.echo("(Bracket 1 also requires no infinite combos, no mass land denial,")
        typer.echo(" no extra turns — verify those manually.)")


def _all_entries(deck):
    """Deck entries including the commander and companion slots, as
    oracle_id -> qty (you own and pay for the companion too)."""
    entries = dict(deck.entries)
    for oid in (deck.commander, deck.partner, deck.companion):
        if oid:
            entries[oid] = entries.get(oid, 0) + 1
    return entries


@deck_app.command("analyze")
def deck_analyze(path: Path = typer.Argument(..., exists=True, readable=True)):
    """How does this deck function and win? Roles vs Commander targets, mana
    curve, color balance, interaction speed, win conditions, market price."""
    conn = db.connect()
    deck = decks.Deck.load(path)
    report = analysis.deck_report(conn, _all_entries(deck))
    by_role = report.by_role

    def count(role):
        return sum(qty for _, qty in by_role.get(role, []))

    typer.echo(f"{deck.format} deck, {deck.size()} cards\n")
    typer.echo("Functional roles (a card can fill several):")
    labels = [
        ("land", "Lands", "lands"),
        ("ramp", "Ramp (extra mana)", "ramp"),
        ("draw", "Card draw", "draw"),
        ("removal", "Removal/interaction", "removal"),
        ("board_wipe", "Board wipes", "board_wipe"),
        ("tutor", "Tutors (search)", None),
    ]
    for role, label, target_key in labels:
        n = count(role)
        target = (
            analysis.COMMANDER_TARGETS.get(target_key)
            if target_key and deck.format == "commander"
            else None
        )
        target_str = f"   (target ~{target})" if target else ""
        typer.echo(f"  {label:<20} {n:>3}{target_str}")

    n_removal, n_instant = count("removal"), count("removal_instant")
    if n_removal:
        typer.echo(
            f"\nInteraction speed: {n_instant} of {n_removal} removal spells"
            " work at instant speed (castable on opponents' turns)."
        )

    if report.curve:
        typer.echo("\nMana curve (nonland cards by mana value):")
        buckets = "  ".join(
            f"{mv if mv < analysis.CURVE_TOP_BUCKET else '7+'}:{report.curve.get(mv, 0)}"
            for mv in range(analysis.CURVE_TOP_BUCKET + 1)
        )
        typer.echo(f"  {buckets}")
        typer.echo(
            f"  average {report.avg_mv:.1f}; cards costing 2 or less"
            f" (your turn-1/2 plays): {report.early_plays}"
        )

    if report.pips and report.sources:
        typer.echo("\nColor balance (mana symbols in costs vs lands producing it):")
        pip_total = sum(report.pips.values())
        source_total = sum(report.sources.values())
        short = set(analysis.short_colors(report))
        for color in "WUBRG":
            if not report.pips.get(color) and not report.sources.get(color):
                continue
            need = report.pips.get(color, 0) / pip_total
            have = report.sources.get(color, 0) / source_total
            flag = "  ← not enough lands make this color" if color in short else ""
            typer.echo(
                f"  {_COLOR_NAMES[color]:<6} {need:4.0%} of symbols,"
                f" {have:4.0%} of sources{flag}"
            )

    typer.echo("\nWays to win:")
    wincons = by_role.get("wincon", [])
    threats = by_role.get("threat", [])
    n_evasive = count("evasive")
    n_poison = count("poison")
    n_mill = count("mill")
    for name, _ in wincons:
        typer.echo(f'  "{name}" wins the game directly')
    if threats:
        typer.echo(
            f"  {sum(q for _, q in threats)} big creatures (power "
            f"{analysis.BIG_THREAT_POWER}+) that can win through combat"
        )
    if n_evasive:
        typer.echo(
            f"  {n_evasive} creatures with evasion (flying, trample, ...) that"
            " can get damage through blockers"
        )
    if n_poison:
        typer.echo(
            f"  {n_poison} cards give poison counters (10 poison = a player loses)"
        )
    if n_mill:
        typer.echo(f"  {n_mill} cards mill libraries (a player who can't draw loses)")
    if not any((wincons, threats, n_evasive, n_poison, n_mill)):
        typer.echo(
            "  none detected — every deck needs a plan: reduce opponents' life"
            " to 0 (creatures, damage), poison, mill, or a card that says"
            ' "you win the game"'
        )

    price_note = f" ({report.unpriced} cards unpriced)" if report.unpriced else ""
    typer.echo(f"\nMarket price: ${report.total_price:,.2f}{price_note}")


@deck_app.command("price")
def deck_price(
    path: Path = typer.Argument(..., exists=True, readable=True),
    top: int = typer.Option(10, help="How many of the most expensive cards to list"),
):
    """Total market price of a deck (Scryfall USD) and its most expensive cards."""
    import json as _json

    conn = db.connect()
    deck = decks.Deck.load(path)
    priced, unpriced = [], []
    total = 0.0
    for oid, qty in _all_entries(deck).items():
        row = conn.execute(
            "SELECT name, json FROM cards WHERE oracle_id = ?", (oid,)
        ).fetchone()
        if row is None:
            continue
        name, raw = row
        price = analysis.card_price(_json.loads(raw))
        if price is None:
            unpriced.append(name)
        else:
            priced.append((price * qty, price, qty, name))
            total += price * qty
    typer.echo(f"Total market price: ${total:,.2f} ({deck.size()} cards)")
    if unpriced:
        typer.echo(f"No price data for: {', '.join(sorted(unpriced))}")
    typer.echo(f"\nMost expensive ({min(top, len(priced))}):")
    for line_total, price, qty, name in sorted(priced, reverse=True)[:top]:
        qty_str = f" x{qty}" if qty > 1 else ""
        typer.echo(f"  ${line_total:>8,.2f}  {name}{qty_str}")


@deck_app.command("validate")
def deck_validate(path: Path = typer.Argument(..., exists=True, readable=True)):
    """Check a deck JSON against its format's construction and legality rules.
    Always shows the deck's identity (commander, colors, size) first, then
    every violation — one problem never hides the rest."""
    conn = db.connect()
    deck = decks.Deck.load(path)
    fmt = formats.get_format(deck.format)

    size_note = (
        f" (needs exactly {fmt.deck_size})"
        if fmt.exact_size and deck.size() != fmt.deck_size
        else f" (needs at least {fmt.deck_size})"
        if not fmt.exact_size and deck.size() < fmt.deck_size
        else ""
    )
    typer.echo(f"{deck.format} deck, {deck.size()} cards{size_note}")
    if deck.commander:
        label = _card_name_by_oid(conn, deck.commander)
        identity = set(formats.get_card(conn, deck.commander)["color_identity"])
        if deck.partner:
            label += f" + {_card_name_by_oid(conn, deck.partner)}"
            identity |= set(formats.get_card(conn, deck.partner)["color_identity"])
        typer.echo(f"Commander: {label}")
        typer.echo(f"Color identity: {_colors_label(identity)}")
    elif fmt.requires_commander:
        typer.echo("Commander: none set (use: doubletap deck commander <deck> <name>)")
    if deck.companion:
        typer.echo(f"Companion: {_card_name_by_oid(conn, deck.companion)}")

    violations = formats.validate(conn, deck)
    if not violations:
        typer.echo("\nValid — no violations.")
        return
    typer.echo(f"\nViolations ({len(violations)}):")
    for v in violations:
        typer.echo(f"  {v.code:16} {v.message}")
    raise typer.Exit(code=1)
