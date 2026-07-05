import sys
from pathlib import Path

import typer

from . import archidekt, db, decks, formats, names, scryfall

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


@cards_app.command("lookup")
def cards_lookup(name: str):
    """Resolve a card name (exact or fuzzy) against the local cache."""
    conn = db.connect()
    matches = names.lookup(conn, name)
    if not matches:
        typer.echo("No match found. Is the cache synced? (doubletap cards sync)")
        raise typer.Exit(code=1)
    for m in matches:
        typer.echo(f"{m.score:5.1f}  {m.name}  ({m.oracle_id})")


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
        ..., "--format", "-f", help="e.g. commander, modern"
    ),
    out: Path = typer.Option(None, "--out", "-o", help="Write the deck as JSON"),
    commander: str = typer.Option(None, help="Commander card name (Commander format)"),
    threshold: float = typer.Option(90.0, help="Fuzzy auto-accept score threshold"),
    interactive: bool = typer.Option(True, help="Prompt on ambiguous/unmatched names"),
):
    """Import a deck from a CSV, plain-text decklist, or decklist photo."""
    conn = db.connect()
    parsed = decks.load_lines(path)
    if commander:
        parsed.append(
            decks.ParsedLine(raw=commander, qty=1, name=commander, is_commander=True)
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
        + (f", commander set" if deck.commander else "")
    )
    if not result.ok:
        typer.echo("Import incomplete: fix ambiguous/unmatched lines above.", err=True)
        raise typer.Exit(code=1)
    if out:
        deck.save(conn, out)
        typer.echo(f"Wrote {out}")


@corpus_app.command("crawl")
def corpus_crawl(
    deck_format: str = typer.Option(..., "--format", "-f"),
    max_decks: int = typer.Option(1000, "--max", help="Deck ids to queue this run"),
):
    """Crawl public Archidekt decks (rate-limited, resumable) into the corpus."""
    conn = db.connect()

    def progress(done, total):
        if done % 25 == 0 or done == total:
            typer.echo(f"fetched {done}/{total}")

    outcomes = archidekt.crawl(conn, deck_format, max_decks, progress=progress)
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


@app.command("recommend")
def recommend(
    deck_path: Path = typer.Option(..., "--deck", exists=True, readable=True),
    k: int = typer.Option(20, "-k", help="Number of suggestions"),
    model_path: Path = typer.Option(
        None,
        "--model",
        help="Checkpoint; defaults to cql_<format>.pt, falling back to bc_<format>.pt",
    ),
):
    """Suggest the top-k additions for a (partial) deck, with synergy rationale
    and a structural gap report. Lands are handled by the gap report, not the
    model."""
    import numpy as np

    from .ml.data import build_vocab
    from .ml.eval import score_state
    from .ml.model import load_checkpoint
    from .ml.reward import PMIModel

    conn = db.connect()
    deck = decks.Deck.load(deck_path)
    fmt = formats.get_format(deck.format)

    if model_path is None:
        models_dir = db.data_home() / "models"
        for candidate in (
            models_dir / f"cql_{fmt.name}.pt",
            models_dir / f"bc_{fmt.name}.pt",
        ):
            if candidate.exists():
                model_path = candidate
                break
        if model_path is None:
            typer.echo("No trained model found; run doubletap train first.", err=True)
            raise typer.Exit(code=1)

    vocab = build_vocab(conn, fmt)
    model, ckpt = load_checkpoint(model_path, vocab)
    pmi_path = db.data_home() / "models" / f"pmi_{fmt.name}.npz"
    pmi = PMIModel.load(pmi_path) if pmi_path.exists() else None

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
    partial = np.array(partial, dtype=np.int64)
    commander_idx = vocab.index.get(deck.commander) if deck.commander else None

    def card_name(idx):
        (name,) = conn.execute(
            "SELECT name FROM cards WHERE oracle_id = ?", (vocab.oracle_ids[idx],)
        ).fetchone()
        return name

    scores = score_state(model, vocab, fmt, partial, commander_idx)
    top = np.argsort(-scores)[:k]
    typer.echo(f"Top {k} additions ({ckpt['algo']} model, {model_path.name}):")
    for rank, idx in enumerate(top, 1):
        line = f"{rank:3}. {card_name(idx):42} {scores[idx]:7.3f}"
        if pmi is not None:
            contributors = pmi.top_contributors(int(idx), partial)
            if contributors:
                line += "  with " + ", ".join(
                    f"{card_name(c)} ({v:.1f})" for c, v in contributors
                )
        typer.echo(line)

    size = deck.size()
    n_lands = int(vocab.land[partial].sum())
    land_frac = n_lands / max(size, 1)
    target_lands = round(fmt.land_fraction_target * fmt.deck_size)
    typer.echo(
        f"\nStructure: {size}/{fmt.deck_size} cards; {n_lands} lands ({land_frac:.0%})"
        f" vs ~{target_lands} target — mana base is yours to tune."
    )


@deck_app.command("validate")
def deck_validate(path: Path = typer.Argument(..., exists=True, readable=True)):
    """Check a deck JSON against its format's construction and legality rules."""
    conn = db.connect()
    deck = decks.Deck.load(path)
    violations = formats.validate(conn, deck)
    if not violations:
        typer.echo(f"Valid {deck.format} deck ({deck.size()} cards)")
        return
    for v in violations:
        typer.echo(f"{v.code:16} {v.message}")
    raise typer.Exit(code=1)
