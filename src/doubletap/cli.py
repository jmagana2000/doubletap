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
