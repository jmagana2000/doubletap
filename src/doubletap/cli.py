import sys
from pathlib import Path

import typer

from . import db, decks, names, scryfall

app = typer.Typer(no_args_is_help=True)
cards_app = typer.Typer(no_args_is_help=True, help="Local Scryfall card cache")
app.add_typer(cards_app, name="cards")
deck_app = typer.Typer(no_args_is_help=True, help="Import and manage decks")
app.add_typer(deck_app, name="deck")


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
