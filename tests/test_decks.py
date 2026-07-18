from pathlib import Path

import pytest

from doubletap import decks
from doubletap.decks import ParsedLine, parse_csv, parse_text_lines, resolve

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_text_lines_quantities_sections_and_tails():
    parsed = parse_text_lines((FIXTURES / "decklist.txt").read_text().splitlines())
    assert [(p.qty, p.name) for p in parsed] == [
        (4, "Lightning Bolt"),
        (4, "Once Upon a Time"),
        (2, "Fire // Ice"),  # "(MH2) 290" set tail stripped
    ]  # comment and sideboard dropped


def test_parse_text_lines_commander_section_and_marker():
    parsed = parse_text_lines(
        ["Commander", "1 Atraxa, Praetors' Voice", "Deck", "1 Sol Ring"]
    )
    assert parsed[0].is_commander and not parsed[1].is_commander

    parsed = parse_text_lines(["1 Atraxa, Praetors' Voice (2XM) 190 *CMDR*"])
    assert parsed[0].is_commander
    assert parsed[0].name == "Atraxa, Praetors' Voice"


def test_parse_moxfield_csv():
    parsed = parse_csv(FIXTURES / "moxfield.csv")
    assert [(p.qty, p.name) for p in parsed] == [
        (1, "Sol Ring"),
        (1, "Atraxa, Praetors' Voice"),
    ]


def test_parse_archidekt_csv():
    parsed = parse_csv(FIXTURES / "archidekt.csv")
    assert [(p.qty, p.name) for p in parsed] == [
        (4, "Lightning Bolt"),
        (2, "Fire // Ice"),
        (1, "Swamp"),
    ]


def test_parse_csv_without_expected_columns():
    with pytest.raises(ValueError):
        parse_csv(FIXTURES / "oracle-cards.jsonl")


def _line(name, qty=1, is_commander=False):
    return ParsedLine(raw=name, qty=qty, name=name, is_commander=is_commander)


def test_resolve_exact_and_commander(loaded_conn):
    result = resolve(
        loaded_conn,
        [_line("Atraxa, Praetors' Voice", is_commander=True), _line("Sol Ring")],
        "commander",
    )
    assert result.ok
    assert len(result.resolved) == 2
    assert result.deck.commander is not None
    assert result.deck.size() == 2
    assert len(result.deck.entries) == 1  # commander not in entries


def test_resolve_assumed_typo(loaded_conn):
    result = resolve(loaded_conn, [_line("Atraxa Praetors Voce")], "commander")
    assert result.ok
    assert len(result.assumed) == 1
    assert result.assumed[0].matches[0].name == "Atraxa, Praetors' Voice"


def test_resolve_ambiguous_between_similar_names(loaded_conn):
    # "lightning blot" is nearly equidistant from Lightning Bolt and Lightning Blow
    result = resolve(loaded_conn, [_line("lightning blot")], "modern")
    assert not result.ok
    assert len(result.ambiguous) == 1
    assert result.deck.size() == 0


def test_resolve_chooser_settles_ambiguity(loaded_conn):
    def chooser(line, matches):
        return next(m for m in matches if m.name == "Lightning Bolt")

    result = resolve(
        loaded_conn, [_line("lightning blot", qty=4)], "modern", chooser=chooser
    )
    assert result.ok
    assert len(result.resolved) == 1
    assert sum(result.deck.entries.values()) == 4


def test_resolve_unmatched(loaded_conn):
    result = resolve(loaded_conn, [_line("Zzyzx Quuxblade")], "modern")
    assert not result.ok
    assert len(result.unmatched) == 1


def test_deck_save_load_round_trip(loaded_conn, tmp_path):
    result = resolve(
        loaded_conn,
        [
            _line("Atraxa, Praetors' Voice", is_commander=True),
            _line("Sol Ring"),
            _line("Swamp", qty=30),
        ],
        "commander",
    )
    path = tmp_path / "deck.json"
    result.deck.save(loaded_conn, path)
    loaded = decks.Deck.load(path)
    assert loaded.format == "commander"
    assert loaded.commander == result.deck.commander
    assert loaded.entries == result.deck.entries


def test_load_lines_routes_images_through_ocr(loaded_conn, tmp_path, monkeypatch):
    from doubletap import ocr

    monkeypatch.setattr(
        ocr,
        "recognize_text",
        lambda path: [
            ("Commander", 0.95),
            ("1 Atraxa, Praetors' Voice", 0.9),
            ("Deck", 0.95),
            ("4 Lightning Bolt", 0.85),
            ("~~ smudge ~~", 0.1),  # below confidence floor, dropped
        ],
    )
    image = tmp_path / "list.png"
    image.write_bytes(b"not a real image")
    parsed = decks.load_lines(image)
    result = resolve(loaded_conn, parsed, "commander")
    assert result.ok
    assert result.deck.commander is not None
    assert sum(result.deck.entries.values()) == 4


@pytest.mark.macos_ocr
def test_real_vision_ocr_smoke(tmp_path):
    import os

    image = os.environ.get("DOUBLETAP_OCR_TEST_IMAGE")
    if not image:
        pytest.skip("set DOUBLETAP_OCR_TEST_IMAGE to a decklist image path")
    from doubletap.ocr import recognize_text

    lines = recognize_text(Path(image))
    assert lines and all(0.0 <= conf <= 1.0 for _, conf in lines)


def test_parse_text_lines_companion_section():
    parsed = parse_text_lines(
        [
            "Companion",
            "1 Lurrus of the Dream-Den",
            "Deck",
            "4 Lightning Bolt",
        ]
    )
    assert parsed[0].is_companion and not parsed[0].is_commander
    assert not parsed[1].is_companion


def test_resolve_companion_outside_deck(loaded_conn):
    result = resolve(
        loaded_conn,
        [
            ParsedLine(
                raw="Lurrus of the Dream-Den",
                qty=1,
                name="Lurrus of the Dream-Den",
                is_companion=True,
            ),
            _line("Lightning Bolt", qty=4),
        ],
        "modern",
    )
    assert result.ok
    assert result.deck.companion is not None
    assert result.deck.companion not in result.deck.entries
    assert result.deck.size() == 4  # companion never counts toward deck size


def test_partner_and_companion_save_load_round_trip(loaded_conn, tmp_path):
    result = resolve(
        loaded_conn,
        [
            _line("Thrasios, Triton Hero", is_commander=True),
            _line("Tymna the Weaver", is_commander=True),
            ParsedLine(
                raw="Lurrus of the Dream-Den",
                qty=1,
                name="Lurrus of the Dream-Den",
                is_companion=True,
            ),
            _line("Sol Ring"),
        ],
        "commander",
    )
    assert result.ok
    path = tmp_path / "deck.json"
    result.deck.save(loaded_conn, path)
    loaded = decks.Deck.load(path)
    assert loaded.commander == result.deck.commander
    assert loaded.partner == result.deck.partner
    assert loaded.companion == result.deck.companion
    assert loaded.size() == 3  # two commanders + Sol Ring, companion excluded


def test_arena_export_import_lines():
    """MTG Arena export: Deck/Sideboard headers, (SET) collector tails."""
    from doubletap.decks import parse_text_lines

    lines = [
        "Deck",
        "4 Standard Strike (FDN) 137",
        "20 Mountain (FDN) 269",
        "Sideboard",
        "3 Negate (FDN) 60",
    ]
    parsed = parse_text_lines(lines)
    names = {(p.name, p.qty) for p in parsed}
    assert names == {("Standard Strike", 4), ("Mountain", 20)}  # sideboard dropped


def test_foil_commander_marker(loaded_conn):
    """Moxfield foil commanders carry two markers: '*F* *CMDR*'."""
    from doubletap.decks import parse_text_lines

    parsed = parse_text_lines(["1 Atraxa, Praetors' Voice (2XM) 190 *F* *CMDR*"])
    assert parsed[0].is_commander
    assert parsed[0].name == "Atraxa, Praetors' Voice"


def test_csv_categories_and_short_rows(tmp_path):
    from doubletap.decks import parse_csv

    p = tmp_path / "export.csv"
    p.write_text(
        "Count,Name,Category\n"
        "1,Sol Ring,Deck\n"
        "1,Negate,Sideboard\n"
        "1,Atraxa Praetors Voice,Commander\n"
        "4\n"  # short row: fewer fields than the header
    )
    parsed = parse_csv(p)
    names = {(pl.name, pl.is_commander) for pl in parsed}
    assert ("Sol Ring", False) in names
    assert ("Atraxa Praetors Voice", True) in names
    assert all(pl.name != "Negate" for pl in parsed)  # sideboard dropped


def test_ocr_quantityless_decklist_keeps_all_lines(monkeypatch, tmp_path):
    """A screenshot listing bare card names must not collapse to one card."""
    from doubletap import decks as decks_mod

    lines = ["Sol Ring", "Juzám Djinn", "Relentless Rats", "Lightning Bolt", "Negate"]
    monkeypatch.setattr(
        "doubletap.ocr.recognize_text", lambda p: [(t, 0.9) for t in lines]
    )
    img = tmp_path / "list.png"
    img.write_bytes(b"fake")
    parsed = decks_mod.load_lines(img)
    assert len(parsed) == 5


def test_third_commander_line_reported_not_dropped(loaded_conn):
    from doubletap.decks import ParsedLine, resolve

    def line(name):
        return ParsedLine(raw=name, qty=1, name=name, is_commander=True)

    result = resolve(
        loaded_conn,
        [line("Atraxa, Praetors' Voice"), line("Thrasios, Triton Hero"),
         line("Tymna the Weaver")],
        "commander",
    )
    assert not result.ok  # the third commander is surfaced, not silently lost
    assert len(result.ambiguous) == 1


def test_cmdr_marker_ignored_outside_commander(loaded_conn):
    from doubletap.decks import ParsedLine, resolve

    result = resolve(
        loaded_conn,
        [ParsedLine(raw="x", qty=1, name="Lightning Bolt", is_commander=True)],
        "modern",
    )
    assert result.deck.commander is None
    assert sum(result.deck.entries.values()) == 1  # a plain main-deck card
