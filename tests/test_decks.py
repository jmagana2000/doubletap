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
