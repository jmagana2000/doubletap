from doubletap.names import lookup, normalize


def test_normalize():
    assert normalize("Lightning Bolt") == "lightning bolt"
    assert normalize("Juzám Djinn") == "juzam djinn"
    assert normalize("Atraxa, Praetors' Voice") == "atraxa praetors voice"
    assert normalize("Fire // Ice") == "fire ice"
    assert normalize("  MANY   spaces  ") == "many spaces"


def test_exact_lookup_prefers_full_name_over_face_collision(loaded_conn):
    # "Lightning Bolt" is also a face of "Emeritus of Conflict // Lightning Bolt";
    # the standalone card must come first, but both are returned
    matches = lookup(loaded_conn, "Lightning Bolt")
    assert [m.name for m in matches] == [
        "Lightning Bolt",
        "Emeritus of Conflict // Lightning Bolt",
    ]
    assert all(m.score == 100.0 for m in matches)


def test_lookup_is_diacritics_insensitive(loaded_conn):
    (m,) = lookup(loaded_conn, "juzam djinn")
    assert m.name == "Juzám Djinn"
    assert m.score == 100.0


def test_face_name_resolves_mdfc(loaded_conn):
    (m,) = lookup(loaded_conn, "Malakir Rebirth")
    assert m.name == "Malakir Rebirth // Malakir Mire"
    assert m.score == 100.0


def test_face_name_resolves_split_card(loaded_conn):
    (m,) = lookup(loaded_conn, "Ice")
    assert m.name == "Fire // Ice"
    assert m.score == 100.0


def test_fuzzy_lookup_typo(loaded_conn):
    # genuinely ambiguous typo: both real candidates must be offered
    matches = lookup(loaded_conn, "lightning blot")
    names = [m.name for m in matches]
    assert "Lightning Bolt" in names
    assert "Lightning Blow" in names
    assert all(m.score < 100.0 for m in matches)


def test_empty_query(loaded_conn):
    assert lookup(loaded_conn, "   ") == []


def test_duplicate_name_ties_prefer_playable_record(loaded_conn):
    """Scryfall sometimes carries two oracle records with one name (playtest/
    promo twins); exact-name resolution must land on the playable one."""
    from doubletap.formats import get_card
    from doubletap.names import lookup

    matches = lookup(loaded_conn, "Twinned Test Mage")
    assert len(matches) == 2 and matches[0].score == 100.0
    top = get_card(loaded_conn, matches[0].oracle_id)
    assert top["legalities"]["commander"] == "legal"
