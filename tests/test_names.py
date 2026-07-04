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
    matches = lookup(loaded_conn, "lightning blot")
    assert matches
    # standalone card outranks the face-name collision at the same score
    assert matches[0].name == "Lightning Bolt"
    assert matches[0].score < 100.0


def test_empty_query(loaded_conn):
    assert lookup(loaded_conn, "   ") == []
