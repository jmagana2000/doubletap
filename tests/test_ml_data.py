import numpy as np
import pytest

from doubletap.formats import COMMANDER, MODERN
from doubletap.ml.data import (
    FEATURE_DIM,
    STATE_DIM,
    action_mask,
    build_vocab,
    load_corpus,
    sample_batch,
    state_features,
)
from doubletap.names import lookup


@pytest.fixture
def vocab(loaded_conn):
    return build_vocab(loaded_conn, COMMANDER)


def vidx(conn, vocab, name):
    return vocab.index[lookup(conn, name)[0].oracle_id]


def test_vocab_covers_commander_legal_cards_only(loaded_conn, vocab):
    names = set()
    for oid in vocab.oracle_ids:
        (name,) = loaded_conn.execute(
            "SELECT name FROM cards WHERE oracle_id = ?", (oid,)
        ).fetchone()
        names.add(name)
    assert "Sol Ring" in names
    assert "Ace Flockbringer" not in names  # digital-only
    assert vocab.features.shape == (len(vocab), FEATURE_DIM)


def test_vocab_flags(loaded_conn, vocab):
    swamp = vidx(loaded_conn, vocab, "Swamp")
    rats = vidx(loaded_conn, vocab, "Relentless Rats")
    bolt = vidx(loaded_conn, vocab, "Lightning Bolt")
    assert vocab.land[swamp] and vocab.basic[swamp]
    assert vocab.any_number[rats] and not vocab.any_number[bolt]
    assert vocab.identity_bits[bolt] == 1 << 3  # R is bit 3 of WUBRG


def test_action_mask_excludes_lands_copies_and_identity(loaded_conn, vocab):
    atraxa = vidx(loaded_conn, vocab, "Atraxa, Praetors' Voice")
    sol_ring = vidx(loaded_conn, vocab, "Sol Ring")
    bolt = vidx(loaded_conn, vocab, "Lightning Bolt")
    swamp = vidx(loaded_conn, vocab, "Swamp")
    rats = vidx(loaded_conn, vocab, "Relentless Rats")

    partial = np.array([sol_ring, rats, rats], dtype=np.int64)
    mask = action_mask(vocab, COMMANDER, partial, commander_idx=atraxa)
    assert not mask[swamp]  # lands are never model actions
    assert not mask[sol_ring]  # singleton copy already in deck
    assert mask[rats]  # any-number exemption
    assert not mask[bolt]  # red, outside GWUB identity
    assert not mask[atraxa]  # the commander itself


def test_action_mask_modern_copy_limit(loaded_conn):
    vocab = build_vocab(loaded_conn, MODERN)
    bolt = vidx(loaded_conn, vocab, "Lightning Bolt")
    three = np.array([bolt] * 3, dtype=np.int64)
    four = np.array([bolt] * 4, dtype=np.int64)
    assert action_mask(vocab, MODERN, three, None)[bolt]
    assert not action_mask(vocab, MODERN, four, None)[bolt]


def test_state_features(loaded_conn, vocab):
    atraxa = vidx(loaded_conn, vocab, "Atraxa, Praetors' Voice")
    sol_ring = vidx(loaded_conn, vocab, "Sol Ring")
    swamp = vidx(loaded_conn, vocab, "Swamp")
    partial = np.array([sol_ring, swamp, swamp], dtype=np.int64)
    feats = state_features(vocab, COMMANDER, partial, atraxa)
    assert feats.shape == (STATE_DIM,)
    assert feats[0:9].sum() == pytest.approx(1 / 100)  # one nonland card
    assert feats[9] == pytest.approx(2 / 100)  # two lands
    assert feats[10] == pytest.approx(3 / 100)
    assert feats[11:16].tolist() == [1, 1, 1, 0, 1]  # WUBRG: Atraxa is WUBG


def _insert_corpus_deck(conn, deck_id, fmt, entries, commander_oid=None):
    conn.execute(
        "INSERT INTO decks (deck_id, source, format, commander_oracle_id, status)"
        " VALUES (?, 'test', ?, ?, 'parsed')",
        (deck_id, fmt, commander_oid),
    )
    for oid, qty in entries.items():
        conn.execute(
            "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, ?)",
            (deck_id, oid, qty),
        )
    if commander_oid:
        conn.execute(
            "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, 1)",
            (deck_id, commander_oid),
        )


@pytest.fixture
def corpus_conn(loaded_conn):
    def o(name):
        return lookup(loaded_conn, name)[0].oracle_id

    _insert_corpus_deck(
        loaded_conn,
        1,
        "commander",
        {o("Sol Ring"): 1, o("Relentless Rats"): 5, o("Swamp"): 10},
        commander_oid=o("Atraxa, Praetors' Voice"),
    )
    _insert_corpus_deck(
        loaded_conn,
        2,
        "commander",
        {o("Sol Ring"): 1, o("Juzám Djinn"): 1, o("Swamp"): 8},
        commander_oid=o("Atraxa, Praetors' Voice"),
    )
    loaded_conn.commit()
    return loaded_conn


def test_load_corpus(corpus_conn, vocab):
    decks = load_corpus(corpus_conn, vocab, COMMANDER)
    assert len(decks) == 2
    d1 = next(d for d in decks if d.deck_id == 1)
    assert d1.commander_idx == vidx(corpus_conn, vocab, "Atraxa, Praetors' Voice")
    assert d1.main_idxs.size == 16  # commander excluded, qty expanded
    assert d1.nonland_positions.size == 6


def test_sample_batch_transitions_are_consistent(corpus_conn, vocab):
    decks = load_corpus(corpus_conn, vocab, COMMANDER)
    rng = np.random.default_rng(7)
    batch = sample_batch(decks, vocab, COMMANDER, 64, rng)
    assert batch.action.shape == (64,)
    assert not vocab.land[batch.action].any()  # targets are always nonland
    assert batch.offsets[0] == 0
    assert np.all(np.diff(batch.offsets) >= 0)
    assert batch.state_feats.shape == (64, STATE_DIM)
    # reproducible with the same seed
    again = sample_batch(decks, vocab, COMMANDER, 64, np.random.default_rng(7))
    assert np.array_equal(batch.action, again.action)
    assert np.array_equal(batch.bag, again.bag)
