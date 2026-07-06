import numpy as np
import pytest

from doubletap.formats import COMMANDER
from doubletap.ml.data import build_vocab
from doubletap.ml.neighbors import blend, neighbor_frequencies
from doubletap.names import lookup

from test_ml_data import corpus_conn  # noqa: F401  (fixture)


def vidx(conn, vocab, name):
    return vocab.index[lookup(conn, name)[0].oracle_id]


def test_neighbor_frequencies_favor_similar_decks(corpus_conn):  # noqa: F811
    vocab = build_vocab(corpus_conn, COMMANDER)
    sol_ring = vidx(corpus_conn, vocab, "Sol Ring")
    rats = vidx(corpus_conn, vocab, "Relentless Rats")
    juzam = vidx(corpus_conn, vocab, "Juzám Djinn")
    bolt = vidx(corpus_conn, vocab, "Lightning Bolt")

    partial = np.array([sol_ring, rats], dtype=np.int64)
    freq = neighbor_frequencies(corpus_conn, vocab, COMMANDER, partial)
    assert freq is not None
    assert freq[sol_ring] == 1.0  # in both corpus decks
    assert freq[juzam] == 0.5  # in one of two
    assert freq[bolt] == 0.0  # in neither


def test_neighbor_frequencies_exclude_near_duplicates(corpus_conn):  # noqa: F811
    vocab = build_vocab(corpus_conn, COMMANDER)

    def o(name):
        return vidx(corpus_conn, vocab, name)

    # exactly deck 1's distinct cards (incl. commander): deck 1 is a near-dup
    # of the query and must not vote for itself
    partial = np.array(
        [o("Sol Ring"), o("Relentless Rats"), o("Swamp"), o("Atraxa, Praetors' Voice")],
        dtype=np.int64,
    )
    freq = neighbor_frequencies(corpus_conn, vocab, COMMANDER, partial, near_dup=0.99)
    assert freq is not None
    assert freq[o("Relentless Rats")] == 0.0  # only in the excluded deck
    assert freq[o("Juzám Djinn")] == 1.0  # deck 2 is now the only neighbor


def test_neighbor_frequencies_empty_inputs(corpus_conn):  # noqa: F811
    vocab = build_vocab(corpus_conn, COMMANDER)
    assert (
        neighbor_frequencies(corpus_conn, vocab, COMMANDER, np.empty(0, dtype=np.int64))
        is None
    )


def test_blend_respects_mask_and_weight():
    scores = np.array([-np.inf, 1.0, 2.0, 3.0], dtype=np.float32)
    freqs = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    pure_model = blend(scores, freqs, 0.0)
    assert np.isneginf(pure_model[0])
    assert pure_model[3] > pure_model[1]  # model order preserved
    heavy = blend(scores, freqs, 1.0)
    assert np.isneginf(heavy[0])  # masked stays masked even with freq 1.0
    assert heavy[1] > heavy[3]  # neighbors flip the order at weight 1
