import numpy as np
import pytest

from doubletap.formats import COMMANDER
from doubletap.ml.data import Vocab
from doubletap.ml.reward import (
    PMIModel,
    build_pmi,
    make_pmi_reranker,
    pool_synergy,
    step_reward,
    structure_reward,
)


def sets(*decks):
    return [np.array(d, dtype=np.int64) for d in decks]


def test_ppmi_rewards_cooccurrence_and_floors_at_zero():
    # A(0) and B(1) always together; C(2) appears alone in other decks
    corpus = sets(*([[0, 1]] * 10 + [[2]] * 10))
    pmi = build_pmi(corpus, vocab_size=3, min_count=2)
    assert pmi.ppmi(0, 1) > 0
    assert pmi.ppmi(0, 2) == 0.0  # never co-occur
    assert pmi.ppmi(0, 0) == 0.0


def test_ppmi_min_count_kills_rare_pairs():
    corpus = sets([0, 1], *[[2, 3]] * 10)
    pmi = build_pmi(corpus, vocab_size=4, min_count=2)
    assert pmi.ppmi(0, 1) == 0.0  # seen once, below min_count
    assert pmi.ppmi(2, 3) > 0


def test_ppmi_staple_scores_below_niche_pair():
    # staple S(0) in every deck; niche N1(1)+N2(2) only ever together
    corpus = sets(*([[0, 1, 2]] * 5 + [[0, 3]] * 15))
    pmi = build_pmi(corpus, vocab_size=4, min_count=2)
    assert pmi.ppmi(1, 2) > pmi.ppmi(0, 3)


def test_synergy_and_contributors():
    corpus = sets(*([[0, 1, 2]] * 10 + [[3]] * 10))
    pmi = build_pmi(corpus, vocab_size=4, min_count=2)
    partial = np.array([1, 2, 3], dtype=np.int64)
    assert pmi.synergy(0, partial) > 0
    assert pmi.synergy(0, np.empty(0, dtype=np.int64)) == 0.0
    contributors = pmi.top_contributors(0, partial, k=2)
    assert [c for c, _ in contributors] == [1, 2] or [c for c, _ in contributors] == [
        2,
        1,
    ]
    assert all(v > 0 for _, v in contributors)


def test_pmi_save_load_round_trip(tmp_path):
    corpus = sets(*([[0, 1]] * 10))
    pmi = build_pmi(corpus, vocab_size=2, min_count=2)
    path = tmp_path / "pmi.npz"
    pmi.save(path)
    loaded = PMIModel.load(path)
    assert loaded.n_decks == pmi.n_decks
    assert loaded.pairs == pytest.approx(pmi.pairs)


def test_pool_synergy_matches_per_target_synergy():
    corpus = sets(*([[0, 1, 2]] * 10 + [[3]] * 10))
    pmi = build_pmi(corpus, vocab_size=4, min_count=2)
    partial = np.array([1, 2], dtype=np.int64)
    pool = np.array([0, 3], dtype=np.int64)
    syn = pool_synergy(pmi, pool, partial)
    assert syn[0] == pytest.approx(pmi.synergy(0, partial))
    assert syn[1] == pytest.approx(pmi.synergy(3, partial))


def test_pool_synergy_empty_partial_is_zero():
    corpus = sets(*([[0, 1]] * 10))
    pmi = build_pmi(corpus, vocab_size=2, min_count=2)
    syn = pool_synergy(pmi, np.array([0, 1]), np.empty(0, dtype=np.int64))
    assert (syn == 0.0).all()


def test_pmi_reranker_boosts_synergistic_candidate():
    # 0 and 1 always co-occur; 2 never co-occurs with either
    corpus = sets(*([[0, 1]] * 15 + [[2]] * 15))
    pmi = build_pmi(corpus, vocab_size=3, min_count=2)
    partial = np.array([0], dtype=np.int64)
    scores = np.array([1.0, 1.0, 5.0], dtype=np.float32)  # model favors 2

    reranker = make_pmi_reranker(pmi, weight=1.0)
    blended = reranker(scores, partial, commander_idx=None)
    assert blended[1] > blended[2]  # synergy flips the model's own ranking


def test_pmi_reranker_zero_weight_is_a_noop():
    corpus = sets(*([[0, 1]] * 15))
    pmi = build_pmi(corpus, vocab_size=2, min_count=2)
    scores = np.array([1.0, 2.0], dtype=np.float32)
    reranker = make_pmi_reranker(pmi, weight=0.0)
    assert np.array_equal(reranker(scores, np.array([0], dtype=np.int64), None), scores)


def test_pmi_reranker_respects_mask():
    corpus = sets(*([[0, 1]] * 15))
    pmi = build_pmi(corpus, vocab_size=2, min_count=2)
    scores = np.array([-np.inf, 2.0], dtype=np.float32)
    reranker = make_pmi_reranker(pmi, weight=1.0)
    blended = reranker(scores, np.array([1], dtype=np.int64), None)
    assert np.isneginf(blended[0])


def _tiny_vocab(land_flags, **over):
    n = len(land_flags)
    land = np.array(land_flags, dtype=bool)
    fields = dict(
        oracle_ids=[f"id{i}" for i in range(n)],
        index={f"id{i}": i for i in range(n)},
        features=np.zeros((n, 1), dtype=np.float32),
        cmc=np.zeros(n, dtype=np.float32),
        identity_bits=np.zeros(n, dtype=np.uint8),
        land=land,
        basic=land.copy(),
        copy_cap=np.zeros(n, dtype=np.int32),
        roles=np.zeros((n, 8), dtype=bool),
        eff_land=land.astype(np.float32),
        cheap_dr=np.zeros(n, dtype=bool),
        src_w=np.zeros((n, 5), dtype=np.float32),
        pips=np.zeros((n, 5), dtype=np.int8),
    )
    fields.update(over)
    return Vocab(**fields)


def test_structure_reward_peaks_at_land_target():
    vocab = _tiny_vocab([True] * 37 + [False] * 63)
    on_target = np.arange(100, dtype=np.int64)  # exactly 37% lands
    assert structure_reward(vocab, COMMANDER, on_target) == pytest.approx(0.0)
    all_spells = np.arange(37, 100, dtype=np.int64)
    assert structure_reward(vocab, COMMANDER, all_spells) == pytest.approx(-0.37)


def test_step_reward_adds_terminal_structure():
    vocab = _tiny_vocab([False, False])
    corpus = sets(*([[0, 1]] * 25))
    pmi = build_pmi(corpus, vocab_size=2, min_count=2)
    partial = np.array([0], dtype=np.int64)
    mid = step_reward(pmi, vocab, COMMANDER, partial, action=1, done=False)
    end = step_reward(pmi, vocab, COMMANDER, partial, action=1, done=True)
    assert mid == pytest.approx(pmi.ppmi(0, 1))
    # terminal adds the (negative) land-fraction distance: no lands at all
    assert end == pytest.approx(mid - 0.37)
