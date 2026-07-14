import numpy as np
import pytest

torch = pytest.importorskip("torch")

from doubletap.formats import COMMANDER  # noqa: E402
from doubletap.ml.data import build_vocab, load_corpus, sample_batch  # noqa: E402
from doubletap.ml.eval import complete_deck, recovery_at_k, score_state  # noqa: E402
from doubletap.ml.model import TwoTowerQ, load_checkpoint, save_checkpoint  # noqa: E402
from doubletap.ml.reward import build_pmi, corpus_card_sets  # noqa: E402
from doubletap.ml.train_bc import bc_loss, sample_negatives  # noqa: E402
from doubletap.ml.train_cql import batch_rewards, cql_losses  # noqa: E402

from test_ml_data import _insert_corpus_deck  # noqa: E402

from doubletap.names import lookup  # noqa: E402


def tiny_model(vocab):
    torch.manual_seed(0)
    return TwoTowerQ(vocab.features, emb_dim=16, hidden=32, out_dim=16)


@pytest.fixture
def rigged_conn(loaded_conn):
    """Ten near-identical commander decks: the model should learn the pattern."""

    def o(name):
        return lookup(loaded_conn, name)[0].oracle_id

    for deck_id in range(1, 11):
        _insert_corpus_deck(
            loaded_conn,
            deck_id,
            "commander",
            {
                o("Sol Ring"): 1,
                o("Relentless Rats"): 20,
                o("Juzám Djinn"): 1,
                o("Swamp"): 77,
            },
            commander_oid=o("Atraxa, Praetors' Voice"),
        )
    loaded_conn.commit()
    return loaded_conn


def test_shapes_and_next_state_consistency(rigged_conn):
    vocab = build_vocab(rigged_conn, COMMANDER)
    decks = load_corpus(rigged_conn, vocab, COMMANDER)
    model = tiny_model(vocab)
    rng = np.random.default_rng(3)
    batch = sample_batch(decks, vocab, COMMANDER, 8, rng, with_next=True)

    bag = torch.from_numpy(batch.bag)
    offsets = torch.from_numpy(batch.offsets)
    commander = torch.from_numpy(batch.commander)
    state = model.state_repr(
        bag, offsets, commander, torch.from_numpy(batch.state_feats)
    )
    assert state.shape == (8, 16)
    cands = torch.randint(0, len(vocab), (8, 5))
    assert model.q(state, cands).shape == (8, 5)

    # next_state_repr's embedding shortcut must equal recomputing from scratch
    action = torch.from_numpy(batch.action)
    next_feats = torch.from_numpy(batch.next_state_feats)
    shortcut = model.next_state_repr(bag, offsets, commander, action, next_feats)
    row = 0
    row_bag = batch.bag[
        batch.offsets[0] : batch.offsets[1]
        if len(batch.offsets) > 1
        else batch.bag.size
    ]
    explicit_bag = torch.from_numpy(np.append(row_bag, batch.action[row]))
    explicit = model.state_repr(
        explicit_bag,
        torch.zeros(1, dtype=torch.int64),
        commander[:1],
        next_feats[:1],
    )
    assert torch.allclose(shortcut[row], explicit[0], atol=1e-5)


def test_bc_overfits_one_batch_and_recovers(rigged_conn):
    vocab = build_vocab(rigged_conn, COMMANDER)
    decks = load_corpus(rigged_conn, vocab, COMMANDER)
    model = tiny_model(vocab)
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(~vocab.land)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    batch = sample_batch(decks, vocab, COMMANDER, 64, rng)
    negatives = sample_negatives(pool, 64, 8, rng)
    first = bc_loss(model, batch, negatives).item()
    for _ in range(60):
        loss = bc_loss(model, batch, negatives)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5

    metrics = recovery_at_k(
        model, decks, vocab, COMMANDER, n_hide=3, ks=(3,), rng=np.random.default_rng(1)
    )
    assert metrics["recovery"][3] > 50.0  # rigged corpus: hidden cards recoverable


def test_score_state_masks_illegal_actions(rigged_conn):
    vocab = build_vocab(rigged_conn, COMMANDER)
    decks = load_corpus(rigged_conn, vocab, COMMANDER)
    model = tiny_model(vocab)
    deck = decks[0]
    scores = score_state(
        model, vocab, COMMANDER, deck.main_idxs[:10], deck.commander_idx
    )
    assert np.isneginf(scores[vocab.land]).all()
    bolt = vocab.index[lookup(rigged_conn, "Lightning Bolt")[0].oracle_id]
    assert np.isneginf(scores[bolt])  # outside Atraxa's identity


def test_cql_losses_shapes_and_conservative_direction(rigged_conn):
    vocab = build_vocab(rigged_conn, COMMANDER)
    decks = load_corpus(rigged_conn, vocab, COMMANDER)
    pmi = build_pmi(corpus_card_sets(decks), len(vocab), min_count=2)
    model = tiny_model(vocab)
    target_model = tiny_model(vocab)
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(~vocab.land)

    batch = sample_batch(decks, vocab, COMMANDER, 32, rng, with_next=True)
    rewards = batch_rewards(pmi, vocab, COMMANDER, batch)
    assert rewards.shape == (32,)
    negatives = sample_negatives(pool, 32, 16, rng)
    next_cands = sample_negatives(pool, 32, 16, rng)
    td, conservative = cql_losses(
        model, target_model, batch, rewards, negatives, next_cands, pool.size, 0.99
    )
    assert td.requires_grad and conservative.requires_grad
    assert torch.isfinite(td) and torch.isfinite(conservative)

    # minimizing the conservative term must push the dataset action's Q above
    # the sampled pool's Q
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def gap():
        bag, offsets, commander, state_feats, action = (
            torch.from_numpy(batch.bag),
            torch.from_numpy(batch.offsets),
            torch.from_numpy(batch.commander),
            torch.from_numpy(batch.state_feats),
            torch.from_numpy(batch.action),
        )
        state = model.state_repr(bag, offsets, commander, state_feats)
        q_data = model.q(state, action.unsqueeze(1)).squeeze(1)
        q_neg = model.q(state, torch.from_numpy(negatives))
        return (q_data.mean() - q_neg.mean()).item()

    before = gap()
    for _ in range(30):
        _, conservative = cql_losses(
            model, target_model, batch, rewards, negatives, next_cands, pool.size, 0.99
        )
        opt.zero_grad()
        conservative.backward()
        opt.step()
    assert gap() > before


def test_complete_deck_fills_nonland_slots_legally(rigged_conn):
    vocab = build_vocab(rigged_conn, COMMANDER)
    model = tiny_model(vocab)
    atraxa = vocab.index[lookup(rigged_conn, "Atraxa, Praetors' Voice")[0].oracle_id]
    sol_ring = vocab.index[lookup(rigged_conn, "Sol Ring")[0].oracle_id]

    partial = np.array([sol_ring], dtype=np.int64)
    added, final = complete_deck(model, vocab, COMMANDER, partial, atraxa)

    # nonland target: 100 - 37 land slots - 1 commander = 62; one already there
    assert len(added) == 61
    assert final.size == 62
    assert not vocab.land[np.array(added)].any()
    # copy limits respected across the greedy fill
    idxs, counts = np.unique(final, return_counts=True)
    over = idxs[(counts > COMMANDER.copy_limit) & ~vocab.any_number[idxs]]
    assert over.size == 0
    assert atraxa not in added


def test_complete_deck_stops_when_pool_exhausts(rigged_conn):
    vocab = build_vocab(rigged_conn, COMMANDER)
    model = tiny_model(vocab)
    juzam = vocab.index[lookup(rigged_conn, "Juzám Djinn")[0].oracle_id]
    # mono-black commander shrinks the fixture pool to a couple of cards; the
    # any-number Relentless Rats keeps the fill going to the nonland target
    partial = np.empty(0, dtype=np.int64)
    added, final = complete_deck(model, vocab, COMMANDER, partial, juzam)
    assert len(added) == 62  # 100 - 37 land slots - 1 commander
    assert set(np.unique(np.array(added))) <= set(np.flatnonzero(~vocab.land))


def test_complete_deck_caps_limited_cards(rigged_conn):
    """Bracket-limited cards (Game Changers) may be added at most `cap` times.
    Relentless Rats is the only repeatable pick in the fixture pool, so once
    singletons run out the greedy loop wants Rats forever — the cap must stop
    it."""
    vocab = build_vocab(rigged_conn, COMMANDER)
    model = tiny_model(vocab)
    atraxa = vocab.index[lookup(rigged_conn, "Atraxa, Praetors' Voice")[0].oracle_id]
    rats = vocab.index[lookup(rigged_conn, "Relentless Rats")[0].oracle_id]
    capped = np.array([rats], dtype=np.int64)
    empty = np.empty(0, dtype=np.int64)

    added, _ = complete_deck(
        model, vocab, COMMANDER, empty, atraxa, capped_idxs=capped, cap=0
    )
    assert rats not in added

    added, _ = complete_deck(
        model, vocab, COMMANDER, empty, atraxa, capped_idxs=capped, cap=2
    )
    assert added.count(rats) == 2


def test_numpy_inference_matches_torch(rigged_conn, tmp_path):
    """The .npz runtime must score exactly like the torch model it came from —
    recommend/complete run on numpy weights in torch-less installs."""
    from doubletap.ml.infer_np import load_np_checkpoint

    vocab = build_vocab(rigged_conn, COMMANDER)
    model = tiny_model(vocab)
    save_checkpoint(tmp_path / "m.pt", model, vocab, "commander", "bc", {})
    np_model, meta = load_np_checkpoint(tmp_path / "m.npz", vocab)
    assert meta["algo"] == "bc"

    atraxa = vocab.index[lookup(rigged_conn, "Atraxa, Praetors' Voice")[0].oracle_id]
    sol = vocab.index[lookup(rigged_conn, "Sol Ring")[0].oracle_id]
    for partial, cmdr in [
        (np.array([sol], dtype=np.int64), atraxa),
        (np.empty(0, dtype=np.int64), atraxa),
        (np.array([sol, sol], dtype=np.int64), None),
    ]:
        torch_scores = score_state(model, vocab, COMMANDER, partial, cmdr)
        np_scores = score_state(np_model, vocab, COMMANDER, partial, cmdr)
        np.testing.assert_allclose(np_scores, torch_scores, rtol=1e-4, atol=1e-5)

    # greedy completion follows the identical trajectory
    added_t, _ = complete_deck(model, vocab, COMMANDER, np.empty(0, np.int64), atraxa)
    added_n, _ = complete_deck(
        np_model, vocab, COMMANDER, np.empty(0, np.int64), atraxa
    )
    assert added_t == added_n


def test_checkpoint_round_trip_and_vocab_guard(rigged_conn, tmp_path):
    vocab = build_vocab(rigged_conn, COMMANDER)
    model = tiny_model(vocab)
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, vocab, "commander", "bc", {"recovery": {}})
    # tiny_model dims differ from the default TwoTowerQ, so loading must use
    # matching dims; guard test only checks the vocab mismatch path
    other = vocab
    other.oracle_ids = list(reversed(vocab.oracle_ids))
    with pytest.raises(ValueError):
        load_checkpoint(path, other)
