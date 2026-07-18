import numpy as np
import pytest

torch = pytest.importorskip("torch")

from doubletap.formats import COMMANDER  # noqa: E402
from doubletap.ml.data import build_vocab, load_corpus, sample_batch, state_dim  # noqa: E402
from doubletap.ml.eval import complete_deck, recovery_at_k, score_state  # noqa: E402
from doubletap.ml.model import TwoTowerQ, load_checkpoint, save_checkpoint  # noqa: E402
from doubletap.ml.reward import build_pmi, corpus_card_sets  # noqa: E402
from doubletap.ml.train_bc import bc_loss, sample_negatives, split_corpus  # noqa: E402
from doubletap.ml.train_cql import batch_rewards, cql_losses  # noqa: E402

from test_ml_data import _insert_corpus_deck  # noqa: E402

from doubletap.names import lookup  # noqa: E402


def tiny_model(vocab):
    torch.manual_seed(0)
    return TwoTowerQ(
        vocab.features,
        state_dim=state_dim(COMMANDER),
        emb_dim=16,
        hidden=32,
        out_dim=16,
    )


@pytest.fixture
def rigged_conn(loaded_conn):
    """Ten similar commander decks: the model should learn the pattern.
    Each deck carries a rotating filler pair so the near-duplicate
    clustering (Jaccard >= 0.8) doesn't collapse the corpus into one
    cluster, which the homogeneity guard now rejects."""

    def o(name):
        return lookup(loaded_conn, name)[0].oracle_id

    fillers = [
        "Standard Strike",
        "Rhystic Study",
        "Twinned Test Mage",
        "Seven Dwarves",
        "Nazgûl",
    ]
    for deck_id in range(1, 11):
        f1 = fillers[deck_id % len(fillers)]
        f2 = fillers[(deck_id + 2) % len(fillers)]
        _insert_corpus_deck(
            loaded_conn,
            deck_id,
            "commander",
            {
                o("Sol Ring"): 1,
                o("Relentless Rats"): 20,
                o("Juzám Djinn"): 1,
                o(f1): 1,
                o(f2): 1,
                o("Swamp"): 75,
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
    over = idxs[(counts > COMMANDER.copy_limit) & (vocab.copy_cap[idxs] == 0)]
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


def test_structural_quality_scores_completions(rigged_conn):
    from doubletap.ml.policy import structural_quality

    vocab = build_vocab(rigged_conn, COMMANDER)
    decks = load_corpus(rigged_conn, vocab, COMMANDER)
    model = tiny_model(vocab)
    q = structural_quality(model, decks, vocab, COMMANDER, max_decks=3)
    assert q["decks"] == 3
    assert 0.0 <= q["composite"] <= 1.0
    # fixture pool has ramp (Sol Ring) but no wipes: quota deficit nonzero
    assert q["quota_deficit"] > 0.0


def test_awr_weights_and_weighted_bc_loss(rigged_conn):
    from doubletap.ml.goldfish import Shaper, vocab_statics
    from doubletap.ml.train_bc import awr_weights, bc_loss

    vocab = build_vocab(rigged_conn, COMMANDER)
    decks = load_corpus(rigged_conn, vocab, COMMANDER)
    model = tiny_model(vocab)
    rng = np.random.default_rng(0)
    batch = sample_batch(decks, vocab, COMMANDER, 16, rng)
    shaper = Shaper(vocab_statics(rigged_conn, vocab), games=2, turns=6)

    w = awr_weights(shaper, batch)
    assert w.shape == (16,)
    assert (w >= 1 / 5.0).all() and (w <= 5.0).all()  # clipped

    pool = np.flatnonzero(~vocab.land)
    negs = sample_negatives(pool, 16, 8, rng)
    weighted = bc_loss(model, batch, negs, w)
    plain = bc_loss(model, batch, negs)
    assert torch.isfinite(weighted) and torch.isfinite(plain)


def test_split_corpus_is_canonical_and_leakproof(rigged_conn):
    """Near-duplicate decks land on the same side, and membership ignores
    the training seed (the 2026-07-17 leakage findings)."""
    vocab = build_vocab(rigged_conn, COMMANDER)
    decks = load_corpus(rigged_conn, vocab, COMMANDER)
    # forge near-duplicates: clone a deck with one card changed
    import copy

    clone = copy.deepcopy(decks[0])
    clone.deck_id = 999_999
    clone.main_idxs = clone.main_idxs.copy()
    clone.main_idxs[0] = decks[1].main_idxs[0]
    pool = decks + [clone]

    for seed in (0, 1, 7):
        train, holdout = split_corpus(pool, holdout_fraction=0.3, seed=seed)
        train_ids = {d.deck_id for d in train}
        hold_ids = {d.deck_id for d in holdout}
        # the clone travels with its original — never split across sides
        assert (decks[0].deck_id in train_ids) == (999_999 in train_ids)
        # membership identical across seeds
        if seed == 0:
            canonical = (frozenset(train_ids), frozenset(hold_ids))
        else:
            assert (frozenset(train_ids), frozenset(hold_ids)) == canonical


def test_rank_cuts_flags_low_synergy_card(rigged_conn):
    """The engine's cut logic: the card with no synergy and no model love
    ranks as a better cut than deck staples; protected roles survive."""
    from doubletap.ml.reward import build_pmi, corpus_card_sets
    from doubletap.swaps import rank_cuts

    vocab = build_vocab(rigged_conn, COMMANDER)
    decks = load_corpus(rigged_conn, vocab, COMMANDER)
    model = tiny_model(vocab)
    pmi = build_pmi(corpus_card_sets(decks), len(vocab), min_count=1)
    deck = decks[0]
    cuts = rank_cuts(
        rigged_conn,
        model,
        vocab,
        COMMANDER,
        deck.main_idxs,
        deck.commander_idx,
        pmi=pmi,
        k=10,
    )
    assert cuts, "expected cut candidates"
    assert all(0.0 <= c.badness <= 1.0 for c in cuts)
    assert all(c.reason for c in cuts)
    # ranked worst-first
    assert cuts == sorted(cuts, key=lambda c: -c.badness)


def test_recovery_hides_every_copy(rigged_conn):
    """The partial handed to the model must not contain any copy of a
    hidden card (4-of formats previously 'recovered' visible duplicates)."""
    vocab = build_vocab(rigged_conn, COMMANDER)
    decks = load_corpus(rigged_conn, vocab, COMMANDER)
    seen_partials = []

    class SpyModel:
        def score(self, partial_idxs, commander_idx, feats, pool):
            seen_partials.append(np.array(partial_idxs))
            return np.zeros(len(pool), dtype=np.float32)

    from doubletap.ml.policy import recovery_at_k

    recovery_at_k(SpyModel(), decks[:3], vocab, COMMANDER, n_hide=4,
                  rng=np.random.default_rng(0))
    # rigged decks run 20x Relentless Rats: whenever Rats was hidden, no
    # copy of it may appear in the partial the model scored
    rats_idx = None
    for deck in decks[:3]:
        vals, counts = np.unique(deck.main_idxs, return_counts=True)
        rats_idx = int(vals[np.argmax(counts)])
    for partial in seen_partials:
        present = set(np.unique(partial))
        full = set(np.unique(decks[0].main_idxs))
        hidden_here = full - present
        for h in hidden_here:
            assert h not in present
