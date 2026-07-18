import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..db import data_home
from ..formats import FormatConfig
from .data import CorpusDeck, build_vocab, load_corpus, sample_batch, state_dim
from .eval import recovery_at_k
from .model import TwoTowerQ, save_checkpoint


def _dup_clusters(decks: list[CorpusDeck]) -> np.ndarray:
    """Cluster id per deck: near-duplicates (card-set Jaccard >= 0.8) share a
    cluster via union-find. Comparison is bucketed by commander (copied
    commander decks keep their commander) with one bucket for commander-less
    formats — full pairwise there is fine at current corpus sizes.
    ponytail: near-dupes with different commanders slip through; rare."""
    parent = list(range(len(decks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    buckets: dict[int | None, list[int]] = {}
    for i, d in enumerate(decks):
        buckets.setdefault(d.commander_idx, []).append(i)
    for members in buckets.values():
        if len(members) < 2:
            continue
        sets = [set(map(int, np.unique(decks[i].main_idxs))) for i in members]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                inter = len(sets[a] & sets[b])
                if inter / (len(sets[a]) + len(sets[b]) - inter) >= 0.8:
                    ra, rb = find(members[a]), find(members[b])
                    if ra != rb:
                        parent[rb] = ra
    return np.array([find(i) for i in range(len(decks))])


def split_corpus(decks: list[CorpusDeck], holdout_fraction: float = 0.1, seed: int = 0):
    """Canonical train/holdout split: near-duplicate clusters stay on one
    side (no leakage), and membership is derived from a FIXED rng — never
    the training seed — so every model and seed sweep shares one holdout
    (the 2026-07-17 leakage check caught both failure modes)."""
    del seed  # kept for call-site compatibility; membership must not vary
    clusters = _dup_clusters(decks)
    ids = list(dict.fromkeys(clusters.tolist()))  # cluster ids, stable order
    order = np.random.default_rng(0).permutation(len(ids))
    n_holdout = max(1, int(len(decks) * holdout_fraction))
    holdout_clusters, count = set(), 0
    for j in order:
        if count >= n_holdout or len(holdout_clusters) == len(ids) - 1:
            break  # train must keep at least one cluster
        cid = ids[j]
        holdout_clusters.add(cid)
        count += int((clusters == cid).sum())
    holdout = [d for d, c in zip(decks, clusters) if c in holdout_clusters]
    train = [d for d, c in zip(decks, clusters) if c not in holdout_clusters]
    if not holdout or not train:
        raise RuntimeError(
            "corpus too homogeneous to split: near-duplicate clustering left"
            f" {len(train)} train / {len(holdout)} holdout decks — crawl more"
            " varied decks before training"
        )
    return train, holdout


def batch_to_tensors(batch):
    return (
        torch.from_numpy(batch.bag),
        torch.from_numpy(batch.offsets),
        torch.from_numpy(batch.commander),
        torch.from_numpy(batch.state_feats),
        torch.from_numpy(batch.action),
    )


def sample_negatives(
    pool: np.ndarray, batch_size: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    return pool[rng.integers(pool.size, size=(batch_size, k))]


def bc_loss(
    model: TwoTowerQ, batch, negatives: np.ndarray, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Sampled-softmax cross-entropy: the human's pick vs uniform negatives.
    Optional per-example weights turn this into advantage-weighted
    regression (AWR): imitate the functionally better picks harder."""
    bag, offsets, commander, state_feats, action = batch_to_tensors(batch)
    state = model.state_repr(bag, offsets, commander, state_feats)
    candidates = torch.cat(
        [action.unsqueeze(1), torch.from_numpy(negatives)], dim=1
    )  # target at column 0
    logits = model.q(state, candidates)
    targets = torch.zeros(len(batch.action), dtype=torch.int64)
    if weights is None:
        return F.cross_entropy(logits, targets)
    per_example = F.cross_entropy(logits, targets, reduction="none")
    return (per_example * weights).mean()


def awr_weights(shaper, batch, clip: float = 5.0) -> torch.Tensor:
    """AWR advantages from goldfish deltas: standardize the batch's deltas,
    exponentiate, clip. Picks that improve how the deck goldfishes get
    imitated harder; picks that hurt it get discounted."""
    bounds = np.append(batch.offsets, batch.bag.size)
    deltas = np.array(
        [
            shaper.delta(
                batch.bag[bounds[i] : bounds[i + 1]],
                int(batch.action[i]),
                int(batch.commander[i]) if batch.commander[i] >= 0 else None,
            )
            for i in range(len(batch.action))
        ],
        dtype=np.float32,
    )
    std = deltas.std() or 1.0
    weights = np.exp((deltas - deltas.mean()) / std).clip(1.0 / clip, clip)
    return torch.from_numpy(weights.astype(np.float32))


def train_awr(
    conn: sqlite3.Connection,
    fmt: FormatConfig,
    shaper,
    steps: int = 1500,
    batch_size: int = 256,
    k_negatives: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    log=print,
) -> Path:
    """Advantage-weighted regression: BC re-weighted by goldfish deltas.
    Stays inside the data distribution (no TD, no conservatism knob) while
    leaning toward functionally better picks — the Tier-1 training
    experiment from the RL-models review (docs/goldfish-sim-design.md)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = build_vocab(conn, fmt)
    decks = load_corpus(conn, vocab, fmt)
    if len(decks) < 20:
        raise RuntimeError(
            f"only {len(decks)} parsed {fmt.name} decks; crawl more first"
        )
    train, holdout = split_corpus(decks, seed=seed)
    pool = np.flatnonzero(~vocab.land)

    model = TwoTowerQ(vocab.features, state_dim=state_dim(fmt))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        batch = sample_batch(train, vocab, fmt, batch_size, rng)
        weights = awr_weights(shaper, batch)
        loss = bc_loss(
            model, batch, sample_negatives(pool, batch_size, k_negatives, rng), weights
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps:
            log(f"step {step}/{steps} loss {loss.item():.4f}")

    metrics = recovery_at_k(model, holdout, vocab, fmt, rng=np.random.default_rng(seed))
    metrics["train_decks"] = len(train)
    log(f"holdout recovery@k: {metrics['recovery']} over {metrics['decks']} decks")

    out = data_home() / "models"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"awr_{fmt.name}.pt"
    save_checkpoint(path, model, vocab, fmt.name, "awr", metrics)
    return path


def train_bc(
    conn: sqlite3.Connection,
    fmt: FormatConfig,
    steps: int = 1500,
    batch_size: int = 256,
    k_negatives: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    log=print,
) -> Path:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = build_vocab(conn, fmt)
    decks = load_corpus(conn, vocab, fmt)
    if len(decks) < 20:
        raise RuntimeError(
            f"only {len(decks)} parsed {fmt.name} decks; crawl more first"
        )
    train, holdout = split_corpus(decks, seed=seed)
    pool = np.flatnonzero(~vocab.land)  # negatives: any legal nonland card

    model = TwoTowerQ(vocab.features, state_dim=state_dim(fmt))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        batch = sample_batch(train, vocab, fmt, batch_size, rng)
        loss = bc_loss(
            model, batch, sample_negatives(pool, batch_size, k_negatives, rng)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps:
            log(f"step {step}/{steps} loss {loss.item():.4f}")

    metrics = recovery_at_k(model, holdout, vocab, fmt, rng=np.random.default_rng(seed))
    metrics["train_decks"] = len(train)
    log(f"holdout recovery@k: {metrics['recovery']} over {metrics['decks']} decks")

    out = data_home() / "models"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"bc_{fmt.name}.pt"
    save_checkpoint(path, model, vocab, fmt.name, "bc", metrics)
    return path
