import copy
import math
import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..db import data_home
from ..formats import FormatConfig
from .data import build_vocab, load_corpus, sample_batch
from .eval import recovery_at_k
from .model import TwoTowerQ, load_checkpoint, save_checkpoint
from .reward import PMIModel, step_reward
from .train_bc import batch_to_tensors, sample_negatives, split_corpus


def batch_rewards(pmi, vocab, fmt, batch, shaper=None) -> torch.Tensor:
    bounds = np.append(batch.offsets, batch.bag.size)
    rewards = []
    for i in range(len(batch.action)):
        partial = batch.bag[bounds[i] : bounds[i + 1]]
        r = step_reward(
            pmi, vocab, fmt, partial, int(batch.action[i]), bool(batch.done[i])
        )
        if shaper is not None:
            cmdr = int(batch.commander[i])
            r += shaper.delta(
                partial, int(batch.action[i]), cmdr if cmdr >= 0 else None
            )
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32)


def cql_losses(
    model: TwoTowerQ,
    target_model: TwoTowerQ,
    batch,
    rewards: torch.Tensor,
    negatives: np.ndarray,
    next_candidates: np.ndarray,
    pool_size: int,
    gamma: float,
):
    bag, offsets, commander, state_feats, action = batch_to_tensors(batch)
    state = model.state_repr(bag, offsets, commander, state_feats)
    q_data = model.q(state, action.unsqueeze(1)).squeeze(1)

    # conservative term: sampled logsumexp over the legal pool, first-order
    # corrected for subsampling (log(N/K)), minus the dataset action's Q
    candidates = torch.cat([action.unsqueeze(1), torch.from_numpy(negatives)], dim=1)
    q_sampled = model.q(state, candidates)
    correction = math.log(pool_size / candidates.shape[1])
    conservative = (torch.logsumexp(q_sampled, dim=1) + correction - q_data).mean()

    with torch.no_grad():
        next_state = target_model.next_state_repr(
            bag, offsets, commander, action, torch.from_numpy(batch.next_state_feats)
        )
        q_next = (
            target_model.q(next_state, torch.from_numpy(next_candidates))
            .max(dim=1)
            .values
        )
        target = rewards + gamma * (1.0 - torch.from_numpy(batch.done)) * q_next
    td = F.smooth_l1_loss(q_data, target)
    return td, conservative


def train_cql(
    conn: sqlite3.Connection,
    fmt: FormatConfig,
    pmi: PMIModel,
    steps: int = 1500,
    batch_size: int = 256,
    k_negatives: int = 512,
    lr: float = 3e-4,
    alpha: float = 1.0,
    gamma: float = 0.99,
    tau: float = 0.005,
    seed: int = 0,
    init_from: Path | None = None,
    shaper=None,
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
    pool = np.flatnonzero(~vocab.land)

    if init_from:
        model, _ = load_checkpoint(init_from, vocab)
        model.train()
        log(f"initialized from {init_from}")
    else:
        model = TwoTowerQ(vocab.features)
    target_model = copy.deepcopy(model)
    target_model.eval()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        batch = sample_batch(train, vocab, fmt, batch_size, rng, with_next=True)
        rewards = batch_rewards(pmi, vocab, fmt, batch, shaper)
        negatives = sample_negatives(pool, batch_size, k_negatives, rng)
        next_candidates = sample_negatives(pool, batch_size, k_negatives, rng)
        td, conservative = cql_losses(
            model,
            target_model,
            batch,
            rewards,
            negatives,
            next_candidates,
            pool.size,
            gamma,
        )
        loss = td + alpha * conservative
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, tp in zip(model.parameters(), target_model.parameters()):
                tp.lerp_(p, tau)
        if step % 100 == 0 or step == steps:
            log(
                f"step {step}/{steps} td {td.item():.4f} conservative {conservative.item():.4f}"
            )

    metrics = recovery_at_k(model, holdout, vocab, fmt, rng=np.random.default_rng(seed))
    metrics["train_decks"] = len(train)
    log(f"holdout recovery@k: {metrics['recovery']} over {metrics['decks']} decks")

    out = data_home() / "models"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cql_{fmt.name}.pt"
    save_checkpoint(path, model, vocab, fmt.name, "cql", metrics)
    return path
