from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import STATE_DIM, Vocab


class TwoTowerQ(nn.Module):
    """Q(state, candidate) scorer. The state tower aggregates the partial deck
    (sum of card embeddings, EmbeddingBag layout) plus commander embedding and
    structural features; the action tower combines the candidate's learned
    embedding with its structured Scryfall features. Q is a dot product, so one
    state forward scores the whole candidate pool.

    The same network trains as behavior cloning (logits) or CQL (Q-values)."""

    def __init__(
        self,
        card_features: np.ndarray,
        emb_dim: int = 64,
        hidden: int = 256,
        out_dim: int = 128,
    ):
        super().__init__()
        n_cards, feature_dim = card_features.shape
        self.card_emb = nn.EmbeddingBag(n_cards, emb_dim, mode="sum")
        self.register_buffer("card_features", torch.from_numpy(card_features))
        self.state_tower = nn.Sequential(
            nn.Linear(emb_dim * 2 + STATE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.action_tower = nn.Sequential(
            nn.Linear(emb_dim + feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def _tower(
        self, deck_emb: torch.Tensor, commander: torch.Tensor, state_feats: torch.Tensor
    ) -> torch.Tensor:
        has_commander = (commander >= 0).unsqueeze(1).float()
        cmd_emb = self.card_emb.weight[commander.clamp(min=0)] * has_commander
        return self.state_tower(torch.cat([deck_emb, cmd_emb, state_feats], dim=1))

    def state_repr(
        self,
        bag: torch.Tensor,
        offsets: torch.Tensor,
        commander: torch.Tensor,
        state_feats: torch.Tensor,
    ) -> torch.Tensor:
        return self._tower(self.card_emb(bag, offsets), commander, state_feats)

    def next_state_repr(
        self,
        bag: torch.Tensor,
        offsets: torch.Tensor,
        commander: torch.Tensor,
        action: torch.Tensor,
        next_state_feats: torch.Tensor,
    ) -> torch.Tensor:
        # sum-mode EmbeddingBag is linear, so adding the action embedding to the
        # partial-deck sum gives the next state's deck embedding for free
        deck_emb = self.card_emb(bag, offsets) + self.card_emb.weight[action]
        return self._tower(deck_emb, commander, next_state_feats)

    def action_repr(self, candidates: torch.Tensor) -> torch.Tensor:
        emb = self.card_emb.weight[candidates]
        feats = self.card_features[candidates]
        return self.action_tower(torch.cat([emb, feats], dim=-1))

    def q(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """state: (B, out); candidates: (B, K) -> (B, K) Q-values."""
        return torch.einsum("bd,bkd->bk", state, self.action_repr(candidates))

    def score_pool(self, state: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
        """state: (out,) single state; pool: (n,) candidate indices -> (n,) scores."""
        return self.action_repr(pool) @ state

    def score(
        self,
        partial_idxs: np.ndarray,
        commander_idx: int | None,
        state_feats: np.ndarray,
        pool: np.ndarray,
    ) -> np.ndarray:
        """Numpy-in/numpy-out scorer — the interface ml.policy drives, shared
        with the torch-free NpTwoTowerQ."""
        self.eval()
        with torch.no_grad():
            bag = torch.from_numpy(partial_idxs)
            offsets = torch.zeros(1, dtype=torch.int64)
            commander = torch.tensor(
                [commander_idx if commander_idx is not None else -1]
            )
            feats = torch.from_numpy(state_feats).unsqueeze(0)
            state = self.state_repr(bag, offsets, commander, feats)[0]
            return self.score_pool(state, torch.from_numpy(pool)).numpy()


def save_checkpoint(
    path: Path,
    model: TwoTowerQ,
    vocab: Vocab,
    format_name: str,
    algo: str,
    metrics: dict,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "oracle_ids": vocab.oracle_ids,
            "format": format_name,
            "algo": algo,
            "metrics": metrics,
        },
        path,
    )
    # sibling .npz: raw weight arrays so recommend/complete run without torch
    from .infer_np import save_np_checkpoint

    save_np_checkpoint(
        Path(path).with_suffix(".npz"),
        model.state_dict(),
        vocab.oracle_ids,
        format_name,
        algo,
    )


def load_checkpoint(path: Path, vocab: Vocab) -> tuple[TwoTowerQ, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if ckpt["oracle_ids"] != vocab.oracle_ids:
        raise ValueError(
            f"{path} was trained on a different card vocab; re-train after cards sync"
        )
    model = TwoTowerQ(vocab.features)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
