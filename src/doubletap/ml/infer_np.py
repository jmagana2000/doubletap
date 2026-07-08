"""Torch-free inference: the TwoTowerQ forward pass in plain numpy.

Checkpoints saved during training carry a sibling `.npz` with the raw weight
arrays; this module loads that and scores candidates identically to the torch
model, so recommend/complete need no torch at runtime. Training still uses
torch — see ml/model.py."""

import json
from pathlib import Path

import numpy as np

from .data import Vocab


class NpTwoTowerQ:
    """Numpy twin of TwoTowerQ's inference path. Same weights, same math:
    deck embedding is a sum, towers are Linear→ReLU→Linear, Q is a dot."""

    def __init__(self, weights: dict[str, np.ndarray], card_features: np.ndarray):
        self.emb = weights["card_emb.weight"]
        self.st = [
            (weights["state_tower.0.weight"], weights["state_tower.0.bias"]),
            (weights["state_tower.2.weight"], weights["state_tower.2.bias"]),
        ]
        self.at = [
            (weights["action_tower.0.weight"], weights["action_tower.0.bias"]),
            (weights["action_tower.2.weight"], weights["action_tower.2.bias"]),
        ]
        self.card_features = card_features

    @staticmethod
    def _mlp(layers, x):
        (w0, b0), (w1, b1) = layers
        return np.maximum(x @ w0.T + b0, 0.0) @ w1.T + b1

    def score(
        self,
        partial_idxs: np.ndarray,
        commander_idx: int | None,
        state_feats: np.ndarray,
        pool: np.ndarray,
    ) -> np.ndarray:
        deck_emb = (
            self.emb[partial_idxs].sum(axis=0)
            if partial_idxs.size
            else np.zeros(self.emb.shape[1], dtype=self.emb.dtype)
        )
        cmd_emb = (
            self.emb[commander_idx]
            if commander_idx is not None
            else np.zeros_like(deck_emb)
        )
        state = self._mlp(self.st, np.concatenate([deck_emb, cmd_emb, state_feats]))
        actions = self._mlp(
            self.at,
            np.concatenate([self.emb[pool], self.card_features[pool]], axis=1),
        )
        return (actions @ state).astype(np.float32)


def save_np_checkpoint(
    path: Path, state_dict: dict, oracle_ids: list, format_name: str, algo: str
) -> None:
    """Write weights as an .npz next to the torch checkpoint."""
    arrays = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
    arrays["__meta__"] = np.frombuffer(
        json.dumps(
            {"oracle_ids": oracle_ids, "format": format_name, "algo": algo}
        ).encode(),
        dtype=np.uint8,
    )
    np.savez_compressed(path, **arrays)


def load_np_checkpoint(path: Path, vocab: Vocab) -> tuple[NpTwoTowerQ, dict]:
    data = np.load(path)
    meta = json.loads(bytes(data["__meta__"]).decode())
    if meta["oracle_ids"] != vocab.oracle_ids:
        raise ValueError(
            f"{path} was trained on a different card vocab; re-train after cards sync"
        )
    weights = {k: data[k] for k in data.files if k != "__meta__"}
    return NpTwoTowerQ(weights, vocab.features), meta
