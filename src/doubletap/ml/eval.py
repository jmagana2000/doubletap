"""Inference and evaluation entry points (torch backend).

The actual masking/greedy/recovery logic lives in ml/policy.py, which is
torch-free and shared with the numpy runtime (ml/infer_np.py). This module
exists so training-time code and tests keep their historical imports."""

from .policy import complete_deck, recovery_at_k, score_state, structural_quality

__all__ = ["score_state", "complete_deck", "recovery_at_k", "structural_quality"]
