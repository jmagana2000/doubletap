# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DoubleTap is an MTG deck-building CLI (`doubletap`, Typer app in `src/doubletap/cli.py`): Scryfall card sync, deck import (CSV/photo/text), Commander/Modern/Standard rules validation, mana-base and goldfish-simulation analytics, and ML-based card recommendation (`recommend`/`complete`) trained offline on public decklists — no game simulator drives the model, no self-play. A local web UI (`doubletap web`) wraps the same CLI commands.

Full user-facing docs: `README.md` (quickstart) and `docs/operating-manual.md` (complete reference — install, every command, maintenance, troubleshooting). Design history and experiment log: `docs/rl-strategy-research.md` (§Results is the running, date-stamped ledger of every model/reward/feature experiment — always read the **latest** dated section; earlier numbers are explicitly superseded there when metrics change) and `docs/goldfish-sim-design.md`.

## Commands

```bash
uv sync --extra dev --extra ocr --extra ml   # full dev environment (Windows: drop --extra ocr, macOS-only)
uv run pytest                                 # full suite, no network needed
uv run pytest -m macos_ocr                    # manual Vision-framework smoke test (macOS only)
uv run ruff check src tests                   # lint (CI-gated)
```

CI (`.github/workflows/test.yml`) runs the suite on **ubuntu-latest, macos-latest, and windows-latest** on every push — a real Windows install failure (encoding + doc gaps) is what added the Windows leg; keep it green, don't special-case it away.

## Architecture

- `db.py` — single SQLite file at `~/.doubletap/doubletap.db` (`DOUBLETAP_HOME` overridable). Tables: `cards` (full Scryfall JSON blob + normalized name index), `decks`/`deck_cards`, `meta`.
- `scryfall.py` — bulk `oracle_cards` sync; `names.py` — fuzzy card-name resolution (rapidfuzz), never silently picks a best match on ambiguity.
- `formats.py` — per-format rules (Commander/Modern/Standard: deck size, singleton vs 4-of, color identity, legality, bracket/Game-Changer detection). **Game Changers are read from Scryfall's `game_changer` field, never hardcoded** — the list changes every few months and a hardcoded snapshot has already gone stale once.
- `decks.py` — CSV/text/photo import, deck save/load. All text file I/O must pass `encoding="utf-8"` explicitly — Windows defaults to cp1252 otherwise, which breaks on any non-ASCII card name (this shipped as a real bug, 0.1.10).
- `archidekt.py` — rate-limited public-deck crawler feeding the training corpus.
- `manabase.py`, `swaps.py`, `analysis.py` — Karsten mana-math, swap recommendations, goldfish-adjacent analytics. Colorless commanders get **Wastes**, not off-identity colored basics.
- `ml/` — the recommendation engine:
  - `data.py` — `Vocab` (per-format legal-card feature arrays) and `card_features`/`state_features`. Feature/state dims are **static, per-card or per-partial-deck** — nothing here can depend on a specific candidate-vs-state relationship (see below).
  - `model.py` — `TwoTowerQ`: state tower sums card embeddings (order-agnostic bag) + structural features; action tower embeds each candidate independent of state. This independence is load-bearing: it's what lets `score_pool` rank the entire legal pool with one state forward pass. **Any new signal that depends on both the candidate and the current partial deck cannot be a static feature — it has to be a reranker** (see `ml/policy.py`'s `reranker` param on `score_state`/`complete_deck`, and examples `make_goldfish_reranker` in `policy.py` and `make_pmi_reranker` in `reward.py`).
  - `reward.py` — `PMIModel`: smoothed pairwise PPMI over deck co-occurrence — literally a card-synergy graph stored as a flat `{(a,b): score}` dict. Used as the CQL training reward, as swap-cut rationale (`swaps.py`), and (`make_pmi_reranker`) as a `--synergy-weight` reranker for `recommend`/`complete` (default 0.3, cleared the keep-bar on a 200-deck Commander holdout — see `docs/rl-strategy-research.md`; Modern/Standard share the default value unvalidated).
  - `neighbors.py` — brute-force O(n_decks) Jaccard similarity for the `--personalize` blend; fine at the current corpus size (thousands of decks), will need an inverted index if the corpus grows past that.
  - `train_bc.py`/`train_cql.py` — BC (imitation, cross-entropy) and CQL (conservative Q-learning); `infer_np.py` — torch-free numpy inference so `recommend`/`complete` never need torch installed, only training does.

## Keep-bar discipline (non-negotiable for anything touching the model)

Every model/reward/feature experiment is gated against the **current champion** with a bar set **before** training, never softened after seeing results:
- Bar: `recovery@50 ≥ champion − 1pt`, AND the metric the change claims to improve must beat the champion by the pre-committed margin (currently +2pts to flip the shipped default; see `docs/rl-strategy-research.md` §Results for the exact current numbers — they change, don't hardcode them here).
- A single-seed miss under ~0.5pt: rerun 2 more seeds, decide on the 3-seed mean against the same bar.
- **Never train an experiment against the live `~/.doubletap/models` dir** — `save_checkpoint` overwrites the serving model in place. Train into a scratch dir, or restore from backup and re-verify `recommend` prints the expected checkpoint afterward.
- Champions must be reproducible: seed-0 retrain at the tagged commit.
- Structural/goldfish quality are tracked, not gating, unless a format's `FormatConfig.pip_state` says otherwise.

## Release protocol

1. CI green on HEAD first — CI is torch-less, so any test importing torch needs `pytest.importorskip("torch")`.
2. Bump `version` in `pyproject.toml`, `uv lock`, commit.
3. `git tag vX.Y.Z && git push && git push origin vX.Y.Z` — the tag push triggers `publish.yml` (Trusted Publishing, no tokens, ever).
4. `gh release create vX.Y.Z` with a human-voiced blurb (not a changelog dump).
5. Verify: PyPI's simple index (`Accept: application/vnd.pypi.simple.v1+json`) or `/pypi/<pkg>/<version>/json` — never the bare `/pypi/<pkg>/json`, and don't trust the CDN for a few minutes either way. Then `uv tool install doubletap --reinstall --refresh` and confirm the version.

**Docs are part of the release, not an afterthought:** any version bump, new/changed CLI option or flag, or behavior change that makes existing README/manual text wrong or incomplete must update `README.md` **and** `docs/operating-manual.md` in the same change — before considering the work done. A shipped feature with stale docs is an incomplete release. This applies even to changes that don't bump the version if they alter documented behavior (new flags, renamed commands, changed defaults).

## Environment

- Managed with **uv**; `uv.lock` is committed intentionally. `.python-version` pins 3.11.
- Intel Mac ceiling: `torch==2.2.2` + `numpy<2` (last x86_64 macOS wheel) — encoded as markers in the `ml` extra, do not "helpfully" bump torch here.
- `recommend`/`complete` run on torch-free `.npz` weights; only `train bc`/`train cql`/`train export` need torch.
