# DoubleTap

MTG deck-building assistant with offline-RL card recommendations. Give it a
partial (or complete) deck and it suggests the top-k additions, learned from
thousands of human-built decklists — no game simulator, no self-play.

- **Format-parameterized**: card legality comes from Scryfall's per-card
  legalities; construction rules (deck size, copy limits, singleton, commander
  color identity) come from per-format config. Ships with **Commander** and
  **Modern**.
- **Two models, honestly compared**: a behavior-cloning baseline (BC) and
  Conservative Q-Learning (CQL) trained on PPMI-synergy + structural rewards,
  A/B'd on the same held-out recovery@k harness.
- **Deck import three ways**: CSV (Moxfield/Archidekt exports), plain-text
  decklists, or a photo/screenshot of a decklist (Apple Vision OCR + fuzzy
  matching — no image classification).
- **Non-goals**: gameplay simulation, pricing, collection management.

## Installation

Requires Python ≥ 3.11. macOS is needed for photo import (Apple Vision) —
everything else is cross-platform.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,ocr]"       # core + tests + photo import
.venv/bin/pip install -e ".[ml]"            # torch, for training/recommending
```

> **Intel Macs**: the last torch wheel for x86_64 macOS is 2.2.2 (Python 3.11
> only) and it predates the numpy 2 ABI. Create the venv with `python3.11` and
> install `pip install "torch==2.2.2" "numpy<2"` instead of the `ml` extra.

All data lives in `~/.doubletap/` (override with `DOUBLETAP_HOME`): the SQLite
card/corpus database, the cached Scryfall bulk file, raw crawl shards, and
trained model checkpoints. Nothing is written inside the repo.

## Quick start

```bash
# 1. Build the local card cache (~180 MB download from Scryfall, refreshed
#    only when their bulk data changes)
doubletap cards sync

# 2. Import your deck — CSV, text list, or a photo of a decklist
doubletap deck import mydeck.csv --format commander -o mydeck.json
doubletap deck import decklist.png --format commander -o mydeck.json

# 3. Check it against format rules
doubletap deck validate mydeck.json

# 4. Get suggestions (needs a trained model — see below)
doubletap recommend --deck mydeck.json -k 20
```

## Command reference

### Card cache

```bash
doubletap cards sync [--force]        # download/refresh Scryfall oracle cards
doubletap cards lookup "lightning blot"   # exact + fuzzy name resolution
```

Lookup is diacritics-insensitive and face-aware: `malakir rebirth` finds the
MDFC "Malakir Rebirth // Malakir Mire", and face-name collisions (a card face
named "Lightning Bolt") never shadow the real card.

### Deck import

```bash
doubletap deck import <file> --format <commander|modern> [-o deck.json]
                      [--commander "Card Name"] [--threshold 90]
                      [--no-interactive]
```

Input is routed by extension: `.csv` (Moxfield `Count`/Archidekt `Quantity`
headers), images (`.png .jpg .heic ...` → Vision OCR), anything else as a
plain-text list (`4 Lightning Bolt`, `4x ...`, section headers, `*CMDR*`
markers; sideboards are dropped).

Name resolution never guesses silently: exact matches resolve; fuzzy matches
≥ threshold with a clear gap are accepted but printed as `assumed`; everything
else is reported `ambiguous`/`unmatched` and the import exits non-zero without
writing (in a terminal you'll be prompted to settle ambiguous lines
interactively).

### Deck validation

```bash
doubletap deck validate deck.json
```

Checks Scryfall legality (banned/not-legal), deck size (exact 100 for
Commander, min 60 for Modern), copy limits (with basic-land and "a deck can
have any number..." exemptions), commander eligibility, and color identity.
Exit 0 when clean, 1 with a violation list otherwise.

### Training corpus

```bash
doubletap corpus crawl --format commander --max 20000 [--order-by -viewCount]
doubletap corpus stats
doubletap corpus pmi --format commander [--min-count 20] [--top 20]
```

The crawler pulls public Archidekt decks (politely: 1 req/s + jitter,
identified User-Agent, exponential backoff, hard stop on repeated 429s). It is
fully resumable — deck ids are queued in SQLite, fetched decks are never
re-requested, and trimmed raw responses are kept in
`~/.doubletap/corpus/raw/*.jsonl.gz` so tables can be rebuilt without
re-crawling. Only decks that pass full format validation enter the corpus
(partner commanders and >2%-unresolvable decks are rejected).

For a large crawl, keep the machine awake:

```bash
caffeinate -is doubletap corpus crawl --format commander --max 20000
```

`corpus pmi` builds the smoothed PPMI synergy table (co-occurrence lift over
popularity, min-count filtered) used for CQL rewards and recommendation
rationale. Its top-pairs printout is a good corpus sanity check — you should
see real packages (Blood Artist + Viscera Seer, Thoughtseize + Inquisition of
Kozilek).

### Training and evaluation

```bash
doubletap train bc  --format commander [--steps 2000]
doubletap corpus pmi --format commander            # prerequisite for cql
doubletap train cql --format commander [--steps 1500] [--alpha 1.0]
doubletap eval --model ~/.doubletap/models/cql_commander.pt
```

Both algorithms share one architecture: a two-tower Q(state, candidate)
network (state = sum of card embeddings + curve/land/identity features;
candidate = card embedding + structured Scryfall features). BC trains with
sampled-softmax cross-entropy on "what card did the human add next"; CQL adds
a conservative penalty (sampled logsumexp with `log(N/K)` correction) and TD
targets on PPMI-synergy + land-fraction rewards, initialized from the BC
checkpoint.

Evaluation is **held-out deck completion**: hide 10 random nonland cards from
each holdout deck and measure what fraction appear in the model's top-k
(recovery@10/50/100). Decision rule: CQL ships only if it beats BC by ≥2
points recovery@50 (or matches it with better structural stats).

### Recommendations

```bash
doubletap recommend --deck mydeck.json -k 20 [--model path.pt] [--personalize 0.3]
doubletap complete --deck mydeck.json -o full.json
```

Defaults to `bc_<format>.pt` (CQL missed the keep-bar on the full corpus),
falling back to `cql_<format>.pt`. Suggestions
are always legal: nonland, within copy limits, inside the commander's color
identity. Each line shows the Q-score and the top PPMI contributors already in
your deck ("with Heroic Intervention (11.1), ..."). Lands are deliberately
excluded from suggestions — random-order training data carries no mana-base
signal — so the output ends with a structural gap report (card count and land
count vs. the format target) instead.

`--personalize` (default 0.3, 0 disables) blends the model score with card
frequencies among the corpus decks most similar to yours (Jaccard nearest
neighbors, near-duplicates excluded). This counters the global popularity
skew: cards common in decks *like yours* rank higher even when globally niche.

`complete` greedily fills the deck's nonland slots (deck size minus the land
target) with the model's top pick, re-scoring after each add, and tells you
how many lands remain to add.

## Development

```bash
.venv/bin/pytest                 # full suite; HTTP mocked, no network needed
.venv/bin/pytest -m macos_ocr    # manual Vision smoke test
                                 # (set DOUBLETAP_OCR_TEST_IMAGE=/path/to.png)
```

Layout:

```
src/doubletap/
├── cli.py          # typer app: cards / deck / corpus / train / eval / recommend
├── db.py           # SQLite schema + DOUBLETAP_HOME resolution
├── scryfall.py     # bulk data sync
├── names.py        # normalization + fuzzy lookup
├── decks.py        # Deck model, CSV/text parsing, resolution pipeline
├── ocr.py          # Apple Vision wrapper (one mockable function)
├── formats.py      # per-format configs + validator
├── archidekt.py    # rate-limited resumable crawler
└── ml/
    ├── data.py     # vocab, card/state features, action masks, transition sampler
    ├── reward.py   # smoothed PPMI + structural reward
    ├── model.py    # two-tower Q network, checkpoints
    ├── train_bc.py / train_cql.py / eval.py
```

## Known limitations

- Partner commanders, companions, and MDFC commanders are rejected from the
  corpus and unsupported in validation (v1 scope).
- Photo import targets decklist photos/screenshots, not spreads of physical
  cards on a table.
- PPMI scores staples near zero (Sol Ring co-occurs with everything, which is
  exactly what its popularity predicts), and recommendations skew toward
  popular cards — rare cards are scored mostly from structured features.
- The reward is a statistic of the same corpus the behavior policy comes
  from, so expect the CQL-over-BC margin to be modest; the BC baseline is the
  honest yardstick, not a fallback.
