# DoubleTap Operating Manual

The complete reference for operating DoubleTap: every command and option,
data locations, standard workflows, maintenance procedures, and failure
recovery. For a gentle introduction, read the [README](../README.md) first.

---

## 1. System overview

DoubleTap is a local, single-user CLI. There is no server and no account;
everything lives on your machine in two places:

| Location | Contents |
|---|---|
| The repo directory | Code and the virtualenv (`.venv/`). Nothing else is written here. |
| `~/.doubletap/` (the **data home**) | Everything the tool creates. Override with the `DOUBLETAP_HOME` environment variable. |

Data home layout:

```
~/.doubletap/
├── doubletap.db          # SQLite: card cache + training corpus
├── decks/                # your saved decks, one JSON file each
├── models/               # trained checkpoints and synergy tables
│   ├── bc_commander.pt   #   behavior-cloning model (the default)
│   ├── cql_commander.pt  #   CQL model (experimental alternative)
│   └── pmi_commander.npz #   card-synergy table
└── corpus/raw/           # compressed raw crawl shards (rebuildable source)
```

**Formats:** every deck has a format, `commander` or `modern`. Rules the
format controls: deck size (exactly 100 vs at least 60), copy limit (1 vs 4),
whether a commander is required, and which Scryfall legality column applies.

**Deck JSON:** a saved deck is a readable JSON file:

```json
{
  "format": "commander",
  "commander": {"oracle_id": "...", "name": "Atraxa, Praetors' Voice"},
  "partner": null,
  "companion": null,
  "cards": [{"oracle_id": "...", "name": "Sol Ring", "qty": 1}]
}
```

The companion never counts toward deck size. Cards are identified by Scryfall
`oracle_id`, so a deck file survives card database refreshes.

---

## 2. Installation and first run

Requirements: Python ≥ 3.11; macOS for photo import (everything else is
cross-platform); ~200 MB disk for the card database.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,ocr]"    # core + tests + photo import
.venv/bin/pip install -e ".[ml]"         # torch, for training/recommending
source .venv/bin/activate                # so `doubletap` works directly
doubletap cards sync                     # ~180 MB download from Scryfall
```

> **Intel Macs:** the last x86_64 torch wheel is 2.2.2 (Python 3.11 only,
> numpy < 2). Create the venv with `python3.11` and run
> `pip install "torch==2.2.2" "numpy<2"` instead of the `ml` extra.

Verify the install: `doubletap cards lookup "sol ring"` should print a match.

---

## 3. Command reference

Every command exits `0` on success and `1` on failure (bad input, unresolved
cards, validation violations, missing prerequisites). `--help` on any command
shows its options.

### 3.1 `cards` — the local card database

| Command | Options | Behavior |
|---|---|---|
| `cards sync` | `--force` | Downloads Scryfall bulk data into the cache. Skips the download when Scryfall's data hasn't changed; `--force` re-downloads regardless (use to refresh **prices**). |
| `cards lookup NAME` | — | Resolves a name (typo-tolerant, accent-insensitive, face-aware). Prints score (string-match confidence, 0–100), name, color identity, and oracle_id per candidate. Exits 1 if nothing matches. |

### 3.2 `deck` — building and inspecting decks

**`deck import PATH`** — routes by file extension: `.csv` (Moxfield/Archidekt
exports), images (`.heic .jpg .png .webp .tiff .bmp` → OCR), anything else as
a plain-text list.

| Option | Default | Meaning |
|---|---|---|
| `--format`, `-f` | `commander` | `commander` or `modern`; changeable later at merge |
| `--out`, `-o` | auto | Output path. Default: `~/.doubletap/decks/<name>.json`, where `<name>` is the card name for single-card photo imports (collision-safe `-2`, `-3` suffixes), else the source file's stem |
| `--commander` | — | Set the commander by name |
| `--companion` | — | Set the companion by name |
| `--threshold` | 90.0 | Fuzzy auto-accept score; matches at/above it (with a clear gap to the runner-up) import as `assumed` |
| `--interactive/--no-interactive` | interactive | Prompt to settle ambiguous/unmatched names in a terminal |

Text-list syntax: `4 Lightning Bolt` or `4x ...` or a bare name (qty 1);
`# comments` and `// comments`; section headers `Deck`, `Commander`,
`Companion`, `Sideboard` (sideboards are dropped); `*CMDR*` marker; Moxfield
`(SET) 123` tails are stripped. Two `*CMDR*` lines = partner commanders.
Imports never guess silently: ambiguous/unmatched lines abort the import
(exit 1) unless settled interactively.

| Command | Behavior |
|---|---|
| `deck list` | Table of every deck in `~/.doubletap/decks/`: file, format, card count, commander (or contents for small commander-less files) |
| `deck show PATH` | Every card in one deck: commander/partner/companion slots, then quantities, alphabetical |
| `deck add PATH NAME` | Add a card (`-n`/`--qty` copies, default 1). Exact name required. Warns on rule violations the add causes (copy limit, color identity) but saves anyway. Exit 1 on unresolved names, nothing written |
| `deck remove PATH NAME` | Remove `-n` copies (default 1; capped at what's there). Removing the commander/partner/companion by name clears that slot. Exit 1 if the card isn't in the deck |
| `deck commander PATH [NAME]` | Without NAME: show current commander and its color identity. With NAME: set/change the commander (`--partner "Name"` for partners). The old commander returns to the main deck; a promoted card leaves it — count is preserved. Requires an exact name; typos get suggestions, not guesses |
| `deck merge PATH PATH...` | Combine ≥2 deck files. `-o` output (default `decks/merged.json`); `--format` overrides (required when inputs disagree). First commander/companion wins; others are noted |
| `deck validate PATH` | Format legality: banned/not-legal cards, size, copy limits (basic-land and "any number" exemptions), commander eligibility, color identity, partner legality, all ten companion restrictions. Exit 0 clean / 1 with violations |
| `deck bracket PATH` | Commander Bracket (1–5) from the Game Changers count: 0 → Bracket 1/2, 1–3 → 3, 4+ → 4. Lists the Game Changers present |
| `deck analyze PATH` | Structural report: role counts vs Commander targets, tutor count, interaction speed, mana curve, color balance (flags colors your lands can't support), ways to win (direct/combat/evasion/poison/mill), market price. Heuristic — see `docs/gameplay-blindspots.md` |
| `deck price PATH` | Total USD (Scryfall market, cheapest finish) + the `--top` (default 10) most expensive cards; lists unpriced cards |

### 3.3 `corpus` — training data (advanced)

| Command | Options | Behavior |
|---|---|---|
| `corpus crawl` | `-f` (required), `--max` (default 1000), `--order-by` (default `-viewCount`) | Crawls public Archidekt decks at ~1/s with backoff. Fully resumable; fetched decks are never re-requested. Only decks passing full format validation enter the corpus |
| `corpus stats` | — | Per-format counts by status: `parsed` (usable), `rejected` (failed a filter), `gone` (deleted/private, skipped forever) |
| `corpus pmi` | `-f` (required), `--min-count` (default 20), `--top` (default 20) | Builds the card-synergy (PPMI) table used for training rewards and recommendation rationale; prints top pairs as a sanity check |

For long crawls keep the machine awake: `caffeinate -is doubletap corpus crawl ...`

### 3.4 `train` / `eval` — models (advanced)

| Command | Options | Behavior |
|---|---|---|
| `train bc` | `-f` (required), `--steps` (default 1500), `--seed` | Behavior-cloning baseline. Refuses to run on < 20 parsed decks. Writes `models/bc_<format>.pt` |
| `train cql` | `-f` (required), `--steps`, `--alpha` (default 1.0), `--seed`, `--init-from-bc/--no-init-from-bc` | CQL variant; requires the PMI table first. Writes `models/cql_<format>.pt` |
| `eval --model PATH` | `--n-hide` (default 10), `--seed` | Held-out recovery@k: hides cards from unseen decks, measures how many the model ranks highly. Higher is better |

### 3.5 `recommend` / `complete` — suggestions

| Command | Options | Behavior |
|---|---|---|
| `recommend --deck PATH` | `-k` (default 20), `--model`, `--personalize` (default 0.3), `--max-card-price` | Top-k legal, nonland additions with synergy rationale ("with Sol Ring (10.5)") and a land-count gap report. Model defaults to `bc_<format>.pt`, falling back to `cql_<format>.pt` |
| `complete --deck PATH` | `-o`, `--model`, `--max-card-price` | Greedily fills nonland slots (re-scoring after each add), tells you how many lands remain. `-o` writes the result |

`--personalize` (0–1) blends the model score with card frequencies among
corpus decks most similar to yours. `--max-card-price` is a per-card USD cap;
unpriced cards stay eligible.

Lands are never suggested by design — add them yourself using the gap report
and the color-balance section of `deck analyze`.

---

## 4. Standard workflows

**Photograph a physical deck → analyzed, legal deck file**

```bash
doubletap deck import IMG_001.HEIC          # once per card photo
doubletap deck merge ~/.doubletap/decks/*.json -o ~/.doubletap/decks/mydeck.json
doubletap deck commander ~/.doubletap/decks/mydeck.json "Your Commander"
doubletap deck analyze  ~/.doubletap/decks/mydeck.json
doubletap deck validate ~/.doubletap/decks/mydeck.json
```

**Finish a partial deck on a budget**

```bash
doubletap recommend --deck mydeck.json -k 30 --max-card-price 2.00
doubletap complete  --deck mydeck.json --max-card-price 2.00 -o finished.json
doubletap deck price finished.json
```

**Tune a deck to a table's power level**

```bash
doubletap deck bracket mydeck.json     # which bracket am I?
# swap out listed Game Changers to drop a bracket, re-run to confirm
```

**Build the recommendation engine from scratch** (one-time, hours)

```bash
caffeinate -is doubletap corpus crawl -f commander --max 20000
doubletap corpus stats                       # watch parsed count grow
doubletap corpus pmi -f commander
doubletap train bc -f commander              # 10–30 min
doubletap eval --model ~/.doubletap/models/bc_commander.pt
```

---

## 5. Maintenance

| Task | When | How |
|---|---|---|
| Refresh card database | New set releases; a name won't resolve | `doubletap cards sync` |
| Refresh **prices** | Before budget decisions (prices are frozen at last sync) | `doubletap cards sync --force` |
| Grow the corpus | Occasionally; after set releases | Re-run `corpus crawl` — it resumes, never re-fetches |
| Retrain models | After meaningful corpus growth or a card-db refresh | `corpus pmi` then `train bc` (order matters for CQL) |
| Back up | Anytime | Copy `~/.doubletap/decks/` (tiny, irreplaceable). The card db and models are rebuildable; `corpus/raw/` shards avoid a re-crawl |
| Reset completely | Corruption, fresh start | Delete `~/.doubletap/` — decks included, so back those up first |
| Run the test suite | After pulling changes | `.venv/bin/pytest` (no network needed) |

**Update the Game Changers list** when WotC revises it: edit
`GAME_CHANGERS` in `src/doubletap/formats.py`.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `command not found: doubletap` | venv not active | `source .venv/bin/activate` or use `.venv/bin/doubletap` |
| `No module named 'torch'` on recommend/complete | Launcher points at a Python without torch (multi-Python venv) | `.venv/bin/python -m pip install -e . --no-deps --force-reinstall` |
| `No trained model found` | No checkpoint for this format | Run the training workflow (§4), or check `~/.doubletap/models/` |
| `only N parsed <format> decks; crawl more first` | Corpus below the 20-deck training minimum | `corpus crawl` more decks |
| `Missing pmi_<format>.npz; run corpus pmi first` | CQL needs the synergy table | `doubletap corpus pmi -f <format>` |
| Import exits 1 with `unmatched`/`ambiguous` lines | Typos, proxies, or a stale card db | Fix the listed lines, run interactively to settle them, or `cards sync` |
| Photo import reads the wrong card | OCR picked another text line | Shoot straight-on with the name band clear; check `deck show`, re-import if wrong |
| Photo import: `Could not read image` | Corrupt/unsupported file | Re-export the photo; confirm the extension is a supported image type |
| Crawl stops with repeated 429s | Archidekt rate-limiting | Wait and re-run — the crawl resumes where it stopped |
| Suggestions feel generic | Popularity skew in public decks | Raise `--personalize` (e.g. 0.5–0.7) |
| A price is missing/zero | Scryfall has no USD price (new/digital card) | Listed as "unpriced"; re-sync later |
| Deck shows `(unreadable)` in `deck list` | Hand-edited/corrupt JSON | Fix or delete the file; re-import from source |

---

## 7. Performance expectations

| Operation | Typical duration |
|---|---|
| `cards sync` (fresh) | Minutes (~180 MB download + load) |
| `deck import` (photo) | Seconds per photo |
| `corpus crawl --max 20000` | Hours (~1 request/second, resumable) |
| `corpus pmi` | Under a minute |
| `train bc` (full corpus) | 10–30 minutes on CPU |
| `recommend` / `complete` | Seconds |
| Full test suite | Seconds |
