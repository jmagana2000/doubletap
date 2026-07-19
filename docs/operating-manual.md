# DoubleTap Operating Manual

The complete reference for operating DoubleTap, the Magic: The Gathering
(MTG) deck-building assistant: every command and option,
data locations, standard workflows, maintenance procedures, and failure
recovery. For a gentle introduction, read the [README](../README.md) first.

---

## 1. System overview

DoubleTap is a local, single-user tool — a CLI plus an optional local web UI
(`doubletap web`, §3.0). There is no cloud service and no account; everything
lives on your machine in two places:

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
│   ├── cql_commander.pt  #   CQL model (the default since it cleared the keep-bar)
│   ├── bc_commander.pt   #   behavior-cloning baseline (fallback)
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

Recommended (uv reads `.python-version` and `uv.lock`, so the environment is
always right — `dev` = tests, `ocr` = photo import, `ml` = training). This
works the same on Windows — `uv` handles the venv either way:

```bash
uv sync --extra dev --extra ocr --extra ml
source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1 (PowerShell) or .venv\Scripts\activate.bat (cmd)
doubletap cards sync
```

The last command downloads ~180 MB of card data from Scryfall.

Plain pip alternative — use **python3.11 specifically** (a newer system
`python3` will build a venv PyTorch can't install into on Intel Macs), and
don't paste the block with comments into zsh. macOS/Linux:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,ocr]"
.venv/bin/pip install -e ".[ml]"
source .venv/bin/activate
doubletap cards sync
```

Windows (PowerShell; the `ocr` extra is macOS-only, so it's skipped here):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\pip install -e ".[ml]"
.venv\Scripts\Activate.ps1
doubletap cards sync
```

> **Why the 3.11 pin, and when you can ignore it:** PyTorch is needed only
> for **training**. Suggestions (`recommend`/`complete`) run on torch-free
> numpy weights (`.npz`), so any Python ≥ 3.11 works for everyday use. The
> pin exists because on Intel Macs the last x86_64 torch wheel is 2.2.2
> (Python 3.11 only, numpy < 2) — encoded as dependency markers in
> `pyproject.toml`, so `uv sync --extra ml` just works. Skip the `ml` extra
> entirely if you never retrain.

Verify the install: `doubletap cards lookup "sol ring"` should print a match.

---

## 3. Command reference

Every command exits `0` on success and `1` on failure (bad input, unresolved
cards, validation violations, missing prerequisites). `--help` on any command
shows its options.

### 3.0 `web` — the browser UI

**`web`** — serves the local web UI. Every action in the UI runs the real CLI
in-process (same code path, guaranteed parity). Binds `127.0.0.1` only;
requests require an `X-DoubleTap` header and only known subcommands are
accepted, so other websites cannot drive it. Long commands (crawl, train)
block until done — leave the tab open.

| Parameter | Default | Description |
|---|---|---|
| `--port` | 8787 | Port to serve on; open `http://127.0.0.1:<port>` in a browser |

### 3.1 `cards` — the local card database

**`cards sync`** — downloads Scryfall bulk data into the cache. Skips the
download when Scryfall's data hasn't changed.

| Parameter | Default | Description |
|---|---|---|
| `--force` | off | Re-download even when the cache is current. Use this to refresh **prices**, which are frozen at the last sync |

**`cards lookup NAME`** — resolves a card name (typo-tolerant,
accent-insensitive, face-aware). Prints score (string-match confidence,
0–100), name, color identity, oracle_id, and — for mana producers — the
card's fractional colored-source weights (Karsten: lands 1.0, mana
artifacts 0.75, mana creatures 0.5 per produced color, e.g. Arcane
Signet shows `sources B:0.75 G:0.75 R:0.75 U:0.75 W:0.75`). Exits 1 if
nothing matches.

| Parameter | Default | Description |
|---|---|---|
| `NAME` | required | The card name to look up; typos and partial names are fine |

### 3.2 `deck` — building and inspecting decks

**`deck import PATH`** — routes by file extension: `.csv` (Moxfield/Archidekt
exports), images (`.heic .jpg .jpeg .png .webp .tiff .bmp` → OCR), anything
else as a plain-text list.

| Parameter | Default | Description |
|---|---|---|
| `PATH` | required | The file to import: CSV export, photo/screenshot, or text decklist |
| `--format`, `-f` | `commander` | Deck format, `commander` or `modern`; changeable later at merge |
| `--out`, `-o` | auto | Output path. Default: `~/.doubletap/decks/<name>.json`, where `<name>` is the card name for single-card photo imports (collision-safe `-2`, `-3` suffixes), else the source file's stem |
| `--commander` | — | Card name to set as the commander |
| `--companion` | — | Card name to set as the companion (sits outside the deck) |
| `--threshold` | 90.0 | Fuzzy auto-accept score (0–100); matches at/above it (with a clear gap to the runner-up) import as `assumed` |
| `--interactive/--no-interactive` | interactive | Whether to prompt to settle ambiguous/unmatched names in a terminal |

Text-list syntax: `4 Lightning Bolt` or `4x ...` or a bare name (qty 1);
`# comments` and `// comments`; section headers `Deck`, `Commander`,
`Companion`, `Sideboard` (sideboards are dropped); `*CMDR*` marker; Moxfield
`(SET) 123` tails are stripped. Two `*CMDR*` lines = partner commanders.
Imports never guess silently: ambiguous/unmatched lines abort the import
(exit 1) unless settled interactively.

**`deck list`** — table of every deck in `~/.doubletap/decks/`: file, format,
card count, commander (or contents for small commander-less files). No
parameters.

**`deck show NAME`** — every card in one deck: commander/partner/companion
slots, then quantity, name, mana cost, and type line per card, alphabetical.

| Parameter | Default | Description |
|---|---|---|
| `NAME` | required | A saved deck name (`deck show my-deck` finds `~/.doubletap/decks/my-deck.json`) or an explicit file path; the `.json` extension is optional either way |

**`deck add PATH NAME`** — add a card. Adding past the copy limit
(singleton in Commander, 4-of elsewhere; card-text caps like Seven
Dwarves/Nazgûl respected) is **refused** — exit 1, nothing written —
unless `--force`. Other rule breaks (color identity) warn but save.
Exit 1 on unresolved names, nothing written.

| Parameter | Default | Description |
|---|---|---|
| `PATH` | required | The deck JSON file to modify |
| `NAME` | required | Card to add; must match exactly (typos get suggestions) |
| `--qty`, `-n` | 1 | How many copies to add |
| `--force` | off | Add even past the copy limit |

**`deck remove PATH NAME`** — remove a card. Removing the
commander/partner/companion by name clears that slot. Exit 1 if the card
isn't in the deck.

| Parameter | Default | Description |
|---|---|---|
| `PATH` | required | The deck JSON file to modify |
| `NAME` | required | Card to remove; must match exactly |
| `--qty`, `-n` | 1 | How many copies to remove (capped at what's there) |

**`deck commander PATH [NAME]`** — show, set, or change the commander. When
changing, the old commander returns to the main deck and a promoted card
leaves it, so the card count is preserved.

| Parameter | Default | Description |
|---|---|---|
| `PATH` | required | The deck JSON file |
| `NAME` | — | Card to make the commander; must match exactly (typos get suggestions). Omit to just show the current commander and its color identity |
| `--partner` | — | Second commander's name (both cards need the Partner ability) |

**`deck merge PATH PATH...`** — combine two or more deck files into one.
First commander/companion wins; extras are noted.

| Parameter | Default | Description |
|---|---|---|
| `PATH...` | required | Two or more deck JSON files to merge |
| `--out`, `-o` | `decks/merged.json` | Where to write the merged deck |
| `--format`, `-f` | inputs' format | Format for the result; required when the inputs disagree |

**`deck validate PATH`** — prints the deck's identity first (format, size,
commander + color identity, companion — even when the deck is incomplete),
then every rule problem: banned/not-legal cards, size, copy limits
(basic-land and "any number" exemptions), commander eligibility, color
identity, partner legality, all ten companion restrictions. Exit 0 clean /
1 with violations.

| Parameter | Default | Description |
|---|---|---|
| `PATH` | required | The deck JSON file to check |

**`deck bracket PATH`** — Commander Bracket (1–5) from the Game Changers
count: 0 → Bracket 1/2, 1–3 → 3, 4+ → 4. Lists the Game Changers present.

| Parameter | Default | Description |
|---|---|---|
| `PATH` | required | The deck JSON file to rate |

**`deck analyze PATH`** — structural report: role counts vs Commander
targets, tutor count, interaction speed, mana curve, color balance (flags
colors your lands can't support), ways to win (direct/combat/evasion/
poison/mill), market price. Heuristic — see `docs/gameplay-blindspots.md`.

| Parameter | Default | Description |
|---|---|---|
| `PATH` | required | The deck JSON file to analyze |

**`deck swaps NAME`** — recommend (cut, add) swap pairs, valued by the
measured swap delta Δ(cut, add) = model score of the add minus the cut's
own re-add score, both evaluated on the deck *without* the cut — so
pairwise interactions count, and a pair only appears when the model
scores the swap as a strict upgrade (Δ > 0). Cut candidates are ranked
by three signals: the model's own score for the card in deck context,
total PMI synergy with the deck, and role-quota surplus; scarce win
conditions are never offered as cuts, and lands are `deck manabase`'s
job. Each pair states why the cut was chosen.

| Parameter | Default | Description |
|---|---|---|
| `NAME` | required | Deck file path or saved deck name (`.json` optional) |
| `-k` | 5 | Number of swap pairs |
| `--max-card-price` | none | Budget cap on suggested additions |

In the Builder, the ⇄ Swaps button shows the same pairs with a one-click
Go, and the manual swap picker lists your cards best-cut-first.

**`deck format NAME [FORMAT]`** — show or change a deck's format. Away
from commander, the commander/partner move into the main deck; toward
commander, the slot is left unset (the Builder's warning points at the
fix). The deck is re-validated against the new format immediately, so a
conversion reports exactly what became illegal.

**`deck manabase NAME`** — recommend a complete mana base for the deck's
spells: Karsten land count (curve regression), per-color source
requirements (hypergeometric tables), real lands chosen greedily to cover
the deficits (untapped preferred, identity/budget/bracket respected),
basics filling the rest — then goldfished against an all-basics baseline
so the value of the nonbasics is measured. Pure math, no ML.

| Parameter | Default | Description |
|---|---|---|
| `NAME` | required | Deck file path or saved deck name (`.json` optional) |
| `--budget` | none | Max USD per land (strict: unpriced lands excluded) |
| `--bracket` | 3 | ≤3 excludes Game Changer lands (Ancient Tomb, Gaea's Cradle, …) |
| `--lands` | Karsten target | Override the land count |
| `--goldfish/--no-goldfish` | on | Simulate recommended vs all-basics |
| `--apply` | off | Replace the deck's lands with the recommendation |

Known v1 ceiling: conditionally-restricted producers (Ancient Ziggurat)
are valued at face; check the list before sleeving.

**`deck goldfish PATH`** — solitaire simulation of how the deck actually
plays: shuffle, mulligan (Karsten's model), play lands, cast greedily.
Reports a 0–1 goldfish score plus mana efficiency, curve-out rate, dead
turns, commander-on-curve, and land-drop rates. No opponent — it measures
function, not matchups. Calibrated against Frank Karsten's published
land-drop tables (docs/goldfish-sim-design.md).

| Parameter | Default | Description |
|---|---|---|
| `PATH` | required | The deck JSON file to goldfish |
| `--games` | 200 | Simulated games (more = tighter estimates, slower) |
| `--turns` | 10 | Turns per game |
| `--draw` | off | Simulate on the draw instead of the play |

**`deck price PATH`** — total USD (Scryfall market, cheapest finish) plus
the most expensive cards; lists unpriced cards.

| Parameter | Default | Description |
|---|---|---|
| `PATH` | required | The deck JSON file to price |
| `--top` | 10 | How many of the most expensive cards to list |

### 3.3 `corpus` — training data (advanced)

**`corpus crawl`** — crawls public Archidekt decks at ~1/s with backoff.
Fully resumable; fetched decks are never re-requested. Only decks passing
full format validation enter the corpus.

| Parameter | Default | Description |
|---|---|---|
| `--format`, `-f` | required | Which format's decks to crawl (`commander` or `modern`) |
| `--max` | 1000 | How many new deck ids to queue this run (0 = just fetch the existing queue) |
| `--order-by` | `-viewCount` | Archidekt search ordering; different orderings reach different decks |

**`corpus stats`** — per-format counts by status: `parsed` (usable),
`rejected` (failed a filter), `gone` (deleted/private, skipped forever). No
parameters.

**`corpus pmi`** — builds the card-synergy (PPMI) table used for training
rewards and recommendation rationale; prints top pairs as a sanity check.

| Parameter | Default | Description |
|---|---|---|
| `--format`, `-f` | required | Which format's corpus to analyze |
| `--min-count` | 20 | Minimum number of decks a card pair must share to count as synergy (higher = fewer, more reliable pairs) |
| `--top` | 20 | How many top synergy pairs to print |

For long crawls keep the machine awake: `caffeinate -is doubletap corpus crawl ...`

### 3.4 `train` / `eval` — models (advanced)

**BC vs CQL — what the two trainers actually do.** Both train the same
network (a two-tower scorer: one tower summarizes your partial deck, the
other summarizes a candidate card; the match between them is the score).
They differ in what the network is taught:

- **BC (behavior cloning)** is supervised imitation. It is shown thousands
  of human-built decks and learns to answer one question: *given this
  partial deck, which card did a human actually add?* Its strength is that
  it directly models what real, functioning decks look like. Its limits:
  it can only imitate — it inherits the popularity bias of public decklists
  and has no notion of whether a pick actually makes the deck better.
- **CQL (conservative Q-learning)** is offline reinforcement learning.
  Instead of imitating picks, it learns a *value* for each candidate: how
  much long-term deck quality does adding this card buy? The reward signal
  is built from card-synergy statistics (the PMI table — which is why
  `corpus pmi` must run first) plus deck-structure terms. The "conservative"
  part is a penalty that keeps value estimates anchored to cards actually
  seen in human decks — without it, offline RL notoriously overrates cards
  it has never seen tried.
- **Why CQL is the default (and wasn't always).** The two are compared on
  the same held-out test (`eval`: hide cards from unseen decks, measure
  recovery@k) under a decision rule committed *before* looking at results:
  CQL ships as default only if it beats BC by at least 2 points of
  recovery@50. On earlier corpora it missed the bar and BC shipped. On the
  2026-07-13 full-corpus retrain, CQL cleared it — 20.79 vs 18.47
  recovery@50 (+2.32) on the 200-deck holdout, winning at every k and with
  a slightly cheaper suggestion curve — so `recommend` now prefers
  `cql_<format>` with BC as the fallback. One honest caveat stands: CQL's
  reward is a statistic of the same corpus BC imitates, so the margin was
  never going to be huge — the keep-bar exists to stop the fancier model
  from shipping on vibes, and equally to promote it the moment the
  evidence is real. *(Update 2026-07-16: the commander champion now also
  carries mana-math features — pip demand on the state side, Karsten
  fractional sources on the card side — recovery@50 21.17 with the best
  completion quality recorded (structural 0.7176); modern rejected
  mana-math features in a 3-seed sweep and keeps its original champion.
  Full history: docs/rl-strategy-research.md §Results.)*

Training order: `corpus crawl` → `corpus pmi` → `train bc` → (optionally)
`train cql`, which initializes from the BC checkpoint.

**Checkpoints and torch.** Training requires PyTorch; **suggestions do
not**. Every training run writes two files: `<algo>_<format>.pt` (torch,
for further training) and `<algo>_<format>.npz` (plain numpy weights).
`recommend`/`complete` prefer the `.npz` and run torch-free — the numpy
scorer is bit-for-bit equivalent to the torch model (pinned by a test).

**`train bc`** — trains the behavior-cloning baseline (the fallback model,
and the initialization for CQL). Refuses to run on fewer than 20 parsed
decks. Writes `models/bc_<format>.pt` + `.npz`.

| Parameter | Default | Description |
|---|---|---|
| `--format`, `-f` | required | Which format to train for |
| `--steps` | 1500 | Training steps; more = longer training, usually better up to a point |
| `--seed` | 0 | Random seed, for reproducible training runs |

**`train cql`** — trains the CQL model (the default for suggestions since
it cleared the keep-bar; see the explainer above); requires the PMI table
first. Writes `models/cql_<format>.pt` + `.npz`.

| Parameter | Default | Description |
|---|---|---|
| `--format`, `-f` | required | Which format to train for |
| `--steps` | 1500 | Training steps |
| `--alpha` | 1.0 | Conservative-penalty weight; higher keeps the model closer to what human decks actually do |
| `--seed` | 0 | Random seed |
| `--init-from-bc/--no-init-from-bc` | on | Start from the BC checkpoint's weights when one exists |

**`train export`** — converts existing `.pt` checkpoints in
`~/.doubletap/models/` to torch-free `.npz` weights (needs torch; new
training runs write both automatically). No parameters.

**`eval`** — held-out recovery@k: hides cards from unseen decks and measures
how many the model ranks highly. Higher is better. Also reports a
**structural quality** composite: half of each holdout deck is masked, the
model completes it greedily, and the completion is scored on color
sufficiency, role quotas, and win-condition presence. Use both numbers to
compare a BC and a CQL checkpoint on equal terms.

| Parameter | Default | Description |
|---|---|---|
| `--model` | required | Path to the checkpoint to evaluate |
| `--n-hide` | 10 | How many cards to hide from each test deck |
| `--seed` | 0 | Random seed controlling which cards are hidden |

### 3.5 `recommend` / `complete` — suggestions

**`recommend`** — top-k legal, nonland additions for a partial deck, with
synergy rationale ("with Sol Ring (10.5)") and a land-count gap report.

| Parameter | Default | Description |
|---|---|---|
| `--deck` | required | The deck JSON file to suggest additions for |
| `-k` | 20 | How many suggestions to show |
| `--model` | auto | Checkpoint to use; defaults to `cql_<format>`, falling back to `bc_<format>` (CQL cleared the keep-bar on 2026-07-13) |
| `--personalize` | 0.3 | 0–1 blend of the model score with card frequencies among corpus decks most similar to yours; higher favors decks like yours, 0 disables |
| `--max-card-price` | none | Per-card USD budget cap; only cards at or under this market price are suggested (unpriced cards stay eligible) |

**`complete`** — greedily fills the deck's nonland slots (re-scoring after
each add) and tells you how many lands remain to add.

| Parameter | Default | Description |
|---|---|---|
| `--deck` | required | The deck JSON file to complete |
| `--out`, `-o` | none | Write the completed deck to this path (omit for a dry run) |
| `--model` | auto | Checkpoint to use; same default chain as `recommend` |
| `--max-card-price` | none | Per-card USD budget cap on added cards |
| `--bracket` | 3 | Target Commander Bracket for the result: 1–2 add no Game Changers, 3 caps the deck at three total (counting existing ones), 4–5 unrestricted. Ignored for Modern |
| `--goldfish` | off | After completing, goldfish the result with the land gap filled by basics split proportionally to the deck's colored pips (simulation only — the saved deck still leaves the mana base to you) |

Lands are never suggested by design — add them yourself using the gap report
and the color-balance section of `deck analyze`.

---

## 4. Standard workflows

**Photograph a physical deck → analyzed, legal deck file**

```bash
doubletap deck import IMG_001.HEIC
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
doubletap deck bracket mydeck.json
# swap out listed Game Changers to drop a bracket, re-run to confirm
```

**Build the recommendation engine from scratch** (one-time, hours)

```bash
caffeinate -is doubletap corpus crawl -f commander --max 20000
doubletap corpus stats
doubletap corpus pmi -f commander
doubletap train bc -f commander
doubletap train cql -f commander
doubletap eval --model ~/.doubletap/models/bc_commander.pt
doubletap eval --model ~/.doubletap/models/cql_commander.pt
```

`recommend` automatically prefers the CQL checkpoint when both exist,
falling back to BC. Compare the two `eval` outputs yourself — if BC wins
recovery@50 by more than the keep-bar margin on your corpus, pass
`--model ~/.doubletap/models/bc_<format>.npz` explicitly.

---

## 5. Maintenance

| Task | When | How |
|---|---|---|
| Refresh card database | New set releases; a name won't resolve | `doubletap cards sync` |
| Refresh **prices** | Before budget decisions (prices are frozen at last sync) | `doubletap cards sync --force` |
| Grow the corpus | Occasionally; after set releases | Re-run `corpus crawl` — it resumes, never re-fetches |
| Retrain models | After meaningful corpus growth or a card-db refresh | `corpus pmi`, then `train bc`, then `train cql` (that order — CQL needs both) |
| Back up | Anytime | Copy `~/.doubletap/decks/` (tiny, irreplaceable). The card db and models are rebuildable; `corpus/raw/` shards avoid a re-crawl |
| Reset completely | Corruption, fresh start | Delete `~/.doubletap/` — decks included, so back those up first |
| Run the test suite | After pulling changes | `.venv/bin/pytest` (no network needed); Windows: `.venv\Scripts\pytest` |

**Update the Game Changers list** when WotC revises it: edit
`GAME_CHANGERS` in `src/doubletap/formats.py`.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `command not found: doubletap` | venv not active | `source .venv/bin/activate` or use `.venv/bin/doubletap`; Windows: `.venv\Scripts\Activate.ps1` or `.venv\Scripts\doubletap` |
| `No module named 'torch'` on recommend/complete | Only legacy `.pt` checkpoints present; torch-free `.npz` weights missing | Run `doubletap train export` once on a torch-enabled setup (`uv sync --extra ml`); afterwards suggestions never need torch |
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
