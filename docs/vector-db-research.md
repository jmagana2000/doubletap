# Vector database research (2026-07-23)

**Verdict: no.** Not a vector database, and — separately — not oracle-text
embeddings as a model feature either. One small, optional, standalone
lookup feature is documented below in case a real need shows up later, but
nothing in this research found one.

## What problem, if any, is real

Three existing signals were checked against "would a vector DB help this":

- `names.py` — rapidfuzz edit-distance on card **names**. Typo/misspelling
  correction. Not semantic, not a vector-search candidate.
- `ml/reward.py`'s `PMIModel` — PPMI over deck **co-occurrence** ("played
  together"), not "functionally alike." Two near-substitute removal spells
  (you'd only ever run one) show ~0 PPMI with each other — the metric only
  sees pairs that appear in the same decklist.
- `ml/neighbors.py`'s `neighbor_frequencies` — brute-force Jaccard between
  the query deck and every corpus deck, over **sparse card-ID sets**, not
  dense vectors. This never becomes a vector-DB problem, at any corpus size
  — Jaccard-over-sets has no cosine/ANN formulation without inventing a
  deck embedding (a separate, bigger, unrequested change). If this scan
  ever needs to be faster, the fix is a card→deck inverted index or
  MinHash-LSH, not pgvector/FAISS.

That leaves genuine oracle-text semantic similarity ("cards that function
like X") as the only real candidate — already named and deferred in this
doc's own "Explicitly rejected" list (§3, "Oracle-text embeddings —
deferred; heavy dependency, revisit if Phase B plateaus").

## Does it need a database

Measured in this repo's own environment (Intel Mac, CPU numpy, no
acceleration): brute-force cosine similarity of one query vector against a
30,000×384 float32 matrix (`mat @ q` + `argpartition` for top-50) took
**2.6ms mean / 4.0ms max over 20 runs**; at D=768, 3.4ms. For comparison,
`neighbor_frequencies`'s existing Python-loop Jaccard scan over ~9.3k decks
(the documented commander-corpus size) took **47.8ms** in the same
environment — and that's already shipped, already tolerated (CLAUDE.md
calls it "fine at current scale"). A dense 30k×384 matvec is 6x faster than
a feature this project already ships.

ANN indexes (HNSW/IVF) start paying for themselves in the
hundreds-of-thousands-to-millions-of-vectors range. DoubleTap's vocab
(~30k Commander-legal cards, fewer for Modern/Standard) grows by a few
hundred cards per Scryfall set release, not exponentially. A plain
`np.ndarray` — the exact pattern `Vocab.features` already uses — is
correct now and for the foreseeable future. Same shape of conclusion the
project already reached on GNNs (`docs/rl-strategy-research.md` — a 1-hop
PPMI graph did the job, no graph database needed).

## If embeddings were generated at all

Oracle text is already sitting unused in the DB: `cards` stores the full
Scryfall JSON blob (`db.py`), including `oracle_text` — no new fetch
required, only new parsing of already-synced data. Three ways to turn it
into vectors, in order of how well they fit this repo:

1. **Zero-dependency bag-of-words/TF-IDF in plain numpy.** Tokenize oracle
   text, build a sparse-then-dense TF-IDF matrix — no ML library at all.
   Weaker semantics than a trained embedding model but catches obvious
   phrase overlap ("destroy target creature," "exile target permanent").
   Costs nothing new in `pyproject.toml`. The only option consistent with
   this repo's "no speculative dependency" convention if this were built.
2. **Hosted embeddings API** (OpenAI/Voyage/Cohere). No torch conflict,
   reuses the already-present `httpx`. Privacy risk is near-zero (oracle
   text is public Scryfall data, not user decks). Cost is trivial (~30k
   cards × ~40–60 tokens ≈ 1.5M tokens, one-time, batch). Adds a network
   dependency and an API key requirement — a real cost for a tool that is
   otherwise fully offline after `cards sync`.
3. **Local sentence-transformers** (e.g. all-MiniLM-L6-v2, 384-dim,
   ~90MB). Pulls a `sentence-transformers` + `torch` dependency chain into
   the *base* install, not just the `ml` extra, and risks the already-
   fragile Intel-Mac ceiling (`torch==2.2.2` pinned, no newer x86_64
   wheel). Also breaks the project's hard invariant that `recommend`/
   `complete` never need torch (`ml/infer_np.py`) unless embeddings are
   baked once into a `.npz` and torch never touches inference again. The
   worst fit of the three, if this were built at all.

Whichever is chosen, it runs once at `cards sync` time, cached as an
`.npz` sidecar keyed by `oracle_ids` — the same shape `save_np_checkpoint`
already uses.

## Did Phase B actually plateau the way the deferred note assumed

The deferred note's condition was "revisit if Phase B plateaus." Phase B
(structured role one-hots + fractional-source features on both towers)
**shipped as an experiment and was reverted**
(`docs/rl-strategy-research.md`, "Results (2026-07-15) — keep-bar FAILED").
Structural composite target was ≥0.819; every variant (default reward
weight, ×5, ×15) landed at 0.696–0.701. Recovery never regressed. The
weight sweep specifically falsified "signal too small."

That result's own three candidate explanations are all about **credit
assignment and training mechanics** — a terminal-only reward giving too
little credit across ~60-step episodes, CQL's conservative penalty
anchoring the policy to human behavior, greedy completions inheriting
structure from the unmasked human half. None of them blame the *features*
for lacking information. That matters: "revisit if Phase B plateaus" was
written assuming a plateau would mean "features aren't rich enough, try
embeddings." What actually happened contradicts that premise. Meanwhile,
every actual win since then — the pip-demand state feature, the
fractional-source action feature, the PMI-synergy reranker — was a cheap
structured/domain feature or a co-occurrence graph, not a semantic
embedding, and each cleared its keep-bar on the first or third seed.
**There is no evidence in this repo that oracle-text embeddings would add
usable signal; there is direct evidence that a richer feature set already
failed to move the metric that would matter.**

## Implementation sketch — documented, not built, ship only on request

The one genuinely justified-if-ever-needed piece is a **lookup feature**,
not a training feature, and even it is speculative rather than requested
anywhere in the README/CLAUDE.md/docs read for this research:

- `ml/embeddings.py`: `build_embeddings(conn, vocab) -> np.ndarray`
  (TF-IDF-in-numpy per the option above), saved to
  `~/.doubletap/models/card_embeddings.npz` (`oracle_ids` + float32
  matrix), mirroring `ml/infer_np.py`'s checkpoint pattern.
- CLI: `doubletap cards similar <name>` — resolve via `names.lookup`,
  brute-force cosine top-k against the cached matrix (~3ms per the
  measurement above), print names + scores. No torch at query time.
- **No integration with `make_pmi_reranker`/`recommend`/`complete`, and no
  keep-bar**: this doesn't touch model scores or reward, so it isn't
  subject to the recovery@k/structural-quality gate — a discovery tool
  alongside `names.py`, not a ranking signal. Only build a keep-bar-gated
  reranker version if a future experiment specifically tests whether
  semantic similarity improves `recommend`/`complete` recovery, and even
  then, the Phase B result above predicts it likely won't move the metric
  that failed.
- Regeneration: hook into `cards sync` (`cli.py`), regenerate only cards
  whose `oracle_text` changed since the last cached run, not the whole
  vocab on every sync.

## Honest tradeoffs

- A vector database (Chroma/Qdrant/LanceDB/pgvector) is unjustified
  infrastructure at this scale — a new dependency and, for most of those
  options, a new service to run and keep in sync with SQLite, to solve a
  problem a 3ms numpy matvec already solves.
- Local sentence-transformers is the riskiest embedding source given the
  project already fights an Intel-Mac torch ceiling and has a hard
  invariant that inference stays torch-free.
- No user-facing request for "similar cards" exists anywhere in this
  repo's docs — this would be a speculative feature, not a fix for an
  observed gap, and the "revisit if Phase B plateaus" condition was met in
  form but not in the way that would predict embeddings help.

**Net recommendation:** don't build a vector database; don't add
embeddings as a model feature. Leave the standalone lookup sketch above
as documentation only — build it if a real "find similar cards" need
shows up, not speculatively.
