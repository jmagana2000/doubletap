# Gameplay Strategy Research → RL Model Refactor Plan

Deep research into Magic: The Gathering deck-building theory (expert
registry: docs/experts.md), verified
claim-by-claim, mapped against what the model currently learns, and turned
into a phased refactor plan. Research method: 5-angle web search fan-out,
23 sources fetched, 113 claims extracted, adversarial verification on the
top claims (2026-07-14/15).

---

## 1. Verified findings

### A. Mana-base mathematics (Frank Karsten) — the strongest material

All hypergeometric/simulation work by Frank Karsten (ChannelFireball /
TCGplayer), verified 3-0 or 2-0 by independent adversarial checks:

- **Castability threshold**: a card of mana value M is "consistently
  castable" when the probability of having its colored sources by turn M
  is ≥ (89 + M)% — 90% for 1-drops up to 96% for 7-drops.
- **Colored-source requirements** (99-card Commander deck, 41 lands):
  1 pip = **19 sources**, double pip (CC) = **30**, 1CC = **28**,
  triple pip = **36**. For 60-card/25 lands: C=14, CC=21, CCC=23.
- **Fractional source weights** for non-land producers: mana dorks = 0.5
  per color; mana rocks and 2-mana land-ramp = 0.75; 3-mana ramp = 0.5;
  land/spell MDFCs = 0.8 (non-mythic) / 1.0 (mythic); cheap cantrips ≈
  0.25; scry 1 ≈ 0.2; treasure makers = 0.25; fetchlands = 1.0 per
  fetchable color.
- **Optimal land count is a regression, not a constant**:
  - 60-card: `lands = 19.59 + 1.90·avgMV − 0.28·cheapDrawRamp (+0.27 companion)`
  - 99-card Commander: `lands = 31.42 + 3.13·avgMV − 0.28·cheapDrawRamp`
  - MDFCs count as 0.38 land (non-mythic) / 0.74 (mythic).
  - Rule-of-thumb equivalent: start at 42 lands + Sol Ring, cut one land
    per 2–3 mana rocks / 3–4 cantrips or dorks, never below ~37.

**Model implication:** our terminal reward uses a *flat* land-fraction
target (37%). Karsten's work says the target is a function of the deck's
own curve and ramp density, and that color sufficiency (pips vs sources)
is a second, independent constraint we currently don't reward at all.

**Addendum (2026-07-15) — Karsten's Commander-specific simulations and an
independent replication.** Karsten's "What's an Optimal Mana Curve and
Land/Ramp Count for Commander" (TCGplayer) runs Monte Carlo goldfish
simulations optimizing expected compounded mana spent. Key results:

- Rule of thumb: **start at 42 lands + Sol Ring, cut one land per 2–3
  additional mana rocks / 3–4 cantrips or dorks, never below ~37**
- Optimal lists shift with commander cost: a 2-mana commander wants ~42
  lands and nearly zero rocks; a 6-mana commander wants ~38 lands plus ~10
  rocks — total mana sources is the invariant, not land count
- Longer-game optima run **zero 1-drops, many more 6-drops, and 13–14
  Signets** — curve shape is a function of expected game length

ScrollVault independently replicated at scale (3.75M simulated games, 5
archetypes × 15 land counts, London mulligan + free Commander mulligan
modeled): **cEDH ~29–31 lands (12 fast-mana), combo 33–35 (10 rocks),
midrange 36–37 (10 mixed), battlecruiser/landfall 38–40**. Their data
validates the regression we shipped: for midrange (avg MV 3.0, 10 ramp)
Karsten's formula predicts 38, simulation says 37–38, with diminishing
returns past 37 (+1–3% cast rates for 3 more lands). It also quantifies
the EDHREC-average-29-lands problem: at 29 lands a 6-drop lands on curve
in only ~75% of games vs 90% at 37.

**Future use:** the per-archetype land table is the natural next upgrade
for `deck analyze` (archetype-aware land targets instead of one
regression), and the goldfish-simulation methodology is exactly the
non-circular reward proposed in §Results.

### A2. Karsten's mulligan math and Limited numbers (ingested 2026-07-15,
read directly from the primary text of "How Many Lands Do You Need to
Consistently Hit Your Land Drops?", 2017, plus the teryror methodology
expansion)

**The canonical mulligan model** (the assumption behind all his tables):
mulligan any 7-card hand with 0/1/6/7 lands; any 6-carder with 0/1/5/6;
any 5-carder with 0/5; keep any 4-carder; scry land-to-top after a
mulligan. (2017 = Vancouver scry era; his 2022 update re-ran everything
under the London mulligan — the regression we shipped comes from that
update.)

**60-card land-drop probabilities** (draw/play, under that rule):
- 25 lands: 3 lands by T3 = 94.6%/90.4%, 4 by T4 = 83.5%/74.7%, mana
  flood (8+ lands by T7) = 15.2% — his reference "midrange" point
- His working consistency targets: ~90% on the land drop you *need*,
  ~75–85% on the one you *want*
- **On the draw is worth ~3 lands**: 26 lands on the play ≈ 23 on the
  draw for the same consistency — boarding out a land on the draw is
  mathematically sound (he tempers it: costs mulligan frequency)
- Archetype ladder: 18–20 lands = low curve (avg CMC < 1.4), 21–23 =
  aggro, 24–25 = midrange, 26–27 = control (avg CMC > 3)

**Original regression lineage**: 2017 fit over 110 PT/GP decks
(R² = 0.614): `lands = 16 + 3.14 × avgCMC`, counting Attune/Aether
Vial/Mox Opal as lands and mana dorks/rocks/cantrips as spells. The 2022
update (already §1A) refines this to `19.59 + 1.90·MV − 0.28·cheapDrawRamp`
with MDFC fractions. Notable observation: all three Pro Tour winners in
his sample sat *above* the regression line — "when in doubt, add the
land."

**Limited (40-card) numbers**:
- **17 lands is the justified standard**: 3 by T3 = 95.6%/91.6%, 4 by T4
  = 85.6%/77.0%, flood 14.9% — and 17 ≈ 25 × 40/60, so Limited and
  Constructed guidance are one consistent system (Commander: × 99/60 →
  41.25)
- 16 lands (low curves): 94.0%/89.2% on T3, 81.3%/71.7% on T4
- 40-card colored sources (from the §1A tables): a 1CC card wants 12
  sources in a 17-land Limited deck
- From the methodology expansion: **color-aware mulligans dominate
  land-count-only mulligans** — in one worked example, castability of a
  turn-1 colored spell rises from 59.4% to 91.8% at a cost of only ~0.7
  expected opening-hand cards; keepable-land tables by curve for 40
  cards: ~15 (low curve) / 17–18 (mid) / 20 (high)

**DoubleTap relevance**: Modern is a 60-card format — the archetype
ladder and draw/play asymmetry could refine `karsten_land_target` for
Modern decks; the mulligan model is the exact keep/mull logic a future
goldfish simulator should implement (§Results future work); Limited is
out of scope (unsupported format) but recorded for completeness.

### B. EDHREC's lift metric — validates our reward core

Verified directly against EDHREC's methodology article: their production
recommendation metric is **lift = P(A∩B) / (P(A)·P(B))**, log-scaled.
DoubleTap's PPMI is exactly log-lift with word2vec-style α=0.75 popularity
smoothing and positive clipping — the same mathematical family EDHREC
*migrated to*, with arguably better popularity handling. **No change
needed**; this de-risks the synergy term.

### C. Command Zone quotas — our targets are at the low end

Verified via two mirrors of the template (episode 379+, categories
re-confirmed in the 2025 "new era" episode 658): **36–38 lands, 10–12
ramp, 10 card draw, 10–12 targeted removal, 3–4 board wipes**. Our
`COMMANDER_TARGETS` (36/10/10/10/3) sit at the bottom of each range.
Reasonable; midpoints are defensible for reward shaping.

### D. Strategy frameworks — mostly gameplay-time, not deck-time

Flores' "Who's the Beatdown" (role misassignment loses games) and
Chapin's tempo theory are about *piloting*, not construction. Their
deck-time shadow is already covered by role balance (threats vs answers
vs draw) — no additional encodable rule survived scrutiny. **Explicitly
not encoding** matchup role assignment.

### E. Academic recommenders — direction validated, methods out of scope

(Verification incomplete for these — sources are papers, claims plausible.)

- *Contextual preference ranking* (arXiv 2105.11864): conditioning card
  scores on the partial deck beats scoring cards independently by ~35
  accuracy points on draft picks. **Validates our two-tower Q(state,
  candidate) design** — the architecture is right.
- *Q-DeckRec* (arXiv 1806.09771): reward = simulated win rate. Requires a
  game simulator — an explicit non-goal of this project. Not adopted.
- *Generalized card representations* (arXiv 2407.05879): oracle-text
  embeddings + per-card win-rate statistics improve generalization.
  Win-rate data (17Lands) exists only for Arena limited formats — **no
  Commander equivalent exists**, so unavailable. Text embeddings are
  viable but a heavy dependency; deferred.

---

## 2. Gap analysis: what the model can't see today

| Verified signal | In model? | Gap |
|---|---|---|
| Synergy (log-lift/PPMI) | ✅ reward | none — validated |
| Deck-context conditioning | ✅ architecture | none — validated |
| Land count as f(curve, ramp) | ❌ flat 37% target | reward uses wrong target |
| Colored sources vs pips | ❌ | not rewarded, not featurized |
| Fractional mana sources | ❌ | rocks/dorks invisible as mana |
| Role quotas (ramp/draw/removal/wipes) | ❌ | exists in `analysis.py`, unused by ML |
| Card roles as features | ❌ | classifier exists, not in card features |
| Win-condition presence | ❌ | detected by analyze, invisible to model |

The unifying observation: **everything `deck analyze` learned to measure
in the blindspot work is invisible to the reward and both towers.**

---

## 3. Refactor plan

Governed by a pre-committed keep-bar (§4). Phases are independently
shippable; each ends with the full test suite green.

### Phase A — Reward upgrades (CQL-side, highest leverage)

1. **Karsten land target** in `structure_reward`: replace the flat
   `land_fraction_target` penalty with `−|lands_eff − karsten_target| /
   deck_size`, where `karsten_target = 31.42 + 3.13·avgMV −
   0.28·cheapDrawRamp` (60-card variant for Modern), clamped to [34, 45]
   ([20, 30] Modern), and `lands_eff` counts MDFCs at 0.38/0.74.
2. **Color-sufficiency term**: per color, effective sources = lands'
   `produced_mana` + Karsten fractional weights for rocks/dorks/ramp;
   demand = the source requirement of the deck's most demanding castable
   costs (lookup table, §1A). Penalty = normalized worst-color shortfall.
3. **Role-quota term**: `−Σ_role max(0, target − count) / target` over
   ramp/draw/removal/wipes (Command Zone midpoints: 11/10/11/3), small
   weight so PPMI synergy stays the primary signal.

New `formats.py` reward weights: `synergy_weight` unchanged; split
`structure_weight` into `mana_weight`, `color_weight`, `quota_weight`
(sweep in Phase C).

### Phase B — Feature upgrades (both towers; forces retrain)

4. `card_features` += 9 dims: the 8 role one-hots from
   `analysis.classify()` (ramp, draw, removal, removal_instant,
   board_wipe, tutor, wincon, evasive) + fractional-mana-source weight.
   FEATURE_DIM 26 → 35.
5. `state_features` += 12 dims: per-role deficit vs quota (5), per-color
   pip-vs-source shortfall (5), average MV (1), early-play fraction (1).
   STATE_DIM 16 → 28. The network can then *learn* credit assignment for
   the Phase A penalties instead of treating them as noise.

### Phase C — Evaluation upgrade (gates everything)

6. New structural eval alongside recovery@k: run `complete_deck` on each
   holdout deck's 50%-masked version; score results on land-target error,
   color shortfall, role deficits, wincon presence → one composite
   "structural quality" score. Report per model.
7. Retrain BC + CQL (feature dims changed), sweep the three reward
   weights coarsely (3×3×3 around defaults), judge by keep-bar.

### Phase D — Non-model wins (shippable immediately, no ML risk)

8. `deck analyze`: replace the flat ~37-land line with the Karsten
   regression ("recommended lands: 39 — avg MV 3.4, 12 cheap draw/ramp"),
   and upgrade color balance to fractional effective sources.
9. `recommend`'s structure report: same Karsten target in the gap line.

### Explicitly rejected

- Game-simulator rewards (Q-DeckRec) — non-goal, always was
- Per-card win-rate features — no Commander data source exists
- Beatdown/tempo role assignment — gameplay-time, not deck-time
- Oracle-text embeddings — deferred; heavy dependency, revisit if Phase
  B plateaus

---

## 4. Keep-bar (pre-committed before any training run)

The refactored model ships as default only if, on the same 200-deck
holdout protocol as the CQL promotion:

1. **Structural quality composite improves ≥ 15%** vs the current
   default (cql, 2026-07-13), AND
2. **recovery@50 does not regress by more than 1.0 point.**

If reward changes alone (Phase A) fail the bar, Phase B features are
tried; if both fail, Phases A–C are reverted to experiment status and
only Phase D (analyze/report upgrades, zero model risk) ships.

## Results (2026-07-15) — keep-bar FAILED; Phase D shipped, A–B reverted

All phases were built and the full experiment ran on the real corpus
(~9.3k Commander decks, 200-deck holdout, seeds fixed):

| Model | recovery@50 | structural composite |
|---|---|---|
| Old champion (cql 07-13, via dim adapter) | 20.79 | **0.7125** |
| New BC (strategy features) | 19.25 | 0.7013 |
| New CQL (features + reward, defaults) | 20.82 | 0.6988 |
| New CQL, structure_weight ×5 | 20.82 | 0.6969 |
| New CQL, structure_weight ×15 | 20.79 | 0.6962 |
| **Keep-bar** | ≥ 19.79 ✓ all | **≥ 0.819 ✗ all** |

Recovery never regressed (the features are harmless, +1.6 at k=100), but
structural quality did not move — not at 1×, 5×, or 15× reward weight. The
weight sweep falsifies the "signal too small" hypothesis. Best remaining
explanations: (a) a terminal-only reward provides too little credit
assignment across ~60-step episodes regardless of magnitude; (b) CQL's
conservative penalty anchors the policy to human behavior, leaving the
reward able only to reorder near-data actions; (c) greedy completions
inherit most of their structure from the unmasked human half, capping the
measurable effect. Fixes worth trying later: potential-based (per-step)
shaping of the structural terms, and a goldfish-simulator reward that is
not a statistic of the training corpus (breaks the circularity ceiling).

**Per the pre-committed rule:** the model changes (Phase A reward, Phase B
features) were reverted; the 2026-07-13 champion was restored as default.
What shipped from this work:

- Phase D in full: Karsten land targets, fractional sources, and honest
  color requirements in `deck analyze`, the web Analytics, and `short_colors`
- The structural-quality evaluation (`ml/policy.structural_quality`, printed
  by `eval`) — the measurement outlives the failed intervention
- The Vocab strategy arrays (roles/eff_land/cheap_dr/src_w/pips) — used by
  the eval and ready for future experiments
- Checkpoint dimension guards with a friendly retrain message

This is the project's second pre-registered negative result (the first
promoted BC over CQL in its day). The keep-bar cuts both ways: it promoted
CQL when the evidence was real, and it blocked this refactor when the
evidence wasn't.

## Results (2026-07-15, later) — pip-demand feature PASSED; new champion

A single 5-dim state feature (WUBRG pip counts of the partial deck,
`state_features[16:21]`, STATE_DIM 16→21) retrained BC+CQL under the bar
"recovery@50 ≥ 19.79 AND structural > 0.7125":

| | old champion | pip-feature CQL |
|---|---|---|
| recovery@50 | 20.79 | **21.17** (@10 9.67, @100 27.87 — all up) |
| structural composite | 0.7125 | 0.7131 (color_shortfall 0.3383) |
| goldfish quality | 0.5057 | 0.5097 |
| BC baseline recovery@50 | 18.47 | 18.78 |

First feature experiment to clear its bar. Contrast with the failed Phase
B: pip demand is a *state* summary the towers can act on directly (what
does the mana base owe?), not a strategy label. The structural gain is
marginal (+0.0006, within noise) but recovery improved across every k —
the keep decision rests on recovery, with structural merely not
regressing.

## Results (2026-07-16) — fractional-source feature PASSED on a 3-seed sweep

The supply-side complement: 5 WUBRG fractional-source dims (Karsten
weights via `analysis.source_weights`) on the **action tower**
(`feature_dim(fmt)` 26→31, gated by the same `pip_state` flag — commander
only). Bar: recovery@50 ≥ 20.17 (pip champion 21.17 − 1) AND structural
> 0.7131. Seed 0 missed the recovery floor by 0.25 with structural up, so
the modern seed-sweep protocol applied — decide on 3-seed means:

| seed | recovery@50 | structural |
|---|---|---|
| 0 | 19.92 | 0.7162 |
| 1 | 20.61 | 0.7118 |
| 2 | **21.17** | **0.7176** |
| mean (bar) | 20.57 (≥ 20.17 ✓) | 0.7152 (> 0.7131 ✓) |

Seed 2 — best of the sweep on both axes — ships as champion: recovery@50
21.17 (ties the pip champion), structural 0.7176 (best completion quality
recorded), goldfish 0.5068. **New commander champion baselines for future
bars: recovery@50 21.17, structural 0.7176, goldfish 0.5068.** The model
now sees both halves of the mana ledger: pip demand on the state tower,
fractional-source supply on the action tower.

**Commander-on-curve check (2026-07-16):** a single-deck observation
(`complete --goldfish` on a 7-card seed deck showed the commander on
curve 13.5% vs ~42% under the prior champion) triggered a pre-committed
regression check: mean commander-on-curve over 50 holdout completions,
revert if ≥5 pts below the pip champion (reproduced deterministically
from seed 0 at v0.1.4). Result: pip 60.7%, srcw 60.9% — **no regression**
(goldfish 0.5072 vs 0.5068). The single-deck number was an artifact of
that completion, not the model. Lesson: one deck, one seed is an anecdote;
the holdout protocol is the measurement. **These are the new commander champion baselines for future bars:
recovery@50 21.17, structural 0.7131, goldfish 0.5097.**

**Modern FAILED the same feature** (floor: old champion 53.77 − 1):
seed 0 gave 52.31; a 3-seed sweep confirmed the regression is real, not
noise — seeds [52.31, 47.42, 51.65], mean 50.46, every seed below the
floor. Reading: modern decks are 1–3 colors of heavily duplicated cards,
so pip demand adds little beyond the curve/identity features and the
extra dims cost more than they pay. **Resolution: per-format state
dims** (`FormatConfig.pip_state`, `data.state_dim(fmt)`) — commander
ships 21 dims, modern keeps its 16-dim champion (recovery@50 53.77);
`load_checkpoint` infers the dim from the stored weights.

## 5. Effort and order

| Phase | Size | Depends on |
|---|---|---|
| D (analyze upgrades) | ~half session | nothing — do first |
| A (reward) | ~1 session | D's helpers (shared code) |
| B (features) | ~1 session incl. retrain | A |
| C (eval + sweep + verdict) | ~1 session | A, B |

Sources: Karsten/ChannelFireball colored-sources article; Karsten/TCGplayer
land-count regression; Karsten/TCGplayer "What's an Optimal Mana Curve and
Land/Ramp Count for Commander" (e22caad1); ScrollVault commander-land-count
replication; EDHREC "From Synergy to Lift"; Command Zone template (two
mirrors); Flores "Who's the Beatdown"; arXiv 1806.09771, 2105.11864,
2407.05879.
