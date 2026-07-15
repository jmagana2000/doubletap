# Gameplay Strategy Research → RL Model Refactor Plan

Deep research into Magic: The Gathering deck-building theory, verified
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

## 5. Effort and order

| Phase | Size | Depends on |
|---|---|---|
| D (analyze upgrades) | ~half session | nothing — do first |
| A (reward) | ~1 session | D's helpers (shared code) |
| B (features) | ~1 session incl. retrain | A |
| C (eval + sweep + verdict) | ~1 session | A, B |

Sources: Karsten/ChannelFireball colored-sources article; Karsten/TCGplayer
land-count regression; TCGplayer Commander land analysis; EDHREC
"From Synergy to Lift"; Command Zone template (two mirrors); Flores
"Who's the Beatdown"; arXiv 1806.09771, 2105.11864, 2407.05879.
