# Goldfish Simulator — Design (approved 2026-07-15)

Solitaire simulation of how a deck actually plays: shuffle, mulligan, play
lands, cast what the mana allows, measure. Purpose: (1) a user-facing
`deck goldfish` analysis; (2) a **non-circular reward** for the CQL model —
the signal comes from the game's mathematics, not the training corpus,
breaking the circularity ceiling documented in rl-strategy-research.md
§Results.

## Modeled (v1)

- Real draws over the actual deck; on play/draw
- Karsten's canonical mulligan model, two modes: `karsten2017` (Vancouver
  scry — calibration against his published tables) and `london` (default;
  judge the 7, bottom excess lands above 3 then priciest spells)
- Lands with colors (`produced_mana`) and unconditional ETB-tapped
- Mana rocks/dorks: cast cost, production amount parsed from "Add …"
  clauses (Sol Ring = 2), summoning sickness for creatures
- Land-ramp spells put a land onto the battlefield (tapped)
- Commander castable from the command zone
- Casting policy: play best land (untapped first), then greedy
  largest-castable-first; pips checked per color

## Not modeled (v1 ceilings, marked `ponytail:` in code)

| Simplification | Effect |
|---|---|
| Conditionally tapped lands treated as untapped | slightly optimistic |
| Draw spells don't draw, tutors don't fetch | understates velocity decks |
| Per-color capacity check, not bipartite matching | rare mis-passes on 4+ color costs |
| No activated abilities / combat / opponent | it's a goldfish |

## Metrics → composite score (weights fixed)

`0.4·mana efficiency + 0.3·curve-out rate + 0.2·(1 − dead turns) +
0.1·commander-on-curve`, plus land-drop rates and mulligan counts.

## Calibration (gate for Stage 2)

Pinned by tests (`tests/test_goldfish.py`): in `karsten2017` mode the sim
reproduces Karsten's published P(3 lands by T3) within ±3 points at
3000 games: 25/60 play 90.4% & draw 94.6%, Limited 17/40 91.6%, 24/60
88.7%. **Status: PASSED on first run.** Speed: ~500µs/game (99 cards,
10 turns) → φ(4 games) ≈ 2ms.

## Stage 2 — reward integration (approved; keep-bar gated)

- φ(partial deck) = goldfish composite of the partial (few games, fewer
  turns, common random numbers across the s/s′ pair for variance
  reduction)
- Per-step potential-based shaping: `r += w·(γ·φ(s′) − φ(s))` — dense
  credit, the fix for the terminal-only failure mode of the previous
  experiment
- **Pre-committed keep-bar**: ships as default only if goldfish quality of
  completions improves ≥10% AND recovery@50 regresses ≤1 point vs the
  current champion; else reverts to experiment status like its
  predecessor.
