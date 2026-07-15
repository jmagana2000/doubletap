# Domain Expert Registry

Subject-matter experts whose published work grounds DoubleTap's analysis
and reward design. Rule of ingestion: experts publishing under
`tcgplayer.com/content/author/` get their quantitative work ingested into
the knowledge base (docs/rl-strategy-research.md); others are recorded
here with their canonical contributions and home venues.

## Frank Karsten — TCGplayer author ✓ (primary quantitative source)

- **Author page**: tcgplayer.com/content/author/frank-karsten
- **Credentials**: MTG Hall of Fame; ~80 Pro Tours; **PhD in game theory
  and probability mathematics** — the only MTG strategy writer whose
  numbers come with proofs, which is why his claims consistently survive
  adversarial verification
- **Canon (ingested)**:
  - *Colored-source requirements* (ChannelFireball): the (89+M)%
    castability threshold and per-cost source tables (1 pip = 19 sources
    in Commander, CC = 30, CCC = 36); fractional source weights (dorks
    0.5, rocks 0.75, MDFC 0.8/1.0, cantrips ~0.25)
  - *Land-count regressions* (TCGplayer): 60-card
    `19.59 + 1.90·avgMV − 0.28·cheapDrawRamp`; Commander
    `31.42 + 3.13·avgMV − 0.28·cheapDrawRamp`; MDFCs as 0.38/0.74 lands
  - *Commander curve/ramp simulations* (TCGplayer, e22caad1): 42 lands +
    Sol Ring baseline, cut 1 per 2–3 rocks, floor 37; optima shift with
    commander cost (2-MV commander → 42 lands/0 rocks; 6-MV → 38/10);
    long-game curves want zero 1-drops and 13–14 Signets
  - *Hypergeometric fundamentals*: 14 single-color sources → 86.1% in
    opening 7; 18 sources → 68.6% for CC; the machinery behind all of the
    above
- **Mulligan math and Limited work (ingested 2026-07-15**, primary text
  read directly): the canonical mulligan model behind his tables, 60-card
  land-drop probability tables, the draw-is-worth-3-lands asymmetry, the
  original 2017 regression (`16 + 3.14·avgCMC`, R²=0.614) and its lineage
  to the 2022 update, the 17-lands-in-Limited justification, and 40-card
  source tables — see docs/rl-strategy-research.md §A2
- **Still not ingested** (low relevance): set-specific Limited tier
  lists, metagame number-crunching, tournament reports
- **Credentials precision** (from his own bio): PhD in *cooperative game
  theory and stochastic operations research*; Pro Tour Hall of Fame 2009
- **Used in DoubleTap**: `analysis.karsten_land_target`, `SOURCES_NEEDED`,
  `source_weights`, `effective_lands` — surfaced in `deck analyze`, the
  web Analytics, and `recommend`'s gap report

## Mike Flores — StarCityGames (not on the TCGplayer author path)

- **Canonical work**: *"Who's the Beatdown?"* (1999) — the most-cited MTG
  strategy essay: every matchup reduces to beatdown vs control roles, and
  misassigning your role is the classic way to lose a winnable game
- **DoubleTap disposition**: role assignment is piloting skill, not deck
  construction — documented as knowably out of scope
  (docs/rl-strategy-research.md §1D)

## Patrick Chapin — StarCityGames/books (not on the TCGplayer author path)

- **Canonical work**: *Next Level Magic* / tempo theory — tempo as
  advantage in per-turn renewable resources (mana) vs card advantage as
  stock resources
- **DoubleTap disposition**: gameplay-time concept; its deck-time shadow
  (mana efficiency, curve) is covered by Karsten's math

## Jimmy Wong & Josh Lee Kwai — The Command Zone (YouTube/EDHREC)

- **Canonical work**: the Command Zone deckbuilding template (ep. 379,
  revised "new era" ep. 658): 36–38 lands, 10–12 ramp, 10 card draw,
  10–12 targeted removal, 3–4 board wipes
- **Used in DoubleTap**: `analysis.COMMANDER_TARGETS` and the roles-vs-
  targets panels in `deck analyze`/Analytics

## EDHREC data team — edhrec.com

- **Canonical work**: the *lift* recommendation metric
  (`P(A∩B)/(P(A)·P(B))`, log-scaled), replacing their asymmetric
  "synergy" score
- **Relevance**: DoubleTap's PPMI reward is log-lift with α-smoothing —
  same family, independently validating the synergy term

## ScrollVault — scrollvault.net (tooling/replication, not an author)

- **Contribution**: 3.75M-game Monte Carlo replication of Karsten's
  Commander land research (per-archetype optima: cEDH 29–31 → 
  battlecruiser 38–40); implements Karsten math as calculators
- **Relevance**: independent validation of the regression DoubleTap ships;
  their goldfish methodology is the blueprint for the proposed
  non-circular simulator reward

---

*Maintenance*: when citing a new source in the knowledge base, check for a
`tcgplayer.com/content/author/<name>` page; if present, ingest their
quantitative work and add them above with status ✓.
