# Magic: The Gathering Gameplay Knowledge → Blindspot Analysis

## The elicitation prompt

The following prompt was used to enumerate gameplay knowledge and audit
DoubleTap against it. Re-run it after major feature work to find new gaps.

> You are a competitive Magic: The Gathering deck-building coach. Enumerate
> everything that decides whether a deck functions in real games, organized
> by: (1) win conditions and how each is actually assembled; (2) resource
> systems (mana curve, color requirements, card advantage, card selection);
> (3) interaction (what to answer, at what speed, at what cost); (4) combat
> dynamics (evasion, board presence, protecting key pieces); (5) consistency
> (tutors, redundancy, land counts vs curve). For each item, state what a
> deck-analysis tool could measure from card data alone (types, mana costs,
> oracle text, keywords) and what it cannot. Then compare against this tool's
> current checks — roles: ramp/draw/removal/board-wipe/wincon/big-threat;
> counts vs Commander quotas; land count; market price — and list the
> highest-impact measurable gaps.

## Findings (blindspots, ranked by impact)

| # | Gap | Why it matters in games | Measurable from card data? |
|---|-----|------------------------|---------------------------|
| 1 | **Mana curve not reported** | A deck of great cards you can't cast until turn 5 loses to a mediocre deck that acts on turns 1–3. Curve is the single strongest structural predictor of "does this deck do something every turn." | Yes — `cmc` per card |
| 2 | **Color balance unchecked** | 10 Mountains in a deck full of `{B}{B}` costs means uncastable hands. The check is pips-in-costs vs lands-producing-that-color. | Yes — `mana_cost` pips vs `produced_mana` on lands |
| 3 | **Poison and mill wincons invisible** | README documents five ways to win; `analyze` detected only two (combat, "you win"). An infect or mill deck was reported as "no way to win detected" — actively wrong. | Yes — Infect/Toxic keywords, mill text |
| 4 | **Interaction speed ignored** | 10 sorcery-speed removal spells and 10 instants are not the same deck. Instant-speed answers hold up mana and answer combos; sorceries can't. | Yes — type line Instant / Flash keyword among removal |
| 5 | **Evasion unmeasured** | Ground creatures get chump-blocked in multiplayer; "big creatures" only win if they connect. Flying/trample/menace decide that. | Yes — keywords |
| 6 | **Tutors/consistency not counted** | Singleton formats live on redundancy or tutors; neither was measured (Game Changers covers only premium tutors). | Yes — "search your library" text |
| 7 | Card selection (scry/surveil) vs raw draw | Selection smooths draws; only raw draw was counted. | Yes — keywords/text |
| 8 | Commander protection | Commander-centric decks fold if the commander dies thrice. | Partially — grant-text regex |
| 9 | Threat assessment / politics | Multiplayer reads, when to hold removal. | No — play skill, not deck data |
| 10 | Mulligan decisions, sequencing | Gameplay skill. | No |

Items 9–10 are knowably unmeasurable (true "unknown unknowns" resolved to
known limits). Items 7–8 are lower-impact; deferred. Items 1–6 are
implemented below.

## The plan (applied)

1. `analysis.py`: `curve_stats()` — nonland MV histogram, average MV, early-play count (MV ≤ 2).
2. `analysis.py`: `mana_balance()` — colored pips in spell costs vs land color sources; flags underserved colors.
3. `analysis.py`: new roles — `poison` (Infect/Toxic), `mill` (mill text) folded into "Ways to win"; `tutor` (library search excluding land-ramp); instant-speed interaction counted within removal.
4. `analysis.py`: `evasive` role — keyword evasion (Flying, Trample, Menace, …) on creatures.
5. `cli.py deck analyze`: report curve, color balance, interaction speed, and the expanded win-condition detection.
6. Tests for every new classifier and stat; CLI test asserts the new sections render.
