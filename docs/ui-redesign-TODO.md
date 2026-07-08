# UI Redesign — Triage & TODO

The redesign brief asked for more than the application currently supports.
This file tracks what was delivered, what's deferred (buildable on existing
data), and what's out of scope until the app itself grows the feature.

## Delivered (Deck Builder pass, this round)

- Design system: obsidian/gold arcane theme, five-mana palette, Cinzel/Inter
  type, glassmorphism panels, gold hairline borders, ambient particle motes
  (disabled under `prefers-reduced-motion`)
- Deck Builder screen: filter rail (glowing WUBRG buttons, type chips,
  mana-value slider), searchable card grid with real Scryfall art, hover
  lift/glow, loading skeletons, designed empty states
- Deck shelf panel: cards grouped by type, hover quantity steppers,
  commander line, action toolbar (show/validate/analyze/bracket/price)
- Sticky deck summary: count, color identity pips, mini mana curve,
  live legality badge
- Card detail modal: large card image, oracle text, mana cost, price,
  add/remove/make-commander, animated open/close, Esc to dismiss
- Keyboard: `⌘K` or `/` focuses search; debounced-as-you-type results
- New structured endpoints: `GET /api/cards` (filtered browse),
  `GET /api/deck` (deck detail + violations); mutations still route through
  `/api/run` so CLI parity holds
- Mobile: rails collapse, nav shrinks to runes, modal stacks

## Delivered (remaining-screens pass)

- **Home**: cinematic hero with the most recent deck's commander art,
  featured-deck line, stat tiles (decks / cards / commanders / formats),
  recent-decks strip — all live from `/api/decks`
- **Deck Gallery**: art-backed deck cards (commander `art_crop`, identity
  pips, format, count); click-through opens the deck in the Builder
- **Analytics**: per-deck curve chart, cost-vs-lands color balance bars
  (short colors flagged red), type breakdown with land/spell ratio,
  roles-vs-targets bars, ways-to-win + violations panel, bracket badge and
  market price — backed by the new `GET /api/analysis` endpoint
- `/api/decks` now carries commander art, identity colors, and mtime

## Deferred — buildable now on existing data, not yet designed

| Item | Backing data that already exists |
|---|---|
| Full command palette (actions, not just search focus) | `/api/run` |
| Recommendations rendered as card tiles instead of text | `recommend` output + `/api/cards` |

## Out of scope — the application has no such feature

| Brief item | Why |
|---|---|
| Win rate on deck cards | No game-result tracking anywhere in the app |
| Card tags | No tagging system in the data model |
| Deck notes | Deck JSON has no notes field |
| React/Next.js + Tailwind + shadcn/ui + Framer Motion | Deliberate architecture decision, not a gap: the UI ships inside the Python package with zero build toolchain; a Node stack would break `uv tool install` distribution. The design goals were met in a self-contained SPA instead |
