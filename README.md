# FIFA World Cup 2026 — Exact-Score Prediction Engine

A professional-grade prediction system for international football that outputs,
for any World Cup match, **the single most likely exact final score**, a
confidence label, and an analyst-grade explanation — backed internally by a
fully probabilistic scoreline distribution.

```
Mexico vs South Africa  (Mexico City)
  Predicted final score : Mexico 1 - 0 South Africa
  Confidence            : medium (modal probability 13.5%)
  Why                   : Mexico are favourites on opponent-adjusted attack/defense
                          ratings (1.47 vs 0.88 expected goals). Mexico benefit from a
                          home-country venue (+33% on goal rate). Most likely score 1-0
                          at 13.5%; outcome split 50%/27%/22% (win/draw/loss).
```

**Target definition.** Group stage: score after 90' + stoppage. Knockouts: score
after 120' if the match reaches extra time. Penalty shootouts are excluded — a
`1-1` knockout prediction is valid and often the honest modal answer.

The full design (modeling architecture, data strategy, backtesting protocol,
risks) is in [docs/PLAN.md](docs/PLAN.md).

## Quickstart

```bash
make install     # pip install -e ".[dev]"  (includes Flask for the UI)
make ingest      # download + clean the open datasets (martj42, openfootball)
make test        # full test suite: leakage guards, label regression, gradient + UI checks
make backtest    # walk-forward backtest 2006-2024 -> reports/backtest_report.md
wc26 predict --date 2026-06-11   # predict that day's WC 2026 fixtures
make serve       # build the prediction artifact + launch the analyst UI at :8000
```

## Analyst UI

A thin, read-only Flask app for viewing and evaluating predictions — professional
and narrow by design (one exact score per match, never a betting/dashboard app).

```bash
wc26 build-ui-data        # fit once, predict all fixtures -> data/processed/ui_predictions.json
wc26 serve                # http://127.0.0.1:8000  (use --rebuild to refresh the artifact)
```

Four pages: **Matches** (all 104 fixtures, filterable by status/stage/team/
confidence; predicted score, confidence, status, actual + exact-hit when played),
**Match detail** (predicted score front-and-centre, explanation, collapsible
analyst section with expected goals + top-5 internal scorelines + extra-time
handling, and a per-match prediction-history timeline), **Evaluation** (model vs
baselines by stage/holdout, confidence calibration, and the *disabled*
experimental layers with their measured null results), and **Data health** (sync
times, source status, and an explicit "live/lineup data: not configured"). The UI
reads a precomputed artifact, so the engine stays the single source of truth and
freshness is honest. Uncertainty (stale ratings, capped blowouts, dead rubbers)
is surfaced, never hidden.

## How it works (Phase 1)

1. **Layer 1 — time-decayed Dixon-Coles** attack/defense ratings, penalized
   weighted MLE over ~20 years of all internationals: exponential time decay,
   match-importance information weights (friendly 0.4 → World Cup 1.3),
   confederation shrinkage for small-sample teams, host advantage, and the
   Dixon-Coles low-score adjustment. Analytic gradients; fits in seconds.
2. **Scoreline matrix** on a 0-10 grid with tail folding; knockout matches get
   an extra-time convolution (90' draws redistributed through a 30' reduced-
   intensity Poisson process), producing the 120-minute distribution.
3. **Selection**: argmax of the matrix (Bayes-optimal under exact-score 0-1
   loss), deterministic tie-breaking, interval-band confidence tuned on the
   backtest, template explanations built from exact model quantities.
4. **Walk-forward backtest** over 15 tournaments (WC 2006-2022, Euros
   2008-2024, Copa América 2015-2024): refit before every match date, no
   leakage by construction (`asof` discipline everywhere, Elo recomputed
   in-repo, ET-score labels regression-tested against hand-checked matches).

## Backtest results (679 tournament matches, walk-forward, calibrated)

| System | Exact-score hit | Score logloss | RPS (1X2) | Total-goals MAE |
|---|---|---|---|---|
| **Dixon-Coles model** | **15.8%** | **2.794** | **0.1941** | **1.51** |
| Elo-Poisson baseline | 15.5% | 2.806 | 0.1944 | 1.53 |
| Constant modal score | 9.9% | 4.796 | 0.2365 | 1.65 |

Exact-score prediction is inherently hard — the best *constant* guess (the
historical modal score) tops out near 10% in our data, so read all numbers
against baselines, not intuition. (A common rule of thumb puts strong
models/markets in the low-to-mid teens; we have no market-odds data in-repo to
verify it, so it's context, not a measured ceiling.) The calibrated model leads
the Elo-Poisson baseline on every metric. Full report: `reports/backtest_report.md`.

### What each layer buys (honest ablations)

- **Calibration** (3 scalars): on the untouched holdout, log loss 2.780 → 2.776
  and total-goals MAE 1.553 → 1.491. Modest and real.
- **Extra-time κ**: fitted at ~0.90 from 268 complete-goal ET matches
  (`wc26 fit-kappa`), confirming the prior rather than guessing.
- **Layer-2 context GLM** (rest, form-residual): leave-one-tournament-out CV
  shows **−0.05%** per-side deviance — negligible. International goal-based
  context features don't move exact scores; the layer is built and tested as
  the integration point for Tier-B features (off by default, `adjust.use_glm`).
- **Tier-B xG-form** (StatsBomb open data, 6 tournaments): within-tournament,
  opponent-adjusted xG residuals with the theoretically-correct coefficient
  signs (chance creation persists `+`, finishing reverts `−`). LOTO-CV verdict:
  exact-hit **unchanged** (16.0%), log loss flat-to-slightly-worse, total-goals
  MAE marginally better (~0.01), a knockout-only hint (12.8%→14.0%, ~1 match).
  The window (1–4 matches) is too short to move the exact-score distribution, so
  this also ships **off by default** — a clean negative result with the pipeline
  ready for broad-calendar xG. See `wc26 eval-xg` / `reports/tier_b_report.md`.

### Why lineup / player-quality / injury features aren't here yet

These were prioritized, but there is **no free, leakage-clean historical source**
of (expected XI, actual XI, per-player quality, availability) across the
international calendar. StatsBomb gives starting XIs for 6 tournaments but no
player-quality metric, and joining Transfermarkt values historically is a
fragile name-matching project with its own leakage risks. Per the evaluation
gate — *don't keep features you can't validate* — these are deferred rather than
shipped unproven. They become viable for **live** 2026 use (API-Football free
tier provides confirmed lineups ~20–40 min pre-kickoff) and that is where they
belong: the simulator/live phase, not the backtested pre-match model.

Confidence labels are calibrated and monotone: high → 17.5% realized hit,
medium → 14.4%, low → 5.9%. (Counter-intuitively, a *very* high modal
probability marks a blowout favourite whose exact margin is inherently
uncertain — those are labelled low.)

## Qualification-scenario engine (matchday 3)

`wc26.features.incentives` reconstructs group standings from results and
classifies each team's matchday-3 situation (secured top-2 / contention /
third-place contention / eliminated) under the 2026 format (top 2 + 8 best
thirds). It powers a dead-rubber flag and the analyst explanation; it is
deliberately *not* fed as a fitted goal feature, since the Layer-2 ablation
showed goal-based context features add nothing and we don't overclaim.

## Roadmap

Done: extra-time κ fit, calibration scalars, Layer-2 GLM (measured null for
goal-based features), qualification-scenario engine, Tier-B xG-form layer
(measured null over the short within-tournament window; pipeline ready for
broad-calendar xG). Next per [docs/PLAN.md](docs/PLAN.md): the minute-by-minute
simulator that becomes the live in-match engine, with lineup/availability
features entering there (live data, not backtestable history). A broad-calendar
xG source (e.g. FBref) remains the most promising path to make Tier-B earn its
place in the pre-match model.

## Data

Free, open sources only: [martj42 international results](https://github.com/martj42/international_results)
(CC0, updated daily), [openfootball worldcup.json](https://github.com/openfootball/worldcup.json)
(CC0). Raw files are downloaded at ingest and never committed. Elo and FIFA-era
ratings are recomputed in-repo for point-in-time correctness.
