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
make install     # pip install -e ".[dev]"
make ingest      # download + clean the open datasets (martj42, openfootball)
make test        # 20 tests: leakage guards, label regression, gradient checks
make backtest    # walk-forward backtest 2006-2024 -> reports/backtest_report.md
wc26 predict --date 2026-06-11   # predict that day's WC 2026 fixtures
```

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

## Backtest results (679 tournament matches, walk-forward)

| System | Exact-score hit | Score logloss | Outcome acc |
|---|---|---|---|
| **Dixon-Coles model** | **15.2%** (16.4% holdout) | **2.799** | 54.9% |
| Elo-Poisson baseline | 15.5% (14.2% holdout) | 2.806 | 55.8% |
| Constant modal score | 9.9% | 4.796 | 41.8% |

Exact-score prediction has a hard ceiling (betting markets hit ~11-12%); read
all numbers against baselines. The model leads on the untouched holdout set
and on scoreline log loss everywhere. Full report: `reports/backtest_report.md`.

Confidence labels are calibrated and monotone: high → 17.5% realized hit,
medium → 14.4%, low → 5.9%. (Counter-intuitively, a *very* high modal
probability marks a blowout favourite whose exact margin is inherently
uncertain — those are labelled low.)

## Roadmap

Per [docs/PLAN.md](docs/PLAN.md): Layer-2 context GLM (rest/travel/altitude/
incentives), calibration scalars, fitted ET intensity, the qualification-
scenario incentive engine for MD3, then the minute-by-minute simulator that
becomes the live in-match engine, and player/lineup-level adjustments.

## Data

Free, open sources only: [martj42 international results](https://github.com/martj42/international_results)
(CC0, updated daily), [openfootball worldcup.json](https://github.com/openfootball/worldcup.json)
(CC0). Raw files are downloaded at ingest and never committed. Elo and FIFA-era
ratings are recomputed in-repo for point-in-time correctness.
