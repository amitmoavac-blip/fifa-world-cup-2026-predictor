# World Cup 2026 Exact-Score Prediction Engine — Full Plan

## Context

Build, from scratch in this empty repo, a professional-grade exact-score prediction engine for international football, targeted at the 2026 World Cup (48 teams, 104 matches, USA/Canada/Mexico — the tournament began June 11, 2026). The user's constraints, confirmed by Q&A:

- **Free data sources only** (paid APIs noted as optional upgrades, never required).
- **Build it right, no rush** — quality over catching this tournament's group stage; the system targets the knockout rounds, post-hoc validation, and future tournaments.
- **Live in-match updating is a later phase**, but the architecture must be designed for it from day one.
- Output per match: **one most likely exact score**, a confidence label, an analyst-grade explanation, optionally the modal probability. Internally fully probabilistic: a complete scoreline distribution, argmax selected.

**Honest expectation-setting (a design principle, not a footnote):** exact-score prediction has a hard ceiling. Betting markets hit ~11–12% on exact scores; an elite model reaches ~11–13%. Quality therefore means a *calibrated scoreline distribution* whose argmax beats strong baselines — not a high hit rate in absolute terms. Every report shows baselines alongside.

---

## 1. Exact prediction target definition

- **Group stage:** final score after 90' + stoppage.
- **Knockout:** final score after 120' if the match reaches extra time, otherwise after 90'. Penalty shootouts are excluded — a knockout prediction of `1-1` is valid and will often be the honest modal answer.
- Formally: the model estimates `P(X=x, Y=y)` over a 0–10 grid per team (tail mass ≥10 folded into the top bin, renormalized), where (X, Y) is the regulation score for group matches and the 120-minute score for knockouts. The prediction is `argmax P(x,y)` after calibration.
- Label hygiene: the chosen results dataset records ET-inclusive, shootout-exclusive scores — exactly this target. A regression test asserts ~20 hand-checked known ET matches (e.g., WC 2022 final = 3-3) at ingest, because inconsistent ET labeling would silently corrupt both training and evaluation.

## 2. Modeling architecture (five layers)

### Layer 1 — Team strength: time-decayed Dixon-Coles with confederation shrinkage
The core. For match m, team i vs j, goals (X, Y):

```
X ~ Poisson(λ),  Y ~ Poisson(μ)
log λ = c + a_i − d_j + h·H_m        # H_m = playing in own country and not neutral-flagged
log μ = c + a_j − d_i
```

Joint pmf gets the Dixon-Coles low-score adjustment τ(x,y;ρ) (corrects 0-0/1-0/0-1/1-1 mass — the known failure of independent Poisson). Fit by **penalized weighted MLE** (scipy L-BFGS-B, fits in seconds, refit nightly):

```
max Σ_m w(Δt_m)·ω(importance_m)·log P(x_m, y_m | θ)  −  ridge penalties
```

- Time decay `w(Δt) = exp(−Δt·ln2/halflife)`, halflife grid-searched in {1.5, 2.5, 4} years via backtest.
- Importance information-weights ω (tuned): friendly ≈ 0.4, Nations League ≈ 0.7, qualifier ≈ 1.0, continental finals ≈ 1.25, World Cup ≈ 1.3.
- **Confederation shrinkage:** parameterize `a_i = a_conf(i) + a_offset_i` with a stronger ridge on the offset — debutants (Jordan, Uzbekistan, Cape Verde…) shrink toward their confederation mean instead of getting noise ratings. Sum-to-zero constraint for identifiability.
- Why MLE, not full Bayes: weighted MLE with decay is the published, battle-tested workhorse and gives ~95% of the predictive value; PyMC/Stan hierarchical random-walk is a v2 research comparison, not the foundation.
- Why no bivariate-Poisson λ₃: the shared component is empirically ~0 on international data and destabilizes the fit; DC τ already fixes the draw mass, which is the actual problem.

### Layer 2 — Match-context correction: constrained Poisson GLM (offset model)
`log μ_final = log μ_L1 + θᵀz`, per side. Ridge-penalized, **sign-constrained** (box bounds: rest advantage ≥ 0, altitude-unfamiliarity ≤ 0, …), trained with Poisson deviance on **walk-forward out-of-fold Layer-1 offsets** (Layer 2 never sees in-sample Layer-1 fits — leakage rule).

Features in three honesty tiers:
- **Tier A (reconstructible historically → fully backtested):** rest-day differential (capped ±4), log travel distance since last match, altitude × altitude-familiarity interaction, refined host flag, stage flags (MD1/MD2/MD3/knockout), dead-rubber/incentive flags from the qualification-scenario engine, form residual (goal diff vs Layer-1 expectation, last 10, shrunk).
- **Tier B (2018+ only, with missing-indicator):** xG-residual form (recent goals minus xG, expected *negative* sign — finishing over/under-performance mean-reverts), lineup-strength delta where historical lineups exist.
- **Tier C (live only, never claimed as validated):** injury/absence and confirmed-lineup expert adjustments, **hard-capped at |Δlog μ| ≤ 0.20** (±20% on goal rate), every manual adjustment logged into the prediction snapshot.

Why GLM, not LightGBM: only ~1,500 major-tournament matches have honest Tier-A features; a GBM will overfit and is unexplainable for free. LightGBM is the v2 upgrade once player/lineup features exist at scale.

### Layer 3 — Scoreline distribution + extra time
- 90' matrix: `P(x,y) = τ(x,y)·Pois(x;λ)·Pois(y;μ)` on the 0–10 grid. Negative-binomial dispersion only if backtest total-goals calibration demands it (international totals are near-Poisson).
- **Extra time (knockouts):** `P₁₂₀(x,y) = P₉₀(x,y)` for x≠y; each 90' draw (d,d) is convolved with a 30' Poisson at rates `κ·λ·(30/90)`, `κ·μ·(30/90)`. A **single global κ** is fitted by MLE on pooled ET periods from major tournaments 2000–2024 (~100+ ET periods; expected κ ≈ 0.85–1.1; the data supports no more granularity). Output = argmax of the 120' matrix.
- Blowout guard: cap μ at 4.5 (high-μ Poisson modes go flat and modal scores get silly in 48-team-era mismatches); blowout predictions auto-labeled low-confidence.

### Layer 3b — Minute-by-minute simulator (knockout milestone; becomes the live engine)
State `(t, score_x, score_y, red_diff)`, per-minute intensities:

```
λ_i(t, state) = (μ_i_final / 90) · m(t) · g(score_diff) · r(red_diff) · e(ET_flag)
```

- `m(t)` minute profile, `g` game-state multipliers (trailing teams push, leaders sit), `r` red-card effect (~+0.9 goals/game swing per man), `e` = κ.
- Multipliers fitted from **StatsBomb open event data** (verified coverage: WC 2018/2022, Euro 2020/2024, Copa América 2024, AFCON 2023) shrunk toward club-football priors — these describe football generally, not specific teams, so borrowing is defensible. The minute profile `m(t)` is additionally fit on decades of goal minutes from martj42's `goalscorers.csv`, which also supplies penalty-frequency features.
- **Consistency guard:** base intensities rescaled so the simulator's neutral-state 90' marginal matches the calibrated closed-form matrix (mean and draw-prob within tolerance). The simulator may never silently disagree with the validated model.
- Live updating (later phase) falls out for free: start the simulator from the current `(t, score, reds)`.

### Layer 4 — Calibration: three structural scalars, never the 121-class simplex
Fitted on pooled walk-forward predictions across all backtest tournaments (1,000+ matches):
1. global intercept `c_cal` on log-μ (total-goals bias),
2. temperature `T` on log-μ spread (over/under-confidence in mismatches),
3. draw-inflation: re-fit ρ_cal on out-of-sample predictions.

Dirichlet/temperature scaling on 121 score classes with ~64 matches per tournament is guaranteed noise — rejected by design. Diagnostics (not fitting): reliability curves with bootstrap CIs for modal-score probability, draw probability, P(over 2.5), sliced by stage and rating mismatch.

**In-tournament:** after MD2 (~48 matches), update `c_cal` with shrinkage `n/(n+n₀)`, n₀ = 150 effective matches — group stage informs but cannot whipsaw the model.

### Layer 5 — Selection, confidence, explanation
- **Selection = argmax** of the calibrated matrix (Bayes-optimal under exact-score 0-1 loss, which is the stated objective). Deterministic tie-break (within 0.005): prefer the score consistent with sign(μ_x − μ_y), then lower total goals. Logged.
- **Confidence label** from modal probability, thresholds tuned on backtest so bands are monotone in realized hit rate (start: high ≥ 0.13, med 0.09–0.13, low < 0.09).
- **Explanation:** template-driven from exact, inherently interpretable pieces (see §11). No SHAP needed — GLM deltas are exact.

## 3. Data strategy (free sources)

All sources below were **verified live on June 11, 2026** by web research (availability, coverage, update cadence).

| Data | Importance | Source | Cost | Verified status / notes |
|---|---|---|---|---|
| International results 1872–present: `results.csv` (date, teams, scores incl. ET / excl. shootouts, tournament, **neutral flag**), `shootouts.csv`, `goalscorers.csv` (**scorer, minute, own-goal, penalty flags**) | **Essential** — Layer-1 training set | martj42 GitHub/Kaggle (CC0) | Free | Updated daily, incl. June 11 2026. No importance column — derive from tournament name. `goalscorers.csv` gives goal minutes + penalty flags over decades → minute profiles `m(t)` and penalty-frequency features without event data |
| Self-computed Elo | Essential (baseline + sanity feature) | computed in-repo from results | Free | Never use published eloratings.net snapshots in backtests — they embed future info. Kaggle Elo mirror ("2026 FIFA WC Historical Elo Ratings", nightly) used only as a cross-check of our implementation |
| FIFA rankings history (Dec 1992→) | Useful (baseline only) | Kaggle `cashncarry/fifaworldranking`, cnc8 scraper | Free | Verified through ≥June 2024; check 2025–26 freshness at ingest |
| Event data + xG + lineups | **Essential for Tier-B features + simulator multipliers** | StatsBomb open data (GitHub, attribution license) | Free | Verified competition list: WC 2018, **WC 2022 (with 360)**, Euro 2020, **Euro 2024**, **Copa América 2024**, **AFCON 2023**, plus historical WCs 1958–1990 — 6 modern majors, more than assumed |
| Tournament xG/shots/possession tables, lineups, referees; **live WC 2026 stats pages (already up)** | Essential (Tier-B extension + in-tournament stats) | FBref (Opta) | Free | Hard limit **10 req/min** (ban risk) — use `soccerdata` package + heavy caching, nightly cron |
| 104-match schedule with venues, groups, kickoffs | **Essential** | openfootball `worldcup.json` (CC0, raw GitHub JSON) + Football-Data.org free tier as redundancy (10 calls/min) | Free | 2026 file verified live, ~daily manual updates (hours of lag on results — results truth comes from martj42/FBref) |
| Venue facts: altitude, **roof/climate-control flag**, local kickoff tz; R32 best-thirds rules | **Essential** | hand-entered config (facts, not data) | Free | Rules encoded as code |
| Player market values + club minutes | Useful (squad-quality feature) | Transfermarkt via `dcaribou/transfermarkt-datasets` Kaggle mirror (updated June 2026) | Free | Mirror is club-centric: use for player values/minutes; build WC squad lists by joining FBref/StatsBomb squad pages — national-team tables aren't fully mirrored |
| Confirmed lineups + match events (goals/cards/subs) for WC 2026 | Useful (Tier C + later live phase) | API-Football **free tier: 100 req/day, all endpoints** (lineups ~20–40 min pre-kickoff); FBref next-day as backup | Free | 100 req/day comfortably covers ~4–6 matches/day; Pro $19/mo is the optional upgrade if live polling is ever wanted |
| Weather forecast + ERA5 history by venue coords | Optional | Open-Meteo (free non-commercial, 10k calls/day, no key) | Free | Only for open stadia at daytime kickoffs |
| Historical correct-score odds | Optional (ceiling benchmark only) | Betfair Historical Data (Basic free/cheap; correct-score markets) — football-data.co.uk is **club-only**, ruled out | Free-ish | Never a feature; do not block on it |
| Benchmark win probabilities | Optional | Opta Analyst "supercomputer" WC 2026 page (free, no bulk download) | Free | Sanity cross-check only; 538/SPI is dead (shut down 2025) |
| FotMob / Sofascore xG & lineups | Backup only | unofficial APIs | Free | Sofascore ToS explicitly bans scraping; never on the critical path |
| Referee tendencies | **Skipped** | — | — | Pure noise at international sample sizes |

**Team registry:** a canonicalization module mapping name variants ("USA"/"United States", Czechia, successor states like Serbia/Yugoslavia) to stable IDs with validity dates. The single most likely source of silently wrong ratings — gets its own tests.

**Point-in-time discipline (anti-leakage core):** every feature function has signature `f(match_id, asof_ts)` and may only read rows with `event_ts < asof_ts`. Backtests pass historical timestamps; live passes now. One code path, no leakage by construction.

**Missing/stale data → confidence:** missing-indicator columns for Tier B; staleness (e.g., team's last competitive match > 9 months ago) widens Layer-1 ridge-implied uncertainty and demotes the confidence label; the explanation states it.

## 4. Feature engineering plan

Covered by the tier system in §2-Layer-2. Key engineering details:
- **Form** is a *residual* vs Layer-1 expectation (not raw results — that double-counts strength), shrunk toward 0.
- **Altitude:** applies to Mexico City (2240m), Guadalajara (1566m), Monterrey; interacted with team altitude-familiarity (CONMEBOL/CONCACAF vs others). Tight ridge — small effect, must not be allowed to be big.
- **Heat:** venue-and-kickoff aware, never city-aware. Dallas, Houston, Atlanta, Vancouver are roofed/climate-controlled; naive "Texas in June" features are wrong for exactly the hottest cities. Applies only to open stadia (Miami, Kansas City, Philadelphia, NY/NJ…) at daytime kickoffs.
- **Host advantage in 2026 is special:** three co-hosts, and US venues will often host effectively *away* crowds (e.g., Mexico-supporting crowds in LA/Houston). Per-host-country coefficient shrunk toward the pooled historical host effect, with a halved prior for USA; host-residual monitor after MD1 with a pre-committed adjustment rule.
- **Incentives:** a qualification-scenario engine enumerates group outcomes at prediction time (12 groups of 4, top 2 + 8 best thirds → narrow losses/low-scoring draws can have positive value on MD3) and emits incentive/dead-rubber flags. Mandatory before MD3, not polish.
- **Fatigue/club minutes, GK quality, set-piece strength:** v2 features (player-level data project); set-piece and penalty tendencies enter v2 via StatsBomb event aggregates.

## 5. Generating scoreline probabilities internally
§2 Layers 1–4: DC-adjusted Poisson matrix from calibrated (λ, μ), 0–10 grid with tail folding, ET convolution for knockouts. The matrix is the single internal source of truth; every output (modal score, confidence, win/draw probabilities for diagnostics) derives from it.

## 6. Selecting the single final score
Argmax of the calibrated matrix (120' matrix for knockouts), deterministic tie-breaking, confidence from modal probability + monotonicity-tuned thresholds. §2-Layer-5.

## 7. Live updating method
- **Pre-tournament:** nightly Layer-1 refit; predictions for all 104 matches from historical data + squad values.
- **Pre-match epochs, each snapshotted:** `pre_lineup` (form, injuries, rest/travel/venue/weather) → `post_lineup` (confirmed XI → Tier-C capped adjustment) → kickoff freeze.
- **In-match (later phase):** the Layer-3b simulator started from current `(minute, score, red cards)`, optionally modulated by live xG/shots; same engine, zero new modeling.
- **Post-match:** results appended → nightly refit updates ratings; prediction vs outcome logged; MD2 calibration-intercept update with shrinkage; all future-match predictions regenerate.
- Snapshots are append-only JSONL: `{match_id, epoch, asof_ts, model_version (git sha + config hash), μ_x, μ_y, matrix_topk, modal_score, modal_prob, confidence, explanation, manual_adjustments[]}` — enabling exactly the pre-lineup/post-lineup/live/final comparisons required.

## 8. Extra-time handling
§2-Layer-3: identity on non-draws, 30' κ-scaled Poisson convolution on 90' draws; κ fitted globally on pooled 2000–2024 ET periods (golden-goal-era handled separately/excluded). v2: the simulator runs ET minutes explicitly with fatigue-adjusted `m(t)`, replacing the convolution with the same consistency guard.

## 9. Backtesting strategy
- **Walk-forward over majors:** for each tournament in {WC 2006→2022, Euros 2008→2024, Copa 2019/2021/2024, AFCON & Asian Cup editions}: train on all internationals strictly before its first match; predict its matches sequentially with rating updates from already-played tournament matches (mirrors live operation).
- **Hyperparameter hygiene:** halflife, ω, ridge strengths selected on tournaments ≤ 2018 only; 2022–2024 held out untouched for the final report.
- **Metrics:** exact-score hit rate (primary); scoreline log loss on the capped grid; RPS on derived 1X2; per-team goal MAE; total-goals MAE; calibration diagnostics. Sliced by stage, rating-gap terciles, group vs knockout, 90' vs 120' targets.
- **Baselines:** (a) constant modal score (~9–10% hit rate — surprisingly hard to beat, and the report says so), (b) independent Poisson from self-computed Elo, (c) FIFA-ranking Poisson, (d) market correct-score odds as ceiling if cheaply available. Success = beat (b) on log loss + hit rate, approach (d).
- **Ablations:** `--features tier_a|tier_ab|all` so the report shows exactly what each feature tier buys. Tier C is never claimed as validated.

## 10. Calibration strategy
§2-Layer-4: three structural scalars on pooled walk-forward predictions; reliability diagnostics with bootstrap CIs; in-tournament shrunk intercept update after MD2. Per-tournament calibration curves are noise and are not fitted to.

## 11. Interpretability strategy
Every prediction's explanation is assembled from exact quantities, not post-hoc ML explanations:
- Layer-1 rating gap → "Argentina's attack vs France's defense implies 1.6 vs 1.1 expected goals."
- Layer-2 GLM deltas are exact per-feature contributions → "one extra rest day and no travel: +0.06 log-goals."
- Tier-C lineup/injury adjustments reported as explicit before/after μ ("confirmed XI moved France from 1.31 to 1.18").
- Draw-adjustment and ET notes for knockouts ("38% chance this needs extra time; ET adds 0.4 expected goals").
- Uncertainty flags: stale data, debutant shrinkage, blowout-flatness, inherently high-variance matchups (high total-μ, small gap).
Epoch snapshots make "how much did lineups/injuries/form move the prediction" a query, not an estimate.

## 12. MVP version (prediction quality first)
1. Ingest + team registry + self-computed Elo + label verification tests.
2. Layer-1 weighted-MLE Dixon-Coles with confederation shrinkage.
3. Closed-form score matrix + argmax + confidence + template explanation.
4. Layer-2 GLM with Tier-A features (rest/travel/altitude/host/stage/incentive/form-residual).
5. Full walk-forward backtest harness with baselines, metrics, ablations, hyperparameter selection.
6. Calibration scalars + diagnostics.
7. Snapshot system + CLI (`wc26 ingest | fit | predict --date | backtest | report`).
8. ET handling + κ fit + 120' targets (knockout-ready).
9. Qualification-scenario incentive engine (MD3-ready).

## 13. Later advanced version
- Minute-by-minute simulator as primary engine; **live in-match updating** from it.
- Per-player lineup index (market values + club minutes + position weights), GK quality as separate defensive adjustment, key-absence marginal values.
- xG auxiliary likelihood / finishing-quality layer; set-piece & penalty tendencies from event data.
- LightGBM Layer 2 (once player features exist), monotonic constraints.
- Full-Bayes (PyMC) Layer 1 as research comparison; negative-binomial dispersion if diagnostics demand.
- Optional cheap API (API-Football/Sportmonks) for confirmed lineups + live events if budget ever changes.

## 14. Main risks and mitigations
1. **Label corruption on ET scores** → hand-checked regression tests at ingest.
2. **Elo/rating leakage in backtests** → everything point-in-time recomputed in-repo; `asof_ts` API everywhere.
3. **Host-advantage misspecification (3 co-hosts, away-crowd venues)** → shrunk per-host coefficients, halved USA prior, MD1 residual monitor with pre-committed rule.
4. **Heat/altitude feature wrongness** → venue config with roof/AC flags; effects only where physically real; tight ridge.
5. **48-team format novelty** (debutant mismatches, best-thirds incentives) → confederation shrinkage, μ cap 4.5, scenario engine.
6. **Overfitting tiny samples** → tiered features, sign constraints, ridge everywhere, scalars-only calibration, ≤2018 hyperparameter wall.
7. **Team-name drift / duplicate fixtures** → registry + ingest invariant tests.
8. **Misread success criteria** → baselines and the ~12% ceiling stated in every report.
9. **Tier-C subjectivity** → hard ±20% cap, full logging, never claimed as validated.

## 15. Data sources — verification outcome (researched June 11, 2026)
The full per-source table is in §3. Key verified facts that shaped the plan:
- **martj42 results dataset is alive and updated daily** (commits June 9/10/11, 2026) — safe as the training backbone during the tournament. Its `goalscorers.csv` (goal minutes, penalty flags, decades of coverage) is a quiet gem: it lets us fit minute profiles and penalty-frequency features far beyond the event-data era.
- **StatsBomb open data covers 6 modern majors** (WC 2018/2022, Euro 2020/2024, Copa América 2024, AFCON 2023) — more than assumed; game-state/red-card multipliers and the xG-residual feature get a broader fit.
- **FBref already has live WC 2026 stat pages** (Opta xG, lineups, referees) — daily in-tournament advanced stats are free, subject to the 10 req/min limit (use `soccerdata` + caching).
- **openfootball worldcup.json** (CC0) is the machine-readable 2026 schedule; Football-Data.org free tier is the redundancy feed.
- **API-Football's free tier includes all endpoints at 100 req/day** — confirmed lineups ~20–40 min pre-kickoff fit the free-only constraint; its $19/mo Pro tier is the documented upgrade path if the live phase ever wants real-time polling.
- **football-data.co.uk has no international odds** (club only) — the odds ceiling-benchmark, if pursued, comes from Betfair Historical Data (has correct-score markets). Optional; never blocks.
- **538/SPI is dead** (platform shut 2025); Opta Analyst's free WC 2026 probabilities serve as an external sanity cross-check only.
- Sofascore/FotMob scraping is a ToS risk (Sofascore explicitly bans it) — backup only, never critical path.

## 16. Repo structure & first build phase

```
fifa-world-cup-2026-predictor/
├── pyproject.toml                  # package: wc26; numpy scipy pandas typer pyyaml matplotlib
├── Makefile                        # make ingest / fit / predict DATE=… / backtest / report
├── configs/  model.yaml  wc2026.yaml  backtest.yaml
├── data/     raw/ (gitignored)  processed/  snapshots/ (append-only JSONL)
├── src/wc26/
│   ├── data/      ingest.py schema.py registry.py elo.py
│   ├── features/  context.py form.py incentives.py store.py (asof API)
│   ├── ratings/   dixon_coles.py priors.py
│   ├── adjust/    glm.py
│   ├── scoreline/ matrix.py extra_time.py simulate.py (phase 2)
│   ├── calibrate/ scalars.py diagnostics.py
│   ├── predict/   engine.py select.py explain.py snapshot.py
│   ├── backtest/  walkforward.py metrics.py baselines.py report.py
│   └── cli.py
└── tests/         # registry, asof-leakage, matrix invariants, known-ET-score regression
```

**Recommended first build phase (maximizes prediction quality earliest):** Steps 1–5 of §12 as one vertical slice — ingestion with verified labels, Layer-1 Dixon-Coles, score matrix + argmax, and the *backtest harness with baselines* — because until the harness exists, no other improvement is measurable. Calibration, Tier-A GLM, ET handling, and the incentive engine follow in that order, each justified by a measured backtest delta.

**Verification:** `make backtest` reproduces the full walk-forward report (hit rates vs baselines, calibration curves); `wc26 predict --date <date>` emits predictions + explanations for that day's WC 2026 fixtures; tests cover registry canonicalization, asof-leakage guards, matrix invariants (sums to 1, tail folding), and known ET scores.

## Next implementation prompt (suggested)
> "Implement Phase 1 of the plan: repo scaffold, data ingestion with team registry and ET-label verification tests, self-computed Elo, the weighted-MLE Dixon-Coles model with time decay/importance weights/confederation shrinkage, the DC-adjusted score matrix with argmax selection, and the walk-forward backtest harness with the constant-modal-score, Elo-Poisson, and FIFA-ranking baselines. Run the backtest on WC 2014–2022 and show me the report."
