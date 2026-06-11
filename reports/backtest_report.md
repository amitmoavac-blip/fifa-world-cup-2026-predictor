# Walk-forward backtest report

Target: exact final score (90' group stage; 120' knockouts, shootouts excluded).
Exact-score prediction has a hard ceiling — betting markets hit ~11-12%. Read
every number against the baselines below, not against intuition (plan section 9).

## All tournaments
| System | N | Exact-score hit | Score logloss | RPS (1X2) | Outcome acc | Total-goals MAE | GD MAE |
|---|---|---|---|---|---|---|---|
| Dixon-Coles model | 679 | 15.2% | 2.799 | 0.1953 | 54.9% | 1.57 | 1.21 |
| Elo-Poisson baseline | 679 | 15.5% | 2.806 | 0.1944 | 55.8% | 1.53 | 1.20 |
| Constant modal score | 679 | 9.9% | 4.796 | 0.2365 | 41.8% | 1.65 | 1.50 |

## Selection set (hyperparameters tuned here)
| System | N | Exact-score hit | Score logloss | RPS (1X2) | Outcome acc | Total-goals MAE | GD MAE |
|---|---|---|---|---|---|---|---|
| Dixon-Coles model | 453 | 14.6% | 2.809 | 0.1964 | 55.0% | 1.58 | 1.22 |
| Elo-Poisson baseline | 453 | 16.1% | 2.814 | 0.1956 | 55.6% | 1.49 | 1.21 |
| Constant modal score | 453 | 9.3% | 4.796 | 0.2388 | 41.9% | 1.62 | 1.51 |

## Holdout set (untouched by tuning)
| System | N | Exact-score hit | Score logloss | RPS (1X2) | Outcome acc | Total-goals MAE | GD MAE |
|---|---|---|---|---|---|---|---|
| Dixon-Coles model | 226 | 16.4% | 2.780 | 0.1931 | 54.9% | 1.55 | 1.17 |
| Elo-Poisson baseline | 226 | 14.2% | 2.789 | 0.1919 | 56.2% | 1.61 | 1.16 |
| Constant modal score | 226 | 11.1% | 4.796 | 0.2321 | 41.6% | 1.70 | 1.46 |

## Group stage only
| System | N | Exact-score hit | Score logloss | RPS (1X2) | Outcome acc | Total-goals MAE | GD MAE |
|---|---|---|---|---|---|---|---|
| Dixon-Coles model | 500 | 16.2% | 2.754 | 0.1957 | 55.2% | 1.51 | 1.19 |
| Elo-Poisson baseline | 500 | 16.6% | 2.758 | 0.1957 | 55.6% | 1.45 | 1.20 |
| Constant modal score | 500 | 10.0% | 4.796 | 0.2378 | 40.4% | 1.60 | 1.53 |

## Knockout only (120-minute target)
| System | N | Exact-score hit | Score logloss | RPS (1X2) | Outcome acc | Total-goals MAE | GD MAE |
|---|---|---|---|---|---|---|---|
| Dixon-Coles model | 179 | 12.3% | 2.925 | 0.1944 | 54.2% | 1.74 | 1.26 |
| Elo-Poisson baseline | 179 | 12.3% | 2.942 | 0.1907 | 56.4% | 1.76 | 1.20 |
| Constant modal score | 179 | 9.5% | 4.796 | 0.2331 | 45.8% | 1.78 | 1.41 |

## By tournament (model only)
| Tournament | N | Exact hit | Logloss | Outcome acc |
|---|---|---|---|---|
| copa2015 | 26 | 11.5% | 2.797 | 53.8% |
| copa2016 | 32 | 12.5% | 3.113 | 59.4% |
| copa2019 | 26 | 15.4% | 2.774 | 57.7% |
| copa2021 | 28 | 21.4% | 2.599 | 57.1% |
| copa2024 | 32 | 25.0% | 2.592 | 56.2% |
| euro2008 | 31 | 12.9% | 2.910 | 51.6% |
| euro2012 | 31 | 12.9% | 2.796 | 54.8% |
| euro2016 | 51 | 17.6% | 2.661 | 41.2% |
| euro2020 | 51 | 15.7% | 2.890 | 58.8% |
| euro2024 | 51 | 21.6% | 2.583 | 52.9% |
| wc2006 | 64 | 17.2% | 2.670 | 60.9% |
| wc2010 | 64 | 20.3% | 2.674 | 56.2% |
| wc2014 | 64 | 9.4% | 3.016 | 56.2% |
| wc2018 | 64 | 12.5% | 2.817 | 56.2% |
| wc2022 | 64 | 6.2% | 3.023 | 51.6% |

## Modal-probability calibration (model)
| Modal-prob bin | N | Predicted | Realized hit rate |
|---|---|---|---|
| (0.07, 0.09] | 1 | 8.0% | 0.0% |
| (0.09, 0.11] | 24 | 10.3% | 20.8% |
| (0.11, 0.13] | 105 | 12.3% | 14.3% |
| (0.13, 1.0] | 549 | 15.5% | 15.1% |

## Confidence label vs realized exact-hit rate
| Label | N | Hit rate |
|---|---|---|
| high | 263 | 17.5% |
| medium | 382 | 14.4% |
| low | 34 | 5.9% |
