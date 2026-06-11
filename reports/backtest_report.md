# Walk-forward backtest report

Target: exact final score (90' group stage; 120' knockouts, shootouts excluded).
Exact-score prediction has a hard ceiling — betting markets hit ~11-12%. Read
every number against the baselines below, not against intuition (plan section 9).

## All tournaments
| System | N | Exact-score hit | Score logloss | RPS (1X2) | Outcome acc | Total-goals MAE | GD MAE |
|---|---|---|---|---|---|---|---|
| Dixon-Coles model | 679 | 15.8% | 2.794 | 0.1941 | 54.9% | 1.51 | 1.19 |
| Elo-Poisson baseline | 679 | 15.5% | 2.806 | 0.1944 | 55.8% | 1.53 | 1.20 |
| Constant modal score | 679 | 9.9% | 4.796 | 0.2365 | 41.8% | 1.65 | 1.50 |

## Selection set (hyperparameters tuned here)
| System | N | Exact-score hit | Score logloss | RPS (1X2) | Outcome acc | Total-goals MAE | GD MAE |
|---|---|---|---|---|---|---|---|
| Dixon-Coles model | 453 | 15.4% | 2.804 | 0.1947 | 54.8% | 1.52 | 1.20 |
| Elo-Poisson baseline | 453 | 16.1% | 2.814 | 0.1956 | 55.6% | 1.49 | 1.21 |
| Constant modal score | 453 | 9.3% | 4.796 | 0.2388 | 41.9% | 1.62 | 1.51 |

## Holdout set (untouched by tuning)
| System | N | Exact-score hit | Score logloss | RPS (1X2) | Outcome acc | Total-goals MAE | GD MAE |
|---|---|---|---|---|---|---|---|
| Dixon-Coles model | 226 | 16.4% | 2.776 | 0.1930 | 55.3% | 1.49 | 1.16 |
| Elo-Poisson baseline | 226 | 14.2% | 2.789 | 0.1919 | 56.2% | 1.61 | 1.16 |
| Constant modal score | 226 | 11.1% | 4.796 | 0.2321 | 41.6% | 1.70 | 1.46 |

## Group stage only
| System | N | Exact-score hit | Score logloss | RPS (1X2) | Outcome acc | Total-goals MAE | GD MAE |
|---|---|---|---|---|---|---|---|
| Dixon-Coles model | 500 | 17.0% | 2.751 | 0.1946 | 55.2% | 1.44 | 1.17 |
| Elo-Poisson baseline | 500 | 16.6% | 2.758 | 0.1957 | 55.6% | 1.45 | 1.20 |
| Constant modal score | 500 | 10.0% | 4.796 | 0.2378 | 40.4% | 1.60 | 1.53 |

## Knockout only (120-minute target)
| System | N | Exact-score hit | Score logloss | RPS (1X2) | Outcome acc | Total-goals MAE | GD MAE |
|---|---|---|---|---|---|---|---|
| Dixon-Coles model | 179 | 12.3% | 2.916 | 0.1929 | 54.2% | 1.70 | 1.24 |
| Elo-Poisson baseline | 179 | 12.3% | 2.942 | 0.1907 | 56.4% | 1.76 | 1.20 |
| Constant modal score | 179 | 9.5% | 4.796 | 0.2331 | 45.8% | 1.78 | 1.41 |

## By tournament (model only)
| Tournament | N | Exact hit | Logloss | Outcome acc |
|---|---|---|---|---|
| copa2015 | 26 | 7.7% | 2.797 | 53.8% |
| copa2016 | 32 | 12.5% | 3.053 | 59.4% |
| copa2019 | 26 | 15.4% | 2.768 | 57.7% |
| copa2021 | 28 | 17.9% | 2.602 | 57.1% |
| copa2024 | 32 | 28.1% | 2.578 | 56.2% |
| euro2008 | 31 | 12.9% | 2.904 | 48.4% |
| euro2012 | 31 | 12.9% | 2.779 | 54.8% |
| euro2016 | 51 | 17.6% | 2.672 | 41.2% |
| euro2020 | 51 | 13.7% | 2.879 | 58.8% |
| euro2024 | 51 | 19.6% | 2.586 | 54.9% |
| wc2006 | 64 | 18.8% | 2.655 | 60.9% |
| wc2010 | 64 | 21.9% | 2.676 | 56.2% |
| wc2014 | 64 | 9.4% | 3.028 | 56.2% |
| wc2018 | 64 | 17.2% | 2.816 | 56.2% |
| wc2022 | 64 | 9.4% | 3.021 | 51.6% |

## Modal-probability calibration (model)
| Modal-prob bin | N | Predicted | Realized hit rate |
|---|---|---|---|
| (0.07, 0.09] | 1 | 8.5% | 0.0% |
| (0.09, 0.11] | 19 | 10.2% | 21.1% |
| (0.11, 0.13] | 110 | 12.2% | 13.6% |
| (0.13, 1.0] | 549 | 15.3% | 16.0% |

## Confidence label vs realized exact-hit rate
| Label | N | Hit rate |
|---|---|---|
| high | 244 | 18.0% |
| medium | 405 | 15.1% |
| low | 30 | 6.7% |
