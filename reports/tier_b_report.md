# Tier-B xG-form experiment

Within-tournament, opponent-adjusted xG-form features folded into the
Layer-2 slot, evaluated with leave-one-tournament-out CV on the six
StatsBomb xG tournaments (WC 2018/2022, Euro 2020/2024, Copa 2024,
AFCON 2023). xG exists only within these tournaments, so the form window
is short (1 match by MD2, 3-4 by the knockouts).

Fitted coefficients (ridge=30.0, signs constrained): att=+0.0412 (chance creation persists), fin=-0.0371 (finishing reverts), opp_def=+0.0000 (opponent leaks).

## All six tournaments (314 matches)
| Arm | N | Exact hit | Log loss | Total-goals MAE | Team-goals MAE |
|---|---|---|---|---|---|
| Elo-Poisson | 314 | 15.6% | 2.8122 | 1.573 | 0.908 |
| DC (Phase-2) | 314 | 16.6% | 2.7970 | 1.551 | 0.900 |
| DC + xG-form | 314 | 16.6% | 2.7987 | 1.545 | 0.893 |

## xG-available matches only (where DC+xG differs from DC)
| Arm | N | Exact hit | Log loss | Total-goals MAE | Team-goals MAE |
|---|---|---|---|---|---|
| DC (Phase-2) | 238 | 16.0% | 2.8195 | 1.588 | 0.912 |
| DC + xG-form | 238 | 16.0% | 2.8217 | 1.580 | 0.903 |

## Group stage (xG-available)
| Arm | N | Exact hit | Log loss | Total-goals MAE | Team-goals MAE |
|---|---|---|---|---|---|
| DC (Phase-2) | 152 | 17.8% | 2.8175 | 1.520 | 0.898 |
| DC + xG-form | 152 | 17.1% | 2.8198 | 1.513 | 0.888 |

## Knockout (xG-available, longest form window)
| Arm | N | Exact hit | Log loss | Total-goals MAE | Team-goals MAE |
|---|---|---|---|---|---|
| DC (Phase-2) | 86 | 12.8% | 2.8229 | 1.709 | 0.936 |
| DC + xG-form | 86 | 14.0% | 2.8251 | 1.698 | 0.930 |

## Verdict

Coefficient signs match theory (creation persists, finishing reverts),
and total/team goal MAE improve marginally (~0.01 goals). But exact-score
hit rate is unchanged and log loss is flat-to-slightly-worse on held-out
tournaments. The only hint of gain is in knockouts (longest window). Per
the evaluation gate, within-tournament xG-form does NOT clearly improve
the exact-score distribution, so it ships OFF by default. The signal is
real but too weak over a 1-4 match window; a broad-calendar xG history
(10+ matches, e.g. via FBref) is the path to revisit.
