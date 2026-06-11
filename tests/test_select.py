import numpy as np

from wc26.predict.select import confidence_label, select_score
from wc26.data.importance import importance_class
from wc26.scoreline.matrix import score_matrix


def test_argmax_selection():
    m = score_matrix(2.2, 0.6, rho=-0.06)
    (x, y), p, _ = select_score(m, 2.2, 0.6)
    assert (x, y) == tuple(int(i) for i in np.unravel_index(m.argmax(), m.shape))
    assert p == m.max()
    assert x > y  # heavy favourite at home


def test_tie_break_deterministic():
    m = np.zeros((11, 11))
    m[1, 0] = 0.101
    m[0, 1] = 0.1
    m[1, 1] = 0.099
    m /= m.sum()
    # lam > mu: prefer the home-win score among near-ties
    (x, y), _, cand = select_score(m, 1.4, 1.0, tie_epsilon=0.05)
    assert (x, y) == (1, 0)
    assert len(cand) == 3


def test_confidence_labels():
    assert confidence_label(0.16) == "high"
    assert confidence_label(0.12) == "medium"
    assert confidence_label(0.05) == "low"
    assert confidence_label(0.22) == "low"   # blowout regime: exact margin uncertain
    assert confidence_label(0.16, demote=True) == "medium"


def test_importance_classes():
    assert importance_class("FIFA World Cup") == "world_cup"
    assert importance_class("FIFA World Cup qualification") == "qualifier"
    assert importance_class("UEFA Euro") == "continental_finals"
    assert importance_class("UEFA Nations League") == "nations_league"
    assert importance_class("Friendly") == "friendly"
    assert importance_class("Gulf Cup") == "minor"
    assert importance_class("CONCACAF Nations League qualification") == "nations_league"
