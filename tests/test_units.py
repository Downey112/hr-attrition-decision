import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrsurv import data as D  # noqa: E402
from hrsurv import decision as R  # noqa: E402
from hrsurv import model as M  # noqa: E402


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.exponential(scale=30 * np.exp(-0.7 * x))
    c = rng.uniform(0, 80, size=n)
    d = pd.DataFrame({"stag": np.minimum(t, c), "event": (t <= c).astype(int), "person_group": np.arange(n) // 2})
    return pd.DataFrame({"x": x}), d


def test_threshold_and_break_even():
    assert R.threshold(k=2, c=0.1, e=0.5) == pytest.approx(0.1)
    p = np.array([0.1, 0.2, 0.0, np.nan])
    be = R.break_even_effectiveness(p, k=1, c=0.1)
    assert be[0] == pytest.approx(1.0) and be[1] == pytest.approx(0.5) and np.isinf(be[2]) and np.isnan(be[3])
    # at exactly the break-even effectiveness, intervening and not intervening cost the same
    assert R.threshold(1, 0.1, be[1]) == pytest.approx(p[1])


def test_realised_cost():
    left = np.array([1, 0, 1, 0])
    assert R.realised_cost([0, 0, 0, 0], left, k=1, c=0.1, e=0.5) == pytest.approx(0.5)
    assert R.realised_cost([1, 1, 1, 1], left, k=1, c=0.1, e=0.5) == pytest.approx(0.1 + 0.25)
    # weights: only the first row counts
    assert R.realised_cost([1, 0, 0, 0], left, 1, 0.1, 0.5, w=[1, 0, 0, 0]) == pytest.approx(0.6)


def test_group_folds_never_split_groups():
    groups = np.repeat(np.arange(50), 3)
    seen = np.zeros(len(groups), int)
    for tr, te in M.group_folds(groups, 5, seed=1):
        assert not set(groups[tr]) & set(groups[te])
        seen[te] += 1
    assert (seen == 1).all()


def test_conditional_prob_matches_survival_ratio():
    X, d = toy()
    cph = M.fit_cox(X, d, 0.0)
    s = M.survival_at(cph, X.iloc[:3], [5.0, 17.0])
    p = M.conditional_leave_prob(cph, X.iloc[:3], np.array([5.0, 5.0, 5.0]))
    assert np.allclose(p, 1 - s[:, 1] / s[:, 0])
    # higher x = higher hazard = higher conditional risk
    assert cph.params_["x"] > 0


def test_ipcw_brier_perfect_and_null():
    t = np.array([5.0, 20.0, 8.0, 30.0])
    e = np.array([1, 1, 1, 0])
    truth = np.array([1.0, 0.0, 1.0, 0.0])
    assert M.ipcw_brier(t, e, truth) == pytest.approx(0.0)
    assert M.ipcw_brier(t, e, 1 - truth) > 0.9


def test_harrell_c_direction():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.ones(4)
    assert M.harrell_c(t, e, [4, 3, 2, 1]) == pytest.approx(1.0)  # highest risk leaves first


def test_linked_groups_chain():
    base = {"gender": "f", "extraversion": 1, "independ": 1, "selfcontrol": 1, "anxiety": 1, "novator": 1,
            "stag": 1.0, "event": 1, "age": 30, "industry": "IT", "profession": "HR"}
    rows = [dict(base), dict(base, stag=2.0), dict(base, extraversion=9), dict(base, extraversion=8, stag=5.0)]
    g = D.linked_groups(pd.DataFrame(rows))
    # rows 0,1 share personality; row 2 shares tenure/event/age/job with row 0; row 3 is alone
    assert g[0] == g[1] == g[2] and g[3] != g[0]


def test_lump():
    s = pd.Series(["a"] * 40 + ["b"] * 5)
    assert set(D.lump(s)) == {"a", "other"}


def test_ipcw_weights_with_early_censoring():
    # row 1 censored at 6 (before the horizon): weight 0, and the censoring KM drops to 3/4 after t=6
    t = np.array([2.0, 6.0, 10.0, 12.0, 20.0])
    e = np.array([1, 0, 1, 0, 0])
    w, y = M.ipcw_weights(t, e, 12.0)
    assert list(y) == [1, 0, 1, 0, 0]
    assert w[1] == 0.0
    assert w[0] == pytest.approx(1.0)            # G(2-) = 1
    assert w[2] == pytest.approx(4 / 3)          # G(10-) = 1 - 1/4 (one censoring among 4 at risk at t=6)
    assert w[3] == pytest.approx(w[4]) and w[3] > 0  # censored exactly at the horizon still counts as stayed
    # weighted outcome reproduces the Kaplan-Meier 12-month leaving probability for this sample
    from lifelines import KaplanMeierFitter
    km = 1 - KaplanMeierFitter().fit(t, e).predict(12.0)
    assert np.sum(w * y) / len(t) == pytest.approx(km)
