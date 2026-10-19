"""Cox model fitting, grouped cross-validation and survival metrics."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

HORIZON = 12.0  # months


def fit_cox(X: pd.DataFrame, d: pd.DataFrame, penalizer: float, robust: bool = False) -> CoxPHFitter:
    df = pd.concat([X, d[["stag", "event", "person_group"]]], axis=1)
    kw = {"cluster_col": "person_group", "robust": True} if robust else {}
    if not robust:
        df = df.drop(columns="person_group")
    return CoxPHFitter(penalizer=penalizer).fit(df, "stag", "event", **kw)


def group_folds(groups: np.ndarray, k: int, seed: int):
    """k folds that never split a person group across train and test."""
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    fold_of = dict(zip(uniq, np.arange(len(uniq)) % k))
    f = np.array([fold_of[g] for g in groups])
    for i in range(k):
        yield np.flatnonzero(f != i), np.flatnonzero(f == i)


def survival_at(cph: CoxPHFitter, X: pd.DataFrame, times) -> np.ndarray:
    """S(t | x) for each row (rows) at each time (columns)."""
    return cph.predict_survival_function(X, times=np.atleast_1d(times)).to_numpy().T


def conditional_leave_prob(cph: CoxPHFitter, X: pd.DataFrame, tenure: np.ndarray,
                           horizon: float = HORIZON) -> np.ndarray:
    """P(leave within `horizon` months | still employed at `tenure`) = 1 - S(t+h)/S(t)."""
    out = np.empty(len(X))
    for i in range(len(X)):
        s = survival_at(cph, X.iloc[[i]], [tenure[i], tenure[i] + horizon])[0]
        out[i] = 1 - s[1] / s[0] if s[0] > 0 else np.nan
    return out


def ipcw_brier(time: np.ndarray, event: np.ndarray, p_event: np.ndarray, t: float = HORIZON,
               train_time=None, train_event=None) -> float:
    """Brier score at t for predicted P(T <= t), weighting by the inverse censoring probability.

    Rows censored before t carry no information and get weight 0; the censoring distribution G
    is a Kaplan-Meier fit on the training rows (or on the evaluated rows if none are given).
    """
    gt = train_time if train_time is not None else time
    ge = train_event if train_event is not None else event
    G = KaplanMeierFitter().fit(gt, 1 - np.asarray(ge))

    w, y = ipcw_weights(time, event, t, G)
    return float(np.sum(w * (y - p_event) ** 2) / len(time))


def ipcw_weights(time, event, t: float = HORIZON, G: KaplanMeierFitter = None):
    """Inverse-probability-of-censoring weights for the binary outcome 'left by t'.

    Leavers by t get 1/G(T-), people known to be there at t (including censored exactly at t)
    get 1/G(t), people censored before t get 0. Returns (weights, outcome).
    """
    time, event = np.asarray(time, float), np.asarray(event)
    if G is None:
        G = KaplanMeierFitter().fit(time, 1 - event)

    def g(x):
        return np.clip(G.survival_function_at_times(np.maximum(x, 0)).to_numpy(), 1e-6, None)

    left = (time <= t) & (event == 1)
    stayed = (time >= t) & ~left
    w = np.where(left, 1 / g(time - 1e-9), 0.0) + np.where(stayed, 1 / g(t), 0.0)
    return w, left.astype(float)


def harrell_c(time, event, risk) -> float:
    # lifelines expects scores where higher = longer survival
    return concordance_index(time, -np.asarray(risk), event)


def oof_predict(X: pd.DataFrame, d: pd.DataFrame, penalizer: float, k: int = 5, seed: int = 0):
    """Out-of-fold linear predictor and P(leave by HORIZON from hire) for every row."""
    lp = np.empty(len(d))
    p12 = np.empty(len(d))
    for tr, te in group_folds(d["person_group"].to_numpy(), k, seed):
        Xtr = X.iloc[tr].loc[:, X.iloc[tr].std() > 0]
        cph = fit_cox(Xtr, d.iloc[tr], penalizer)
        Xte = X.iloc[te][Xtr.columns]
        lp[te] = cph.predict_log_partial_hazard(Xte).to_numpy()
        p12[te] = 1 - survival_at(cph, Xte, HORIZON)[:, 0]
    return lp, p12


def km_calibration(time, event, p, bins: int = 5) -> pd.DataFrame:
    """Mean predicted vs Kaplan-Meier observed P(leave by HORIZON) within quantile bins of p."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    rows = []
    for b in np.unique(q):
        m = q == b
        km = KaplanMeierFitter().fit(time[m], event[m])
        rows.append({"bin": int(b) + 1, "n": int(m.sum()), "predicted": float(p[m].mean()),
                     "observed_km": 1 - float(km.predict(HORIZON))})
    return pd.DataFrame(rows)
