"""Cox proportional-hazards model of time to leaving.

Two covariate sets are compared under grouped 5-fold CV (5 repeats, folds never split a person group):
  decision  job, hiring and personality covariates only (what the retention rule may use)
  full      decision + gender, age and supervisor gender (to measure what excluding them costs)
plus a null model (Kaplan-Meier, same risk for everyone). Metrics: Harrell's C and the IPCW
Brier score for P(leave within 12 months of hire). The final decision model is refit on all rows
with cluster-robust standard errors and checked for proportional hazards.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrsurv import data as D  # noqa: E402
from hrsurv import model as M  # noqa: E402
from hrsurv import plotstyle as ps  # noqa: E402

FIG = D.ROOT / "reports" / "figures"
TAB = D.ROOT / "reports" / "tables"
OUT = D.ROOT / "data" / "processed"
PENALTIES = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
SEEDS = range(5)
TUNE_SEED = 100


def cv_scores(X, d, penalizer, seed):
    lp, p12 = M.oof_predict(X, d, penalizer, seed=seed)
    t, e = d["stag"].to_numpy(), d["event"].to_numpy()
    return M.harrell_c(t, e, lp), M.ipcw_brier(t, e, p12), lp, p12


def null_p12(d, seed):
    t, e = d["stag"].to_numpy(), d["event"].to_numpy()
    p = np.empty(len(d))
    for tr, te in M.group_folds(d["person_group"].to_numpy(), 5, seed):
        p[te] = 1 - KaplanMeierFitter().fit(t[tr], e[tr]).predict(M.HORIZON)
    return p


def grouped_bootstrap(oof, B=1000, seed=241):
    """CIs for out-of-fold C-index and the Brier gain over the null model, resampling person groups.
    Predictions are held fixed, so these reflect sampling of people, not refitting."""
    t, e = oof["stag"].to_numpy(), oof["event"].to_numpy()
    groups = oof["person_group"].to_numpy()
    uniq = np.unique(groups)
    rows_of = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        i = np.concatenate([rows_of[g] for g in rng.choice(uniq, len(uniq))])
        out.append({"c": M.harrell_c(t[i], e[i], oof["lp"].to_numpy()[i]),
                    "brier_gain": M.ipcw_brier(t[i], e[i], oof["p12_null"].to_numpy()[i])
                    - M.ipcw_brier(t[i], e[i], oof["p12"].to_numpy()[i])})
    return pd.DataFrame(out).quantile([0.025, 0.975])


def main():
    d = D.clean(D.load_raw())
    X = D.design(d)
    Xf = D.design(d, protected=True)

    # tune on a split that is not reused for evaluation
    tune = pd.DataFrame([{"penalizer": pen, "c_index": cv_scores(X, d, pen, TUNE_SEED)[0]} for pen in PENALTIES])
    pen = float(tune.loc[tune["c_index"].idxmax(), "penalizer"])
    print(tune.to_string(index=False), f"\nchosen penalizer {pen}")
    tune.to_csv(TAB / "cox_penalizer_cv.csv", index=False, float_format="%.4f")

    rows = []
    for s in SEEDS:
        cd, bd, lp, p12 = cv_scores(X, d, pen, s)
        cf, bf, _, _ = cv_scores(Xf, d, pen, s)
        pn = null_p12(d, s)
        rows.append({"seed": s, "c_decision": cd, "c_full": cf,
                     "brier_null": M.ipcw_brier(d["stag"].to_numpy(), d["event"].to_numpy(), pn),
                     "brier_decision": bd, "brier_full": bf})
        if s == 0:
            oof = d[["stag", "event", "person_group", "gender", "age"]].assign(lp=lp, p12=p12, p12_null=pn)
    cv = pd.DataFrame(rows)
    cv.to_csv(TAB / "cox_cv.csv", index=False, float_format="%.4f")
    # sd here is split-to-split variation, not a standard error; the bootstrap below gives the CI
    summary = cv.drop(columns="seed").agg(["mean", "std"]).T
    print(summary.round(4))
    oof.to_csv(OUT / "oof_predictions.csv", index=False)
    ci = grouped_bootstrap(oof)
    print("bootstrap 95% CI (seed-0 OOF):\n", ci.round(4))

    # sensitivity: does the person-grouping rule change the CV estimate?
    sens = {}
    for name, groups in {"personality_key_only": D.linked_groups(d, keys=("personality",)),
                         "no_grouping": np.arange(len(d))}.items():
        sens[name] = cv_scores(X, d.assign(person_group=groups), pen, 0)[0]
    sens["chained_default"] = rows[0]["c_decision"]
    print("C-index by grouping rule:", {k: round(v, 4) for k, v in sens.items()})

    cal = M.km_calibration(oof["stag"].to_numpy(), oof["event"].to_numpy(), oof["p12"].to_numpy())
    cal.to_csv(TAB / "calibration_12m.csv", index=False, float_format="%.4f")
    print(cal.round(3).to_string(index=False))
    fig, ax = ps.plt.subplots(figsize=(4.5, 4.5))
    lim = max(cal["predicted"].max(), cal["observed_km"].max()) * 1.15
    ax.plot([0, lim], [0, lim], color=ps.GRID, lw=1.5, zorder=0)
    ax.plot(cal["predicted"], cal["observed_km"], "o-", color=ps.SERIES[0], ms=8)
    ax.set(title="12-month calibration (out-of-fold quintiles)", xlabel="predicted P(leave by 12 months)",
           ylabel="observed (Kaplan-Meier)", xlim=(0, lim), ylim=(0, lim))
    ps.save(fig, FIG / "calibration_12m.png")

    # inference uses the unpenalised fit (the penalty shrinks hazard ratios and their CIs); prediction uses pen
    cph = M.fit_cox(X, d, 0.0, robust=True)
    hr = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    hr.columns = ["hazard_ratio", "hr_lo", "hr_hi", "p_robust"]
    hr.to_csv(TAB / "cox_hazard_ratios.csv", float_format="%.4g")
    print(hr.round(3).to_string())

    ph = proportional_hazard_test(M.fit_cox(X, d, pen), pd.concat([X, d[["stag", "event"]]], axis=1),
                                  time_transform="rank")
    ph_t = ph.summary[["test_statistic", "p"]].sort_values("p")
    ph_t.to_csv(TAB / "ph_test.csv", float_format="%.4g")
    print("PH violations p<0.05:\n", ph_t[ph_t["p"] < 0.05].round(4).to_string())

    fig, ax = ps.plt.subplots(figsize=(7, 8))
    h = hr.sort_values("hazard_ratio")
    y = np.arange(len(h))
    ax.hlines(y, h["hr_lo"], h["hr_hi"], color=ps.TEXT2, lw=1.5)
    sig = h["p_robust"] < 0.05
    ax.plot(h["hazard_ratio"][~sig], y[~sig], "o", color=ps.GRID, mec=ps.TEXT2, ms=7, label="p >= 0.05")
    ax.plot(h["hazard_ratio"][sig], y[sig], "o", color=ps.SERIES[0], ms=8, label="p < 0.05")
    ax.axvline(1, color=ps.TEXT2, lw=1)
    ax.set_xscale("log")
    ticks = [0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(y, h.index, fontsize=8)
    ax.set(title="Hazard ratios, decision model\n(unpenalised fit, cluster-robust 95% CI)",
           xlabel="hazard ratio (log scale)")
    ax.legend(loc="lower right")
    ps.save(fig, FIG / "hazard_ratios.png")

    res = {"penalizer": pen, "cv": {k: {"mean": round(v["mean"], 4), "sd": round(v["std"], 4)}
                                    for k, v in summary.iterrows()},
           "bootstrap_ci_seed0": {c: [round(float(ci[c].iloc[0]), 4), round(float(ci[c].iloc[1]), 4)]
                                  for c in ci.columns},
           "c_by_grouping_rule": {k: round(float(v), 4) for k, v in sens.items()},
           "ph_violations": ph_t[ph_t["p"] < 0.05].index.tolist(), "ph_tests": len(ph_t)}
    (D.ROOT / "reports" / "cox.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
