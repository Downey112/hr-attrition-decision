"""Descriptive survival analysis: Kaplan-Meier curves, log-rank tests and univariate Cox tests.

Rows that may be the same person (see data.linked_groups) are not independent, so the naive
log-rank p-value is shown next to a cluster-robust Wald p-value from a univariate Cox model.
Benjamini-Hochberg is applied to the robust p-values across all factors tested.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrsurv import data as D  # noqa: E402
from hrsurv import plotstyle as ps  # noqa: E402

FIG = D.ROOT / "reports" / "figures"
TAB = D.ROOT / "reports" / "tables"
FACTORS = ["traffic", "coach", "greywage", "industry", "profession", "way", "gender", "head_gender", "age_band"]
LABEL = {"gender": "gender (protected)", "head_gender": "supervisor gender (protected)",
         "age_band": "age band (protected)"}


def robust_wald(d, cols):
    """Joint Wald test of a block of Cox coefficients with a cluster-robust covariance."""
    df = pd.concat([d[["stag", "event", "person_group"]], cols], axis=1)
    cph = CoxPHFitter().fit(df, "stag", "event", cluster_col="person_group", robust=True)
    b = cph.params_.to_numpy()
    v = cph.variance_matrix_.to_numpy()
    w = float(b @ np.linalg.solve(v, b))
    return w, len(b), stats.chi2.sf(w, len(b)), cph


def main():
    d = D.clean(D.load_raw())
    d["age_band"] = D.age_band(d["age"]).astype(str)
    km = KaplanMeierFitter().fit(d["stag"], d["event"])
    at = [6, 12, 24, 36, 60]
    overall = {
        "rows": len(d), "exact_duplicates_dropped": int(len(D.load_raw()) - len(d)),
        "person_groups": int(d["person_group"].nunique()),
        "rows_in_multi_row_groups": int((d.groupby("person_group")["stag"].transform("size") > 1).sum()),
        "events": int(d["event"].sum()), "median_tenure_months": float(km.median_survival_time_),
        "survival": {str(t): round(float(km.predict(t)), 3) for t in at},
    }

    fig, ax = ps.plt.subplots(figsize=(7, 4))
    km.plot_survival_function(ax=ax, color=ps.SERIES[0], legend=False)
    ax.set(title="Share still employed by tenure (Kaplan-Meier, 95% CI)", xlabel="tenure (months)",
           ylabel="still employed", ylim=(0, 1.02), xlim=(0, 120))
    ps.save(fig, FIG / "km_overall.png")

    rows = []
    for f in FACTORS:
        s = D.lump(d[f]) if f in D.CATEGORICAL else d[f]
        lr = multivariate_logrank_test(d["stag"], s, d["event"])
        ref = s.value_counts().index[0]
        dummies = pd.get_dummies(s, prefix=f, dtype=float).drop(columns=f"{f}_{ref}")
        w, k, p, _ = robust_wald(d, dummies)
        rows.append({"factor": LABEL.get(f, f), "levels": int(s.nunique()), "logrank_chi2": lr.test_statistic,
                     "logrank_p_naive": lr.p_value, "robust_wald_chi2": w, "df": k, "robust_p": p})
        levels = s.value_counts().index[: len(ps.SERIES)]
        fig, ax = ps.plt.subplots(figsize=(7, 4))
        for i, lvl in enumerate(levels):
            m = s == lvl
            k_ = KaplanMeierFitter().fit(d.loc[m, "stag"], d.loc[m, "event"], label=f"{lvl} (n={m.sum()})")
            k_.plot_survival_function(ax=ax, ci_show=False, color=ps.SERIES[i])
        ax.set(title=f"Still employed by {LABEL.get(f, f)}", xlabel="tenure (months)",
               ylabel="still employed", ylim=(0, 1.02), xlim=(0, 120))
        ax.legend(fontsize=8, loc="lower left")
        ps.save(fig, FIG / f"km_{f}.png")
    for c in D.PERSONALITY:
        z = ((d[c] - d[c].mean()) / d[c].std()).rename(c).to_frame()
        w, k, p, cph = robust_wald(d, z)
        rows.append({"factor": f"{c} (per SD)", "levels": np.nan, "logrank_chi2": np.nan,
                     "logrank_p_naive": np.nan, "robust_wald_chi2": w, "df": k, "robust_p": p,
                     "hazard_ratio": float(np.exp(cph.params_.iloc[0]))})
    t = pd.DataFrame(rows)
    t["robust_q_bh"] = multipletests(t["robust_p"], method="fdr_bh")[1]
    t = t.sort_values("robust_p")
    t.to_csv(TAB / "factor_tests.csv", index=False, float_format="%.4g")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    # turnover by level (12-month leaving risk from KM) for the factors that survive BH
    lvl_rows = []
    for f in FACTORS:
        s = D.lump(d[f]) if f in D.CATEGORICAL else d[f]
        for lvl in s.value_counts().index:
            m = s == lvl
            k_ = KaplanMeierFitter().fit(d.loc[m, "stag"], d.loc[m, "event"])
            lvl_rows.append({"factor": f, "level": lvl, "n": int(m.sum()), "events": int(d.loc[m, "event"].sum()),
                             "left_by_12m": 1 - float(k_.predict(12)), "median_months": k_.median_survival_time_})
    pd.DataFrame(lvl_rows).to_csv(TAB / "levels_km.csv", index=False, float_format="%.4g")
    (D.ROOT / "reports" / "overview.json").write_text(json.dumps(overall, indent=2))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
