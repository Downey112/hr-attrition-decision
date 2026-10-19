"""Write the CSV extracts the Power BI report reads (powerbi/data/).

Everything here is derived from the outputs of 02-04; nothing is re-estimated. Codes are replaced
by readable labels. These extracts contain row-level data derived from the CC BY-NC-SA 4.0 source,
so they carry the same licence (see powerbi/data/LICENSE.md).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrsurv import data as D  # noqa: E402

BI = D.ROOT / "powerbi" / "data"
TAB = D.ROOT / "reports" / "tables"
REP = D.ROOT / "reports"

FACTORS = {  # column -> display name; protected ones are shown for the audit only
    "industry": "Industry", "profession": "Profession", "traffic": "Hiring channel",
    "coach": "Probation mentor", "greywage": "Wage type", "way": "Commute",
    "gender": "Gender (protected)", "head_gender": "Supervisor gender (protected)",
    "age_band": "Age at hire (protected)",
}
LEVELS = {
    "traffic": {"youjs": "Applied via job site", "empjs": "Found by employer on job site",
                "rabrecNErab": "Approached on a recommendation", "referal": "Employee referral",
                "recNErab": "Recommended by non-employee", "friends": "Knew the employer",
                "advert": "Applied directly", "KA": "Recruitment agency"},
    "coach": {"no": "No mentor", "yes": "Mentor (colleague)", "my head": "Mentor (own manager)"},
    "greywage": {"white": "Fully declared", "grey": "Partly undeclared (grey)"},
    "way": {"bus": "Bus", "car": "Car", "foot": "On foot"},
    "gender": {"f": "Female", "m": "Male"},
    "head_gender": {"f": "Female", "m": "Male"},
    "industry": {"other": "Other", "manufacture": "Manufacturing", "transport": "Transport",
                 "PowerGeneration": "Power generation", "Consult": "Consulting", "State": "State sector"},
    "profession": {"other": "Other"},
}
REFERENCE = {"industry": "Retail", "profession": "HR", "traffic": "youjs", "coach": "no", "greywage": "white",
             "way": "bus"}
MONTHS = np.arange(0, 121)
MAX_CURVES = 8


def label(col, value):
    return LEVELS.get(col, {}).get(value, value)


def factor_series(d, col):
    s = D.lump(d[col]) if col in D.CATEGORICAL else d[col]
    return s.map(lambda v: label(col, v))


def p_text(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def covariate_label(name):
    if name in D.PERSONALITY:
        return {"independ": "Independence", "selfcontrol": "Self-control", "novator": "Openness"}.get(
            name, name.capitalize()) + " (per SD)"
    col, _, lvl = name.partition("_")
    return f"{FACTORS[col]}: {label(col, lvl)} (vs {label(col, REFERENCE[col])})"


def main():
    BI.mkdir(parents=True, exist_ok=True)
    d = D.clean(D.load_raw())
    d["age_band"] = D.age_band(d["age"]).astype(str)

    km_rows, factor_rows = [], []
    for i, (col, name) in enumerate(FACTORS.items()):
        factor_rows.append({"factor": name, "factor_order": i, "protected": "protected" in name})
        s = factor_series(d, col)
        # at most 8 curves per chart (one per palette colour): pool the smallest levels
        top = s.value_counts().index[:MAX_CURVES - 1] if s.nunique() > MAX_CURVES else s.unique()
        s = s.where(s.isin(top), "Other")
        for lvl in s.value_counts().index:
            m = s == lvl
            k = KaplanMeierFitter().fit(d.loc[m, "stag"], d.loc[m, "event"])
            surv = k.survival_function_at_times(MONTHS).to_numpy()
            km_rows += [{"factor": name, "level": f"{lvl} (n={m.sum()})", "month": int(t), "still_employed": v}
                        for t, v in zip(MONTHS, surv)]
    pd.DataFrame(factor_rows).to_csv(BI / "factors.csv", index=False)
    pd.DataFrame(km_rows).to_csv(BI / "km_curves.csv", index=False, float_format="%.4f")

    lv = pd.read_csv(TAB / "levels_km.csv")
    lv["factor_col"] = lv["factor"]
    lv["factor"] = lv["factor_col"].map(FACTORS)
    lv["level"] = [label(c, v) for c, v in zip(lv["factor_col"], lv["level"])]
    lv.drop(columns="factor_col").to_csv(BI / "group_levels.csv", index=False, float_format="%.4f")

    ft = pd.read_csv(TAB / "factor_tests.csv")
    test_names = {**{c: n for c, n in FACTORS.items()}, "gender (protected)": FACTORS["gender"],
                  "supervisor gender (protected)": FACTORS["head_gender"],
                  "age band (protected)": FACTORS["age_band"],
                  **{f"{p} (per SD)": covariate_label(p) for p in D.PERSONALITY}}
    ft["factor"] = ft["factor"].map(test_names)
    ft["p_text"] = ft["robust_p"].map(p_text)
    ft["q_text"] = ft["robust_q_bh"].map(p_text)
    ft["p_rank"] = ft["robust_p"].rank(method="first").astype(int)
    ft[["factor", "df", "robust_p", "robust_q_bh", "p_text", "q_text", "p_rank"]].to_csv(BI / "factor_tests.csv", index=False, float_format="%.4g")

    hr = pd.read_csv(TAB / "cox_hazard_ratios.csv")
    hr["label"] = hr["covariate"].map(covariate_label)
    hr["p_text"] = hr["p_robust"].map(p_text)
    hr["significant"] = np.where(hr["p_robust"] < 0.05, "p < 0.05", "not significant")
    hr.to_csv(BI / "hazard_ratios.csv", index=False, float_format="%.4f")

    staff = pd.read_csv(D.ROOT / "data" / "processed" / "staff_scores.csv")
    for col in D.CATEGORICAL:
        # pool with the full-data counts the model used, not the smaller staff subset's
        kept = set(D.lump(d[col]))
        staff[col] = staff[col].where(staff[col].isin(kept), "other").map(lambda v: label(col, v))
    staff["gender"] = staff["gender"].map(LEVELS["gender"])
    # small epsilon so values on a bin edge (e.g. 0.075) are not pushed down by float error
    staff["risk_bin"] = np.floor(staff["p_leave_12m"] / 0.025 + 1e-9) * 0.025
    keep = ["row_id", "tenure_months", "industry", "profession", "traffic", "coach", "greywage", "way",
            "p_leave_12m", "risk_bin", "gender", "age_band"]
    staff[keep].to_csv(BI / "staff.csv", index=False, float_format="%.4f")

    grid = pd.read_csv(TAB / "decision_grid_bootstrap.csv")
    for c in ["k", "c", "e"]:
        grid[f"{c}_x100"] = (grid[c] * 100).round().astype(int)
    grid.to_csv(BI / "backtest_grid.csv", index=False, float_format="%.4f")

    pd.read_csv(TAB / "calibration_12m.csv").to_csv(BI / "calibration.csv", index=False, float_format="%.4f")
    fc = pd.read_csv(TAB / "fairness_calibration.csv")
    fc["group"] = fc["group"].replace(LEVELS["gender"])
    fc["attribute"] = fc["attribute"].map({"gender": "Gender", "age_band": "Age at hire"})
    fc = fc.sort_values(["attribute", "group"], key=lambda s: s.replace({"<=25": "0"}))
    fc["order"] = range(len(fc))
    fc.to_csv(BI / "fairness_calibration.csv", index=False, float_format="%.4f")

    ov = json.loads((REP / "overview.json").read_text())
    cox = json.loads((REP / "cox.json").read_text())
    metrics = {
        "employees": ov["rows"], "leavers": ov["events"], "median_tenure_months": ov["median_tenure_months"],
        "left_by_12m": 1 - ov["survival"]["12"], "c_index": cox["cv"]["c_decision"]["mean"],
        "c_index_lo": cox["bootstrap_ci_seed0"]["c"][0], "c_index_hi": cox["bootstrap_ci_seed0"]["c"][1],
        "brier_null": cox["cv"]["brier_null"]["mean"], "brier_model": cox["cv"]["brier_decision"]["mean"],
        "staff_scored": int(staff["p_leave_12m"].notna().sum()),
    }
    pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())}).to_csv(
        BI / "metrics.csv", index=False, float_format="%.6g")
    print("wrote", sorted(p.name for p in BI.glob("*.csv")))


if __name__ == "__main__":
    main()
