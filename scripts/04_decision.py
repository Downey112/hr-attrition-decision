"""Turn survival predictions into a retention decision and test it.

1. Backtest: out-of-fold P(leave within 12 months of hire) from 03_cox, with IPCW weights for rows
   censored before 12 months. Compare three policies: intervene on nobody, on everybody, or on
   those above the cost threshold c/(e*k). Repeated over a grid of assumptions.
2. Current staff (event = 0): P(leave in the next 12 months | still there at current tenure)
   from the decision model refit on all rows, and each person's break-even effectiveness.
3. Fairness audit: calibration and targeting rates by gender and age band. These attributes
   are not model inputs.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrsurv import data as D  # noqa: E402
from hrsurv import decision as R  # noqa: E402
from hrsurv import model as M  # noqa: E402
from hrsurv import plotstyle as ps  # noqa: E402

FIG = D.ROOT / "reports" / "figures"
TAB = D.ROOT / "reports" / "tables"
OUT = D.ROOT / "data" / "processed"
BASE = {"k": 1.0, "c": 0.10, "e": 0.30}  # base case; see README for why
OPTIMISTIC = {"k": 2.0, "c": 0.10, "e": 0.50}  # a case where the rule targets a real share of staff
GRID = {"k": [0.5, 1.0, 2.0], "c": [0.05, 0.10, 0.20], "e": [0.1, 0.2, 0.3, 0.5, 0.7]}
B = 1000  # bootstrap resamples


def policies(p, left, w, k, c, e):
    """Costs are IPCW means: the weighted rows with known 12-month status stand in for everyone,
    including the cost c of targeting people who were later censored."""
    tau = R.threshold(k, c, e)
    target = p > tau
    nobody = R.realised_cost(np.zeros_like(target), left, k, c, e, w)
    everyone = R.realised_cost(np.ones_like(target), left, k, c, e, w)
    model = R.realised_cost(target, left, k, c, e, w)
    return {"k": k, "c": c, "e": e, "threshold": tau, "share_targeted": float(np.mean(target)),
            "cost_nobody": nobody, "cost_everyone": everyone, "cost_model": model,
            "model_saving_vs_best_simple": min(nobody, everyone) - model}


def main():
    d = D.clean(D.load_raw())
    oof = pd.read_csv(OUT / "oof_predictions.csv")
    t, ev, p = oof["stag"].to_numpy(), oof["event"].to_numpy(), oof["p12"].to_numpy()
    w, left = M.ipcw_weights(t, ev)

    grid = pd.DataFrame([policies(p, left, w, k, c, e)
                         for k, c, e in itertools.product(GRID["k"], GRID["c"], GRID["e"])])
    grid.to_csv(TAB / "decision_grid.csv", index=False, float_format="%.4f")
    base = policies(p, left, w, **BASE)
    opt = policies(p, left, w, **OPTIMISTIC)
    print("base", {k: round(v, 4) for k, v in base.items()})
    print("optimistic", {k: round(v, 4) for k, v in opt.items()})
    print(grid[grid["share_targeted"].between(0.001, 0.999)].round(4).to_string(index=False))

    # grouped bootstrap of the saving (person groups resampled; predictions and weights held fixed,
    # so the intervals cover sampling of people only, not model refitting: they are too narrow if anything)
    groups = oof["person_group"].to_numpy()
    uniq = np.unique(groups)
    rows_of = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(241)
    draws = [np.concatenate([rows_of[g] for g in rng.choice(uniq, len(uniq))]) for _ in range(B)]
    boot = []
    for _, r in grid.iterrows():
        s = [policies(p[i], left[i], w[i], r["k"], r["c"], r["e"])["model_saving_vs_best_simple"] for i in draws]
        boot.append({"k": r["k"], "c": r["c"], "e": r["e"], "threshold": r["threshold"],
                     "share_targeted": r["share_targeted"], "saving": r["model_saving_vs_best_simple"],
                     "lo95": np.percentile(s, 2.5), "hi95": np.percentile(s, 97.5)})
    boot = pd.DataFrame(boot)
    boot.to_csv(TAB / "decision_grid_bootstrap.csv", index=False, float_format="%.4f")
    # only scenarios where the rule actually splits staff (5-95% targeted) can show a gain from targeting;
    # elsewhere the rule collapses to 'nobody' or 'everybody' by construction
    live = boot[boot["share_targeted"].between(0.05, 0.95)]
    grid_summary = {"scenarios": len(boot), "rule_splits_staff": len(live),
                    "saving_ci_above_0": int((live["lo95"] > 0).sum()),
                    "saving_ci_below_0": int((live["hi95"] < 0).sum()),
                    "max_saving": float(live["saving"].max())}
    print(grid_summary)
    print(live.sort_values("saving", ascending=False).round(4).to_string(index=False))

    # cost vs threshold, base case and optimistic case
    taus = np.linspace(0, 0.4, 81)
    fig, axes = ps.plt.subplots(1, 2, figsize=(10, 4))
    for ax, sc, name in [(axes[0], BASE, "Base case"), (axes[1], OPTIMISTIC, "Optimistic case")]:
        costs = [R.realised_cost(p > x, left, sc["k"], sc["c"], sc["e"], w) for x in taus]
        tau = R.threshold(**sc)
        ax.plot(taus, costs, color=ps.SERIES[0])
        if tau <= taus[-1]:
            ax.axvline(tau, color=ps.SERIES[1], lw=1.5, ls="--")
        ax.set(title=f"{name}: k={sc['k']}, c={sc['c']}, e={sc['e']}\n(dashed: rule threshold c/(e*k) = {tau:.2f})",
               xlabel="intervene if P(leave in 12m) >",
               ylabel="cost per employee (salary-years)")
    ps.save(fig, FIG / "cost_vs_threshold.png")

    # where does targeting add value? saving vs best simple policy across effectiveness
    fig, ax = ps.plt.subplots(figsize=(7, 4))
    sub = grid[grid["k"] == 1.0]
    for i, c in enumerate(GRID["c"]):
        s = sub[sub["c"] == c]
        ax.plot(s["e"], s["model_saving_vs_best_simple"] * 100, "o-", color=ps.SERIES[i], ms=7, label=f"c = {c}")
    ax.axhline(0, color=ps.TEXT2, lw=1)
    ax.set(title="Gain from targeting over the better of 'nobody' and 'everybody' (k = 1)",
           xlabel="intervention effectiveness e", ylabel="saving, % of one salary per employee")
    ax.legend()
    ps.save(fig, FIG / "targeting_gain.png")

    # current staff
    X = D.design(d)
    pen = json.loads((D.ROOT / "reports" / "cox.json").read_text())["penalizer"]
    cph = M.fit_cox(X, d, pen)
    staff = d[d["event"] == 0].copy()
    staff["p_leave_12m"] = M.conditional_leave_prob(cph, X.loc[staff.index], staff["stag"].to_numpy())
    last_event = d.loc[d["event"] == 1, "stag"].max()
    # the Cox baseline hazard is zero after the last observed exit, so these risks would read as 0
    staff["beyond_followup"] = staff["stag"] + M.HORIZON > last_event
    staff.loc[staff["beyond_followup"], "p_leave_12m"] = np.nan
    staff["break_even_e"] = R.break_even_effectiveness(staff["p_leave_12m"], BASE["k"], BASE["c"])
    staff["intervene_base"] = staff["p_leave_12m"] > R.threshold(**BASE)
    staff["intervene_optimistic"] = staff["p_leave_12m"] > R.threshold(**OPTIMISTIC)
    staff["age_band"] = D.age_band(staff["age"]).astype(str)
    cols = ["stag", "industry", "profession", "traffic", "coach", "greywage", "way", "p_leave_12m",
            "break_even_e", "intervene_base", "intervene_optimistic", "beyond_followup", "gender", "age_band"]
    staff[cols].rename(columns={"stag": "tenure_months"}).rename_axis("row_id").to_csv(
        OUT / "staff_scores.csv", float_format="%.4f")
    print(staff["p_leave_12m"].describe().round(3), "\nbeyond follow-up:", int(staff["beyond_followup"].sum()))

    # fairness audit, two populations kept apart:
    # (a) calibration of the from-hire 12-month risk, all rows, out-of-fold
    # (b) who the rule would target among current staff (conditional risk at their tenure)
    oof["age_band"] = D.age_band(oof["age"]).astype(str)
    cal_rows, tgt_rows = [], []
    for attr in ["gender", "age_band"]:
        for lvl, g in oof.groupby(attr):
            km = KaplanMeierFitter().fit(g["stag"], g["event"])
            cal_rows.append({"attribute": attr, "group": lvl, "n": len(g), "predicted_12m": g["p12"].mean(),
                             "observed_12m_km": 1 - float(km.predict(M.HORIZON))})
        for lvl, s in staff.groupby(attr):
            tgt_rows.append({"attribute": attr, "group": lvl, "staff_n": len(s),
                             "mean_p_leave_12m": s["p_leave_12m"].mean(),
                             "targeted_base": s["intervene_base"].mean(),
                             "targeted_optimistic": s["intervene_optimistic"].mean()})
    fair_cal, fair_tgt = pd.DataFrame(cal_rows), pd.DataFrame(tgt_rows)
    fair_cal.to_csv(TAB / "fairness_calibration.csv", index=False, float_format="%.4f")
    fair_tgt.to_csv(TAB / "fairness_targeting.csv", index=False, float_format="%.4f")
    print(fair_cal.round(3).to_string(index=False))
    print(fair_tgt.round(3).to_string(index=False))

    res = {"base_case": BASE, "base_result": base, "optimistic_case": OPTIMISTIC, "optimistic_result": opt,
           "grid_summary": grid_summary,
           "staff_n": len(staff), "staff_p12": staff["p_leave_12m"].describe().round(4).to_dict(),
           "staff_beyond_followup": int(staff["beyond_followup"].sum()),
           "staff_targeted_base": int(staff["intervene_base"].sum()),
           "staff_targeted_optimistic": int(staff["intervene_optimistic"].sum())}
    (D.ROOT / "reports" / "decision.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
