"""Generate the Power BI project (PBIP: TMDL semantic model + PBIR report) under powerbi/.

The report reads the CSVs written by 05_export_bi.py from the folder in the DataFolder parameter.
Re-run this script after changing the layout; edits made in Power BI Desktop are overwritten.
Lineage tags and logical ids are uuid5 values, so regenerating gives identical files.
"""
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrsurv.data import ROOT  # noqa: E402

NAME = "HRAttrition"
PBI = ROOT / "powerbi"
DATA = PBI / "data"
NS = uuid.UUID("6f1c2b0e-4a57-4d7e-9a37-2f0d6c1b8e11")
SCHEMA = "https://developer.microsoft.com/json-schemas/fabric"
BLUE, GREY, INK, MUTED = "#1D4ED8", "#CBD5E1", "#0F172A", "#64748B"
SERIES = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#6250D6", "#E34948"]


def tag(*parts):
    return str(uuid.uuid5(NS, "/".join(parts)))


# ---------------------------------------------------------------- semantic model (TMDL)

def csv_table(name, cols, measures=(), calc_cols=(), sort_by=None):
    """cols: [(column, tmdl type, M type, format or None)]; sort_by: {column: column it sorts by}"""
    sort_by = sort_by or {}
    out = [f"table {name}", f"\tlineageTag: {tag(name)}", ""]
    for m in measures:
        out += measure_lines(name, *m)
    for c, typ, _, fmt in cols:
        out += [f"\tcolumn {q(c)}", f"\t\tdataType: {typ}"]
        if fmt:
            out.append(f"\t\tformatString: {fmt}")
        out += [f"\t\tlineageTag: {tag(name, c)}", "\t\tsummarizeBy: none", f"\t\tsourceColumn: {c}"]
        if c in sort_by:
            out.append(f"\t\tsortByColumn: {sort_by[c]}")
        out.append("")
    for c, typ, expr, fmt in calc_cols:
        out += [f"\tcolumn {q(c)} = {expr}", f"\t\tdataType: {typ}"]
        if fmt:
            out.append(f"\t\tformatString: {fmt}")
        out += [f"\t\tlineageTag: {tag(name, c)}", "\t\tsummarizeBy: none", ""]
    types = ",\n".join(f'\t\t\t\t        {{"{c}", {mt}}}' for c, _, mt, _ in cols)
    out += [f"\tpartition {name} = m", "\t\tmode: import", "\t\tsource =", "\t\t\t\tlet",
            f'\t\t\t\t    Source = Csv.Document(File.Contents(DataFolder & "\\{name}.csv"), '
            '[Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),',
            "\t\t\t\t    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),",
            "\t\t\t\t    Typed = Table.TransformColumnTypes(Promoted, {", types, '\t\t\t\t    }, "en-US")',
            "\t\t\t\tin", "\t\t\t\t    Typed", ""]
    return "\n".join(out)


def param_table(name, col, start, stop, step, fmt, measures=()):
    """What-if parameter: an integer series (value x 100, so lookups are exact) plus the real value."""
    out = [f"table {q(name)}", f"\tlineageTag: {tag(name)}", ""]
    for m in measures:
        out += measure_lines(name, *m)
    out += ["\tcolumn Value", "\t\tdataType: int64", "\t\tformatString: 0", f"\t\tlineageTag: {tag(name, 'Value')}",
            "\t\tsummarizeBy: none", "\t\tisNameInferred", "\t\tsourceColumn: [Value]", "",
            f"\tcolumn {q(col)} = [Value] / 100", "\t\tdataType: double", f"\t\tformatString: {fmt}",
            f"\t\tlineageTag: {tag(name, col)}", "\t\tsummarizeBy: none", "",
            f"\tpartition {q(name)} = calculated", "\t\tmode: import",
            f"\t\tsource = GENERATESERIES ( {start}, {stop}, {step} )", ""]
    return "\n".join(out)


def measure_lines(table, name, expr, fmt=None, folder=None, hidden=False):
    lines = expr.strip().splitlines()
    if len(lines) == 1:
        out = [f"\tmeasure {q(name)} = {lines[0]}"]
    else:
        out = [f"\tmeasure {q(name)} ="] + [f"\t\t\t{ln}" for ln in lines]
    if fmt:
        out.append(f"\t\tformatString: {fmt}")
    if hidden:
        out.append("\t\tisHidden")
    if folder:
        out.append(f"\t\tdisplayFolder: {folder}")
    out += [f"\t\tlineageTag: {tag(table, 'm', name)}", ""]
    return out


def q(name):
    return f"'{name}'" if any(ch in name for ch in " -()/%") else name


def metric(name):
    return f'CALCULATE ( MAX ( metrics[value] ), metrics[metric] = "{name}" )'


METRIC_MEASURES = [
    ("Employees", metric("employees"), "#,0", "KPI"),
    ("Leavers", metric("leavers"), "#,0", "KPI"),
    ("Median Tenure", metric("median_tenure_months"), "0", "KPI"),
    ("Left by 12m", metric("left_by_12m"), "0.0%", "KPI"),
    ("C Index", metric("c_index"), "0.00", "KPI"),
    ("Staff Scored KPI", metric("staff_scored"), "#,0", "KPI"),
    ("Label Employees", '"real survey respondents · " & FORMAT ( [Leavers], "#,0" ) & " have left"', None, "Labels"),
    ("Label Tenure", '"months, half have left by then"', None, "Labels"),
    ("Label Left 12m", '"of new hires leave in their first year"', None, "Labels"),
    ("Label C Index", '"chance = 0.50 · 95% CI " & FORMAT ( ' + metric("c_index_lo") + ', "0.00" ) & "–" & FORMAT ( '
     + metric("c_index_hi") + ', "0.00" )', None, "Labels"),
    ("Label Staff", '"still in the job, with a 12-month risk score"', None, "Labels"),
]

DECISION_MEASURES = [
    ("Sel k", "SELECTEDVALUE ( 'Loss multiple'[k], 1 )", "0.00", "Assumptions"),
    ("Sel c", "SELECTEDVALUE ( 'Intervention cost'[c], 0.10 )", "0.00", "Assumptions"),
    ("Sel e", "SELECTEDVALUE ( 'Effectiveness'[e], 0.30 )", "0%", "Assumptions"),
    ("Risk Threshold", "DIVIDE ( [Sel c], [Sel e] * [Sel k] )", "0.0%", "Decision"),
    ("Staff Scored", "COUNTROWS ( FILTER ( staff, NOT ISBLANK ( staff[p_leave_12m] ) ) )", "#,0", "Decision"),
    ("Staff Targeted", "VAR t = [Risk Threshold]\nRETURN\n    COUNTROWS ( FILTER ( staff, staff[p_leave_12m] > t ) ) + 0",
     "#,0", "Decision"),
    ("Share Targeted", "DIVIDE ( [Staff Targeted], [Staff Scored] )", "0.0%", "Decision"),
    ("Mean Risk", "AVERAGE ( staff[p_leave_12m] )", "0.0%", "Decision"),
    ("Expected Saving",
     "VAR k = [Sel k]\nVAR c = [Sel c]\nVAR e = [Sel e]\nVAR t = [Risk Threshold]\nRETURN\n"
     "    DIVIDE (\n        SUMX ( FILTER ( staff, staff[p_leave_12m] > t ), staff[p_leave_12m] * e * k - c ),\n"
     "        [Staff Scored]\n    ) + 0", "0.0%", "Decision"),
    ("Median Break-even e",
     "VAR k = [Sel k]\nVAR c = [Sel c]\nRETURN\n"
     "    MEDIANX ( FILTER ( staff, staff[p_leave_12m] > 0 ), DIVIDE ( c, staff[p_leave_12m] * k ) )", "0%", "Decision"),
    ("Backtest Saving",
     "LOOKUPVALUE (\n    backtest_grid[saving],\n    backtest_grid[k_x100], ROUND ( [Sel k] * 100, 0 ),\n"
     "    backtest_grid[c_x100], ROUND ( [Sel c] * 100, 0 ),\n    backtest_grid[e_x100], ROUND ( [Sel e] * 100, 0 )\n)",
     "0.0%", "Decision"),
    ("Backtest Value",
     'IF ( ISBLANK ( [Backtest Saving] ), "n/a", FORMAT ( [Backtest Saving], "+0.0%;-0.0%;0.0%" ) )', None, "Decision"),
    ("Row Risk If Targeted",
     "VAR p = MAX ( staff[p_leave_12m] )\nRETURN\n    IF ( p > [Risk Threshold], p )", "0.0%", "Decision"),
    ("Row Net Value",
     "VAR p = MAX ( staff[p_leave_12m] )\nRETURN\n    IF ( p > [Risk Threshold], p * [Sel e] * [Sel k] - [Sel c] )", "0.000", "Decision"),
    ("Row Break-even e",
     "VAR p = MAX ( staff[p_leave_12m] )\nRETURN\n    IF ( p > [Risk Threshold], DIVIDE ( [Sel c], p * [Sel k] ) )", "0%", "Decision"),
        ("Bin Color", f'IF ( MAX ( staff[risk_bin] ) + 0.0125 > [Risk Threshold], "{BLUE}", "{GREY}" )', None, "Colors"),
    ("Label Threshold", '"c ÷ (e × k): intervene above this"', None, "Labels"),
    ("Label Targeted", 'FORMAT ( [Share Targeted], "0.0%" ) & " of " & FORMAT ( [Staff Scored], "#,0" ) & " scored staff"',
     None, "Labels"),
    ("Label Expected", '"net saving per employee, × salary"', None, "Labels"),
    ("Label Backtest",
     "VAR lo = LOOKUPVALUE ( backtest_grid[lo95], backtest_grid[k_x100], ROUND ( [Sel k] * 100, 0 ), "
     "backtest_grid[c_x100], ROUND ( [Sel c] * 100, 0 ), backtest_grid[e_x100], ROUND ( [Sel e] * 100, 0 ) )\n"
     "VAR hi = LOOKUPVALUE ( backtest_grid[hi95], backtest_grid[k_x100], ROUND ( [Sel k] * 100, 0 ), "
     "backtest_grid[c_x100], ROUND ( [Sel c] * 100, 0 ), backtest_grid[e_x100], ROUND ( [Sel e] * 100, 0 ) )\nRETURN\n"
     '    IF ( ISBLANK ( [Backtest Saving] ), "not a back-tested scenario",\n'
     '        "95% CI " & FORMAT ( lo, "0.0%" ) & " to " & FORMAT ( hi, "0.0%" ) )',
     None, "Labels"),
    ("Label Break-even", '"median needed to break even"', None, "Labels"),
    ("Title Histogram",
     '"12-month leaving risk, current staff · blue = above " & FORMAT ( [Risk Threshold], "0.0%" )', None, "Titles"),
    ("Title Staff Table",
     'FORMAT ( [Staff Targeted], "#,0" ) & " staff to target at k = " & FORMAT ( [Sel k], "0.00" ) & ", c = " '
     '& FORMAT ( [Sel c], "0.00" ) & ", e = " & FORMAT ( [Sel e], "0%" ) & ", highest risk first"', None, "Titles"),
]


def model_files():
    files = {}
    files["definition/database.tmdl"] = "database\n\tcompatibilityLevel: 1567\n"
    tables = ["metrics", "factors", "km_curves", "group_levels", "factor_tests", "hazard_ratios", "staff",
              "backtest_grid", "calibration", "fairness_calibration", "Loss multiple", "Intervention cost",
              "Effectiveness"]
    files["definition/model.tmdl"] = "\n".join([
        "model Model", "\tculture: en-US", "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
        "\tsourceQueryCulture: en-US", "\tdataAccessOptions", "\t\tlegacyRedirects", "\t\treturnErrorValuesAsNull", "",
        "annotation __PBI_TimeIntelligenceEnabled = 0", "",
        f"annotation PBI_QueryOrder = {json.dumps(['DataFolder'] + tables[:10])}", "",
        'annotation PBI_ProTooling = ["DevMode"]', "", *[f"ref table {q(t)}" for t in tables], ""])
    folder = str(DATA).replace("\\", "\\\\") if False else str(DATA)
    files["definition/expressions.tmdl"] = (
        f'expression DataFolder = "{folder}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n'
        f"\tlineageTag: {tag('DataFolder')}\n\n\tannotation PBI_ResultType = Text\n")
    files["definition/relationships.tmdl"] = "\n".join([
        f"relationship {tag('rel', 'km')}", "\tfromColumn: km_curves.factor", "\ttoColumn: factors.factor", "",
        f"relationship {tag('rel', 'levels')}", "\tfromColumn: group_levels.factor", "\ttoColumn: factors.factor", ""])

    txt, i64, dbl = ("string", "type text", None), ("int64", "Int64.Type", "#,0"), ("double", "type number", None)

    def cols(*spec):
        return [(n, t[0], t[1], fmt if fmt is not None else t[2]) for n, t, fmt in spec]

    T = {}
    T["metrics"] = csv_table("metrics", cols(("metric", txt, None), ("value", dbl, "0.0000")),
                             [(n, e, f, d) for n, e, f, d in METRIC_MEASURES])
    T["factors"] = csv_table("factors", cols(("factor", txt, None), ("factor_order", i64, None),
                                             ("protected", ("boolean", "type logical", None), None)))
    T["km_curves"] = csv_table("km_curves", cols(("factor", txt, None), ("level", txt, None), ("month", i64, "0"),
                                                 ("still_employed", dbl, "0%")),
                               [("Still Employed", "AVERAGE ( km_curves[still_employed] )", "0%", "Base")])
    T["group_levels"] = csv_table("group_levels", cols(("factor", txt, None), ("level", txt, None), ("n", i64, None),
                                           ("events", i64, None), ("left_by_12m", dbl, "0.0%"),
                                           ("median_months", dbl, "0")),
                            [("Left Within 12 Months", "AVERAGE ( group_levels[left_by_12m] )", "0%", "Base")])
    T["factor_tests"] = csv_table("factor_tests", cols(("factor", txt, None), ("df", i64, None),
                                                       ("robust_p", dbl, "0.0000"), ("robust_q_bh", dbl, "0.0000"), ("p_text", txt, None), ("q_text", txt, None),
                                                       ("p_rank", i64, None)), sort_by={"factor": "p_rank"})
    T["hazard_ratios"] = csv_table("hazard_ratios", cols(
        ("covariate", txt, None), ("hazard_ratio", dbl, "0.00"), ("hr_lo", dbl, "0.00"), ("hr_hi", dbl, "0.00"),
        ("p_robust", dbl, "0.000"), ("label", txt, None), ("significant", txt, None), ("p_text", txt, None)))
    T["staff"] = csv_table("staff", cols(
        ("row_id", i64, "0"), ("tenure_months", dbl, "0"), ("industry", txt, None), ("profession", txt, None),
        ("traffic", txt, None), ("coach", txt, None), ("greywage", txt, None), ("way", txt, None),
        ("p_leave_12m", dbl, "0.0%"), ("risk_bin", dbl, "0.0%"), ("gender", txt, None), ("age_band", txt, None)),
        DECISION_MEASURES)
    T["backtest_grid"] = csv_table("backtest_grid", cols(
        ("k", dbl, "0.0"), ("c", dbl, "0.00"), ("e", dbl, "0%"), ("threshold", dbl, "0.0%"),
        ("share_targeted", dbl, "0.0%"), ("saving", dbl, "0.0%"), ("lo95", dbl, "0.0%"), ("hi95", dbl, "0.0%"),
        ("k_x100", i64, "0"), ("c_x100", i64, "0"), ("e_x100", i64, "0")))
    T["calibration"] = csv_table("calibration", cols(("bin", i64, "0"), ("n", i64, None), ("predicted", dbl, "0.0%"),
                                                     ("observed_km", dbl, "0.0%")),
                                 [("Predicted Rate", "AVERAGE ( calibration[predicted] )", "0.0%", "Base"),
                                  ("Observed Rate", "AVERAGE ( calibration[observed_km] )", "0.0%", "Base")],
                                 [("quintile", "string", '"Q" & calibration[bin]', None)])
    T["fairness_calibration"] = csv_table("fairness_calibration", cols(
        ("attribute", txt, None), ("group", txt, None), ("n", i64, None), ("predicted_12m", dbl, "0.0%"),
        ("observed_12m_km", dbl, "0.0%"), ("order", i64, None)), sort_by={"group": "order"})
    T["Loss multiple"] = param_table("Loss multiple", "k", 50, 300, 25, "0.00")
    T["Intervention cost"] = param_table("Intervention cost", "c", 1, 30, 1, "0.00")
    T["Effectiveness"] = param_table("Effectiveness", "e", 5, 100, 5, "0%")
    for name, body in T.items():
        files[f"definition/tables/{name}.tmdl"] = body
    files[".platform"] = platform("SemanticModel")
    files["definition.pbism"] = json.dumps({
        "$schema": f"{SCHEMA}/item/semanticModel/definitionProperties/1.0.0/schema.json",
        "version": "4.2", "settings": {}}, indent=2)
    return files


def platform(kind):
    return json.dumps({
        "$schema": f"{SCHEMA}/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": kind, "displayName": NAME},
        "config": {"version": "2.0", "logicalId": tag("platform", kind)}}, indent=2)


# ---------------------------------------------------------------- report (PBIR)

def lit(v):
    return {"expr": {"Literal": {"Value": v}}}


def s(text):
    return lit("'" + text.replace("'", "''") + "'")


def color(hexv):
    return {"solid": {"color": lit(f"'{hexv}'")}}


def measure_ref(entity, prop):
    return {"Measure": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def col_ref(entity, prop):
    return {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def proj(entity, prop, display=None, measure=False, active=None):
    p = {"field": measure_ref(entity, prop) if measure else col_ref(entity, prop),
         "queryRef": f"{entity}.{prop}", "nativeQueryRef": prop}
    if display:
        p["displayName"] = display
    if active:
        p["active"] = True
    return p


def M(entity, prop, display=None):
    return proj(entity, prop, display, measure=True)


def C(entity, prop, display=None, active=None):
    return proj(entity, prop, display, active=active)


def padding(t, b, l, r):
    return [{"properties": {"top": lit(f"{t}D"), "bottom": lit(f"{b}D"), "left": lit(f"{l}D"), "right": lit(f"{r}D")}}]


class Page:
    def __init__(self, name, display):
        self.name, self.display, self.visuals, self.z, self.interactions = name, display, {}, 1000, []

    def add(self, name, pos, visual, title=None, subtitle=None, alt=None, background=True, pad=None):
        x, y, w, h = pos
        self.z += 1000
        vco = {"title": [{"properties": {"show": lit("true" if title else "false")}}],
               "background": [{"properties": {"show": lit("true" if background else "false")}}]}
        if title:
            t = vco["title"][0]["properties"]
            t["text"] = title if isinstance(title, dict) else s(title)
        if subtitle:
            vco["subTitle"] = [{"properties": {"show": lit("true"), "text": s(subtitle)}}]
        if alt:
            vco["general"] = [{"properties": {"altText": s(alt)}}]
        if pad:
            vco["padding"] = padding(*pad)
        self.visuals[name] = {
            "$schema": f"{SCHEMA}/item/report/definition/visualContainer/2.3.0/schema.json",
            "name": name, "position": {"x": x, "y": y, "z": self.z, "height": h, "width": w, "tabOrder": self.z},
            "visual": {**visual, "visualContainerObjects": vco, "drillFilterOtherVisuals": True}}

    def text(self, name, pos, paragraphs, background=False):
        runs = [{"textRuns": [{"value": v, "textStyle": {"fontFamily": f, "fontSize": sz, "color": c}}]}
                for v, f, sz, c in paragraphs]
        self.add(name, pos, {"visualType": "textbox", "objects": {"general": [{"properties": {"paragraphs": runs}}]}},
                 background=background, pad=(0, 0, 0, 0) if not background else (12, 12, 16, 16))

    def header(self, pos, title, subtitle):
        self.text("header", pos, [(title, "Segoe UI Semibold", "18pt", INK), (subtitle, "Segoe UI", "9pt", MUTED)])

    def kpi(self, key, pos, entity, value, label, title, alt):
        x, y, w, h = pos
        self.add(f"kpi_{key}_tile", pos, {"visualType": "textbox",
                                          "objects": {"general": [{"properties": {"paragraphs": []}}]}},
                 pad=(0, 0, 0, 0))
        self.add(f"kpi_{key}_value", (x, y, w, 76),
                 {"visualType": "card", "query": {"queryState": {"Values": {"projections": [M(entity, value)]}}},
                  "objects": {"labels": [{"properties": {"fontSize": lit("24D")}}]}},
                 title=title, alt=alt, background=False, pad=(8, 0, 16, 16))
        v = self.visuals[f"kpi_{key}_value"]["visual"]["visualContainerObjects"]["title"][0]["properties"]
        v.update({"fontSize": lit("10D"), "fontColor": color(MUTED), "alignment": s("left")})
        self.add(f"kpi_{key}_label", (x, y + 76, w, h - 76),
                 {"visualType": "card", "query": {"queryState": {"Values": {"projections": [M(entity, label)]}}},
                  "objects": {"labels": [{"properties": {"fontSize": lit("9D"), "color": color(MUTED)}}]}},
                 background=False, pad=(0, 0, 16, 16))

    def slicer(self, name, pos, entity, column, display, default=None, alt=None):
        objects = {"data": [{"properties": {"mode": s("Dropdown")}}],
                   "selection": [{"properties": {"singleSelect": lit("true")}}],
                   "header": [{"properties": {"show": lit("true"), "text": s(display)}}]}
        if default is not None:
            objects["general"] = [{"properties": {"filter": {"filter": {
                "Version": 2, "From": [{"Name": "p", "Entity": entity, "Type": 0}],
                "Where": [{"Condition": {"In": {
                    "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "p"}}, "Property": column}}],
                    "Values": [[{"Literal": {"Value": default}}]]}}}]}}}}]
        self.add(name, pos, {"visualType": "slicer",
                             "query": {"queryState": {"Values": {"projections": [C(entity, column, display, True)]}}},
                             "objects": objects}, alt=alt, pad=(4, 4, 12, 12))

    def chart(self, name, pos, vtype, query, title, subtitle=None, alt=None, objects=None, sort=None, pad=None):
        v = {"visualType": vtype, "query": {"queryState": query}, "objects": objects or {}}
        if sort:
            v["query"]["sortDefinition"] = {"sort": [{"field": sort[0], "direction": sort[1]}]}
        self.add(name, pos, v, title=title, subtitle=subtitle, alt=alt, pad=pad)

    def table(self, name, pos, projections, title, subtitle=None, alt=None, sort=None):
        self.chart(name, pos, "tableEx", {"Values": {"projections": projections}}, title, subtitle, alt,
                   objects={"total": [{"properties": {"totals": lit("false")}}],
                            "columnHeaders": [{"properties": {"wordWrap": lit("true")}}]}, sort=sort,
                   pad=(8, 8, 16, 16))

    def files(self):
        bg = {"color": color("#F4F6F9"), "transparency": lit("0D")}
        out = {f"definition/pages/{self.name}/page.json": {
            "$schema": f"{SCHEMA}/item/report/definition/page/2.0.0/schema.json",
            "name": self.name, "displayName": self.display, "displayOption": "FitToPage", "height": 720, "width": 1280,
            "objects": {"background": [{"properties": bg}], "outspace": [{"properties": bg}]},
            **({"visualInteractions": self.interactions} if self.interactions else {})}}
        for vn, v in self.visuals.items():
            out[f"definition/pages/{self.name}/visuals/{vn}/visual.json"] = v
        return out


KPI_X = [24, 272, 520, 768, 1016]  # 5 x 240 + 4 x 8 gutter + 2 x 24 margin = 1280


def page_overview():
    p = Page("who_leaves", "Who leaves")
    p.header((24, 24, 984, 56), "Employee turnover · who leaves, and how fast",
             "Real survey of 1,116 Russian office workers (Babushkin, 2015–17), one job each · Kaplan–Meier survival "
             "and a Cox model · pick a factor on the right")
    p.slicer("slicer_factor", (1016, 24, 240, 56), "factors", "factor", "Factor", default="'Industry'",
             alt="Choose which factor splits the survival curves and the bar chart.")
    kp = [("emp", "Employees", "Label Employees", "Employees in the data"),
          ("tenure", "Median Tenure", "Label Tenure", "Median tenure"),
          ("left", "Left by 12m", "Label Left 12m", "Leave within 12 months"),
          ("c", "C Index", "Label C Index", "Model ranking accuracy (C-index)"),
          ("staff", "Staff Scored KPI", "Label Staff", "Current staff scored")]
    for x, (key, val, lab, title) in zip(KPI_X, kp):
        p.kpi(key, (x, 88, 240, 104), "metrics", val, lab, title, f"{title}.")
    p.chart("km_chart", (24, 200, 600, 296), "lineChart",
            {"Category": {"projections": [C("km_curves", "month", active=True)]},
             "Y": {"projections": [M("km_curves", "Still Employed")]},
             "Series": {"projections": [C("km_curves", "level")]}},
            "Share still employed by months of tenure",
            "Kaplan–Meier estimate per group · steeper = people leave sooner",
            "Line chart of the share still employed against tenure in months, one line per group of the chosen factor.")
    bar_sort = (measure_ref("group_levels", "Left Within 12 Months"), "Descending")
    p.chart("levels_bar", (632, 200, 624, 296), "clusteredBarChart",
            {"Category": {"projections": [C("group_levels", "level", active=True)]},
             "Y": {"projections": [M("group_levels", "Left Within 12 Months")]}},
            "Left within the first 12 months, by group", "Kaplan–Meier, all groups of the chosen factor",
            "Bar chart of the share leaving within 12 months for each group of the chosen factor.",
            objects={"dataPoint": [{"properties": {"fill": color(BLUE)}}],
                     "labels": [{"properties": {"show": lit("true")}}]},
            sort=bar_sort)
    p.table("hr_table", (24, 504, 800, 192),
            [C("hazard_ratios", "label", "Covariate (Cox model, adjusted for all others)"),
             C("hazard_ratios", "hazard_ratio", "Hazard ratio"), C("hazard_ratios", "hr_lo", "95% CI low"),
             C("hazard_ratios", "hr_hi", "95% CI high"), C("hazard_ratios", "p_text", "p")],
            "What raises or lowers the leaving rate", "Hazard ratio > 1 = leave sooner · cluster-robust CIs · "
            "associations, not causes", "Table of Cox model hazard ratios with confidence intervals.",
            sort=(col_ref("hazard_ratios", "hazard_ratio"), "Descending"))
    p.table("tests_table", (832, 504, 424, 192),
            [C("factor_tests", "factor", "Factor"), C("factor_tests", "p_text", "p"),
             C("factor_tests", "q_text", "q (BH)")],
            "Does the factor matter on its own?", "Cluster-robust Wald test · q corrects for testing 14 factors",
            "Table of univariate test p-values and false discovery rate q-values by factor.",
            sort=(col_ref("factor_tests", "factor"), "Ascending"))
    return p


def page_decision():
    p = Page("decision", "Decision simulator")
    p.header((24, 24, 488, 56), "Retention decision simulator",
             "Intervene when risk × effectiveness × loss > cost · costs in multiples of one annual salary")
    p.slicer("slicer_k", (520, 24, 240, 56), "Loss multiple", "k", "Loss if they leave, k",
             default="1D", alt="Choose the cost of losing an employee as a multiple of salary.")
    p.slicer("slicer_c", (768, 24, 240, 56), "Intervention cost", "c", "Programme cost, c",
             default="0.1D", alt="Choose the cost of the retention intervention as a multiple of salary.")
    p.slicer("slicer_e", (1016, 24, 240, 56), "Effectiveness", "e", "Effectiveness, e",
             default="0.3D", alt="Choose the share of would-be leavers the intervention keeps.")
    kp = [("thr", "Risk Threshold", "Label Threshold", "Risk threshold"),
          ("tgt", "Staff Targeted", "Label Targeted", "Staff to target"),
          ("exp", "Expected Saving", "Label Expected", "Expected saving (model)"),
          ("bt", "Backtest Value", "Label Backtest", "Back-tested gain from targeting"),
          ("be", "Median Break-even e", "Label Break-even", "Break-even effectiveness")]
    for x, (key, val, lab, title) in zip(KPI_X, kp):
        p.kpi(key, (x, 88, 240, 104), "staff", val, lab, title, f"{title}.")
    p.chart("risk_hist", (24, 200, 600, 304), "clusteredColumnChart",
            {"Category": {"projections": [C("staff", "risk_bin", "Predicted risk (2.5-point bins)", True)]},
             "Y": {"projections": [M("staff", "Staff Scored", "Staff")]}},
            {"expr": measure_ref("staff", "Title Histogram")},
            "Cox model, P(leave in next 12 months | tenure so far) · nobody is far from the 14% base rate",
            "Histogram of predicted 12-month leaving risk for current staff, bars above the threshold in blue.",
            objects={"dataPoint": [{"properties": {"fill": {"solid": {"color": {"expr": measure_ref(
                "staff", "Bin Color")}}}},
                "selector": {"data": [{"dataViewWildcard": {"matchingOption": 1}}]}}],
                "categoryAxis": [{"properties": {"axisType": s("Categorical")}}]},
            sort=(col_ref("staff", "risk_bin"), "Ascending"))
    group_cols = [M("staff", "Staff Scored", "Staff"), M("staff", "Mean Risk", "Mean risk"),
                  M("staff", "Share Targeted", "Targeted")]
    p.table("fair_gender", (632, 200, 624, 112), [C("staff", "gender", "Gender (not a model input)")] + group_cols,
            "Who gets targeted · by gender", None, "Table of targeting rate by gender.")
    p.table("fair_age", (632, 320, 624, 184), [C("staff", "age_band", "Age at hire (not a model input)")] + group_cols,
            "By age at hire · risk under-predicted for 36+", None, "Table of targeting rate by age band at hire.",
            sort=(col_ref("staff", "age_band"), "Ascending"))
    p.table("staff_table", (24, 512, 1232, 184),
            [C("staff", "row_id", "Row"), C("staff", "tenure_months", "Tenure (months)"),
             C("staff", "industry", "Industry"), C("staff", "profession", "Profession"),
             C("staff", "traffic", "Hiring channel"), C("staff", "greywage", "Wage type"),
             M("staff", "Row Risk If Targeted", "12-month risk"), M("staff", "Row Net Value", "Net value (× salary)"),
             M("staff", "Row Break-even e", "Break-even e")],
            {"expr": measure_ref("staff", "Title Staff Table")},
            "Survey respondents still in their job, not one company's staff · net value = risk × e × k − c",
            "Table of staff above the threshold with their risk, expected net value and break-even effectiveness.",
            sort=(measure_ref("staff", "Row Risk If Targeted"), "Descending"))
    p.interactions = [{"source": f"slicer_{x}", "target": "risk_hist", "type": "NoFilter"} for x in "kce"]
    return p


def page_model():
    p = Page("model_check", "Model check")
    p.header((24, 24, 1232, 56), "Model check · is the risk score good enough to act on?",
             "Out-of-fold predictions (grouped 5-fold CV) · back-test weights people with unknown 12-month status "
             "(IPCW) · bootstrap CIs resample people")
    p.chart("calibration", (24, 88, 600, 304), "clusteredColumnChart",
            {"Category": {"projections": [C("calibration", "quintile", "Risk quintile", True)]},
             "Y": {"projections": [M("calibration", "Predicted Rate", "Predicted"), M("calibration", "Observed Rate", "Observed")]}},
            "Predicted vs observed 12-month leaving, by risk quintile",
            "Bars should match · predictions are squeezed: 9–21% predicted vs 7–24% observed",
            "Column chart comparing predicted and observed 12-month leaving rates in five risk quintiles.",
            objects={"dataPoint": [{"properties": {"fill": color(BLUE)},
                                    "selector": {"metadata": "calibration.Predicted Rate"}},
                                   {"properties": {"fill": color(GREY)},
                                    "selector": {"metadata": "calibration.Observed Rate"}}],
                     "labels": [{"properties": {"show": lit("true")}}]},
            sort=(col_ref("calibration", "quintile"), "Ascending"))
    p.table("backtest", (632, 88, 624, 304),
            [C("backtest_grid", "k", "k"), C("backtest_grid", "c", "c"), C("backtest_grid", "e", "e"),
             C("backtest_grid", "threshold", "Threshold"), C("backtest_grid", "share_targeted", "Targeted"),
             C("backtest_grid", "saving", "Gain"), C("backtest_grid", "lo95", "CI low"),
             C("backtest_grid", "hi95", "CI high")],
            "Back-test: gain from targeting over the better simple policy",
            "45 scenarios · gain in share of one salary per employee · CI above 0 in only 4",
            "Table of the 45 back-tested cost scenarios with the gain from targeting and its confidence interval.",
            sort=(col_ref("backtest_grid", "saving"), "Descending"))
    p.table("fair_cal", (24, 400, 600, 296),
            [C("fairness_calibration", "attribute", "Attribute"), C("fairness_calibration", "group", "Group"),
             C("fairness_calibration", "n", "n"), C("fairness_calibration", "predicted_12m", "Predicted"),
             C("fairness_calibration", "observed_12m_km", "Observed")],
            "Calibration by group (attributes not used by the model)",
            "Leaving age out under-predicts risk for those hired at 36–40 and 41+",
            "Table of predicted and observed 12-month leaving rates by gender and age band.",
            sort=(col_ref("fairness_calibration", "group"), "Ascending"))
    p.text("notes", (632, 400, 624, 296), [
        ("How to read this report", "Segoe UI Semibold", "12pt", INK),
        ("The model ranks people only a little better than chance (C-index 0.60). Its 12-month error is about 1.7% "
         "lower than giving everyone the same 14% risk.", "Segoe UI", "10pt", INK),
        ("So the decision is driven by the assumptions. Targeting beats both 'nobody' and 'everybody' only when "
         "the threshold c ÷ (e·k) lands near the 14% base rate, and the best gain is 4.3% of one salary per employee.",
         "Segoe UI", "10pt", INK),
        ("The number most worth measuring is e, the intervention's effectiveness, with a small randomised pilot.",
         "Segoe UI", "10pt", INK),
        ("Caveats: a self-selected survey, not one company · personality was measured at survey time · "
         "associations, not causes · data CC BY-NC-SA 4.0, © Edward Babushkin.", "Segoe UI", "9pt", MUTED),
    ], background=True)
    return p


def report_files():
    pages = [page_overview(), page_decision(), page_model()]
    files = {
        ".platform": platform("Report"),
        "definition.pbir": {"$schema": f"{SCHEMA}/item/report/definitionProperties/2.0.0/schema.json",
                            "version": "4.0", "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}}},
        "definition/version.json": {"$schema": f"{SCHEMA}/item/report/definition/versionMetadata/1.0.0/schema.json",
                                    "version": "2.0.0"},
        "definition/report.json": {
            "$schema": f"{SCHEMA}/item/report/definition/report/3.0.0/schema.json",
            "themeCollection": {"customTheme": {"name": f"{NAME}.json", "reportVersionAtImport": {
                "visual": "2.3.0", "report": "3.0.0", "page": "2.0.0"}, "type": "RegisteredResources"}},
            "objects": {"outspacePane": [{"properties": {"expanded": lit("false")}}]},
            "resourcePackages": [{"name": "RegisteredResources", "type": "RegisteredResources", "items": [
                {"name": f"{NAME}.json", "path": f"{NAME}.json", "type": "CustomTheme"}]}],
            "settings": {"useEnhancedTooltips": True}},
        "definition/pages/pages.json": {"$schema": f"{SCHEMA}/item/report/definition/pagesMetadata/1.0.0/schema.json",
                                        "pageOrder": [p.name for p in pages], "activePageName": pages[0].name},
        f"StaticResources/RegisteredResources/{NAME}.json": theme(),
    }
    for p in pages:
        files.update(p.files())
    return files


def theme():
    axis = {"labelColor": {"solid": {"color": MUTED}}, "fontSize": 9, "fontFamily": "Segoe UI", "showAxisTitle": False}
    return {
        "name": NAME, "dataColors": SERIES, "background": "#FFFFFF", "foreground": INK, "tableAccent": BLUE,
        "good": "#16A34A", "neutral": MUTED, "bad": "#DC2626",
        "textClasses": {"callout": {"fontSize": 26, "fontFace": "Segoe UI Semibold", "color": INK},
                        "title": {"fontSize": 12, "fontFace": "Segoe UI Semibold", "color": INK},
                        "header": {"fontSize": 10, "fontFace": "Segoe UI Semibold", "color": INK},
                        "label": {"fontSize": 9, "fontFace": "Segoe UI", "color": MUTED}},
        "visualStyles": {
            "page": {"*": {"background": [{"color": {"solid": {"color": "#F4F6F9"}}, "transparency": 0}],
                           "outspace": [{"color": {"solid": {"color": "#F4F6F9"}}, "transparency": 0}]}},
            "*": {"*": {"background": [{"show": True, "color": {"solid": {"color": "#FFFFFF"}}, "transparency": 0}],
                        "border": [{"show": False, "radius": 6}], "dropShadow": [{"show": False}],
                        "padding": [{"top": 12, "bottom": 12, "left": 16, "right": 16}],
                        "title": [{"show": True, "fontColor": {"solid": {"color": INK}}, "fontSize": 12,
                                   "fontFamily": "Segoe UI Semibold", "alignment": "left"}],
                        "subTitle": [{"fontColor": {"solid": {"color": MUTED}}, "fontSize": 9}]}},
            "card": {"*": {"labels": [{"color": {"solid": {"color": INK}}, "fontSize": 26,
                                       "fontFamily": "Segoe UI Semibold"}], "categoryLabels": [{"show": False}]}},
            "lineChart": {"*": {"lineStyles": [{"strokeWidth": 2}], "categoryAxis": [{**axis, "gridlineShow": False}],
                                "valueAxis": [{**axis, "gridlineShow": True,
                                               "gridlineColor": {"solid": {"color": "#E2E8F0"}}}],
                                "legend": [{"show": True, "position": "Top", "fontSize": 9, "showTitle": False,
                                            "labelColor": {"solid": {"color": MUTED}}}]}},
            "clusteredBarChart": {"*": {"categoryAxis": [axis], "valueAxis": [{**axis, "show": False}]}},
            "clusteredColumnChart": {"*": {"categoryAxis": [axis], "valueAxis": [{**axis, "gridlineShow": True,
                                     "gridlineColor": {"solid": {"color": "#E2E8F0"}}}],
                                     "legend": [{"show": True, "position": "Top", "fontSize": 9, "showTitle": False}]}},
            "tableEx": {"*": {"grid": [{"gridHorizontal": True, "gridHorizontalColor": {"solid": {"color": "#E2E8F0"}},
                                        "gridVertical": False, "rowPadding": 2}],
                              "columnHeaders": [{"fontColor": {"solid": {"color": MUTED}}, "fontSize": 9,
                                                 "backColor": {"solid": {"color": "#FFFFFF"}}}],
                              "values": [{"fontSize": 9, "fontColorPrimary": {"solid": {"color": INK}},
                                          "backColorPrimary": {"solid": {"color": "#FFFFFF"}},
                                          "backColorSecondary": {"solid": {"color": "#FFFFFF"}}}]}},
            "slicer": {"*": {"header": [{"fontColor": {"solid": {"color": MUTED}}, "fontSize": 9}],
                             "items": [{"fontColor": {"solid": {"color": INK}}, "fontSize": 10}]}},
        }}


def write(base: Path, files: dict):
    for rel, content in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False)
        path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")


def main():
    for part in ["Report", "SemanticModel"]:
        d = PBI / f"{NAME}.{part}"
        if d.exists():
            for sub in d.iterdir():  # keep Desktop's .pbi cache folder, replace the definition
                if sub.name != ".pbi":
                    shutil.rmtree(sub) if sub.is_dir() else sub.unlink()
    write(PBI / f"{NAME}.SemanticModel", model_files())
    write(PBI / f"{NAME}.Report", report_files())
    write(PBI, {f"{NAME}.pbip": {
        "$schema": f"{SCHEMA}/pbip/pbipProperties/1.0.0/schema.json", "version": "1.0",
        "artifacts": [{"report": {"path": f"{NAME}.Report"}}], "settings": {"enableAutoRecovery": True}},
        ".gitignore": "**/.pbi/localSettings.json\n**/.pbi/cache.abf\n"})
    n = sum(1 for _ in (PBI / f"{NAME}.Report").rglob("visual.json"))
    print(f"wrote {PBI / (NAME + '.pbip')} ({n} visuals)")


if __name__ == "__main__":
    main()
