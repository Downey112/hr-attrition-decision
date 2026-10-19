"""Load, clean and encode the Babushkin employee-turnover data."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "turnover.csv"
SHA256 = "5098600655806f73c151b30432cf6697437928a30283e88c5c0a1cffeac65ceb"

PERSONALITY = ["extraversion", "independ", "selfcontrol", "anxiety", "novator"]
CATEGORICAL = ["industry", "profession", "traffic", "coach", "greywage", "way"]
# protected or third-party attributes: used for auditing and comparison, never in the decision model
PROTECTED = ["gender", "age", "head_gender"]
MIN_LEVEL = 30  # rarer categories are pooled into "other"


def load_raw(path: Path = RAW) -> pd.DataFrame:
    # the file is single-byte encoded; one Cyrillic letter in "Finance" does not survive any codec
    return pd.read_csv(path, encoding="latin-1")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    for c in ["gender", "industry", "profession", "traffic", "coach", "head_gender", "greywage", "way"]:
        d[c] = d[c].astype(str).str.strip()
    d.loc[d["profession"].str.startswith("Finan"), "profession"] = "Finance"
    d["industry"] = d["industry"].replace({"etc": "other"})
    d["profession"] = d["profession"].replace({"etc": "other"})
    d = d.drop_duplicates().reset_index(drop=True)
    d["person_group"] = linked_groups(d)
    return d


LINK_KEYS = {"personality": ["gender", *PERSONALITY], "spell": ["stag", "event", "age", "industry", "profession"]}


def linked_groups(d: pd.DataFrame, keys=("personality", "spell")) -> np.ndarray:
    """Connected components of rows that may describe the same person.

    By default two rows are linked if they share gender plus all five personality scores, or share
    tenure, event, age, industry and profession. Grouped CV and cluster-robust errors use this id.
    Chaining can merge different people; that makes CV and robust errors more conservative, not
    invalid, and 03_cox reports a sensitivity check with the personality key alone.
    """
    keys = [d[LINK_KEYS[k]] for k in keys]
    parent = np.arange(len(d))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in keys:
        first = {}
        for i, row in enumerate(map(tuple, k.to_numpy())):
            j = first.setdefault(row, i)
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    roots = np.array([find(i) for i in range(len(d))])
    return pd.factorize(roots)[0]


def lump(s: pd.Series, min_count: int = MIN_LEVEL) -> pd.Series:
    counts = s.value_counts()
    return s.where(s.map(counts) >= min_count, "other")


def design(d: pd.DataFrame, protected: bool = False) -> pd.DataFrame:
    """Covariate matrix for the Cox model: dummy-coded categoricals (largest level is the reference)
    plus standardised personality scores. With protected=True, gender, age and head_gender are added."""
    parts = []
    for c in CATEGORICAL:
        s = lump(d[c])
        ref = s.value_counts().index[0]
        dummies = pd.get_dummies(s, prefix=c, dtype=float)
        parts.append(dummies.drop(columns=f"{c}_{ref}"))
    z = (d[PERSONALITY] - d[PERSONALITY].mean()) / d[PERSONALITY].std()
    parts.append(z)
    if protected:
        parts.append(pd.DataFrame({
            "male": (d["gender"] == "m").astype(float),
            "age_z": (d["age"] - d["age"].mean()) / d["age"].std(),
            "head_male": (d["head_gender"] == "m").astype(float),
        }, index=d.index))
    return pd.concat(parts, axis=1)


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 25, 30, 35, 40, 100], labels=["<=25", "26-30", "31-35", "36-40", "41+"])
