import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrsurv import data as D  # noqa: E402

pytestmark = pytest.mark.skipif(not D.RAW.exists(), reason="run scripts/01_fetch_data.py first")


@pytest.fixture(scope="module")
def d():
    return D.clean(D.load_raw())


def test_clean_invariants(d):
    assert len(D.load_raw()) == 1129
    assert len(d) == 1116 and not d.drop(columns="person_group").duplicated().any()
    assert d["event"].isin([0, 1]).all() and (d["stag"] > 0).all()
    assert d["event"].sum() == 560
    assert "Finance" in set(d["profession"]) and not d["profession"].str.contains("�|\xf1").any()
    assert not d["industry"].str.startswith(" ").any()


def test_design_excludes_protected(d):
    X = D.design(d)
    assert not {"male", "age_z", "head_male"} & set(X.columns)
    assert not [c for c in X.columns if c.startswith(("gender", "age", "head_gender"))]
    assert X.notna().all().all() and len(X) == len(d)
    assert {"male", "age_z", "head_male"} <= set(D.design(d, protected=True).columns)
