"""Download turnover.csv from Kaggle (davinwijaya/employee-turnover) and verify its checksum.

Needs a Kaggle API token in ~/.kaggle/access_token (or KAGGLE_API_TOKEN). Skips the download
if the file is already present with the expected hash.
"""
import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrsurv.data import RAW, SHA256  # noqa: E402

URL = "https://www.kaggle.com/api/v1/datasets/download/davinwijaya/employee-turnover"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token() -> str:
    t = os.environ.get("KAGGLE_API_TOKEN")
    if t:
        return t.strip()
    p = Path.home() / ".kaggle" / "access_token"
    if not p.exists():
        sys.exit("No Kaggle token: set KAGGLE_API_TOKEN or create ~/.kaggle/access_token")
    return p.read_text().strip()


def main():
    if RAW.exists() and sha256(RAW) == SHA256:
        print(f"{RAW.name} present, checksum ok")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {token()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        RAW.write_bytes(z.read("turnover.csv"))
    got = sha256(RAW)
    if got != SHA256:
        sys.exit(f"checksum mismatch: {got}")
    print(f"downloaded {RAW.name}, checksum ok")


if __name__ == "__main__":
    main()
