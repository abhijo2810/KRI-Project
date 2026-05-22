# scripts/50_merge_all_metrics.py
from pathlib import Path
import pandas as pd

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT    = SCRIPT_DIR.parent
CSV_DIR = ROOT / "outputs" / "csv"
OUT_FILE = CSV_DIR / "kelp_metrics_master.csv"

# ---------- Load each CSV if present ----------
files = {
    "area_lightness": CSV_DIR / "area_lightness.csv",
    "fraying":        CSV_DIR / "fraying.csv",
    "invasion_depth": CSV_DIR / "invasion_depth.csv",
    "bry_fast":       CSV_DIR / "bryozoan_fast_metrics.csv",
}

dfs = {}
for name, path in files.items():
    if path.exists():
        dfs[name] = pd.read_csv(path)
        print(f"Loaded {name}: {len(dfs[name])} rows")
    else:
        print(f"[warn] Missing file, will skip: {path.name}")

if not dfs:
    raise RuntimeError("No CSVs found to merge.")

# ---------- Merge on image name ----------
# Use the first available CSV as the base so the script is robust to
# any single upstream file being absent.
base_name, merged = next(iter(dfs.items()))
for name, df in dfs.items():
    if name == base_name:
        continue
    merged = merged.merge(df, on="image", how="outer")

merged.to_csv(OUT_FILE, index=False)
print(f"\nMaster file written to: {OUT_FILE}")
print(f"Total images: {len(merged)}")
