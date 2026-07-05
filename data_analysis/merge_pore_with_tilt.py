"""
Merge weighted_proximity_sum_P from the per-replica
*tilt_multipeptide_metrics*.csv files into the per-replica
centered_per_lipid.csv files, matched by `resid`.

Layout assumed (matches `seaborn_jupyter.ipynb`):

    <BASE_DIR>/<system>/<rep>/centered_per_lipid.csv
    <BASE_DIR>/<system>/Clustering_data/multi_peptide_200ns/<rep>/*tilt_multipeptide_metrics*.csv

    <BASE_DIR>/<system>_long/<rep>/centered_per_lipid.csv
    <BASE_DIR>/<system>_long/multi_peptide_long/<rep>/*tilt_multipeptide_metrics*.csv

For every centered_per_lipid.csv it finds, the script writes a
centered_per_lipid_merged.csv next to it, with one extra column
(`weighted_proximity_sum_P`) sourced by resid lookup.

Usage:
    python merge_pore_with_tilt.py                       # uses default BASE_DIR
    python merge_pore_with_tilt.py /path/to/data         # custom base
    python merge_pore_with_tilt.py --dry-run             # report only
    python merge_pore_with_tilt.py --overwrite           # overwrite existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_BASE_DIR = Path("/home/jacob/data_processing/data")
SOURCE_COLUMN = "weighted_proximity_sum_P"
MERGE_KEY = "resid"
OUTPUT_NAME = "centered_per_lipid_merged.csv"


def locate_tilt_file(per_lipid_path: Path) -> Path | None:
    """
    Given .../<system>/<rep>/centered_per_lipid.csv,
    return the matching *tilt_multipeptide_metrics*.csv path or None.
    """
    rep_dir = per_lipid_path.parent              # .../<system>/<rep>
    system_dir = rep_dir.parent                  # .../<system>
    rep = rep_dir.name

    if system_dir.name.endswith("_long"):
        candidate_dir = system_dir / "multi_peptide_long" / rep
    else:
        candidate_dir = system_dir / "Clustering_data" / "multi_peptide_200ns" / rep

    if not candidate_dir.is_dir():
        return None

    # Ignore Windows-side ":Zone.Identifier" sidecars.
    candidates = [
        p for p in candidate_dir.glob("*tilt_multipeptide_metrics*.csv")
        if "Zone.Identifier" not in p.name
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        # Prefer the shortest filename when multiple variants exist
        # (e.g. `9_tilt_...csv` over `9_fixed_tilt_...csv`).
        candidates.sort(key=lambda p: (len(p.name), p.name))
    return candidates[0]


def merge_one(per_lipid_path: Path,
              tilt_path: Path,
              output_path: Path,
              overwrite: bool) -> tuple[bool, str]:
    """Return (success, message). Skips if output exists and not overwriting."""
    if output_path.exists() and not overwrite:
        return False, f"skip (exists): {output_path}"

    try:
        per_lipid = pd.read_csv(per_lipid_path)
        tilt = pd.read_csv(tilt_path, usecols=[MERGE_KEY, SOURCE_COLUMN])
    except ValueError as e:
        return False, f"read error ({per_lipid_path}): {e}"

    if SOURCE_COLUMN in per_lipid.columns:
        return False, f"already merged: {per_lipid_path}"

    # Drop duplicates on resid in the tilt table so the merge is 1:1.
    tilt = tilt.drop_duplicates(subset=MERGE_KEY, keep="first")

    merged = per_lipid.merge(tilt, on=MERGE_KEY, how="left")

    n_total = len(merged)
    n_filled = int(merged[SOURCE_COLUMN].notna().sum())
    n_missing = n_total - n_filled

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    msg = (
        f"wrote {output_path.relative_to(output_path.parents[2])} "
        f"({n_filled}/{n_total} rows matched"
    )
    if n_missing:
        missing_resids = (
            merged.loc[merged[SOURCE_COLUMN].isna(), MERGE_KEY].tolist()
        )
        msg += f", missing resids: {missing_resids}"
    msg += ")"
    return True, msg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_dir", nargs="?", default=str(DEFAULT_BASE_DIR),
        help=f"Base data directory (default: {DEFAULT_BASE_DIR}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would happen without writing any files.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing centered_per_lipid_merged.csv files.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.is_dir():
        print(f"ERROR: base dir not found: {base}", file=sys.stderr)
        return 2

    per_lipid_files = sorted(p for p in base.glob("*/*/centered_per_lipid.csv"))
    if not per_lipid_files:
        print(f"No centered_per_lipid.csv found under {base}", file=sys.stderr)
        return 1

    n_ok = n_skip = n_fail = 0
    for per_lipid in per_lipid_files:
        tilt = locate_tilt_file(per_lipid)
        if tilt is None:
            n_fail += 1
            print(f"FAIL: no tilt file for {per_lipid}")
            continue

        output = per_lipid.with_name(OUTPUT_NAME)
        if args.dry_run:
            print(f"DRY:  {per_lipid}  <-  {tilt}  ->  {output}")
            n_ok += 1
            continue

        ok, msg = merge_one(per_lipid, tilt, output, args.overwrite)
        print(("OK:   " if ok else "SKIP: ") + msg)
        if ok:
            n_ok += 1
        else:
            n_skip += 1

    print()
    print(f"Summary: {n_ok} merged, {n_skip} skipped, {n_fail} no-tilt-found"
          f" (of {len(per_lipid_files)} per-lipid files)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
