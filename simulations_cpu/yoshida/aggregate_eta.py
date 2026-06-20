import argparse
import glob
import os
import sys

import pandas as pd

# plot_utils lives in visualization/ — add it to path relative to this file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "visualization"))
from plot_utils import get_metadata


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-run Eta into threshold summary CSVs")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw per-task CSVs")
    parser.add_argument("--output-dir", required=True, help="Directory to write summary CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_files:
        print(f"No .csv files found in {args.input_dir}")
        sys.exit(1)

    records = []
    skipped = 0

    for csv_path in csv_files:
        meta = get_metadata(csv_path)

        try:
            df = pd.read_csv(csv_path, comment="#")
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f"WARNING: could not read {os.path.basename(csv_path)}: {e}", file=sys.stderr)
            skipped += 1
            continue

        if "Eta" not in df.columns:
            print(f"WARNING: no Eta column in {os.path.basename(csv_path)}, skipping",
                  file=sys.stderr)
            skipped += 1
            continue

        min_rows = 5
        if len(df) < min_rows:
            print(f"WARNING: only {len(df)} rows in {os.path.basename(csv_path)} "
                  f"(expected >={min_rows}), skipping", file=sys.stderr)
            skipped += 1
            continue

        eta_bar = df["Eta"].iloc[int(0.8 * len(df)):].mean()

        N = int(meta.get("N", -1))
        model = meta.get("Model", "unknown").lower()
        amplitude = float(meta.get("Amplitude", -1))

        records.append({"N": N, "Model": model, "Amplitude": amplitude, "Eta": eta_bar})

    if not records:
        print("No valid records found — nothing to write.")
        sys.exit(1)

    df_all = pd.DataFrame(records)
    groups = df_all.groupby(["N", "Model"])
    files_written = 0

    for (N, model), group in groups:
        summary = group[["Amplitude", "Eta"]].sort_values("Amplitude").reset_index(drop=True)
        fname = f"threshold_summary_{model}_N{N}.csv"
        fpath = os.path.join(args.output_dir, fname)
        summary.to_csv(fpath, index=False)
        files_written += 1
        print(f"  wrote {fname}  ({len(summary)} rows)")

    print(f"\nDone: {files_written} summary files written covering "
          f"{files_written} (N, Model) group(s). "
          f"{skipped} input file(s) skipped.")

    expected = 8
    if files_written < expected:
        print(f"NOTE: expected {expected} groups (4 N values x 2 models), "
              f"got {files_written} — some may be missing or not yet collected.")


if __name__ == "__main__":
    main()
