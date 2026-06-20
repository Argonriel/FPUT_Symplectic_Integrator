import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import get_metadata

MY_DPI = 300


def main():
    parser = argparse.ArgumentParser(description="Batch displacement shape plots")
    parser.add_argument("input_dir", help="Directory containing .csv files")
    parser.add_argument("output_dir", help="Directory to write output figures")
    parser.add_argument("--n-snapshots", type=int, default=8,
                        help="Number of evenly-spaced time snapshots to plot (default: 8)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_files:
        print(f"No .csv files found in {args.input_dir}")
        sys.exit(1)

    for csv_path in csv_files:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Processing {stem}...")

        params = get_metadata(csv_path)
        df = pd.read_csv(csv_path, comment='#')
        df.columns = df.columns.str.strip()

        x_cols = [col for col in df.columns if col.startswith('x') and col[1:].isdigit()]
        if not x_cols:
            print(f"  No displacement data in {stem}, skipping.")
            continue

        N_val = int(params.get('N', len(x_cols) + 1))
        total_rows = len(df)
        snapshot_indices = np.linspace(0, total_rows - 1, args.n_snapshots, dtype=int)

        shape_cmap = plt.cm.viridis
        plt.figure(figsize=(10, 12), dpi=MY_DPI)

        for idx_count, row_idx in enumerate(snapshot_indices):
            t_val = df['Time'].iloc[row_idx]
            displacement = df[x_cols].iloc[row_idx].values
            full_shape = np.pad(displacement, (1, 1), 'constant')
            color = shape_cmap(idx_count / max(len(snapshot_indices) - 1, 1))
            plt.plot(np.arange(N_val + 1), full_shape,
                     color=color, lw=1.5, alpha=0.9,
                     label=f"t = {t_val:.2e}")

        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, lw=1)
        plt.xlabel("Particle Index $i$", fontsize=12)
        plt.ylabel("Displacement $x_i$", fontsize=12)
        plt.legend(loc='upper right', fontsize=10, framealpha=0.8)
        plt.grid(True, alpha=0.2)
        plt.margins(x=0.02)
        plt.tight_layout()

        out_path = os.path.join(args.output_dir, f"{stem}_shape.pdf")
        plt.savefig(out_path)
        plt.close()
        print(f"  -> {stem}_shape.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
