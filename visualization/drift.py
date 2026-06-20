import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import get_metadata, time_axis_scale

MY_DPI = 300


def main():
    parser = argparse.ArgumentParser(description="Batch total-energy drift plots")
    parser.add_argument("input_dir", help="Directory containing .csv files")
    parser.add_argument("output_dir", help="Directory to write output figures")
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

        total_energy = df['TotalEnergy']
        max_time = df['Time'].max()
        time_div, time_label = time_axis_scale(max_time)
        t_scaled = df['Time'] / time_div

        E0 = total_energy.iloc[0]
        rel_error = (total_energy - E0) / E0

        plt.figure(figsize=(10, 6), dpi=MY_DPI)
        plt.plot(t_scaled, rel_error, color='black', linewidth=1.0)
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel(f"Time (×{time_label})", fontsize=12)
        plt.ylabel(r"Relative Error $\Delta E / E_0$", fontsize=12)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.grid(True, alpha=0.3)
        plt.margins(0)
        plt.tight_layout()

        out_path = os.path.join(args.output_dir, f"{stem}_drift.pdf")
        plt.savefig(out_path)
        plt.close()
        print(f"  -> {stem}_drift.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
