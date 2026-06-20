import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from plot_utils import get_metadata, time_axis_scale

MY_DPI = 300


def main():
    parser = argparse.ArgumentParser(description="Batch Mode 1 superperiod envelope plots")
    parser.add_argument("input_dir", help="Directory containing .csv files")
    parser.add_argument("output_dir", help="Directory to write output figures")
    parser.add_argument("--peak-distance", type=int, default=100,
                        help="Minimum distance between peaks (default: 100)")
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

        time = df['Time'].values
        mode1_raw = df['Mode1'].values

        max_time = np.max(time)
        time_div, time_label = time_axis_scale(max_time)

        initial_energy = mode1_raw[0]
        mode1 = mode1_raw / initial_energy

        peaks, _ = find_peaks(mode1, distance=args.peak_distance)
        if 0 not in peaks:
            peaks = np.insert(peaks, 0, 0)

        plt.figure(figsize=(12, 5), dpi=MY_DPI)
        plt.plot(time / time_div, mode1, color='gray', alpha=0.4, lw=0.5,
                 label='Normalized Modal Energy $E_1(t) / E_1(0)$')
        plt.plot(time[peaks] / time_div, mode1[peaks], color='red', lw=1.5,
                 label='Envelope (Superperiod Line)')
        plt.xlabel(f"Physical Time (×{time_label})")
        plt.ylabel("Normalized Energy $E_1(t)/E_1(0)$")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        out_path = os.path.join(args.output_dir, f"{stem}_superperiod.pdf")
        plt.savefig(out_path)
        plt.close()
        print(f"  -> {stem}_superperiod.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
