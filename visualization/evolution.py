import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_utils import get_metadata, time_axis_scale

PLOT_LENGTH = 10
PLOT_HEIGHT = 6
MY_DPI = 300


def plot_spectral(df, params, out_path, window_size, important_modes, log_scale):
    modes_cols = [col for col in df.columns if col.startswith('Mode')]
    initial_e1 = df['Mode1'].iloc[0]
    cmap = plt.cm.turbo

    is_beta = params.get('Model', '').lower() == 'beta'
    active_count = (len(modes_cols) + 1) // 2 if is_beta else len(modes_cols)

    max_time = df['Time'].max()
    time_div, time_label = time_axis_scale(max_time)
    t_scaled = df['Time'] / time_div

    fig, ax = plt.subplots(figsize=(PLOT_LENGTH, PLOT_HEIGHT), dpi=MY_DPI)

    for i, col in enumerate(modes_cols):
        normalized_y = df[col] / initial_e1

        if is_beta and (i + 1) % 2 == 0:
            color = 'gray'
            lw, alpha, label = 0.5, 0.05, None
        else:
            color_idx = i // 2 if is_beta else i
            color = cmap(color_idx / active_count)
            if i == 0:
                lw, alpha, label = 3.0, 1.0, "Mode 1"
            elif i < important_modes:
                lw, alpha, label = 1.5, 0.9, f"Mode {i + 1}"
            else:
                lw, alpha, label = 0.8, 0.4, None

        z_rough = 50 - i
        z_smooth = 100 - i

        if window_size == 1:
            ax.plot(t_scaled, normalized_y, color=color, linewidth=lw,
                    alpha=alpha, label=label, zorder=z_smooth)
        else:
            smooth_y = normalized_y.rolling(window=window_size, center=True, min_periods=1).mean()
            ax.plot(t_scaled, normalized_y, color=color, alpha=0.1,
                    linewidth=0.5, zorder=z_rough)
            ax.plot(t_scaled, smooth_y, color=color, linewidth=lw,
                    alpha=alpha, label=label, zorder=z_smooth)

    ax.set_xlabel(f"Time (×{time_label})")
    if log_scale:
        ax.set_ylabel("Normalized Energy $E_k(t) / E_1(0)$ (Log Scale)", fontsize=12)
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 2.0)
    else:
        ax.set_ylabel("Normalized Energy $E_k(t) / E_1(0)$", fontsize=12)
        ax.set_ylim(0, 1.1)

    lgd = ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
    lgd.set_zorder(1000)
    ax.grid(True, alpha=0.2)
    ax.margins(x=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_heatmap(df, out_path):
    modes_cols = [col for col in df.columns if col.startswith('Mode')]
    heatmap_data = np.log10(df[modes_cols].values.T + 1e-15)

    fig, ax = plt.subplots(figsize=(20, 6), dpi=MY_DPI)
    sns.heatmap(heatmap_data, cmap='magma',
                cbar_kws={'label': r'Log$_{10}$(Energy)'},
                xticklabels=False, yticklabels=1, ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mode Index", fontsize=12)
    plt.margins(0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Batch energy evolution plots (spectral + heatmap)")
    parser.add_argument("input_dir", help="Directory containing .csv files")
    parser.add_argument("output_dir", help="Directory to write output figures")
    parser.add_argument("--skip-spectral", action="store_true")
    parser.add_argument("--skip-heatmap", action="store_true")
    parser.add_argument("--window-size", type=int, default=1,
                        help="Rolling mean window for spectral plot (default: 1 = no smoothing)")
    parser.add_argument("--important-modes", type=int, default=5,
                        help="Number of leading modes with thick/labeled lines (default: 5)")
    parser.add_argument("--log-scale", action="store_true",
                        help="Log-scale spectral y-axis")
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

        if not args.skip_spectral:
            out_path = os.path.join(args.output_dir, f"{stem}_spectral.pdf")
            plot_spectral(df, params, out_path, args.window_size, args.important_modes, args.log_scale)
            print(f"  -> {stem}_spectral.pdf")

        if not args.skip_heatmap:
            out_path = os.path.join(args.output_dir, f"{stem}_heatmap.png")
            plot_heatmap(df, out_path)
            print(f"  -> {stem}_heatmap.png")

    print("Done.")


if __name__ == "__main__":
    main()
