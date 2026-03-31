import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def plot_threshold(csv_file, output_filename):
    df = pd.read_csv(csv_file).sort_values(by='Amplitude')
    x = df['Amplitude'].values
    y = df['Eta'].values

    lower_bounds = [0.4, min(x) - 20, 0.001, 0.0]
    upper_bounds = [1.0, max(x) + 50, 1.0, 0.15]
    p0 = [0.8, np.median(x), 0.05, 0.05]

    try:
        popt, _ = curve_fit(sigmoid, x, y, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)
        Ac = popt[1]
    except Exception as e:
        print(f"Fit failed: {e}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    span = max(x) - min(x)
    x_min_plot = max(0, min(x) - span * 0.20)
    x_max_plot = max(x) + span * 0.20

    x_fit = np.linspace(x_min_plot, x_max_plot, 200)
    y_fit = sigmoid(x_fit, *popt)

    ax.grid(True, linestyle=':', linewidth=1.5, alpha=0.5, zorder=0)
    ax.plot(x_fit, y_fit, color='#DC143C', linestyle='-', linewidth=2.5, label='Logistic Fit', zorder=2)
    ax.axvline(Ac, color='#FF8C00', linestyle='--', linewidth=2, label=rf'$A_c \approx {Ac:.1f}$', zorder=2)
    ax.scatter(x, y, color='#4B0082', s=80, alpha=0.8, edgecolors='w', label='Simulation Data', zorder=3) 
    
    ax.set_xlabel(r'Initial Amplitude ($A$)')
    ax.set_ylabel(r'Normalized Spectral Entropy ($\bar{\eta}$)')
    ax.set_ylim(-0.05, max(y) + 0.1)
    ax.set_xlim(x_min_plot, x_max_plot)
    ax.legend(loc='lower right', frameon=True, shadow=False, edgecolor='black')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
