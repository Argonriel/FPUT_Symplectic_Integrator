import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

# plot_energy_evolution东西太多了，太混乱了
# enter your csv name here:
CSV_FILE = "32_alpha0.25_A1_dt0.35_mtp10_0222_2114.csv"

flag_old_data = 1
dt_val = 0.35

df = pd.read_csv(CSV_FILE, comment='#')
time = df['Time'].values
mode1_raw = df['Mode1'].values

if flag_old_data == 1:
    time = time * dt_val

max_time = np.max(time)
if max_time >= 1e9:
    time_div = 1e9
    time_label = "10$^9$"
elif max_time >= 1e8:
    time_div = 1e8
    time_label = "10$^8$"
elif max_time >= 1e7:
    time_div = 1e7
    time_label = "10$^7$"
elif max_time >= 1e6:
    time_div = 1e6
    time_label = "10$^6$"
elif max_time >= 1e5:
    time_div = 1e5
    time_label = "10$^5$"
else:
    time_div = 1.0
    time_label = "1"

initial_energy = mode1_raw[0]
mode1 = mode1_raw / initial_energy

peaks, _ = find_peaks(mode1, distance=100)
if 0 not in peaks:
    peaks = np.insert(peaks, 0, 0)

plt.figure(figsize=(12, 5), dpi=300)

plt.plot(time / time_div, mode1, color='gray', alpha=0.4, lw=0.5, label='Normalized Modal Energy $E_1(t) / E_1(0)$')
plt.plot(time[peaks] / time_div, mode1[peaks], color='red', lw=1.5, label='Envelope (Superperiod Line)')

plt.xlabel(f"Physical Time ($\\times {time_label}$ dt)")
plt.ylabel("Normalized Energy $E_1(t)/E_1(0)$")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
