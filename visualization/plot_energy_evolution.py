import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns  # heatmap
import numpy as np

# base_name在隔壁，这里的png命名只是换掉后缀
# 统一一下，所有图现在都没有plt.title了
# N太大的话csv爆炸无敌大，最好别算shape
# x轴单位根据最大的t变化，e-369，但是shape那里锁定了k，反正也用不到更大的，让我偷个懒
# spectral图最完善

# enter your csv file name here:
CSV_FILE = "4096_beta1.0_A40_500M.csv"
print(f"Reading data from: {CSV_FILE}...")

plot_length = 10
plot_height = 6
my_dpi = 300
important_modes = 5
window_size = 1  # number of points taken to calculate average，只对spectral生效，会把原数据淡淡地画在下一图层
scale = 300  # 只对mode图生效

# 0不要1要
flag_mode = 0
flag_drift = 0
flag_spectral = 1  # smooth & normalized
flag_log = 0  # spectral图要不要log一下y轴？
flag_heat = 0
flag_old_data = 1  # 老数据开关 (0代表新数据的物理时间，1代表这是老数据用了步数)，只对spectral生效

# 写小N要shape的几个时间点
snapshot_indices = [
        0,
        2400,
        4900,
        7400,
        11100,
        14800,
        17000,
        19000
        # total_rows // 4,
        # total_rows - 1
    ]


def get_metadata(filename):  # read the header of csv
    meta = {}
    with open(filename, 'r') as f:
        for lines in f:
            if lines.startswith('#'):
                # delete #，split by :
                content = lines.replace('#', '').strip()
                if ':' in content:
                    key, val = content.split(':', 1)
                    meta[key.strip()] = val.strip()
            else:
                break  # 碰到非#说明结束了
    return meta


params = get_metadata(CSV_FILE)  # read all parameters from our header

try:
    df = pd.read_csv(CSV_FILE, comment="#")  # ignore lines with #
except FileNotFoundError:
    print("File not found:/")
    sys.exit(1)

df.columns = df.columns.str.strip()
print("Wash:", df.columns.tolist())  # 把空格洗掉
if flag_old_data == 1:
    dt_val = float(params.get('dt', 0.1))
    print(f"Old data! It's not physical time .. (multiply by dt={dt_val})...")
    df['Time'] = df['Time'] * dt_val
max_time = df['Time'].max()
if max_time >= 1e9:
    time_div = 1e9
    time_label = "10$^9$"
    time_unit = "B"
elif max_time >= 1e8:
    time_div = 1e8
    time_label = "10$^8$"
    time_unit = "x10^8"
elif max_time >= 1e7:
    time_div = 1e7
    time_label = "10$^7$"
    time_unit = "x10^7"
elif max_time >= 1e6:
    time_div = 1e6
    time_label = "10$^6$"
    time_unit = "M"
elif max_time >= 1e5:
    time_div = 1e5
    time_label = "10$^5$"
    time_unit = "x10^5"
elif max_time >= 1e3:
    time_div = 1e3
    time_label = "10$^3$"
    time_unit = "k"  # ! kilo
else:
    time_div = 1
    time_label = "1"
    time_unit = ""
print(f"Max time is {max_time:.2e}，so we use x{time_label} scale")


t_scaled = df['Time'] / time_div
total_energy = df['TotalEnergy']
modes_cols = [col for col in df.columns if col.startswith('Mode')]


'''
Energy plot
'''
if flag_mode == 1:
    print("Plotting results...")
    plt.figure(figsize=(plot_length, plot_height), dpi=my_dpi)

    scale_factor = scale / df['Mode1'].iloc[0]
    for i, col in enumerate(modes_cols):  # use enumerate to give i a value
        df[col] = df[col] * scale_factor
        alpha_p = 1.0
        line, = plt.plot(t_scaled, df[col], label=col, linewidth=1.5, alpha=alpha_p)
        peak_idx = df[col].idxmax()  # find peak
        peak_x = t_scaled[peak_idx]
        peak_y = df[col][peak_idx]
        if i == 0 and peak_x == t_scaled.min():  # avoid M1 & 1e-7 overlap
            h_align = 'left'
            x_offset = 10
        elif i >= 9:
            h_align = 'center'
            x_offset = -1  # 两位数得拽回来一点
        else:
            h_align = 'center'
            x_offset = 0
        # ha=horizontal alignment, offset points=peak is (0,0)
        plt.annotate(f"M{i+1}", xy=(peak_x, peak_y), xytext=(x_offset, 3), color=line.get_color(),
                     ha=h_align, textcoords='offset points', fontweight='bold', fontsize=9, alpha=alpha_p)

    plt.xlabel(f"Time (x{time_label})", fontsize=12)
    plt.ylabel("mode energy $E_k(t)$", fontsize=12)
    plt.ylim(0, df[modes_cols].max().max()*1.05)  # add margin above to avoid overlap
    # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.8)
    plt.grid(True, alpha=0.3)  # 透明度
    plt.margins(0)  # margin太丑了
    plt.tight_layout()

    plt.savefig(CSV_FILE.replace(".csv", "_modes.png"))  # 换个后缀
    print("Energy plot saved")


'''
Total energy drift
'''

if flag_drift == 1:
    print("Plotting total energy drift...")
    plt.figure(figsize=(10, 6), dpi=300)

    # use initial total energy E0 to calculate relative error
    E0 = total_energy.iloc[0]  # integer location, 0 is the 1st row
    rel_error = (total_energy - E0) / E0

    plt.plot(t_scaled, rel_error, color='black', linewidth=1.0)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel(f'Time (x{time_label})', fontsize=12)
    plt.ylabel(r"Relative Error $\Delta E / E_0$", fontsize=12)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.grid(True, alpha=0.3)  # 透明度
    plt.margins(0)  # margin太丑了
    plt.tight_layout()

    plt.savefig(CSV_FILE.replace(".csv", "_drift.pdf"))  # 换个后缀
    print("Energy drift plot saved")


'''
Spectral Line Plot
'''

if flag_spectral == 1:
    print("Plotting spectral line results...")
    plt.figure(figsize=(plot_length, plot_height), dpi=my_dpi)

    initial_e1 = df['Mode1'].iloc[0]
    # cmap = plt.cm.RdYlBu  # 红-黄-蓝
    cmap = plt.cm.turbo

    is_beta = params.get('Model').lower() == 'beta'
    if is_beta:
        active_count = (len(modes_cols) + 1) // 2
    else:
        active_count = len(modes_cols)

    for i, col in enumerate(modes_cols):  # use enumerate to give i a value
        normalized_y = df[col] / initial_e1
        smooth_y = normalized_y.rolling(window=window_size, center=True, min_periods=1).mean()

        if is_beta and (i+1) % 2 == 0:  # even modes in beta
            color = 'gray'
            lw, alpha, label = 0.5, 0.05, None
        else:
            if is_beta:
                color_idx = i // 2
            else:
                color_idx = i

            color = cmap(color_idx / active_count)  # use relative position to give a color

            if i == 0:  # first mode
                lw, alpha, label = 3.0, 1.0, "Mode 1"
            elif i < important_modes:
                lw, alpha, label = 1.5, 0.9, f"Mode {i+1}"
            else:  # not important modes
                lw, alpha, label = 0.8, 0.4, None

        # 动态计算 zorder：i 越小（Mode 越靠前），zorder 越大（图层越靠上）
        # 保证平滑曲线 (smooth) 始终在毛刺曲线 (rough) 上方
        z_rough = 50 - i
        z_smooth = 100 - i

        plt.plot(t_scaled, normalized_y, color=color, alpha=0.1, linewidth=0.5, zorder=z_rough)  # original rough
        plt.plot(t_scaled, smooth_y, color=color, linewidth=lw, alpha=alpha, label=label, zorder=z_smooth)  # smooth

    plt.xlabel(f"Time (x{time_label})")

    if flag_log == 1:
        plt.ylabel("Normalized Energy $E_k(t) / E_1(0)$ (Log Scale)", fontsize=12)
        plt.yscale('log')
        plt.ylim(1e-4, 2.0)
    elif flag_log == 0:
        plt.ylabel("Normalized Energy $E_k(t) / E_1(0)$", fontsize=12)
        plt.ylim(0, 1.1)

    lgd=plt.legend(loc='upper right', fontsize=10, framealpha=0.8)
    lgd.set_zorder(1000)  # 强制让图例在最上面
    plt.grid(True, alpha=0.2)
    plt.margins(x=0)
    plt.tight_layout()

    plt.savefig(CSV_FILE.replace(".csv", "_spectral.pdf"))
    print("Spectral plot saved")


'''
Heatmap
'''

if flag_heat == 1:
    print("Plotting Heatmap...")
    plt.figure(figsize=(20, 6), dpi=300)

    # use T to transpose t&E; + 1e-15 to avoid 0
    heatmap_data = np.log10(df[modes_cols].values.T + 1e-15)

    ax = sns.heatmap(heatmap_data, cmap='magma',
                     cbar_kws={'label': r'Log$_{10}$(Energy)'},
                     xticklabels=False, yticklabels=1)

    plt.xlabel(f"Time (x{time_label})")
    plt.ylabel("Mode Index", fontsize=12)
    plt.margins(0)
    plt.tight_layout()

    plt.savefig(CSV_FILE.replace(".csv", "_heatmap.png"))
    print(f"Energy flow heatmap saved")


'''
Actual Shape (Displacement) Plot
'''

x_cols = [col for col in df.columns if col.startswith('x') and col[1:].isdigit()]

if len(x_cols) > 0:
    print(f"Displacement data detected ({len(x_cols)} particles). Plotting Actual Shape...")
    N_val = int(params.get('N', len(x_cols) + 1))
    plt.figure(figsize=(10, 12), dpi=300)

    total_rows = len(df)

    shape_cmap = plt.cm.viridis  # cute color

    for idx_count, row_idx in enumerate(snapshot_indices):
        t_val = df['Time'].iloc[row_idx]
        displacement = df[x_cols].iloc[row_idx].values

        # 补充两端固定的 0 墙壁
        full_shape = np.pad(displacement, (1, 1), 'constant')

        color = shape_cmap(idx_count / (len(snapshot_indices) - 1))
        plt.plot(np.arange(N_val + 1), full_shape,
                 color=color, lw=1.5, alpha=0.9,
                 label=f"step = {t_val / 1000:.1f}k ")

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, lw=1)
    plt.xlabel("Particle Index $i$", fontsize=12)
    plt.ylabel("Displacement $x_i$", fontsize=12)

    plt.legend(loc='upper right', fontsize=10, framealpha=0.8)
    plt.grid(True, alpha=0.2)
    plt.margins(x=0.02)
    plt.tight_layout()

    shape_filename = CSV_FILE.replace(".csv", "_shape.pdf")
    plt.savefig(shape_filename)
    print(f"Actual Shape plot saved as {shape_filename}")

else:
    print("No displacement data found. Skipping Actual Shape plot.")

print("All finished!")
