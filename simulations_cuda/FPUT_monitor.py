import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import time


DATA_DIR = "../data"  # 放csv的文件夹
REFRESH_INTERVAL = 10  # 每？秒刷新一下
MAX_MODES = 5         # 前？个mode

def get_latest_file(path):
    """自动寻找目录下最新修改的 CSV 文件"""
    list_of_files = glob.glob(os.path.join(path, "*.csv"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getmtime)

def plot_live():
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    print(f"Monitor activated! searching for new data in {DATA_DIR} ...")

    while True:
        target_file = get_latest_file(DATA_DIR)
        
        if target_file and os.path.exists(target_file):
            try:
                # 跳过注释行read
                df = pd.read_csv(target_file, comment='#')
                
                if len(df) > 2:
                    ax.clear()
                    
                    mode_cols = [c for c in df.columns if c.startswith('Mode')][:MAX_MODES]
                    
                    for col in mode_cols:
                        ax.plot(df['Time'], df[col], label=col, lw=1.5, alpha=0.8)
                    
                    # 计算进度，之前设定的5000 segments
                    progress = (len(df) / 5000) * 100
                    
                    # 没有title
                    ax.set_xlabel("Physical Time ($t$)")
                    ax.set_ylabel("Modal Energy ($E_k$)")
                    ax.legend(loc='upper right', frameon=True, fontsize='small')
                    ax.grid(True, linestyle=':', alpha=0.6)
                    
                    ax.text(0.02, 0.95, f"Monitoring: {os.path.basename(target_file)}\nProgress: {progress:.1f}%", 
                            transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                    
                    plt.pause(0.1)
                else:
                    print(f"\r Waiting for data...", end="")
            except Exception as e:
                print(f"\n unknown error, maybe just wait and rerun: {e}")
        else:
            print(f"\r where is your csv I cannot find it...", end="")
            
        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    try:
        plot_live()
    except KeyboardInterrupt:
        print("\n You stopped me...")
