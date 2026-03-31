import subprocess
import os
import re
import numpy as np

print("🛠️ Compiling CUDA engine...")
subprocess.run(["nvcc", "-O3", "fput_core.cu", "-o", "./fput_sim"], check=True)

# task list
task_dict = {
    512:  np.arange(1, 7, 0.5),
    1024: np.arange(5, 25, 1),
    2048: np.arange(20, 91, 2.5),
    4096: np.arange(200, 460, 10)
}

DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

for N, A_list in task_dict.items():
    print(f"\n Running N = {N}")
    final_etas = []
    
    for A in A_list:
        print(f"  -> Computing A = {A} ... ", end="", flush=True)
        process = subprocess.Popen(['./fput_sim', str(N), str(A)], 
                                    stdout=subprocess.PIPE, text=True)
        
        etas_this_run = []
        for line in process.stdout:
            match = re.search(r'STEP:\d+,ETA:(\S+)', line)
            if match: etas_this_run.append(float(match.group(1)))
        
        process.wait()
        
        # 取最后20%的tail
        avg_eta = np.mean(etas_this_run[-len(etas_this_run)//5:]) if etas_this_run else 0.0
        final_etas.append(avg_eta)
        print(f"Avg Eta: {avg_eta:.4f}")

    # save as CSV
    csv_path = os.path.join(DATA_DIR, f"fput_N{N}_alpha0.25.csv")
    np.savetxt(csv_path, np.column_stack((A_list, final_etas)), 
               delimiter=",", header="Amplitude,Eta", comments="")

print("\n Tasks completed!")
