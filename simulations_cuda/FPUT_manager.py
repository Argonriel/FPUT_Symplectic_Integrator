import os
import subprocess

EXE_NAME = "./fput_sim"
DATA_DIR = "./data"
LOG_DIR = "./logs"

# 任务清单
tasks = [
    {"n": 1024, "alpha": 0.1, "amp": 0.8, "name": "1024_a0.1_A0.8"},
    {"n": 1024, "alpha": 0.2, "amp": 0.4, "name": "1024_a0.2_A0.4"}
]

def setup():
    for d in [DATA_DIR, LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

def compile_engine():
    # 使用 -O3 & fast-math
    cmd = ["nvcc", "fput_cuda_engine.cu", "-o", EXE_NAME, "-O3", "-use_fast_math", "-arch=native"]
    subprocess.run(cmd, check=True)
    print("成功！")

def run_tasks():
    for task in tasks:
        output_file = os.path.join(DATA_DIR, f"{task['name']}.csv")
        log_file = os.path.join(LOG_DIR, f"{task['name']}.log")
        
        print(f"Start: {task['name']} (N={task['n']}, A={task['amp']})")
        
        cmd = [
            EXE_NAME,
            str(task['n']),
            str(task['alpha']),
            str(task['amp']),
            output_file
        ]
        
        with open(log_file, "w") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            # 顺序执行可以使用 p.wait()
            # 并行可以把 p 存入列表
            p.wait() 
            
        print(f"data saved at: {output_file}")

if __name__ == "__main__":
    setup()
    compile_engine()
    run_tasks()
