import numpy as np
# import cupy as cp
import time
from numba import cuda
import scipy.fft
import os


@cuda.jit(fastmath=True)  # 3. Physics functions
def update_a(x, F, n, value, model_flag):
    j = cuda.grid(1)  # indices

    if j < n-1:  # N-1 particles in total
        curr = x[j]
        prev = 0.0 if j == 0 else x[j - 1]  # fixed ends 首尾 0
        next_val = 0.0 if j == n - 2 else x[j + 1]

        dx_f = next_val - curr  # forward spring displacement
        dx_b = curr - prev  # backward

        force = dx_f - dx_b  # linear force (by Hooke's Law)

        if model_flag == 0:  # alpha: add nonlinear parts
            force += value * (dx_f * dx_f - dx_b * dx_b)  # faster than square
        else:  # beta
            force += value * (dx_f * dx_f * dx_f - dx_b * dx_b * dx_b)

        F[j] = force


@cuda.jit(fastmath=True)  # 之前大loop的前半部分
def update_x(x, v, a, dt, dt2, n):
    j = cuda.grid(1)
    if j < n-1:
        # x_new = x + dt * v + 0.5 * dt2 * a
        x[j] += dt * v[j] + 0.5 * dt2 * a[j]  # 更新x, taylor expansion on x(t+dt), ignore x```/a`


@cuda.jit(fastmath=True)
def update_v(v, a, a_new, dt, n):
    j = cuda.grid(1)
    if j < n-1:
        # v_new = v + 0.5 * dt * (a + a_new)
        v[j] += 0.5 * dt * (a[j] + a_new[j])    # use trapezium rule
        # a_new = get_acceleration(x_new, n, value, model_flag)
        a[j] = a_new[j]


def evolve_gpu(d_x, d_v, d_a, d_a_new, dt, dt2, steps, n, value, model_flag, blocks, threads):
    for _ in range(steps):
        update_x[blocks, threads](d_x, d_v, d_a, dt, dt2, n)  # variable都加个d区分
        update_a[blocks, threads](d_x, d_a_new, n, value, model_flag)
        update_v[blocks, threads](d_v, d_a, d_a_new, dt, n)


def get_energy(x, v, omega, n, modes_count):  # 4. 这一段没改 FFT Helper, bye bye matrix multiplication :)
    # def of type1: 2*sigma(xi*sin)
    a_k = scipy.fft.dst(x, type=1) / n
    a_k_dot = scipy.fft.dst(v, type=1) / n

    # 计算每个 mode 的 total energy = kinetic energy + potential energy
    k_slice = slice(0, modes_count)
    E = (0.5 * a_k_dot[k_slice] * a_k_dot[k_slice]) + (
            0.5 * (omega[k_slice] * omega[k_slice]) * a_k[k_slice] * a_k[k_slice])
    return E


def get_total_energy(x, v, value, model_flag):
    K = 0.5 * np.sum(v * v)  # kinetic energy

    x_full = np.pad(x, (1, 1), 'constant')  # fixed ends = 0
    r = np.diff(x_full)  # r = x[i+1] - x[i]
    V_linear = 0.5 * np.sum(r * r)  # linear potential energy

    # nonlinear potential energy
    if model_flag == 0:  # Alpha: V = r^2/2 + alpha * r^3 / 3
        V_nonlinear = (value / 3.0) * np.sum(r * r * r)
    else:  # Beta:  V = r^2/2 + beta * r^4 / 4
        V_nonlinear = (value / 4.0) * np.sum(r ** 4)

    return K + V_linear + V_nonlinear


def unique_filename(base_name, extension):
    filename = f"{base_name}.{extension}"
    counter = 1
    while os.path.exists(filename):  # no repetition!
        filename = f"{base_name}({counter}).{extension}"
        counter += 1

    return filename


def main():
    # 1. Parameter Setup

    MODEL = "beta"
    VALUE = 1.0
    MODEL_FLAG = 0 if MODEL == "alpha" else 1
    N = 16384
    Dt2 = 0.05
    Dt = np.sqrt(Dt2)
    NUM_STEPS = 2_000_000_000
    STRIDE = 2_000_000  # sampling rate
    IC = "sine"  # initial condition, sine/sawtooth
    MODES_TO_PLOT = 64
    AMPLITUDE = 2.0

    # 2. Initialization

    print(f"Initializing: N={N}, {MODEL}={VALUE}, Amp={AMPLITUDE}, Steps={NUM_STEPS}")
    # fixed ends: x0 = xN = 0
    I = np.arange(1, N)  # xi: x_1, x_2, ... x_N-1
    K = np.arange(1, N)  # ak
    OMEGA = 2.0 * np.sin(K * np.pi / (2 * N))
    # creates 2 arrays, length N-1, all elements 0
    X = np.zeros(N - 1)  # array of initial displacement at t=0
    v = np.zeros(N - 1)  # array of velocity

    if IC == "sine":  # fig 1, 2
        X = AMPLITUDE * np.sin(np.pi * I / N)  # 把[0,pi]的sin进行N等分，取i=1...N-1的点，刚好0和N的sin都是0
    elif IC == "sawtooth":  # fig 3
        xi = I / N
        X = AMPLITUDE * (1.0 - 2.0 * np.abs(xi - 0.5))  # y=1-2|x-0.5|

    d_x = cuda.to_device(X)
    d_v = cuda.to_device(v)

    d_a = cuda.device_array(N - 1, dtype=np.float64)
    d_a_new = cuda.device_array(N - 1, dtype=np.float64)

    threads_per_block = 512
    blocks_per_grid = (N - 1 + threads_per_block - 1) // threads_per_block
    update_a[blocks_per_grid, threads_per_block](d_x, d_a, N, VALUE, MODEL_FLAG)

    # 5. Loop? Loop. Loop! Loop<3

    print(f"Starting simulation loop... Steps={NUM_STEPS}")
    start_time = time.time()
    history_energy = []
    history_time = []
    history_total_energy = []
    num_segments = NUM_STEPS // STRIDE

    for seg in range(num_segments):
        # update and record energies of modes_to_plot
        evolve_gpu(d_x, d_v, d_a, d_a_new, Dt, Dt2, STRIDE, N, VALUE, MODEL_FLAG, blocks_per_grid, threads_per_block)  # use stride as steps!
        current_x = d_x.copy_to_host()  # call device_x to calculate energy on CPU
        current_v = d_v.copy_to_host()
        current_energies = get_energy(current_x, current_v, OMEGA, N, MODES_TO_PLOT)
        current_total_e = get_total_energy(current_x, current_v, VALUE, MODEL_FLAG)

        # record data
        history_energy.append(current_energies)
        history_time.append((seg+1)*STRIDE*Dt)
        history_total_energy.append(current_total_e)

        # print progress
        if seg % (num_segments // 100) == 0:  # 因为num_segments是//来的，这里不是标准的1%，会比1%小一点
            elapsed = time.time() - start_time
            progress = (seg / num_segments) * 100
            print(f"Progress: {progress:.1f}% (Step {(seg+1)*STRIDE*Dt}), Time: {elapsed:.1f}s")

    print(f"Done! Total time: {time.time() - start_time:.1f}s")

    # 6. Saving our data ^^
    arr_time = np.array(history_time)
    arr_energy = np.array(history_energy)
    arr_total_e = np.array(history_total_energy)
    # 对于大N绝不能保存x，否则文件爆炸无敌大，善待电脑从我做起
    output_data = np.column_stack((arr_time, arr_energy, arr_total_e))

    metadata = {
        "Model": "alpha" if MODEL_FLAG == 0 else "beta",
        "N": N,
        "Alpha_Beta_Value": f"{VALUE:.3f}",
        "Amplitude": f"{AMPLITUDE:.2f}",
        "dt": f"{Dt:.2f}",
        "IC_Type": IC
    }
    timestamp = time.strftime("%m%d_%H%M")
    BASE_NAME = f"{N}_{metadata['Model']}{VALUE}_A{AMPLITUDE}_dt{metadata['dt']}_mtp{MODES_TO_PLOT}_{timestamp}"
    # e.g. 1024_alpha0.22_A0.4_dt0.32_mtp20_0209_2315
    csv_filename = unique_filename(BASE_NAME, "csv")
    e_cols = ",".join([f"Mode{i + 1}" for i in range(MODES_TO_PLOT)])
    column_names = "Time," + e_cols + ",TotalEnergy"

    with open(csv_filename, "w") as f:
        for key, val in metadata.items():
            f.write(f"# {key}: {val}\n")
        f.write(column_names + "\n")
        np.savetxt(f, output_data, delimiter=",")

    print(f"Data saved as: {csv_filename}")


if __name__ == "__main__":
    main()
