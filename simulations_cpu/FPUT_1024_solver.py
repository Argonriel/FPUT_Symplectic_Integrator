import numpy as np
import time
from numba import njit
import scipy.fft
import os


@njit(fastmath=True)  # 3. Physics
def get_acceleration(x, n, value, model_flag):  # get acc of all particles
    F = np.zeros(n - 1)  # Array of forces

    for j in range(n - 1):  # N-1 particles in total
        curr = x[j]
        prev = 0.0 if j == 0 else x[j - 1]  # fixed ends = 0
        next_val = 0.0 if j == n - 2 else x[j + 1]

        dx_f = next_val - curr  # forward spring displacement
        dx_b = curr - prev  # backward spring displacement

        force = dx_f - dx_b  # linear force (by Hooke's Law)
        if model_flag == 0:  # alpha: add nonlinear parts
            force += value * (dx_f * dx_f - dx_b * dx_b)  # faster than square
        else:  # beta
            force += value * (dx_f * dx_f * dx_f - dx_b * dx_b * dx_b)

        F[j] = force
    return F


@njit(fastmath=True)  # 之前大loop的前半部分
def evolve(x, v, a, dt, dt2, steps, n, value, model_flag):  # 把一个stride里头的step全跑了
    for _ in range(steps):
        x += v * dt + 0.5 * dt2 * a
        a_new = get_acceleration(x, n, value, model_flag)
        v += 0.5 * dt * (a + a_new)
        a = a_new

    return x, v, a


def get_energy(x, v, omega, n, modes_count):  # 4. FFT Helper, bye bye matrix multiplication :)
    # def of type1: 2*sigma(xi*sin)
    # use np.sqrt(2 * n) to normalize
    a_k = scipy.fft.dst(x, type=1) / np.sqrt(2 * n)
    a_k_dot = scipy.fft.dst(v, type=1) / np.sqrt(2 * n)

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
        V_nonlinear = (value / 3.0) * np.sum(r ** 3)
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

    MODEL = "alpha"
    VALUE = 0.2
    MODEL_FLAG = 0 if MODEL == "alpha" else 1
    SHAPE_FLAG = 0  # 画特定时间点的shape？1要0不要
    N = 1024
    Dt2 = 0.1*0.1
    Dt = np.sqrt(Dt2)
    NUM_STEPS = 2_000_000_000
    STRIDE = 2000000  # sampling rate
    IC = "sine"  # initial condition, sine/sawtooth
    MODES_TO_PLOT = 20
    AMPLITUDE = 0.4

    # 2. Initialization

    print(f"Initializing: N={N}, {MODEL}={VALUE}, Amp={AMPLITUDE}, Steps={NUM_STEPS}")
    # fixed ends: x0 = xN = 0
    I = np.arange(1, N)  # xi: x_1, x_2, ... x_N-1
    K = np.arange(1, N)  # ak
    OMEGA = 2.0 * np.sin(K * np.pi / (2 * N))
    # creates 2 arrays, length N-1, all elements 0
    X = np.zeros(N - 1)  # array of initial displacement at t=0
    V = np.zeros(N - 1)  # array of velocity

    if IC == "sine":  # fig 1, 2
        X = AMPLITUDE * np.sin(np.pi * I / N)  # 把[0,pi]的sin进行N等分，取i=1...N-1的点，刚好0和N的sin都是0
    elif IC == "sawtooth":  # fig 3
        xi = I / N
        X = AMPLITUDE * (1.0 - 2.0 * np.abs(xi - 0.5))  # y=1-2|x-0.5|

    # 5. Loop

    print(f"Starting simulation loop... Steps={NUM_STEPS}")
    start_time = time.time()
    history_energy = []
    history_time = []
    history_total_energy = []
    history_x = []  # only small N
    a = get_acceleration(X, N, VALUE, MODEL_FLAG)  # initial acceleration
    num_segments = NUM_STEPS // STRIDE

    for seg in range(num_segments):
        # update and record energies of modes_to_plot
        X, V, a = evolve(X, V, a, Dt, Dt2, STRIDE, N, VALUE, MODEL_FLAG)  # use stride as steps!
        current_energies = get_energy(X, V, OMEGA, N, MODES_TO_PLOT)
        current_total_e = get_total_energy(X, V, VALUE, MODEL_FLAG)

        # record data
        current_step = (seg + 1) * STRIDE
        history_energy.append(current_energies)
        history_time.append(current_step)
        history_total_energy.append(current_total_e)
        history_x.append(X.copy())  # only small N

        # print progress
        if seg % (num_segments // 100) == 0:  # 因为num_segments是//来的，这里不是标准的1%，会比1%小一点
            elapsed = time.time() - start_time
            progress = (seg / num_segments) * 100
            print(f"Progress: {progress:.1f}% (Step {current_step}), Time: {elapsed:.1f}s")

    print(f"Done! Total time: {time.time() - start_time:.1f}s, processing data...")

    # 6. Saving our data ^^

    arr_time = np.array(history_time)
    arr_energy = np.array(history_energy)
    arr_total_e = np.array(history_total_energy)
    if SHAPE_FLAG == 1:  # 要画特定时间点shape
        arr_x = np.array(history_x)  # only small N
        output_data = np.column_stack((arr_time, arr_energy, arr_total_e, arr_x))
    elif SHAPE_FLAG == 0:  # 不画
        output_data = np.column_stack((arr_time, arr_energy, arr_total_e))

    metadata = {  # dict with all parameters
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
    e_cols = ",".join([f"Mode{i+1}" for i in range(MODES_TO_PLOT)])
    if SHAPE_FLAG == 1:
        x_cols = ",".join([f"x{i+1}" for i in range(N-1)])  # only small N
        column_names = "Time," + e_cols + ",TotalEnergy," + x_cols
    if SHAPE_FLAG == 0:
        column_names = "Time," + e_cols + ",TotalEnergy,"

    with open(csv_filename, "w") as f:
        for key, val in metadata.items():
            f.write(f"# {key}: {val}\n")
        f.write(column_names + "\n")
        np.savetxt(f, output_data, delimiter=",")

    print(f"Data saved as: {csv_filename}")


if __name__ == "__main__":
    main()
