#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        std::cerr << "CUDA Error " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(e) << std::endl; \
        exit(1); \
    } \
}

// kernel
__global__ void get_acceleration_gpu(const double* x, double* a, int n, double alpha) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n - 1) {
        double curr = x[j];
        double prev = (j == 0) ? 0.0 : x[j - 1];
        double next_val = (j == n - 2) ? 0.0 : x[j + 1];
        double dx_f = next_val - curr;
        double dx_b = curr - prev;
        a[j] = (dx_f - dx_b) + alpha * (dx_f * dx_f - dx_b * dx_b);
    }
}

__global__ void update_x_gpu(double* x, const double* v, const double* a, double dt, double dt2, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n - 1) x[j] += v[j] * dt + 0.5 * dt2 * a[j];
}

__global__ void update_v_gpu(double* v, const double* a_old, const double* a_new, double dt, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n - 1) v[j] += 0.5 * dt * (a_old[j] + a_new[j]);
}

// CPU
void get_mode_energies(int N, const std::vector<double>& x, const std::vector<double>& v, std::vector<double>& mode_E, int modes_to_plot) {
    double factor = sqrt(2.0 / N);
    for (int k = 1; k <= modes_to_plot; ++k) {
        double Q_k = 0.0, P_k = 0.0;
        for (int j = 1; j < N; ++j) {
            double sin_val = sin(M_PI * j * k / N);
            Q_k += x[j - 1] * sin_val;
            P_k += v[j - 1] * sin_val;
        }
        Q_k *= factor; P_k *= factor;
        double omega = 2.0 * sin(M_PI * k / (2.0 * N));
        mode_E[k - 1] = 0.5 * (P_k * P_k + omega * omega * Q_k * Q_k);
    }
}

double get_total_energy(int N, double alpha, const std::vector<double>& x, const std::vector<double>& v) {
    double kin = 0.0, pot = 0.0;
    for (double vi : v) kin += vi * vi;
    kin *= 0.5;
    for (int i = 0; i < N; ++i) {
        double curr = (i == 0) ? 0.0 : x[i - 1];
        double next_val = (i == N - 1) ? 0.0 : x[i];
        double dx = next_val - curr;
        pot += 0.5 * dx * dx + (alpha / 3.0) * dx * dx * dx;
    }
    return kin + pot;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "这么个事儿: " << argv[0] << " <N> <Alpha> <Amplitude> <SavePath>" << std::endl;
        return 1;
    }

    const int N = std::stoi(argv[1]);
    const double alpha = std::stod(argv[2]);
    const double amplitude = std::stod(argv[3]);
    const std::string filename = argv[4];

    const int MODES_TO_PLOT = 20;
    const double Dt = 0.20;
    const int STRIDE = 200000;
    const int num_segments = 5000; 

    std::vector<double> h_x(N - 1, 0.0), h_v(N - 1, 0.0);
    for (int j = 1; j < N; ++j) h_x[j - 1] = amplitude * sin(M_PI * j / N);

    double *d_x, *d_v, *d_a, *d_a_new;
    cudaMalloc(&d_x, (N - 1) * sizeof(double));
    cudaMalloc(&d_v, (N - 1) * sizeof(double));
    cudaMalloc(&d_a, (N - 1) * sizeof(double));
    cudaMalloc(&d_a_new, (N - 1) * sizeof(double));
    
    cudaMemcpy(d_x, h_x.data(), (N - 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), (N - 1) * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N - 1 + threadsPerBlock - 1) / threadsPerBlock;
    double dt2 = Dt * Dt;

    get_acceleration_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_a, N, alpha);
    
    std::ofstream outFile(filename);
    outFile << "# Model: alpha\n# N: " << N << "\n# Alpha: " << alpha << "\n# Amplitude: " << amplitude << "\n# dt: " << Dt << "\n";
    outFile << "Time";
    for(int i=1; i<=MODES_TO_PLOT; ++i) outFile << ",Mode" << i;
    outFile << ",TotalEnergy\n" << std::scientific << std::setprecision(15);

    for (int seg = 0; seg < num_segments; ++seg) {
        cudaMemcpy(h_x.data(), d_x, (N - 1) * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v.data(), d_v, (N - 1) * sizeof(double), cudaMemcpyDeviceToHost);

        double t_current = seg * (long long)STRIDE * Dt;
        std::vector<double> mode_E(MODES_TO_PLOT, 0.0);
        get_mode_energies(N, h_x, h_v, mode_E, MODES_TO_PLOT);
        outFile << t_current;
        for (int k = 0; k < MODES_TO_PLOT; ++k) outFile << "," << mode_E[k];
        outFile << "," << get_total_energy(N, alpha, h_x, h_v) << "\n";
        
        if (seg % 50 == 0) outFile.flush();

        for (int step = 0; step < STRIDE; ++step) {
            update_x_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_v, d_a, Dt, dt2, N);
            get_acceleration_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_a_new, N, alpha);
            update_v_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_v, d_a, d_a_new, Dt, N);
            double* temp = d_a; d_a = d_a_new; d_a_new = temp;
        }
    }

    outFile.close();
    cudaFree(d_x); cudaFree(d_v); cudaFree(d_a); cudaFree(d_a_new);
    return 0;
}
