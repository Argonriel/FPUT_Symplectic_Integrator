#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>

const double ALPHA = 0.25;
const double dt = 0.01;
const long long TOTAL_STEPS = 10000000;
const int SNAPSHOT_FREQ = 100000;

__global__ void computeForces(double* x, double* f, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < N) {
        double dx_right = x[i+1] - x[i];
        double dx_left = x[i] - x[i-1];
        f[i] = (dx_right - dx_left) + ALPHA * (dx_right * dx_right - dx_left * dx_left);
    }
}

__global__ void updatePosition(double* x, double* v, double* f, double dt, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < N) { v[i] += 0.5 * f[i] * dt; x[i] += v[i] * dt; }
}

__global__ void updateVelocity(double* v, double* f, double dt, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < N) { v[i] += 0.5 * f[i] * dt; }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    double A = (argc > 2) ? std::atof(argv[2]) : 10.0;

    double *h_x = new double[N+1], *h_v = new double[N+1];
    for(int i = 0; i <= N; ++i) { h_x[i] = A * sin(M_PI * i / N); h_v[i] = 0.0; }
    
    double *d_x, *d_v, *d_f;
    cudaMalloc(&d_x, (N+1)*sizeof(double));
    cudaMalloc(&d_v, (N+1)*sizeof(double));
    cudaMalloc(&d_f, (N+1)*sizeof(double));
    cudaMemcpy(d_x, h_x, (N+1)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, (N+1)*sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize) / blockSize;
    computeForces<<<numBlocks, blockSize>>>(d_x, d_f, N);

    for(long long step = 0; step <= TOTAL_STEPS; ++step) {
        updatePosition<<<numBlocks, blockSize>>>(d_x, d_v, d_f, dt, N);
        computeForces<<<numBlocks, blockSize>>>(d_x, d_f, N);
        updateVelocity<<<numBlocks, blockSize>>>(d_v, d_f, dt, N);

        if (step > 0 && step % SNAPSHOT_FREQ == 0) {
            cudaMemcpy(h_x, d_x, (N+1)*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_v, d_v, (N+1)*sizeof(double), cudaMemcpyDeviceToHost);
            
            double total_E = 0.0;
            std::vector<double> E_k(N, 0.0);
            for(int k = 1; k < N; ++k) {
                double Q_k = 0.0, P_k = 0.0;
                for(int i = 1; i < N; ++i) {
                    double term = sin(M_PI * k * i / (double)N);
                    Q_k += h_x[i] * term; P_k += h_v[i] * term;
                }
                Q_k *= sqrt(2.0 / N); P_k *= sqrt(2.0 / N);
                double omega = 2.0 * sin(M_PI * k / (2.0 * N));
                E_k[k] = 0.5 * P_k * P_k + 0.5 * omega * omega * Q_k * Q_k;
                total_E += E_k[k];
            }
            double entropy = 0.0;
            for(int k = 1; k < N; ++k) {
                if(E_k[k] > 1e-15) {
                    double p_k = E_k[k] / total_E;
                    entropy -= p_k * log(p_k);
                }
            }
            std::cout << "STEP:" << step << ",ETA:" << (entropy / log((double)(N-1))) << std::endl;
        }
    }
    cudaFree(d_x); cudaFree(d_v); cudaFree(d_f);
    delete[] h_x; delete[] h_v;
    return 0;
}
