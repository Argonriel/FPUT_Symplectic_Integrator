// FPUT_yoshida_solver.cpp
// 4th-order Yoshida symplectic integrator for FPUT-alpha/beta lattices
//
// Compile: g++ -O3 -march=native -o fput_yoshida FPUT_yoshida_solver.cpp
// Usage:   ./fput_yoshida <N> <alpha|beta> <Value> <Amplitude> <SavePath> [options]
//
// Positional args (required, in order):
//   N          number of particles + 1 (interior particles: N-1)
//   alpha|beta nonlinearity type
//   Value      alpha or beta coefficient
//   Amplitude  mode-1 initial displacement amplitude
//   SavePath   output CSV file path
//
// Optional flags (any order, defaults reproduce original behavior):
//   --dt <double>    integration time step          (default: 0.1)
//   --stride <int>   Yoshida steps between snapshots (default: 200000)
//   --nseg <int>     number of snapshots             (default: 5000)
//   --shape          append x1..x{N-1} columns; only allowed for N<=256
//   --entropy        append Eta column (normalized spectral entropy, O(N^2)/snapshot)
//
// Physical run length: t_max = stride * nseg * dt
// Examples:
//   ./fput_yoshida 1024 alpha 0.25 0.4  data/1024_a0.25_A0.4.csv
//   ./fput_yoshida 32   alpha 0.25 2.0  data/32_shape.csv --shape
//   ./fput_yoshida 1024 alpha 0.25 20   data/1024_entropy.csv --entropy
//   ./fput_yoshida 1024 alpha 0.1  0.4  data/out.csv --nseg 10000   # t_max = 2e8
//
// Output CSV format is identical to FPUT_cuda_solver.cu (compatible with
// visualization scripts), with an extra "# Integrator: Yoshida4" header line.

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- Yoshida 4th-order coefficients ----------
// Yoshida (1990), Phys. Lett. A 150, 262
// One full step: T(c1) V(d1) T(c2) V(d2) T(c2) V(d1) T(c1)
// where T(s) shifts positions, V(s) shifts momenta via forces.
static const double CBRT2 = 1.2599210498948732;      // 2^(1/3)
static const double W1    =  1.0 / (2.0 - CBRT2);   //  1.3512071919596578
static const double W0    = -CBRT2 / (2.0 - CBRT2); // -1.7024143839193153  (negative: backward sub-step)
static const double C1    = W1 * 0.5;                // position coeff, symmetric: C4 = C1
static const double C2    = (W0 + W1) * 0.5;         // position coeff, symmetric: C3 = C2
static const double D1    = W1;                       // momentum coeff, symmetric: D3 = D1
static const double D2    = W0;                       // momentum coeff

// ---------- Physics ----------

// Forces on all N-1 interior particles (fixed-end BCs: x_0 = x_N = 0).
// FPUT-alpha: f'(r) = r + alpha*r^2  -->  F_j = (dx_f - dx_b) + alpha*(dx_f^2 - dx_b^2)
// FPUT-beta:  f'(r) = r + beta*r^3   -->  F_j = (dx_f - dx_b) + beta*(dx_f^3 - dx_b^3)
static void compute_forces(const std::vector<double>& x, std::vector<double>& F,
                            int N, double value, int model_flag) {
    const int M = N - 1;
    for (int j = 0; j < M; ++j) {
        const double curr  = x[j];
        const double prev  = (j == 0)     ? 0.0 : x[j - 1];
        const double next  = (j == M - 1) ? 0.0 : x[j + 1];
        const double dx_f  = next - curr;
        const double dx_b  = curr - prev;
        double force = dx_f - dx_b;
        if (model_flag == 0)
            force += value * (dx_f * dx_f - dx_b * dx_b);
        else
            force += value * (dx_f * dx_f * dx_f - dx_b * dx_b * dx_b);
        F[j] = force;
    }
}

// ---------- Yoshida step ----------

// Each step costs 3 force evaluations (vs 1 for Verlet).
// The D2 sub-step uses a negative effective step size W0*dt < 0, giving the
// method its 4th-order accuracy through backward-in-time cancellation of errors.
// F is used as a reusable scratch buffer; pass the same vector every call.
static void yoshida_step(std::vector<double>& x, std::vector<double>& v,
                          std::vector<double>& F, int N, double dt,
                          double value, int model_flag) {
    const int    M    = N - 1;
    const double c1dt = C1 * dt;
    const double c2dt = C2 * dt;
    const double d1dt = D1 * dt;
    const double d2dt = D2 * dt;

    // sub-step 1: T(c1) V(d1)
    for (int j = 0; j < M; ++j) x[j] += c1dt * v[j];
    compute_forces(x, F, N, value, model_flag);
    for (int j = 0; j < M; ++j) v[j] += d1dt * F[j];

    // sub-step 2: T(c2) V(d2)
    for (int j = 0; j < M; ++j) x[j] += c2dt * v[j];
    compute_forces(x, F, N, value, model_flag);
    for (int j = 0; j < M; ++j) v[j] += d2dt * F[j];

    // sub-step 3: T(c2) V(d1)   [C3 = C2, D3 = D1 by time-reversal symmetry]
    for (int j = 0; j < M; ++j) x[j] += c2dt * v[j];
    compute_forces(x, F, N, value, model_flag);
    for (int j = 0; j < M; ++j) v[j] += d1dt * F[j];

    // final position half-update: T(c1)   [C4 = C1]
    for (int j = 0; j < M; ++j) x[j] += c1dt * v[j];
}

// ---------- Diagnostics ----------

// Modal energies via direct DST-I: Q_k = sqrt(2/N) * sum_j x_j * sin(pi*j*k/N)
// Equivalent to scipy.fft.dst(x, type=1) / sqrt(2N) used in the Python solver.
static void get_mode_energies(const std::vector<double>& x,
                               const std::vector<double>& v,
                               int N, std::vector<double>& mode_E, int modes_to_plot) {
    const double factor = std::sqrt(2.0 / N);
    for (int k = 1; k <= modes_to_plot; ++k) {
        double Q_k = 0.0, P_k = 0.0;
        for (int j = 1; j < N; ++j) {
            const double s = std::sin(M_PI * j * k / N);
            Q_k += x[j - 1] * s;
            P_k += v[j - 1] * s;
        }
        Q_k *= factor;
        P_k *= factor;
        const double omega = 2.0 * std::sin(M_PI * k / (2.0 * N));
        mode_E[k - 1] = 0.5 * (P_k * P_k + omega * omega * Q_k * Q_k);
    }
}

// Normalized spectral entropy over ALL N-1 normal modes (O(N^2) per call).
// Only called when --entropy is set; the cheap 20-mode path above is unchanged.
// eta = S / ln(N-1),  S = -sum_k p_k*ln(p_k),  p_k = E_k / sum_j(E_j)
// Treats 0*ln(0) as 0 (skip modes with E_k <= 1e-15 to avoid log(0)).
static double full_spectrum_entropy(const std::vector<double>& x,
                                    const std::vector<double>& v, int N) {
    const int    M      = N - 1;
    const double factor = std::sqrt(2.0 / N);
    double E_sum = 0.0;
    std::vector<double> E(M);
    for (int k = 1; k <= M; ++k) {
        double Q_k = 0.0, P_k = 0.0;
        for (int j = 1; j <= M; ++j) {
            const double s = std::sin(M_PI * j * k / N);
            Q_k += x[j - 1] * s;
            P_k += v[j - 1] * s;
        }
        Q_k *= factor;
        P_k *= factor;
        const double omega = 2.0 * std::sin(M_PI * k / (2.0 * N));
        E[k - 1] = 0.5 * (P_k * P_k + omega * omega * Q_k * Q_k);
        E_sum += E[k - 1];
    }
    if (E_sum <= 0.0) return 0.0;
    double S = 0.0;
    for (int k = 0; k < M; ++k) {
        if (E[k] > 1e-15) {
            const double p = E[k] / E_sum;
            S -= p * std::log(p);
        }
    }
    return S / std::log(static_cast<double>(M));
}

// Total Hamiltonian: H = K + V_linear + V_nonlinear
// V = sum_i [ r_i^2/2 + alpha*r_i^3/3 ]  or  [ r_i^2/2 + beta*r_i^4/4 ]
// where r_i = x_{i+1} - x_i, with x_0 = x_N = 0.
static double get_total_energy(const std::vector<double>& x,
                                const std::vector<double>& v,
                                int N, double value, int model_flag) {
    double kin = 0.0;
    for (const double vi : v) kin += vi * vi;
    kin *= 0.5;

    double pot = 0.0;
    for (int i = 0; i < N; ++i) {
        const double xl = (i == 0)     ? 0.0 : x[i - 1];
        const double xr = (i == N - 1) ? 0.0 : x[i];
        const double r  = xr - xl;
        pot += 0.5 * r * r;
        if (model_flag == 0) pot += (value / 3.0) * r * r * r;
        else                 pot += (value / 4.0) * r * r * r * r;
    }
    return kin + pot;
}

// ---------- Main ----------

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <N> <alpha|beta> <Value> <Amplitude> <SavePath> [--shape] [--entropy]\n";
        return 1;
    }

    const int         N         = std::stoi(argv[1]);
    const std::string model_str = argv[2];
    const double      value     = std::stod(argv[3]);
    const double      amplitude = std::stod(argv[4]);
    const std::string filename  = argv[5];

    // Optional flags and named parameters (any order after positional args).
    bool   shape_mode   = false;
    bool   entropy_mode = false;
    double Dt           = 0.1;
    long long stride    = 200000;
    long long nseg      = 5000;

    for (int i = 6; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--shape") {
            shape_mode = true;
        } else if (arg == "--entropy") {
            entropy_mode = true;
        } else if (arg == "--dt") {
            if (i + 1 >= argc) { std::cerr << "Error: --dt requires a value.\n"; return 1; }
            try { Dt = std::stod(argv[++i]); } catch (...) {
                std::cerr << "Error: --dt value is not a valid number.\n"; return 1;
            }
            if (Dt <= 0.0) { std::cerr << "Error: --dt must be > 0.\n"; return 1; }
        } else if (arg == "--stride") {
            if (i + 1 >= argc) { std::cerr << "Error: --stride requires a value.\n"; return 1; }
            try { stride = std::stoll(argv[++i]); } catch (...) {
                std::cerr << "Error: --stride value is not a valid integer.\n"; return 1;
            }
            if (stride < 1) { std::cerr << "Error: --stride must be >= 1.\n"; return 1; }
        } else if (arg == "--nseg") {
            if (i + 1 >= argc) { std::cerr << "Error: --nseg requires a value.\n"; return 1; }
            try { nseg = std::stoll(argv[++i]); } catch (...) {
                std::cerr << "Error: --nseg value is not a valid integer.\n"; return 1;
            }
            if (nseg < 1) { std::cerr << "Error: --nseg must be >= 1.\n"; return 1; }
        } else {
            std::cerr << "Error: unknown flag '" << arg << "'.\n"; return 1;
        }
    }
    if (shape_mode && N > 256) {
        std::cerr << "Error: --shape disabled for N>256 to avoid generating an enormous CSV;"
                     " remove --shape or use a smaller N.\n";
        return 1;
    }

    if (model_str != "alpha" && model_str != "beta") {
        std::cerr << "Error: model must be 'alpha' or 'beta'.\n";
        return 1;
    }
    const int model_flag = (model_str == "alpha") ? 0 : 1;

    // ---- Simulation parameters ----
    const int MODES_TO_PLOT = 20;
    // Dt, stride, nseg already parsed above (defaults: 0.1 / 200000 / 5000).

    const int M = N - 1;
    std::vector<double> x(M, 0.0), v(M, 0.0), F(M, 0.0);

    // Sine mode-1 initial condition (same as both reference solvers)
    for (int j = 0; j < M; ++j)
        x[j] = amplitude * std::sin(M_PI * (j + 1) / N);

    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: cannot open output file: " << filename << "\n";
        return 1;
    }

    // Metadata header (same keys as FPUT_cuda_solver.cu + extra Integrator line)
    outFile << "# Integrator: Yoshida4\n"
            << "# Model: "       << model_str << "\n"
            << "# N: "           << N         << "\n"
            << "# " << (model_flag == 0 ? "Alpha" : "Beta") << ": " << value << "\n"
            << "# Amplitude: "   << amplitude << "\n"
            << "# dt: "          << Dt        << "\n"
            << "# Stride: "      << stride    << "\n"
            << "# NumSegments: " << nseg      << "\n"
            << "# Shape: "       << (shape_mode   ? 1 : 0) << "\n"
            << "# Entropy: "     << (entropy_mode ? 1 : 0) << "\n";
    outFile << "Time";
    for (int i = 1; i <= MODES_TO_PLOT; ++i) outFile << ",Mode" << i;
    outFile << ",TotalEnergy";
    if (entropy_mode) outFile << ",Eta";
    if (shape_mode)
        for (int j = 1; j < N; ++j) outFile << ",x" << j;
    outFile << "\n";
    outFile << std::scientific << std::setprecision(15);

    std::vector<double> mode_E(MODES_TO_PLOT, 0.0);
    const long long total_steps = stride * nseg;
    auto t_wall = std::chrono::high_resolution_clock::now();

    std::cout << "FPUT Yoshida-4  |  N=" << N
              << "  |  " << model_str << "=" << value
              << "  |  A=" << amplitude
              << "  |  dt=" << Dt
              << "  |  total steps=" << total_steps
              << "  |  t_max=" << static_cast<double>(total_steps) * Dt << "\n";

    const int prog_every = std::max(1LL, nseg / 10);
    for (long long seg = 0; seg < nseg; ++seg) {

        // Snapshot before advancing
        const double t_current = static_cast<double>(seg) * static_cast<double>(stride) * Dt;
        get_mode_energies(x, v, N, mode_E, MODES_TO_PLOT);
        outFile << t_current;
        for (int k = 0; k < MODES_TO_PLOT; ++k) outFile << "," << mode_E[k];
        outFile << "," << get_total_energy(x, v, N, value, model_flag);
        if (entropy_mode) outFile << "," << full_spectrum_entropy(x, v, N);
        if (shape_mode)
            for (int j = 0; j < M; ++j) outFile << "," << x[j];
        outFile << "\n";
        if (seg % 50 == 0) outFile.flush();

        // Advance stride Yoshida steps
        for (long long step = 0; step < stride; ++step)
            yoshida_step(x, v, F, N, Dt, value, model_flag);

        if (seg % prog_every == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t_wall).count();
            std::cout << "  " << std::fixed << std::setprecision(1)
                      << (100.0 * seg / nseg) << "%"
                      << "  |  t=" << t_current
                      << "  |  " << elapsed << "s elapsed\n" << std::flush;
        }
    }

    // Final 100% progress line
    {
        auto now = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t_wall).count();
        const double t_final = static_cast<double>(nseg - 1) * static_cast<double>(stride) * Dt;
        std::cout << "  100.0%  |  t=" << t_final
                  << "  |  " << elapsed << "s elapsed\n" << std::flush;
    }

    outFile.close();
    const double total_wall = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall).count();
    std::cout << "Done!  " << total_wall << "s  |  saved to " << filename << "\n";
    return 0;
}
