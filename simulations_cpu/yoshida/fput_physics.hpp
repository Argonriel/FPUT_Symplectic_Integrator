// fput_physics.hpp
// Shared physics kernels for the 4th-order Yoshida FPUT solver.
//
// This header was extracted from FPUT_yoshida_solver.cpp so that BOTH the
// production solver (FPUT_yoshida_solver.cpp) and the correctness self-test
// driver (fput_selftest.cpp) call the *identical* functions. This is
// deliberate: it lets the tests exercise the exact production kernels with
// arbitrary states, rather than re-deriving the indexing in a second place
// (which would risk duplicating any boundary/indexing mistake).
//
// The physical-trajectory kernels (compute_forces, yoshida_step,
// get_mode_energies, full_spectrum_entropy, get_total_energy) are byte-for-byte
// the same arithmetic as the original single-file solver, so a diagnostics-off
// run reproduces the previous trajectory up to floating-point reproducibility.
//
// New, opt-in diagnostics added here:
//   - compute_toda_J        : exact intensive 4th-order Toda integral (alpha=1 form)
//   - compute_toda_J_quad   : low-energy quadratic approximation (validation only)
//   - compute_tangent_force : linearized force (Jacobian-vector product)
//   - yoshida_step_tangent  : one Yoshida step advancing BOTH state and tangent

#ifndef FPUT_PHYSICS_HPP
#define FPUT_PHYSICS_HPP

#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fput {

// Model flags (kept identical to the original solver's convention).
constexpr int MODEL_ALPHA = 0;
constexpr int MODEL_BETA  = 1;

// ---------- Yoshida 4th-order coefficients ----------
// Yoshida (1990), Phys. Lett. A 150, 262
// One full step: T(c1) V(d1) T(c2) V(d2) T(c2) V(d1) T(c1)
static const double CBRT2 = 1.2599210498948732;      // 2^(1/3)
static const double W1    =  1.0 / (2.0 - CBRT2);   //  1.3512071919596578
static const double W0    = -CBRT2 / (2.0 - CBRT2); // -1.7024143839193153
static const double C1    = W1 * 0.5;                // position coeff, symmetric: C4 = C1
static const double C2    = (W0 + W1) * 0.5;         // position coeff, symmetric: C3 = C2
static const double D1    = W1;                       // momentum coeff, symmetric: D3 = D1
static const double D2    = W0;                       // momentum coeff

// ---------- Physics ----------

// Forces on all N-1 interior particles (fixed-end BCs: x_0 = x_N = 0).
// FPUT-alpha: f'(r) = r + alpha*r^2  -->  F_j = (dx_f - dx_b) + alpha*(dx_f^2 - dx_b^2)
// FPUT-beta:  f'(r) = r + beta*r^3   -->  F_j = (dx_f - dx_b) + beta*(dx_f^3 - dx_b^3)
// Optional telemetry: if min_bond != nullptr, the most negative bond strain seen
// in this evaluation is folded into *min_bond. This piggybacks on the bond
// differences already computed here (no separate O(N) scan). It never alters the
// force arithmetic, so trajectories are bit-identical whether or not it is used.
static inline void compute_forces(const std::vector<double>& x, std::vector<double>& F,
                                  int N, double value, int model_flag,
                                  double* min_bond = nullptr) {
    const int M = N - 1;
    for (int j = 0; j < M; ++j) {
        const double curr  = x[j];
        const double prev  = (j == 0)     ? 0.0 : x[j - 1];
        const double next  = (j == M - 1) ? 0.0 : x[j + 1];
        const double dx_f  = next - curr;
        const double dx_b  = curr - prev;
        double force = dx_f - dx_b;
        if (model_flag == MODEL_ALPHA)
            force += value * (dx_f * dx_f - dx_b * dx_b);
        else
            force += value * (dx_f * dx_f * dx_f - dx_b * dx_b * dx_b);
        F[j] = force;
        if (min_bond) {
            // dx_b covers bonds 0..M-1; dx_f at j=M-1 covers the last bond M.
            if (dx_b < *min_bond) *min_bond = dx_b;
            if (dx_f < *min_bond) *min_bond = dx_f;
        }
    }
}

// Linearized force (tangent / Jacobian-vector product):
//   (DF(x) . dx)_j = K_f * (dx_{j+1} - dx_j) - K_b * (dx_j - dx_{j-1})
// with fixed-boundary tangent displacements dx_{-1} = dx_M = 0, and bond
// stiffness  K(r) = f'(r):
//   FPUT-alpha: K(r) = 1 + 2*alpha*r
//   FPUT-beta:  K(r) = 1 + 3*beta*r^2
// dx here is the tangent displacement vector; the physical state x sets the
// stiffnesses. O(N), no allocation.
static inline void compute_tangent_force(const std::vector<double>& x,
                                         const std::vector<double>& dx,
                                         std::vector<double>& DFdx,
                                         int N, double value, int model_flag) {
    const int M = N - 1;
    for (int j = 0; j < M; ++j) {
        const double curr  = x[j];
        const double prev  = (j == 0)     ? 0.0 : x[j - 1];
        const double next  = (j == M - 1) ? 0.0 : x[j + 1];
        const double r_f   = next - curr;   // forward bond displacement
        const double r_b   = curr - prev;   // backward bond displacement

        double K_f, K_b;
        if (model_flag == MODEL_ALPHA) {
            K_f = 1.0 + 2.0 * value * r_f;
            K_b = 1.0 + 2.0 * value * r_b;
        } else {
            K_f = 1.0 + 3.0 * value * r_f * r_f;
            K_b = 1.0 + 3.0 * value * r_b * r_b;
        }

        const double dcurr = dx[j];
        const double dprev = (j == 0)     ? 0.0 : dx[j - 1];
        const double dnext = (j == M - 1) ? 0.0 : dx[j + 1];
        DFdx[j] = K_f * (dnext - dcurr) - K_b * (dcurr - dprev);
    }
}

// ---------- Yoshida step (physical only) ----------
// F is a reusable scratch buffer; pass the same vector every call.
static inline void yoshida_step(std::vector<double>& x, std::vector<double>& v,
                                std::vector<double>& F, int N, double dt,
                                double value, int model_flag,
                                double* min_bond = nullptr) {
    const int    M    = N - 1;
    const double c1dt = C1 * dt;
    const double c2dt = C2 * dt;
    const double d1dt = D1 * dt;
    const double d2dt = D2 * dt;

    for (int j = 0; j < M; ++j) x[j] += c1dt * v[j];
    compute_forces(x, F, N, value, model_flag, min_bond);
    for (int j = 0; j < M; ++j) v[j] += d1dt * F[j];

    for (int j = 0; j < M; ++j) x[j] += c2dt * v[j];
    compute_forces(x, F, N, value, model_flag, min_bond);
    for (int j = 0; j < M; ++j) v[j] += d2dt * F[j];

    for (int j = 0; j < M; ++j) x[j] += c2dt * v[j];
    compute_forces(x, F, N, value, model_flag, min_bond);
    for (int j = 0; j < M; ++j) v[j] += d1dt * F[j];

    for (int j = 0; j < M; ++j) x[j] += c1dt * v[j];
}

// ---------- Yoshida step (physical + tangent) ----------
// Advances (x, v) with the SAME operations and order as yoshida_step (so the
// physical trajectory is bit-identical to a diagnostics-off run) and, in
// lockstep, advances the tangent vector (dx, dv) through the linearized map.
// F and DFdx are reusable scratch buffers.
static inline void yoshida_step_tangent(std::vector<double>& x, std::vector<double>& v,
                                        std::vector<double>& dx, std::vector<double>& dv,
                                        std::vector<double>& F, std::vector<double>& DFdx,
                                        int N, double dt, double value, int model_flag,
                                        double* min_bond = nullptr) {
    const int    M    = N - 1;
    const double c1dt = C1 * dt;
    const double c2dt = C2 * dt;
    const double d1dt = D1 * dt;
    const double d2dt = D2 * dt;

    // drift T(c1)
    for (int j = 0; j < M; ++j) { x[j]  += c1dt * v[j];  dx[j] += c1dt * dv[j]; }
    // kick V(d1)
    compute_forces(x, F, N, value, model_flag, min_bond);
    compute_tangent_force(x, dx, DFdx, N, value, model_flag);
    for (int j = 0; j < M; ++j) { v[j]  += d1dt * F[j];  dv[j] += d1dt * DFdx[j]; }

    // drift T(c2)
    for (int j = 0; j < M; ++j) { x[j]  += c2dt * v[j];  dx[j] += c2dt * dv[j]; }
    // kick V(d2)
    compute_forces(x, F, N, value, model_flag, min_bond);
    compute_tangent_force(x, dx, DFdx, N, value, model_flag);
    for (int j = 0; j < M; ++j) { v[j]  += d2dt * F[j];  dv[j] += d2dt * DFdx[j]; }

    // drift T(c2)
    for (int j = 0; j < M; ++j) { x[j]  += c2dt * v[j];  dx[j] += c2dt * dv[j]; }
    // kick V(d1)
    compute_forces(x, F, N, value, model_flag, min_bond);
    compute_tangent_force(x, dx, DFdx, N, value, model_flag);
    for (int j = 0; j < M; ++j) { v[j]  += d1dt * F[j];  dv[j] += d1dt * DFdx[j]; }

    // final drift T(c1)
    for (int j = 0; j < M; ++j) { x[j]  += c1dt * v[j];  dx[j] += c1dt * dv[j]; }
}

// Euclidean phase-space norm of the tangent vector (interior dof only; the
// fixed boundaries carry zero tangent displacement by construction).
static inline double tangent_norm(const std::vector<double>& dx,
                                  const std::vector<double>& dv) {
    double s = 0.0;
    const std::size_t M = dx.size();
    for (std::size_t j = 0; j < M; ++j) s += dx[j] * dx[j] + dv[j] * dv[j];
    return std::sqrt(s);
}

// ---------- Diagnostics ----------

// Modal energies via direct DST-I (cheap 20-mode path, unchanged).
static inline void get_mode_energies(const std::vector<double>& x,
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

// Normalized spectral entropy over ALL N-1 normal modes (O(N^2), unchanged).
static inline double full_spectrum_entropy(const std::vector<double>& x,
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

// Total Hamiltonian: H = K + V (fixed ends x_0 = x_N = 0), unchanged.
static inline double get_total_energy(const std::vector<double>& x,
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
        if (model_flag == MODEL_ALPHA) pot += (value / 3.0) * r * r * r;
        else                           pot += (value / 4.0) * r * r * r * r;
    }
    return kin + pot;
}

// ---------- Toda integral ----------
//
// Exact intensive 4th-order Toda integral J (alpha = 1 form), following
// Christodoulidi & Flach, Chaos 35, 113127 (2025), Eq. (5), with the
// fixed-boundary continuation of Christodoulidi & Efthymiopoulos (2019).
//
// Solver-convention indexing (M = N-1 moving particles):
//   q_0 = q_{M+1} = 0 ;  q_n = x[n-1]  for n = 1..M
//   p_0 = p_{M+1} = 0 ;  p_n = v[n-1]  for n = 1..M
//   dq_n = q_{n+1} - q_n ,  n = 0..M          (M+1 = N bonds)
//   b_n  = exp(2*dq_n)
//   ghost bonds:  b_{-1} = b_0 ,  b_{M+1} = b_M
//
//   J = 1/(2M) * sum_{n=0}^{M} [ p_n^4
//                                + b_n * (p_n^2 + p_n*p_{n+1} + p_{n+1}^2)
//                                + (b_n/8) * (b_{n-1} + b_n + b_{n+1})
//                                - 3/8 ]
//
// This is the RAW intensive J. It is a bare observable: the exponential carries
// neither alpha nor beta. Its adiabatic-invariant interpretation applies to the
// alpha model (near the Toda-integrable point); for beta it is only a control
// observable. Normalization by J(0) belongs in analysis code, not here.
//
// dq_buf and b_buf are reusable scratch buffers of length >= M+1. O(N).
static inline double compute_toda_J(const std::vector<double>& x,
                                    const std::vector<double>& v,
                                    int N,
                                    std::vector<double>& dq_buf,
                                    std::vector<double>& b_buf) {
    const int M = N - 1;
    // dq_n and b_n for n = 0..M  (indices 0..M into the buffers).
    for (int n = 0; n <= M; ++n) {
        double q_lo = (n == 0) ? 0.0 : x[n - 1];     // q_n
        double q_hi = (n == M) ? 0.0 : x[n];         // q_{n+1}
        const double dq = q_hi - q_lo;
        dq_buf[n] = dq;
        b_buf[n]  = std::exp(2.0 * dq);
    }

    double sum = 0.0;
    for (int n = 0; n <= M; ++n) {
        // p_n and p_{n+1} with p_0 = p_{M+1} = 0.
        const double p_n  = (n == 0) ? 0.0 : v[n - 1];
        const double p_n1 = (n == M) ? 0.0 : v[n];   // p_{n+1}; note p_{M+1}=0

        const double b_n  = b_buf[n];
        const double b_lo = (n == 0) ? b_buf[0] : b_buf[n - 1];   // b_{n-1}, ghost b_{-1}=b_0
        const double b_hi = (n == M) ? b_buf[M] : b_buf[n + 1];   // b_{n+1}, ghost b_{M+1}=b_M

        const double p_n2  = p_n * p_n;
        sum += p_n2 * p_n2                                        // p_n^4
             + b_n * (p_n2 + p_n * p_n1 + p_n1 * p_n1)
             + (b_n / 8.0) * (b_lo + b_n + b_hi)
             - 3.0 / 8.0;
    }
    return sum / (2.0 * M);
}

// Low-energy quadratic approximation to J (validation only; NOT for production
// output). Christodoulidi & Flach Eq. (6), solver convention:
//
//   J_quad = 2*epsilon + 1/(2M) * sum_{n=0}^{M} [ p_n*p_{n+1} + dq_n*dq_{n+1} ]
//   epsilon = H_FPUT / M ,  dq_{M+1} = dq_M (fixed-boundary continuation)
//
// The FPUT energy H_FPUT is passed in so the caller controls the model.
static inline double compute_toda_J_quad(const std::vector<double>& x,
                                         const std::vector<double>& v,
                                         int N, double H_fput,
                                         std::vector<double>& dq_buf) {
    const int M = N - 1;
    const double epsilon = H_fput / static_cast<double>(M);
    for (int n = 0; n <= M; ++n) {
        double q_lo = (n == 0) ? 0.0 : x[n - 1];
        double q_hi = (n == M) ? 0.0 : x[n];
        dq_buf[n] = q_hi - q_lo;
    }
    double corr = 0.0;
    for (int n = 0; n <= M; ++n) {
        const double p_n  = (n == 0) ? 0.0 : v[n - 1];
        const double p_n1 = (n == M) ? 0.0 : v[n];
        const double dq_n  = dq_buf[n];
        const double dq_n1 = (n == M) ? dq_buf[M] : dq_buf[n + 1];  // ghost dq_{M+1}=dq_M
        corr += p_n * p_n1 + dq_n * dq_n1;
    }
    return 2.0 * epsilon + corr / (2.0 * M);
}

}  // namespace fput

#endif  // FPUT_PHYSICS_HPP
