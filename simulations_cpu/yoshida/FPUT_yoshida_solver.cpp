// FPUT_yoshida_solver.cpp
// 4th-order Yoshida symplectic integrator for FPUT-alpha/beta lattices
// with opt-in dynamical diagnostics (Toda integral J, max Lyapunov exponent)
// and checkpoint/restart.
//
// Compile (simple):  g++ -O3 -march=native -o fput_yoshida FPUT_yoshida_solver.cpp
// Compile (with git provenance):  make          (see Makefile)
//
// Usage:   ./fput_yoshida <N> <alpha|beta> <Value> <Amplitude> <SavePath> [options]
//
// Positional args (required, in order):
//   N          number of particles + 1 (interior particles: N-1)
//   alpha|beta nonlinearity type
//   Value      alpha or beta coefficient
//   Amplitude  mode-1 initial displacement amplitude
//   SavePath   output CSV file path (on --resume this is the CONTINUATION file)
//
// Optional flags (any order, defaults reproduce original behavior):
//   --dt <double>          integration time step          (default: 0.1)
//   --stride <int>         Yoshida steps between snapshots (default: 200000)
//   --nseg <int>           number of snapshots             (default: 5000)
//   --shape                append x1..x{N-1} columns; only allowed for N<=256
//   --entropy              append Eta column (normalized spectral entropy)
//   --toda                 append TodaJ column (exact intensive 4th-order Toda integral)
//   --toda-debug           additionally append TodaJ_quad (quadratic approx; diagnostic)
//   --lyapunov             append LyapunovFTLE,LyapunovLocal,LyapRenormCount columns
//   --lyap-renorm-steps <int>  Yoshida steps between tangent renormalizations (default: 100)
//   --lyap-seed <int>          RNG seed for the initial tangent vector       (default: 12345)
//   --checkpoint-file <path>   write a binary checkpoint at segment boundaries
//   --checkpoint-every <int>   segments between checkpoints    (default: 50)
//   --resume <path>            resume from a checkpoint; SavePath is a NEW
//                              continuation CSV holding snapshots from the
//                              checkpoint segment onward (see docs/diagnostics.md).
//
// Diagnostics are opt-in: with none of {--toda,--toda-debug,--lyapunov} the
// numerical trajectory and CSV schema are unchanged from the original solver.
//
// Timing convention (a snapshot is written BEFORE each advance):
//   last saved time         = (NumSegments - 1) * Stride * dt
//   nominal integrated time =  NumSegments      * Stride * dt
// These are NOT equal; both are reported. Do not call both t_max.

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <random>

#include "fput_physics.hpp"
#include "fput_checkpoint.hpp"

// Provenance is injected by the Makefile. Defaults are conservative: an unknown
// build is assumed "dirty" so it cannot masquerade as reproducible.
#ifndef FPUT_GIT_COMMIT
#define FPUT_GIT_COMMIT "unknown"
#endif
#ifndef FPUT_GIT_DIRTY
#define FPUT_GIT_DIRTY 1
#endif

using fput::MODEL_ALPHA;

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <N> <alpha|beta> <Value> <Amplitude> <SavePath>"
                     " [--dt d] [--stride s] [--nseg n] [--shape] [--entropy]"
                     " [--toda] [--toda-debug] [--lyapunov]"
                     " [--lyap-renorm-steps k] [--lyap-seed s]"
                     " [--checkpoint-file p] [--checkpoint-every k] [--resume p]\n";
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
    bool   toda_mode    = false;
    bool   toda_debug   = false;
    bool   lyap_mode    = false;
    double Dt           = 0.1;
    long long stride    = 200000;
    long long nseg      = 5000;
    long long lyap_renorm_steps = 100;
    long long lyap_seed = 12345;
    std::string checkpoint_file;
    long long checkpoint_every = 50;
    std::string resume_file;

    for (int i = 6; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--shape") {
            shape_mode = true;
        } else if (arg == "--entropy") {
            entropy_mode = true;
        } else if (arg == "--toda") {
            toda_mode = true;
        } else if (arg == "--toda-debug") {
            toda_mode = true;
            toda_debug = true;
        } else if (arg == "--lyapunov") {
            lyap_mode = true;
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
        } else if (arg == "--lyap-renorm-steps") {
            if (i + 1 >= argc) { std::cerr << "Error: --lyap-renorm-steps requires a value.\n"; return 1; }
            try { lyap_renorm_steps = std::stoll(argv[++i]); } catch (...) {
                std::cerr << "Error: --lyap-renorm-steps value is not a valid integer.\n"; return 1;
            }
            if (lyap_renorm_steps < 1) { std::cerr << "Error: --lyap-renorm-steps must be >= 1.\n"; return 1; }
        } else if (arg == "--lyap-seed") {
            if (i + 1 >= argc) { std::cerr << "Error: --lyap-seed requires a value.\n"; return 1; }
            try { lyap_seed = std::stoll(argv[++i]); } catch (...) {
                std::cerr << "Error: --lyap-seed value is not a valid integer.\n"; return 1;
            }
        } else if (arg == "--checkpoint-file") {
            if (i + 1 >= argc) { std::cerr << "Error: --checkpoint-file requires a path.\n"; return 1; }
            checkpoint_file = argv[++i];
        } else if (arg == "--checkpoint-every") {
            if (i + 1 >= argc) { std::cerr << "Error: --checkpoint-every requires a value.\n"; return 1; }
            try { checkpoint_every = std::stoll(argv[++i]); } catch (...) {
                std::cerr << "Error: --checkpoint-every value is not a valid integer.\n"; return 1;
            }
            if (checkpoint_every < 1) { std::cerr << "Error: --checkpoint-every must be >= 1.\n"; return 1; }
        } else if (arg == "--resume") {
            if (i + 1 >= argc) { std::cerr << "Error: --resume requires a path.\n"; return 1; }
            resume_file = argv[++i];
        } else {
            std::cerr << "Error: unknown flag '" << arg << "'.\n"; return 1;
        }
    }
    if (shape_mode && N > 256) {
        std::cerr << "Error: --shape disabled for N>256 to avoid generating an enormous CSV;"
                     " remove --shape or use a smaller N.\n";
        return 1;
    }
    if (N < 2) {
        std::cerr << "Error: N must be >= 2 (at least one interior particle).\n";
        return 1;
    }
    if (model_str != "alpha" && model_str != "beta") {
        std::cerr << "Error: model must be 'alpha' or 'beta'.\n";
        return 1;
    }
    const int model_flag = (model_str == "alpha") ? MODEL_ALPHA : fput::MODEL_BETA;

    // ---- Simulation parameters ----
    const int MODES_TO_PLOT = 20;
    const int M = N - 1;
    std::vector<double> x(M, 0.0), v(M, 0.0), F(M, 0.0);

    // ---- Tangent (Lyapunov) state ----
    std::vector<double> dx, dv, DFdx;
    double log_accum = 0.0;         // sum of ln(norm) over completed renormalizations
    double log_at_last_snap = 0.0;  // cumulative log-stretch recorded at previous snapshot
    double t_at_last_snap = 0.0;
    long long renorm_count = 0;

    // Run-summary telemetry: most negative bond strain reached (escape guard).
    double min_bond_strain = std::numeric_limits<double>::infinity();

    long long seg_start = 0;           // first snapshot index to write this invocation
    std::string origin_commit = FPUT_GIT_COMMIT;
    int origin_dirty = FPUT_GIT_DIRTY;

    // ---- Resume path ----
    const bool resuming = !resume_file.empty();
    if (resuming) {
        fput::CheckpointState cp;
        std::string err;
        if (!fput::read_checkpoint(resume_file, cp, err)) {
            std::cerr << "Error: cannot resume: " << err << "\n";
            return 1;
        }
        // Reject resume when dynamically-relevant parameters do not match.
        auto mismatch = [](const std::string& name) {
            std::cerr << "Error: resume parameter mismatch: " << name
                      << " differs from checkpoint.\n";
        };
        bool bad = false;
        if (cp.N != N)                            { mismatch("N"); bad = true; }
        if (cp.model_flag != model_flag)          { mismatch("model"); bad = true; }
        if (cp.value != value)                    { mismatch("Value"); bad = true; }
        if (cp.amplitude != amplitude)            { mismatch("Amplitude"); bad = true; }
        if (cp.dt != Dt)                          { mismatch("dt"); bad = true; }
        if (cp.stride != stride)                  { mismatch("stride"); bad = true; }
        if (cp.entropy != (entropy_mode ? 1 : 0)) { mismatch("--entropy"); bad = true; }
        if (cp.toda != (toda_mode ? 1 : 0))       { mismatch("--toda"); bad = true; }
        if (cp.toda_debug != (toda_debug ? 1 : 0)){ mismatch("--toda-debug"); bad = true; }
        if (cp.lyapunov != (lyap_mode ? 1 : 0))   { mismatch("--lyapunov"); bad = true; }
        if (cp.lyapunov && cp.lyap_renorm_steps != lyap_renorm_steps) { mismatch("--lyap-renorm-steps"); bad = true; }
        if (cp.lyapunov && cp.lyap_seed != lyap_seed) { mismatch("--lyap-seed"); bad = true; }
        if (bad) return 1;
        if (nseg <= cp.next_seg) {
            std::cerr << "Error: --nseg (" << nseg << ") must exceed the checkpoint segment ("
                      << cp.next_seg << "); nothing to do.\n";
            return 1;
        }

        // Restore state.
        x = cp.x; v = cp.v;
        if (lyap_mode) {
            dx = cp.dx; dv = cp.dv; DFdx.assign(M, 0.0);
            log_accum = cp.log_accum;
            renorm_count = cp.renorm_count;
            log_at_last_snap = cp.log_at_last_snap;
            t_at_last_snap = cp.t_at_last_snap;
        }
        min_bond_strain = cp.min_bond_strain;
        seg_start = cp.next_seg;
        origin_commit = cp.origin_git_commit;
        origin_dirty = cp.origin_git_dirty;
        std::cout << "Resuming from " << resume_file << " at segment " << seg_start
                  << " (origin commit " << origin_commit
                  << ", dirty=" << origin_dirty << ")\n";
    } else {
        // ---- Fresh run: sine mode-1 initial condition ----
        for (int j = 0; j < M; ++j)
            x[j] = amplitude * std::sin(M_PI * (j + 1) / N);
        if (lyap_mode) {
            dx.assign(M, 0.0); dv.assign(M, 0.0); DFdx.assign(M, 0.0);
            std::mt19937_64 rng(static_cast<unsigned long long>(lyap_seed));
            std::normal_distribution<double> gauss(0.0, 1.0);
            for (int j = 0; j < M; ++j) { dx[j] = gauss(rng); dv[j] = gauss(rng); }
            const double n0 = fput::tangent_norm(dx, dv);
            if (!(n0 > 0.0)) { std::cerr << "Error: initial tangent vector has zero norm.\n"; return 1; }
            for (int j = 0; j < M; ++j) { dx[j] /= n0; dv[j] /= n0; }
        }
    }

    // Toda scratch buffers (length M+1), only if enabled.
    std::vector<double> dq_buf, b_buf;
    if (toda_mode) { dq_buf.assign(M + 1, 0.0); b_buf.assign(M + 1, 0.0); }

    // Open output. Fresh run truncates; resume writes a fresh continuation file.
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: cannot open output file: " << filename << "\n";
        return 1;
    }

    // Metadata header (original keys preserved; new keys appended only when relevant).
    outFile << "# Integrator: Yoshida4\n"
            << "# Model: "       << model_str << "\n"
            << "# N: "           << N         << "\n"
            << "# " << (model_flag == MODEL_ALPHA ? "Alpha" : "Beta") << ": " << value << "\n"
            << "# Amplitude: "   << amplitude << "\n"
            << "# dt: "          << Dt        << "\n"
            << "# Stride: "      << stride    << "\n"
            << "# NumSegments: " << nseg      << "\n"
            << "# Shape: "       << (shape_mode   ? 1 : 0) << "\n"
            << "# Entropy: "     << (entropy_mode ? 1 : 0) << "\n"
            << "# TodaIntegral: "<< (toda_mode ? 1 : 0)    << "\n"
            << "# Lyapunov: "    << (lyap_mode ? 1 : 0)    << "\n";
    if (lyap_mode) {
        outFile << "# LyapRenormSteps: " << lyap_renorm_steps << "\n"
                << "# LyapSeed: "        << lyap_seed         << "\n";
    }
    if (resuming) {
        // Continuation file: NumSegments above is the TOTAL target; this file
        // holds snapshots [ResumeFromSegment .. NumSegments-1]. The full
        // trajectory is the original CSV's rows [0..ResumeFromSegment-1] followed
        // by this file's rows.
        outFile << "# ResumeFromSegment: "     << seg_start      << "\n"
                << "# CheckpointOriginCommit: " << origin_commit  << "\n"
                << "# CheckpointOriginDirty: "  << origin_dirty   << "\n";
    }
    outFile << "# SolverGitCommit: " << FPUT_GIT_COMMIT << "\n"
            << "# SolverGitDirty: "  << FPUT_GIT_DIRTY  << "\n";

    // Column header.
    outFile << "Time";
    for (int i = 1; i <= MODES_TO_PLOT; ++i) outFile << ",Mode" << i;
    outFile << ",TotalEnergy";
    if (entropy_mode) outFile << ",Eta";
    if (toda_mode)    outFile << ",TodaJ";
    if (toda_debug)   outFile << ",TodaJ_quad";
    if (lyap_mode)    outFile << ",LyapunovFTLE,LyapunovLocal,LyapRenormCount";
    if (shape_mode)
        for (int j = 1; j < N; ++j) outFile << ",x" << j;
    outFile << "\n";
    outFile << std::scientific << std::setprecision(15);

    std::vector<double> mode_E(MODES_TO_PLOT, 0.0);
    const double last_saved_time = static_cast<double>(nseg - 1) * static_cast<double>(stride) * Dt;
    const double nominal_time    = static_cast<double>(nseg)     * static_cast<double>(stride) * Dt;
    auto t_wall = std::chrono::high_resolution_clock::now();

    // Helper: snapshot the current state into a checkpoint at segment boundary `seg`.
    auto do_checkpoint = [&](long long seg) {
        fput::CheckpointState cp;
        std::snprintf(cp.origin_git_commit, sizeof(cp.origin_git_commit), "%s", origin_commit.c_str());
        cp.origin_git_dirty = origin_dirty;
        cp.N = N; cp.model_flag = model_flag; cp.value = value; cp.amplitude = amplitude;
        cp.dt = Dt; cp.stride = stride;
        cp.entropy = entropy_mode ? 1 : 0; cp.toda = toda_mode ? 1 : 0;
        cp.toda_debug = toda_debug ? 1 : 0; cp.lyapunov = lyap_mode ? 1 : 0;
        cp.lyap_renorm_steps = lyap_renorm_steps; cp.lyap_seed = lyap_seed;
        cp.next_seg = seg;
        cp.completed_steps = seg * stride;
        cp.current_time = static_cast<double>(seg) * static_cast<double>(stride) * Dt;
        cp.x = x; cp.v = v;
        if (lyap_mode) { cp.dx = dx; cp.dv = dv; }
        cp.log_accum = log_accum; cp.renorm_count = renorm_count;
        cp.log_at_last_snap = log_at_last_snap; cp.t_at_last_snap = t_at_last_snap;
        cp.min_bond_strain = min_bond_strain;
        if (!fput::write_checkpoint(checkpoint_file, cp))
            std::cerr << "Warning: failed to write checkpoint at seg=" << seg << "\n";
    };

    std::cout << "FPUT Yoshida-4  |  N=" << N
              << "  |  " << model_str << "=" << value
              << "  |  A=" << amplitude
              << "  |  dt=" << Dt
              << "  |  seg " << seg_start << ".." << (nseg - 1)
              << "  |  last_saved_t=" << last_saved_time
              << "  |  nominal_t=" << nominal_time
              << "  |  toda=" << toda_mode << " lyap=" << lyap_mode
              << (resuming ? "  |  RESUME" : "")
              << (checkpoint_file.empty() ? "" : "  |  CKPT") << "\n";

    const long long prog_every = std::max(1LL, nseg / 10);
    for (long long seg = seg_start; seg < nseg; ++seg) {

        // Checkpoint at a clean segment boundary (state here == time seg*stride*dt,
        // and the CSV already holds every earlier snapshot). Not on the very
        // first iteration of this invocation (that state is already the resume
        // point / start).
        if (!checkpoint_file.empty() && seg > seg_start && seg % checkpoint_every == 0) {
            outFile.flush();
            do_checkpoint(seg);
        }

        // ---- Snapshot before advancing ----
        const double t_current = static_cast<double>(seg) * static_cast<double>(stride) * Dt;
        fput::get_mode_energies(x, v, N, mode_E, MODES_TO_PLOT);
        const double H = fput::get_total_energy(x, v, N, value, model_flag);

        outFile << t_current;
        for (int k = 0; k < MODES_TO_PLOT; ++k) outFile << "," << mode_E[k];
        outFile << "," << H;
        if (entropy_mode) outFile << "," << fput::full_spectrum_entropy(x, v, N);

        if (toda_mode) {
            const double J = fput::compute_toda_J(x, v, N, dq_buf, b_buf);
            if (!std::isfinite(J)) {
                std::cerr << "Error: non-finite TodaJ at seg=" << seg << "; aborting.\n";
                return 2;
            }
            outFile << "," << J;
            if (toda_debug) {
                const double Jq = fput::compute_toda_J_quad(x, v, N, H, dq_buf);
                outFile << "," << Jq;
            }
        }

        if (lyap_mode) {
            const double cur = fput::tangent_norm(dx, dv);
            const double total = log_accum + std::log(cur);  // cumulative log-stretch to now
            const double ftle  = (t_current > 0.0) ? total / t_current : 0.0;
            const double dt_snap = t_current - t_at_last_snap;
            const double local = (dt_snap > 0.0) ? (total - log_at_last_snap) / dt_snap : 0.0;
            if (!std::isfinite(ftle) || !std::isfinite(local)) {
                std::cerr << "Error: non-finite Lyapunov value at seg=" << seg << "; aborting.\n";
                return 2;
            }
            outFile << "," << ftle << "," << local << "," << renorm_count;
            log_at_last_snap = total;
            t_at_last_snap   = t_current;
        }

        if (shape_mode)
            for (int j = 0; j < M; ++j) outFile << "," << x[j];
        outFile << "\n";
        if (seg % 50 == 0) outFile.flush();

        // ---- Advance stride Yoshida steps ----
        if (lyap_mode) {
            for (long long step = 0; step < stride; ++step) {
                fput::yoshida_step_tangent(x, v, dx, dv, F, DFdx, N, Dt, value, model_flag,
                                           &min_bond_strain);
                if ((step + 1) % lyap_renorm_steps == 0) {
                    const double nrm = fput::tangent_norm(dx, dv);
                    if (!(nrm > 0.0) || !std::isfinite(nrm)) {
                        std::cerr << "Error: tangent norm non-finite/zero during renorm; aborting.\n";
                        return 2;
                    }
                    log_accum += std::log(nrm);
                    const double inv = 1.0 / nrm;
                    for (int j = 0; j < M; ++j) { dx[j] *= inv; dv[j] *= inv; }
                    ++renorm_count;
                }
            }
        } else {
            for (long long step = 0; step < stride; ++step)
                fput::yoshida_step(x, v, F, N, Dt, value, model_flag, &min_bond_strain);
        }

        if (seg % prog_every == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t_wall).count();
            std::cout << "  " << std::fixed << std::setprecision(1)
                      << (100.0 * seg / nseg) << "%"
                      << "  |  t=" << t_current
                      << "  |  " << elapsed << "s elapsed\n" << std::flush;
        }
    }

    // Final checkpoint (state after the last advance, ready to extend from seg=nseg).
    if (!checkpoint_file.empty()) {
        outFile.flush();
        do_checkpoint(nseg);
    }

    // Final 100% progress line (reports the LAST SAVED time, not nominal).
    {
        auto now = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t_wall).count();
        std::cout << "  100.0%  |  last_saved_t=" << last_saved_time
                  << "  |  " << elapsed << "s elapsed\n" << std::flush;
    }

    outFile.close();
    const double total_wall = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall).count();
    const double reported_min_bond =
        std::isfinite(min_bond_strain) ? min_bond_strain : 0.0;
    std::cout << "Done!  " << total_wall << "s  |  min_bond_strain=" << reported_min_bond
              << "  |  saved to " << filename << "\n";
    return 0;
}
