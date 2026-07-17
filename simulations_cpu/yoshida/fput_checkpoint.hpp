// fput_checkpoint.hpp
// Binary checkpoint/restart for the Yoshida FPUT solver.
//
// Purpose: let a run be extended later (e.g. 1e7 -> 1e8/1.4e8) WITHOUT re-running
// from scratch, and let long jobs survive interruption.
//
// Format: a small binary file written field-by-field (no struct padding
// assumptions), holding all doubles at full precision (fwrite of the raw IEEE-754
// bytes). Checkpoints are machine-local artifacts, so native byte order is used.
//
// Atomicity: the writer writes to "<path>.tmp", flushes and closes it, then
// std::rename()s it into place -- rename is atomic on POSIX, so a reader never
// sees a partially-written checkpoint.
//
// The checkpoint captures a state at a SEGMENT BOUNDARY (the top of the main
// loop, just before snapshot `next_seg` is written), so resume continues the
// loop from `next_seg` with no duplicated or missing snapshot.

#ifndef FPUT_CHECKPOINT_HPP
#define FPUT_CHECKPOINT_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace fput {

// Bump CKPT_VERSION on any incompatible format change.
static const char CKPT_MAGIC[8] = {'F', 'P', 'U', 'T', 'C', 'K', 'P', 'T'};
constexpr int32_t CKPT_VERSION = 1;

struct CheckpointState {
    // provenance
    char origin_git_commit[41] = {0};  // solver hash that first created the run
    int32_t origin_git_dirty = 0;

    // dynamically-relevant parameters (must match on resume)
    int32_t N = 0;
    int32_t model_flag = 0;
    double value = 0.0;
    double amplitude = 0.0;
    double dt = 0.0;
    int64_t stride = 0;
    int32_t entropy = 0;
    int32_t toda = 0;
    int32_t toda_debug = 0;
    int32_t lyapunov = 0;
    int64_t lyap_renorm_steps = 0;
    int64_t lyap_seed = 0;

    // progress (state is at the top of the loop for segment next_seg)
    int64_t next_seg = 0;
    int64_t completed_steps = 0;
    double current_time = 0.0;

    // physical state (length M = N-1)
    std::vector<double> x;
    std::vector<double> v;

    // tangent state + Lyapunov accumulators (only meaningful if lyapunov)
    std::vector<double> dx;
    std::vector<double> dv;
    double log_accum = 0.0;
    int64_t renorm_count = 0;
    double log_at_last_snap = 0.0;
    double t_at_last_snap = 0.0;

    // run-summary telemetry carried across resume
    double min_bond_strain = 0.0;
};

namespace detail {
template <typename T>
inline bool wr(std::FILE* f, const T& v) {
    return std::fwrite(&v, sizeof(T), 1, f) == 1;
}
template <typename T>
inline bool rd(std::FILE* f, T& v) {
    return std::fread(&v, sizeof(T), 1, f) == 1;
}
inline bool wr_vec(std::FILE* f, const std::vector<double>& a) {
    const int64_t n = static_cast<int64_t>(a.size());
    if (!wr(f, n)) return false;
    if (n == 0) return true;
    return std::fwrite(a.data(), sizeof(double), a.size(), f) == a.size();
}
inline bool rd_vec(std::FILE* f, std::vector<double>& a) {
    int64_t n = 0;
    if (!rd(f, n)) return false;
    a.resize(static_cast<size_t>(n));
    if (n == 0) return true;
    return std::fread(a.data(), sizeof(double), a.size(), f) == a.size();
}
}  // namespace detail

// Atomic write: <path>.tmp then rename. Returns false on any I/O error.
inline bool write_checkpoint(const std::string& path, const CheckpointState& s) {
    using namespace detail;
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;

    bool ok = true;
    ok = ok && (std::fwrite(CKPT_MAGIC, 1, 8, f) == 8);
    ok = ok && wr(f, CKPT_VERSION);
    ok = ok && (std::fwrite(s.origin_git_commit, 1, 41, f) == 41);
    ok = ok && wr(f, s.origin_git_dirty);
    ok = ok && wr(f, s.N) && wr(f, s.model_flag) && wr(f, s.value)
            && wr(f, s.amplitude) && wr(f, s.dt) && wr(f, s.stride);
    ok = ok && wr(f, s.entropy) && wr(f, s.toda) && wr(f, s.toda_debug)
            && wr(f, s.lyapunov) && wr(f, s.lyap_renorm_steps) && wr(f, s.lyap_seed);
    ok = ok && wr(f, s.next_seg) && wr(f, s.completed_steps) && wr(f, s.current_time);
    ok = ok && wr_vec(f, s.x) && wr_vec(f, s.v);
    ok = ok && wr_vec(f, s.dx) && wr_vec(f, s.dv);
    ok = ok && wr(f, s.log_accum) && wr(f, s.renorm_count)
            && wr(f, s.log_at_last_snap) && wr(f, s.t_at_last_snap);
    ok = ok && wr(f, s.min_bond_strain);

    // Ensure bytes hit the OS before the rename.
    ok = ok && (std::fflush(f) == 0);
    if (std::fclose(f) != 0) ok = false;
    if (!ok) { std::remove(tmp.c_str()); return false; }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); return false; }
    return true;
}

// Read a checkpoint. Returns false on I/O error, bad magic, or version mismatch
// (err is set to a human-readable reason).
inline bool read_checkpoint(const std::string& path, CheckpointState& s, std::string& err) {
    using namespace detail;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open checkpoint file: " + path; return false; }

    char magic[8];
    bool ok = (std::fread(magic, 1, 8, f) == 8);
    if (ok && std::memcmp(magic, CKPT_MAGIC, 8) != 0) { err = "bad checkpoint magic"; std::fclose(f); return false; }
    int32_t version = 0;
    ok = ok && rd(f, version);
    if (ok && version != CKPT_VERSION) {
        err = "checkpoint format version " + std::to_string(version)
            + " != expected " + std::to_string(CKPT_VERSION);
        std::fclose(f);
        return false;
    }
    ok = ok && (std::fread(s.origin_git_commit, 1, 41, f) == 41);
    ok = ok && rd(f, s.origin_git_dirty);
    ok = ok && rd(f, s.N) && rd(f, s.model_flag) && rd(f, s.value)
            && rd(f, s.amplitude) && rd(f, s.dt) && rd(f, s.stride);
    ok = ok && rd(f, s.entropy) && rd(f, s.toda) && rd(f, s.toda_debug)
            && rd(f, s.lyapunov) && rd(f, s.lyap_renorm_steps) && rd(f, s.lyap_seed);
    ok = ok && rd(f, s.next_seg) && rd(f, s.completed_steps) && rd(f, s.current_time);
    ok = ok && rd_vec(f, s.x) && rd_vec(f, s.v);
    ok = ok && rd_vec(f, s.dx) && rd_vec(f, s.dv);
    ok = ok && rd(f, s.log_accum) && rd(f, s.renorm_count)
            && rd(f, s.log_at_last_snap) && rd(f, s.t_at_last_snap);
    ok = ok && rd(f, s.min_bond_strain);
    std::fclose(f);
    if (!ok) { err = "truncated or unreadable checkpoint"; return false; }
    return true;
}

}  // namespace fput

#endif  // FPUT_CHECKPOINT_HPP
