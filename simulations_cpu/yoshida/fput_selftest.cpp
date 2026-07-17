// fput_selftest.cpp
// Thin command-line driver that exposes the EXACT production physics kernels
// (from fput_physics.hpp) to the pytest suite, evaluated on arbitrary states.
//
// This is a test-only tool. It lets the tests feed random / hand-crafted states
// to the same functions the solver uses, instead of re-implementing the indexing
// (which would risk duplicating a boundary mistake).
//
// Compile: g++ -O3 -o fput_selftest fput_selftest.cpp   (or: make fput_selftest)
//
// Protocol: arguments select the command; all floating-point state is read from
// stdin as whitespace-separated doubles. Output is whitespace-separated doubles
// (or a single scalar) to stdout, at full precision.
//
//   fput_selftest forces        <N> <model> <value>            < [x(M)]
//   fput_selftest tangent_force <N> <model> <value>            < [x(M) dx(M)]
//   fput_selftest toda_J        <N>                            < [x(M) v(M)]
//   fput_selftest toda_Jquad    <N> <model> <value>            < [x(M) v(M)]
//   fput_selftest energy        <N> <model> <value>            < [x(M) v(M)]
//   fput_selftest tangent_step  <N> <model> <value> <dt>       < [x(M) v(M) dx(M) dv(M)]
//   fput_selftest step          <N> <model> <value> <dt>       < [x(M) v(M)]
//
// model is "alpha" or "beta".

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "fput_physics.hpp"

static int model_flag_from(const std::string& s) {
    if (s == "alpha") return fput::MODEL_ALPHA;
    if (s == "beta")  return fput::MODEL_BETA;
    std::cerr << "Error: model must be 'alpha' or 'beta'.\n";
    std::exit(1);
}

static std::vector<double> read_doubles(int count) {
    std::vector<double> out(count);
    for (int i = 0; i < count; ++i) {
        if (!(std::cin >> out[i])) {
            std::cerr << "Error: expected " << count << " doubles on stdin, got " << i << ".\n";
            std::exit(1);
        }
    }
    return out;
}

static void print_vec(const std::vector<double>& a) {
    std::cout << std::scientific << std::setprecision(17);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (i) std::cout << ' ';
        std::cout << a[i];
    }
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <command> <N> [model] [value] [dt]\n";
        return 1;
    }
    const std::string cmd(argv[1]);
    const int N = std::stoi(argv[2]);
    const int M = N - 1;
    if (M < 1) { std::cerr << "Error: N must be >= 2.\n"; return 1; }
    std::cout << std::scientific << std::setprecision(17);

    if (cmd == "toda_J") {
        auto x = read_doubles(M);
        auto v = read_doubles(M);
        std::vector<double> dq(M + 1, 0.0), b(M + 1, 0.0);
        std::cout << fput::compute_toda_J(x, v, N, dq, b) << "\n";
        return 0;
    }

    // Remaining commands need model + value.
    if (argc < 5) { std::cerr << "Error: " << cmd << " requires <model> <value>.\n"; return 1; }
    const int    model = model_flag_from(argv[3]);
    const double value = std::stod(argv[4]);

    if (cmd == "forces") {
        auto x = read_doubles(M);
        std::vector<double> F(M, 0.0);
        fput::compute_forces(x, F, N, value, model);
        print_vec(F);
    } else if (cmd == "tangent_force") {
        auto x  = read_doubles(M);
        auto dx = read_doubles(M);
        std::vector<double> DFdx(M, 0.0);
        fput::compute_tangent_force(x, dx, DFdx, N, value, model);
        print_vec(DFdx);
    } else if (cmd == "toda_Jquad") {
        auto x = read_doubles(M);
        auto v = read_doubles(M);
        std::vector<double> dq(M + 1, 0.0);
        const double H = fput::get_total_energy(x, v, N, value, model);
        std::cout << fput::compute_toda_J_quad(x, v, N, H, dq) << "\n";
    } else if (cmd == "energy") {
        auto x = read_doubles(M);
        auto v = read_doubles(M);
        std::cout << fput::get_total_energy(x, v, N, value, model) << "\n";
    } else if (cmd == "tangent_step") {
        if (argc < 6) { std::cerr << "Error: tangent_step requires <dt>.\n"; return 1; }
        const double dt = std::stod(argv[5]);
        auto x  = read_doubles(M);
        auto v  = read_doubles(M);
        auto dx = read_doubles(M);
        auto dv = read_doubles(M);
        std::vector<double> F(M, 0.0), DFdx(M, 0.0);
        fput::yoshida_step_tangent(x, v, dx, dv, F, DFdx, N, dt, value, model);
        // Output x' v' dx' dv' on one line.
        std::vector<double> out;
        out.insert(out.end(), x.begin(), x.end());
        out.insert(out.end(), v.begin(), v.end());
        out.insert(out.end(), dx.begin(), dx.end());
        out.insert(out.end(), dv.begin(), dv.end());
        print_vec(out);
    } else if (cmd == "step") {
        if (argc < 6) { std::cerr << "Error: step requires <dt>.\n"; return 1; }
        const double dt = std::stod(argv[5]);
        auto x = read_doubles(M);
        auto v = read_doubles(M);
        std::vector<double> F(M, 0.0);
        fput::yoshida_step(x, v, F, N, dt, value, model);
        std::vector<double> out;
        out.insert(out.end(), x.begin(), x.end());
        out.insert(out.end(), v.begin(), v.end());
        print_vec(out);
    } else {
        std::cerr << "Error: unknown command '" << cmd << "'.\n";
        return 1;
    }
    return 0;
}
