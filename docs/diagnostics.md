# Dynamical diagnostics: Toda integral J and maximum Lyapunov exponent

This document describes the opt-in diagnostics added to the canonical Yoshida-4
solver (`simulations_cpu/yoshida/FPUT_yoshida_solver.cpp`): the **intensive
fourth-order Toda integral J** and the **finite-time maximum Lyapunov exponent**.

> **Scope & conventions.** This campaign fixes **alpha = 1** (see below). The
> completed FPUT-beta statistical analysis is frozen and untouched. Diagnostics
> are opt-in: with none enabled, trajectories and the CSV schema are unchanged.

---

## 1. The alpha = 1 convention and the exact rescaling

The reference paper (Christodoulidi & Flach, *Chaos* **35**, 113127, 2025) fixes
`alpha = 1`, so its Toda integral J uses `exp(2·dq)` with no coefficient. We
adopt the same convention for this diagnostics campaign, for **both** models:

- **J** is the paper's exact bare functional (`exp(2·dq)`, no coefficient in the
  exponential) for alpha and for the beta control. The exponential carries
  neither alpha nor beta.

Fixing `alpha = 1` loses no generality for the pure FPUT-alpha model. Under the
exact rescaling `y = alpha·q`, `u = alpha·p`, the alpha-model dynamics map to the
alpha = 1 dynamics with

    A_(alpha=1)       = alpha   · A_alpha
    epsilon_(alpha=1) = alpha^2 · epsilon_alpha

The normalized spectral entropy **eta is exactly invariant** under this rescaling
(mode energies scale by `alpha^2`, so the fractions `p_k` and hence `eta` are
unchanged). This makes the existing **alpha = 0.25** entropy results a lossless
seed for the alpha = 1 pilot range via

    A_1 = 0.25·A_0.25,   epsilon_1 = epsilon_0.25 / 16.

The rescaled data locate the recurrent / crossover / spreading regimes in
epsilon; they are **not** new alpha = 1 diagnostic runs (they contain no J or
Lyapunov data). The completed alpha = 0.25 threshold analysis is a separate,
finished study and is untouched.

Running alpha ≠ 1 would require deriving the `exp(2·alpha·dq)` form and its
prefactors plus a unit test that it reduces to Eq. (5) at alpha = 1. Not done here.

---

## 2. Solver indexing and boundary convention

The solver uses `N` = particles + 1, with `M = N - 1` moving interior particles,
fixed ends `q_0 = q_N = 0`, `p_0 = p_N = 0`. The paper uses `N_paper` moving
particles with `q_0 = q_{N_paper+1} = 0`. The mapping is

    N_paper = M = N_solver - 1.

Consequences, verified against the solver's own N-bond energy loop:

- there are `M + 1 = N_solver` physical bonds `dq_n = q_{n+1} - q_n`, `n = 0..M`;
- the paper's intensive prefactor `1/(2·N_paper)` becomes **`1/(2·M) = 1/(2·(N_solver-1))`** — never `1/(2·N_solver)`;
- the per-bond constant `-3/8` is counted `M + 1 = N_solver` times;
- boundary momenta `p_0 = p_{M+1} = 0`.

C++ array mapping: `x[j] = q_{j+1}`, `v[j] = p_{j+1}`, `j = 0..M-1`.

---

## 3. Toda integral J (exact, alpha = 1 form)

With `b_n = exp(2·dq_n)` and the fixed-boundary ghost continuation
`b_{-1} = b_0`, `b_{M+1} = b_M`:

    J = 1/(2·M) · sum_{n=0}^{M} [ p_n^4
                                  + b_n·(p_n^2 + p_n·p_{n+1} + p_{n+1}^2)
                                  + (b_n/8)·(b_{n-1} + b_n + b_{n+1})
                                  - 3/8 ]

This is the **raw intensive J** (not `J/J(0)`), computed in `O(N)` per snapshot.
Normalization and equilibrium comparisons belong in analysis code.

**Ground state check.** At `dq = 0, p = 0` every bond term is `(1/8)(1+1+1) - 3/8 = 0`,
so `J = 0` exactly (the `-3/8` subtracts the trivial Toda ground-state density).

**Quadratic (low-energy) approximation** (validation only, `--toda-debug`):

    J_quad = 2·epsilon + 1/(2·M)·sum_{n=0}^{M} [ p_n·p_{n+1} + dq_n·dq_{n+1} ],
    epsilon = H_FPUT / M,  dq_{M+1} = dq_M.

Expanding the exact J to quadratic order (the linear-in-`dq` term telescopes to
zero with fixed boundaries) reproduces this form; both are implemented and
cross-checked in the tests.

### Interpretation (read this before using J)

- **FPUT-alpha:** J is a near-conserved **adiabatic invariant** measuring drift
  away from the nearby integrable Toda dynamics and the onset of action
  diffusion. Near equilibrium the paper finds `J_eq ≈ 2·epsilon` (alpha only).
- **FPUT-beta:** J is recorded **only as a control observable**. Beta is not near
  the alpha-Toda point. Do **not** call J an adiabatic invariant for beta, do
  **not** put beta into the exponential, and do **not** draw `J_eq ≈ 2·epsilon`
  on beta plots (that value is an alpha/Toda result). Whether J and spectral
  entropy agree for beta is an open question, not an assumption.

Spectral entropy (harmonic-mode spreading), the Lyapunov exponent (chaos), and
Toda-J drift (integrability breaking) are **distinct** measures; do not conflate
them, and do not infer thermalization from the final value of J or λ alone.

---

## 4. Maximum Lyapunov exponent (Benettin)

A single tangent vector `(δx, δv)` is evolved through the **same** discrete
Yoshida map as the trajectory. For each drift substep `x += c·dt·v`:
`δx += c·dt·δv`; for each kick `v += d·dt·F(x)`: `δv += d·dt·(DF(x)·δx)`, with

    (DF·δx)_j = K_f·(δx_{j+1} - δx_j) - K_b·(δx_j - δx_{j-1}),
    δx_{-1} = δx_M = 0,
    K(r) = 1 + 2·alpha·r   (alpha) ,   K(r) = 1 + 3·beta·r^2   (beta).

Benettin renormalization: initialize a deterministic seeded unit tangent; every
`--lyap-renorm-steps` steps accumulate `ln‖w‖` and renormalize. Reported columns:

- **`LyapunovFTLE`** — cumulative finite-time exponent `= (Σ ln‖w‖ + ln‖w_now‖)/t`.
  This is renormalization-invariant (adding a renorm just moves a term between
  the sum and the trailing partial-block term).
- **`LyapunovLocal`** — **the log-stretch accumulated since the previous saved
  snapshot divided by the elapsed time since that snapshot** (the time-averaged
  local exponent over one snapshot interval). This single definition is used
  throughout — not mixed with any other.
- **`LyapRenormCount`** — cumulative number of renormalizations (validation aid).

Defaults: `--lyap-renorm-steps 100`, `--lyap-seed 12345`. The seed affects only
the tangent transient, never the physical trajectory. Work added is `O(N)` per
step (≈ +130% runtime; measured), using preallocated buffers. Non-finite values
abort the run with an error (never silently written).

> **Note:** the FPUT-alpha cubic potential is unbounded below; strongly-excited
> alpha runs can escape the well, which the finite-value guard catches. Use
> moderate alpha amplitudes, or the bounded beta model, for long tangent runs.

### Escape telemetry: most-negative bond strain

For `alpha = 1` the cubic bond potential `V(r) = r^2/2 + r^3/3` has a barrier in
the compression direction at `r = -1` with height `V(-1) = 1/6`; a bond strain
approaching `r = -1` signals impending escape. The solver tracks the **most
negative bond strain reached over the whole run**, folded into the bond
differences already computed inside `compute_forces` (no extra per-step O(N)
scan, and it never alters the trajectory). It is printed in the final line as
`min_bond_strain=...` and stored in checkpoints. Report this — not just the
initial or `max |r|` value — when assessing escape risk.

---

## 4a. Checkpoint / restart

Long runs can be checkpointed and later resumed or **extended** (e.g. 1e7 →
1e8/1.4e8) without recomputing from scratch.

- **Write:** `--checkpoint-file <path>` writes a binary checkpoint every
  `--checkpoint-every` segments (default 50) and once at the end. Writes are
  **atomic**: the file is written to `<path>.tmp`, flushed, closed, then
  `rename()`d into place, so a reader never sees a partial checkpoint. Checkpoints
  occur only at a **segment boundary** (top of the main loop), where the CSV
  already holds every earlier snapshot.
- **Contents:** format magic + version; originating solver Git commit + dirty
  flag; `N`, model, coefficient, amplitude, `dt`, `stride`; all enabled
  diagnostics and Lyapunov settings; the next segment index, completed step
  count, and physical time; positions and velocities; (when Lyapunov is on) the
  tangent vectors, accumulated log-stretch, renormalization count, and the
  per-snapshot local accumulators; and the running most-negative bond strain. All
  doubles are stored in raw binary (exact), never as truncated text.
- **Resume:** `--resume <path>` restores the state and continues. It **rejects**
  resume if any dynamically-relevant parameter (`N`, model, coefficient,
  amplitude, `dt`, `stride`, the diagnostics set, `--lyap-renorm-steps`,
  `--lyap-seed`) differs from the checkpoint, and requires `--nseg` to exceed the
  checkpoint segment. `--nseg` may be **larger** than the original run — that is
  how a run is extended.
- **Output model:** resume writes a **new continuation CSV** (the `SavePath`
  given on the resume command) containing snapshots from the checkpoint segment
  to `nseg-1`. It does **not** append to or mutate the original CSV. The full
  trajectory is the concatenation of the original CSV's rows `[0 ..
  ResumeFromSegment-1]` and the continuation's rows. The continuation header
  records `NumSegments` = the total target and `ResumeFromSegment` = the split
  point, so `NumSegments` is never silently wrong.
- **Equivalence guarantee:** because the checkpoint preserves every double
  exactly, an uninterrupted run and a checkpoint/resume split produce
  **bitwise-identical** data rows (verified by `tests/test_checkpoint.py`,
  including `TodaJ` and `LyapunovFTLE`).

Example — run to 1e7, then extend to 1e8:

```bash
./fput_yoshida 512 alpha 1.0 17.8356 run.csv \
    --dt 0.1 --stride 200000 --nseg 500 --entropy --toda --lyapunov \
    --checkpoint-file run.ckpt                 # nominal 1e7, checkpoint at end

./fput_yoshida 512 alpha 1.0 17.8356 run_cont.csv \
    --dt 0.1 --stride 200000 --nseg 5000 --entropy --toda --lyapunov \
    --resume run.ckpt                          # extend to nominal 1e8
# full trajectory = run.csv rows [0..499] ++ run_cont.csv rows [500..4999]
```

---

## 5. Output timing convention

A snapshot is written **before** each advance, so distinguish:

    last saved time         = (NumSegments - 1) · Stride · dt
    nominal integrated time =  NumSegments      · Stride · dt

These are not equal; both are printed by the solver and must not both be called
`t_max`.

---

## 6. CLI, metadata, and CSV columns

**New flags** (all optional; defaults reproduce the original behavior):

| Flag | Meaning |
|---|---|
| `--toda` | append `TodaJ` (exact intensive J) |
| `--toda-debug` | additionally append `TodaJ_quad` (quadratic; diagnostic only) |
| `--lyapunov` | append `LyapunovFTLE,LyapunovLocal,LyapRenormCount` |
| `--lyap-renorm-steps <int>` | renormalization interval (default 100) |
| `--lyap-seed <int>` | tangent RNG seed (default 12345) |
| `--checkpoint-file <path>` | write a binary checkpoint at segment boundaries |
| `--checkpoint-every <int>` | segments between checkpoints (default 50) |
| `--resume <path>` | resume from a checkpoint (SavePath = continuation CSV) |

**New metadata lines** (parsed transparently by `plot_utils.get_metadata`):
`# TodaIntegral:`, `# Lyapunov:`, `# LyapRenormSteps:`, `# LyapSeed:`,
`# SolverGitCommit:` (Git short hash; `unknown` if not built via the `Makefile`),
and `# SolverGitDirty:` (`0` = built from a clean tracked worktree, `1` = dirty;
production data must be produced from `SolverGitDirty: 0`). Continuation files
(from `--resume`) additionally carry `# ResumeFromSegment:`,
`# CheckpointOriginCommit:`, and `# CheckpointOriginDirty:`.

**Column order:** `Time, Mode1..Mode20, TotalEnergy, [Eta], [TodaJ],
[TodaJ_quad], [LyapunovFTLE, LyapunovLocal, LyapRenormCount], [x1..]`. Readers
that select columns by name (all repo scripts) are unaffected.

### CLI examples

```bash
# Build (stamps the Git hash into SolverVersion):
cd simulations_cpu/yoshida && make

# Plain solver (unchanged behavior, no diagnostics):
./fput_yoshida 1024 alpha 0.25 20 out.csv --entropy

# alpha=1 with Toda J and Lyapunov:
./fput_yoshida 1024 alpha 1.0 3.0 out.csv --dt 0.1 --stride 200000 --nseg 500 \
    --entropy --toda --lyapunov --lyap-renorm-steps 100 --lyap-seed 12345

# beta control observable (J recorded, NOT an adiabatic invariant here):
./fput_yoshida 4096 beta 1.0 50 out.csv --entropy --toda
```

---

## 7. Tests, build, and validation commands

```bash
# Build both binaries (production + self-test driver):
cd simulations_cpu/yoshida && make

# Run the diagnostics test suite (compiles the binaries into a temp dir):
python -m pytest simulations_cpu/yoshida/tests/ -q

# Short numerical validation (energy drift, J, Lyapunov, overhead, rescaling):
python simulations_cpu/yoshida/run_validation.py --binary ./fput_yoshida \
    --existing-alpha025 ../../data/yoshida_threshold_v2/alpha_N512_A1.0000.csv
```

The test suite covers: diagnostics-off regression vs the previous solver;
schema/`get_metadata` compatibility; C++ J vs an independent Python reference;
exact→quadratic J at low amplitude; ground-state and boundary edge cases;
**J conservation along genuine Toda dynamics**; analytic tangent force vs a
finite-difference Jacobian-vector product; full tangent step vs finite-difference
of the physical step; fixed-boundary tangent; renormalization invariance;
harmonic FTLE decay; nonlinear positive FTLE; seed reproducibility; no NaN/inf;
and **checkpoint/restart bitwise-equivalence** (split-run, extension, periodic
checkpoint) plus resume parameter-mismatch and corrupt-checkpoint rejection.

Independent reference implementations live in
`simulations_cpu/yoshida/toda_reference.py` (exact J, quadratic J, tangent force,
full tangent step, and a minimal Toda-potential integrator for the conservation
test).

---

## 8. Single-trajectory diagnostic plots

```bash
python -m analysis.plot_diagnostics --input <run.csv> --output-dir <dir>
```

Produces (PNG + PDF): `TodaJ(t)`, `TodaJ/J(0)(t)`, `TodaJ - J(0)` vs log time,
`LyapunovFTLE(t)`, `LyapunovLocal(t)`, and an aligned separate-panel figure
(entropy / J / FTLE) sharing only the time axis. The alpha-only `J_eq ≈ 2·epsilon`
line is drawn **only** for alpha runs, never for beta.

---

## 9. Production-run manifest (generate, do not execute)

```bash
python -m analysis.generate_run_manifest \
    --data-dir data/yoshida_threshold_v2 --output-dir analysis/manifests
```

Writes `analysis/manifests/run_manifest.json` (tracked; includes the selection
rationale) and `run_manifest.csv` (gitignored via `*.csv`; flat table). It is
**not** executed. Two groups:

- **alpha pilot (alpha = 1):** matched energy densities across N = 512/1024/2048,
  seeded by the rescaled alpha = 0.25 regimes; amplitudes computed to hit each
  target epsilon exactly. N = 4096 is excluded from the first pilot.
- **beta controls (N = 4096):** low-amplitude points spanning epsilon ≈ 1e-4 up
  to the current lowest sampled N = 4096 beta epsilon, to fill the low-entropy
  gap. Exact epsilon from `H0 = N(z + 1.5·beta·z^2)`, `z = A^2·sin^2(pi/2N)`.
  Labeled as controls, not a new beta study; realized eta is not assumed.

---

## 10. Files

| File | Role |
|---|---|
| `simulations_cpu/yoshida/fput_physics.hpp` | shared physics kernels (state + tangent + J) |
| `simulations_cpu/yoshida/fput_checkpoint.hpp` | atomic binary checkpoint read/write |
| `simulations_cpu/yoshida/FPUT_yoshida_solver.cpp` | production solver (main) |
| `simulations_cpu/yoshida/fput_selftest.cpp` | test driver exposing the kernels |
| `simulations_cpu/yoshida/Makefile` | builds both binaries; stamps Git hash |
| `simulations_cpu/yoshida/toda_reference.py` | independent Python reference (tests) |
| `simulations_cpu/yoshida/run_validation.py` | short numerical validation harness |
| `simulations_cpu/yoshida/tests/` | pytest suite |
| `analysis/plot_diagnostics.py` | single-trajectory diagnostic plots |
| `analysis/generate_run_manifest.py` | proposed run manifest generator |
