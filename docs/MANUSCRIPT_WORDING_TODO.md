# Manuscript Wording & Claims Discipline

Standing reference for how results are phrased in figures, captions, and the manuscript.
Split into (A) open TODO items to apply now, and (B) a permanent claims-discipline checklist
established across the pilot/analysis work.

> **Update 2026-07-18:** eps=1e-4 (all N) and eps=8e-4 (N=2048) extended to 1e8; eps=1e-4
> N=2048 further to 1.4e8; a dt=0.05 convergence check (eps=1e-4, N=512, to 1e8) was run.
> Findings: eps=1e-4 shows a robust slow Toda drift (reproduced at dt=0.05) with only
> marginal/intermittent finite-time chaos; eps=8e-4 reaches near-equilibration by 1e8 at both
> N=512 and N=2048 with comparable (not identical) timescales.

---

## A. Open TODO — apply to alpha figures/captions and manuscript text

- [ ] **Spectral entropy at low energy density.**
  Do **not** write "eta alone would suggest partial thermalization."
  Write instead: **"eta alone indicates substantial harmonic-mode spreading, but does not
  establish chaos or action equilibration."**
  Rationale: eta ~ 0.59-0.75 means the harmonic modal energy has spread; it is *not* evidence of
  half-thermalization.

- [ ] **Low-eps cumulative FTLE (REVISED after the 1e8/1.4e8 extension).**
  The earlier pilot statement ("consistent with decay toward zero over the observed interval")
  was true **only at 1e7** and must NOT be used as a general claim.
  At eps=1e-4 the cumulative FTLE follows the 1/t guide down to a minimum ~1-1.3e-6 near t~1e7,
  then **rises to a small positive floor ~2e-6 and levels off through 1e8-1.4e8** — it does not
  continue decaying to zero.
  Write instead: **"at 1e7 the FTLE is consistent with decay toward zero; extended to 1e8-1.4e8
  it levels at a small positive floor (~2e-6) rather than continuing to decay."**
  Any 1/t reference line remains **a visual guide, not a fit.**

- [ ] **Low-eps regime label (REVISED; use through 1.4e8).**
  Do **not** call eps=1e-4 "near-integrable" and do **not** call it "established chaos."
  Preferred phrasing: **"slow Toda-action drift accompanied by a small finite-time Lyapunov
  growth rate, while the local growth remains strongly fluctuating and does not yet establish a
  stable positive Lyapunov plateau."**
  Supporting numbers: block-mean J/(2eps) decreases monotonically 1.498 -> 1.466 through 1.4e8
  (still far from the 1.0 equilibrium estimate); block Lyapunov rates ~1.4-2.2e-6 but classified
  inconclusive (LyapunovLocal std ~3x its mean).

- [ ] **The eps=1e-4 slow drift is confirmed against dt (keep the caveat on FTLE).**
  The ~1.5-2% Toda drift REPRODUCES at dt=0.05 (dPhi_J ~1.35% vs 1.67%; FTLE ~2e-6 both;
  eta_tail 0.672 vs 0.680; energy-drift ratio 12, ~4th-order). So the slow-onset result is a
  real effect, not a dt=0.1 discretization artifact. **Caveat:** the dt=0.05 run used renorm 100
  (physical interval 5) not 200 (interval 10), so its FTLE / LyapunovLocal comparison to dt=0.1
  is order-of-magnitude only; J, Phi_J, eta, and energy comparisons are unaffected.

- [ ] **FTLE tail standard deviation in the summary figure.**
  The tail std is **temporal variation over the tail window, not statistical uncertainty across
  independent realizations.** State this in the caption.
  To claim a genuine positive plateau, require **all three**: (i) cumulative FTLE flattens;
  (ii) LyapunovLocal tail-mean stably positive; (iii) local fluctuations sit around a positive
  value, not around zero. **eps=1e-4 fails leg (iii)** -> do not call it a plateau.
  Prefer block slopes of accumulated stretch S(t)=t*lambda_FTLE(t) over successive intervals
  (e.g. [1e7,3e7],[3e7,6e7],[6e7,1e8],[1e8,1.4e8]) to the noisy LyapunovLocal.

---

## B. Standing claims-discipline checklist (reference)

- [ ] **J = 2eps is an estimate, not a theorem.** Refer to it as the *alpha equilibrium estimate*
  (J ~ 2eps). Not an exact theorem, and **not** a beta-model result.

- [ ] **J(0) ~ 3eps is a mode-1 initial-condition fact.** The fundamental-mode IC gives J(0) ~
  3eps, so J approaches equilibrium **from above**. Confirmed via the quadratic Toda approximation.

- [ ] **Finite-time categories, not thermodynamic claims.** Present all regime labels as
  **finite-time classifications at the stated nominal duration**; state the duration (1e7, 1e8,
  1.4e8) with each claim. Do not phrase as infinite-time / thermodynamic-limit statements.

- [ ] **Two timescale metrics (define both; fixed physical persistence window).**
  T10_persistent = earliest time at which the smoothed Phi_J stays >= 0.1 for the rest of the
  trajectory (Toda-action drift ONSET). T90_persistent = same with threshold 0.9 (approach to
  equilibrium). Use one fixed physical smoothing window (e.g. 2e6) for all N/eps; report
  ">1e8 (not reached)" honestly rather than lowering the threshold or substituting the final
  time. For eps=1e-4, T90 is currently not meaningful but T10 may be measurable.

- [ ] **Do not call fluctuating raw J pointwise monotone.** Judge Toda drift by block means or a
  smoothed trend, not raw J(t) row-by-row.

- [ ] **Finite-size wording is observable-specific.** Do **not** say "larger N is universally
  more thermalized." Describe each observable's N-trend separately. Strongest claim is
  **finite-size consistency** (same classification across N), not a strong monotone finite-size
  trend in every observable. (At eps=8e-4, T90 timescales for N=512 vs N=2048 are comparable —
  ratio ~1.35 — i.e. weak size dependence, not identical.)

- [ ] **No precise crossover thresholds / no power-law fits yet.** With sparse eps: no fitted
  crossover thresholds, no interpolated phase boundary, no continuous shaded regime bands.
  **Do not fit Td ~ eps^-gamma or Teq scaling yet** — not enough effective eps points or
  well-defined onset times.

- [ ] **Headline finding is separation of stages.** Harmonic-mode spreading, chaos, and
  Toda-action equilibration turn on at different energy densities (eta -> lambda -> J): distinct
  finite-time dynamical stages. The extended low-eps point shows these stages are also separated
  *in time* at fixed eps.

- [ ] **Lyapunov renorm cadence must match across dt for any FTLE/LyapunovLocal comparison.**
  Keep the physical renorm interval fixed (dt * renorm_steps): dt=0.1 -> renorm 100,
  dt=0.05 -> renorm 200 (both = 10). Benettin cumulative FTLE is largely renorm-invariant (the
  25/50/100/200 pilot test confirmed this), but strict apples-to-apples LyapunovLocal comparison
  requires matched cadence. The existing dt=0.05 check used renorm 100; a short renorm-200
  comparison (e.g. to 3e7) is planned before final write-up.

- [ ] **Beta crossover is empirical, not critical.** eps_cross ~ N^-0.33 is an empirical
  fixed-observation-time entropy crossover, not a thermodynamic-limit critical eps_c.

- [ ] **Diagnostic roles.** Spectral entropy = harmonic-mode spreading; max Lyapunov = chaos;
  Toda integral J = drift from the nearby integrable Toda dynamics / onset of action diffusion.
  For **alpha**, J is (near) an adiabatic invariant; for **beta**, J is only a control observable
  — do not call it an adiabatic invariant for beta, nor draw J_eq ~ 2eps on beta plots unless
  independently derived.

- [ ] **Terminology.** Prefer **"quasi-stationary state"** over "metastable" for the FPU context
  (Benettin/Ponno QSS literature).

- [ ] **Reproducibility hygiene.** Every production/continuation CSV carries SolverGitCommit and
  SolverGitDirty=0. Energy density from the CSV's initial total energy: eps = H(0)/(N-1).
  Amplitudes computed exactly, never hard-coded. Independent (from-t=0) runs use their OWN
  TotalEnergy[0] for eps_actual, never another run's. Continuation runs record the machine that
  produced them (cross-machine FP divergence is expected for chaotic trajectories; statistical
  conclusions robust, pointwise cross-machine reproducibility is not).
