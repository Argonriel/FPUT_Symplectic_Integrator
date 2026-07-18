# Manuscript Wording & Claims Discipline

Standing reference for how results are phrased in figures, captions, and the manuscript.
Split into (A) open TODO items to apply now, and (B) a permanent claims-discipline checklist
established across the pilot/analysis work.

---

## A. Open TODO — apply to alpha-pilot figures/captions and manuscript text

- [ ] **Spectral entropy at low energy density.**
  Do **not** write "η alone would suggest partial thermalization."
  Write instead: **"η alone indicates substantial harmonic-mode spreading, but does not
  establish chaos or action equilibration."**
  Rationale: η ≈ 0.59–0.70 means the harmonic modal energy has spread; it is *not* evidence of
  half-thermalization.

- [ ] **Low-ε cumulative FTLE decay.**
  Do **not** state the low-ε FTLE has been *shown* to follow 1/t or (ln t)/t.
  Write instead: **"the cumulative FTLE is consistent with decay toward zero over the observed
  interval."**
  Any 1/t or (ln t)/t reference line in a figure must be labeled **a visual guide, not a fit.**

- [ ] **FTLE tail standard deviation in the summary figure.**
  The tail std is **temporal variation over the tail window, not statistical uncertainty across
  independent realizations.** State this explicitly in the caption.
  When judging whether a point has a genuine positive plateau, require **all three**:
  (i) cumulative FTLE flattens;
  (ii) `LyapunovLocal` tail-mean is stably positive;
  (iii) local fluctuations sit around a positive value, not around zero.

---

## B. Standing claims-discipline checklist (reference)

- [ ] **J = 2ε is an estimate, not a theorem.** Refer to it as the *alpha equilibrium estimate*
  (J ≈ 2ε). It is not an exact theorem and is **not** a beta-model result.

- [ ] **J(0) ≈ 3ε is a mode-1 initial-condition fact.** State that the fundamental-mode IC gives
  J(0) ≈ 3ε, so J approaches equilibrium **from above** (opposite to the random-IC picture,
  where J rises from below). Independently confirmed via the quadratic Toda approximation.

- [ ] **Finite-time categories, not thermodynamic claims.** Present `J-flat` / `J-evolving` /
  `J-near-equilibrium` as **finite-time classifications at the stated nominal duration**
  (currently 1e7). Do not phrase them as infinite-time or thermodynamic-limit statements.

- [ ] **Finite-size wording is observable-specific.** Do **not** say "larger N is universally
  more thermalized." Describe each observable's N-trend separately (η most N-sensitive;
  J nearly N-independent except near equilibrium; FTLE weakly N-dependent). The strongest claim
  is **finite-size consistency** (same dynamical classification across N), not a strong monotone
  finite-size trend in every observable.

- [ ] **No precise crossover thresholds from sparse ε.** With only a few sampled ε levels: do not
  fit or claim precise crossover thresholds, do not interpolate a sharp phase boundary, and do
  not draw continuous shaded "regime boundaries" between sparse points.

- [ ] **The headline finding is separation, not "entropy fails."** Frame the core result as:
  **mode spreading, chaos, and Toda-action equilibration turn on at different energy densities
  (η → λ → J), i.e. they are distinct finite-time dynamical stages** — more informative than a
  single threshold, and more precise than "spectral entropy fails."

- [ ] **Beta crossover is empirical, not critical.** The beta transition scaling is an
  **empirical fixed-observation-time entropy crossover, ε_cross ~ N^-0.33**, not a true critical
  energy density ε_c (which in the KAM/SST literature means a thermodynamic-limit quantity).

- [ ] **Diagnostic roles.** Spectral entropy measures harmonic-mode spreading; the maximum
  Lyapunov exponent measures chaos; the Toda integral J measures drift from the nearby
  integrable Toda dynamics / onset of action diffusion. For **alpha**, J is (near) an adiabatic
  invariant; for **beta**, J is only a control observable — do **not** call it an adiabatic
  invariant for beta, and do not draw J_eq ≈ 2ε on beta plots unless independently derived.

- [ ] **Terminology.** Prefer **"quasi-stationary state"** over "metastable" for the FPU context
  (maps to the Benettin/Ponno QSS literature).

- [ ] **Reproducibility hygiene.** Every production/continuation CSV must carry the solver git
  commit and a clean-worktree flag (`SolverGitCommit`, `SolverGitDirty = 0`). Energy density is
  reported from the CSV's initial total energy: ε = H(0)/(N−1). Amplitudes are computed exactly,
  never hard-coded.
