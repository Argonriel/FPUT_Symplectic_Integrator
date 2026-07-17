"""Reproducible analysis pipeline for FPUT-beta Yoshida4 trajectories.

This package summarizes the *existing* long FPUT-beta runs produced by the
Yoshida-4 symplectic integrator (see ``simulations_cpu/yoshida``). It does not
run simulations, and it never modifies raw CSV data.

Entry point::

    python -m analysis.summarize_beta summarize-beta \
        --input-dir data/yoshida_threshold_v2 --output-dir results

Modules
-------
metadata     : typed parsing of the commented ``# key: value`` CSV header
               (reuses ``visualization/plot_utils.py::get_metadata``).
statistics   : per-run quantities (tail-window statistics, energy density
               ``epsilon``, analytic ``H0``, energy drift).
validation   : accept/reject a candidate trajectory with a recorded reason.
duplicates   : detect and deterministically resolve duplicate runs.
collapse     : descriptive finite-size-collapse metrics (no extrapolation).
plotting     : the required diagnostic figures (matplotlib only).
summarize_beta : the command-line pipeline that ties everything together.
"""

__all__ = [
    "metadata",
    "statistics",
    "validation",
    "duplicates",
    "collapse",
    "plotting",
]
