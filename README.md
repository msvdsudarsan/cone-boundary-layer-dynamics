# cone-boundary-layer-dynamics

Reproducibility code and data for "Thermal Admissibility, Heat-Flux
Bounds, and a Degenerate-Node Transition in the Boundary Layer of an
MHD Ternary Hybrid Nanofluid on a Stretching Cone" (Sri Venkata Durga
Sudarsan Madhyannapu, Lakshmi Appidi, K.G.R. Deepthi, T. Prasanna
Kumar, Devaganugula Naga Venkata Rama Krishna, Kankipati Subbarao),
submitted to International Communications in Heat and Mass Transfer.

**Zenodo DOI (this release, v1.4.0):** https://doi.org/10.5281/zenodo.22729830

**Zenodo concept DOI (always resolves to the latest version):** https://doi.org/10.5281/zenodo.22070673

> **Note:** The code archived at the time of the manuscript's original submission was release
> **v1.0.0** (https://doi.org/10.5281/zenodo.22070674). Release **v1.1.0**
> (https://doi.org/10.5281/zenodo.22099329) extended the second-root search of Section 5.5 from
> two states to six states spanning the full continuation branch (`branch_multiplicity_scan.csv`).
> Release **v1.2.0** (https://doi.org/10.5281/zenodo.22116347) added
> `additional_verification.py` and its saved output, independently verifying the generality
> dichotomy, the projected far-field condition and resolved branch terminus
> lambda_c = -0.46252 +/- 0.00002, the M=0 center-manifold reduction, and the
> variational-sensitivity check. Release **v1.3.0**
> (https://doi.org/10.5281/zenodo.22274708) corrected the author order in this README and in
> the reproducibility-code citation to match the manuscript's final author list. The current
> release, **v1.4.0** (https://doi.org/10.5281/zenodo.22729830), corrects the reduced Nusselt
> number and slip-induced ceiling to use the wall-flux prefactor (A5+Rd), consistent with the
> energy equation's own diffusion coefficient, in place of an earlier A5(1+Rd) form; all
> Nusselt-number values, the ceiling Nu_infinity, and the corresponding figures and saved
> outputs have been recomputed and regenerated accordingly, and this is the version cited in
> the manuscript's final reference list. No other reported values (far-field entrainment, Qc,
> lambda_c, eigenvalues) are affected by this correction.

This archive contains only the reproducibility code and data. The
manuscript LaTeX source is submitted separately through the journal's
own submission system, per the journal's research-data policy of
keeping code/data archives distinct from manuscript source files.

## Contents

```
cone_dynamics.py            hand-rolled RK4 shooting/Newton solver +
                             pseudo-arclength continuation
make_figures.py              regenerates all 7 figures from computed data
residual_diagnostics.py      independent scipy.integrate.solve_bvp
                              collocation cross-check and residual table
additional_verification.py   verifies the generality dichotomy (Sec. 2.4),
                              the projected far-field condition and resolved
                              lambda_c (Sec. 5.7), the M=0 center-manifold
                              reduction (Sec. 3.2), and the variational-
                              sensitivity monotonicity check (Sec. 5.6)
figures/                     regenerated figure output (PDF + PNG)
verification_output/         saved numerical output backing the paper's
                              tables (continuation_diagnostics.csv,
                              branch_multiplicity_scan.csv,
                              residual_diagnostics.txt,
                              python_output_values.json/.txt,
                              additional_verification_log.txt,
                              additional_verification_values.json)
LICENSE                       MIT
```

**Note on `branch_multiplicity_scan.csv`:** the momentum multiplicity search itself always covers the full 2001-point grid on $a\in[-30,30]$ quoted in the manuscript (Section 5.6). The `bounded_range_lo`/`bounded_range_hi` columns record the narrower sub-interval of that grid on which the RK4 shooting trajectory stays numerically finite rather than overflowing; a sign change (and hence a candidate second root) can only be detected inside that finite sub-interval, which is why it is reported alongside the crossing list.

## Reproducing the numerics

```
python3 cone_dynamics.py             # shooting/continuation solver
python3 residual_diagnostics.py      # independent solve_bvp cross-check
python3 make_figures.py              # regenerates figures/*.pdf and *.png
python3 additional_verification.py   # generality, projected BC/lambda_c,
                                      # M=0 center manifold, variational check
```

Requires Python 3 with numpy, scipy and matplotlib. Tested with Python 3.12.3, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.8; no version-specific features are used, so nearby versions should also work.

**Quick verification** (checks the paper's key numerical claims without regenerating figures): run `additional_verification.py` alone (~2 minutes on a single core) -- it independently verifies the generality dichotomy, the projected far-field condition and resolved $\lambda_c=-0.46252\pm0.00002$, the $M=0$ center-manifold coefficient, and the variational-sensitivity positivity check, printing a pass/fail-style log to `verification_output/additional_verification_log.txt`.

**Full reproduction** (regenerates every figure and diagnostic file from scratch): run all four scripts in the order listed below. Measured single-core runtimes: `cone_dynamics.py` ~5 minutes (the continuation branch is the slow part), `residual_diagnostics.py` under 1 minute, `make_figures.py` ~4 minutes (re-solves several parameter sweeps to draw Figures 1-7), `additional_verification.py` ~2 minutes. Total expected runtime for a full reproduction is approximately 11-12 minutes on a single core; none of the scripts require a GPU or parallel hardware.

## Citing this archive

S.V.D.S. Madhyannapu, L. Appidi, K.G.R. Deepthi, T. Prasanna Kumar,
D.N.V.R. Krishna, K. Subbarao, cone-boundary-layer-dynamics:
reproducibility code and numerical verification data [software],
Zenodo, v1.4.0, 2026.
https://doi.org/10.5281/zenodo.22729830

The saved numerical verification output in `verification_output/`
(continuation diagnostics, the six-state second-root scan, the
residual-diagnostics results, and the generality/projected-BC/M=0/
variational verification) is part of this same software record and is
not cited as a separate dataset.

## Companion manuscript

A companion manuscript with an overlapping set of authors, covering the parametric
heat-transfer survey and entropy-generation analysis for the same
physical model, is under review elsewhere. No numerical results,
tables, or figures in this archive are shared with that manuscript.

## License

MIT. See LICENSE.
