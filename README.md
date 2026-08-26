# cone-boundary-layer-dynamics

Reproducibility code and data for "Phase-Space Structure and
Degenerate-Node Transition in an MHD Ternary Hybrid Nanofluid Boundary
Layer on a Stretching Cone" (Lakshmi Appidi, K.G.R. Deepthi, T. Prasanna
Kumar, Devaganugula Naga Venkata Rama Krishna, Sri Venkata Durga
Sudarsan Madhyannapu, Kankipati Subbarao), submitted to Physica D:
Nonlinear Phenomena.

**Zenodo DOI (this release, v1.1.0):** https://doi.org/10.5281/zenodo.22116347

**Zenodo concept DOI (always resolves to the latest version):** https://doi.org/10.5281/zenodo.22070673

> **Note:** The code archived at the time of the manuscript's original submission was release
> **v1.0.0** (https://doi.org/10.5281/zenodo.22070674). Release **v1.1.0**
> (https://doi.org/10.5281/zenodo.22099329) extends the second-root search of Section 5.2 from
> two states to six states spanning the full continuation branch (`branch_multiplicity_scan.csv`)
> and is the version cited in the manuscript's reference list for the original tables and
> figures. A further release, **v1.2.0**, adds `additional_verification.py` and its saved
> output, verifying the generality dichotomy, the projected far-field condition and resolved
> lambda_c, the M=0 center-manifold reduction, and the variational-sensitivity check added to
> the manuscript after v1.1.0; see the Releases tab for the exact DOI once minted. It reproduces
> every previously reported table and figure value unchanged.

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
                              sensitivity monotonicity check (Sec. 5.8)
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

## Reproducing the numerics

```
python3 cone_dynamics.py             # shooting/continuation solver
python3 residual_diagnostics.py      # independent solve_bvp cross-check
python3 make_figures.py              # regenerates figures/*.pdf and *.png
python3 additional_verification.py   # generality, projected BC/lambda_c,
                                      # M=0 center manifold, variational check
```

Requires Python 3 with numpy, scipy and matplotlib.

## Citing this archive

L. Appidi, K.G.R. Deepthi, T. Prasanna Kumar, D.N.V.R. Krishna,
S.V.D.S. Madhyannapu, K. Subbarao, cone-boundary-layer-dynamics:
reproducibility code [software], Zenodo, v1.1.0, 2026.
https://doi.org/10.5281/zenodo.22099329

The saved numerical verification output in `verification_output/`
(continuation diagnostics, the six-state second-root scan, and the
residual-diagnostics results) is additionally citable as a dataset:

L. Appidi, K.G.R. Deepthi, T. Prasanna Kumar, D.N.V.R. Krishna,
S.V.D.S. Madhyannapu, K. Subbarao, Numerical verification data for
"Phase-Space Structure and Degenerate-Node Transition in an MHD Ternary
Hybrid Nanofluid Boundary Layer on a Stretching Cone" [dataset], Zenodo,
v1.1.0, 2026. https://doi.org/10.5281/zenodo.22099329

## Companion manuscript

A companion manuscript by the same authors, covering the parametric
heat-transfer survey and entropy-generation analysis for the same
physical model, is under review elsewhere. No numerical results,
tables, or figures in this archive are shared with that manuscript.

## License

MIT. See LICENSE.
