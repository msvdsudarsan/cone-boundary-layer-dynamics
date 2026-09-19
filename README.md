# cone-boundary-layer-dynamics

Reproducibility code and numerical data for "Interfacial Thermal Resistance and Volumetric Heat Capacity Set Wall Heat-Flux Bounds in an Aqueous Ternary Nanoparticle Dispersion" (Sri Venkata Durga Sudarsan Madhyannapu, Lakshmi Appidi,
K.G.R. Deepthi, T. Prasanna Kumar, Devaganugula Naga Venkata Rama Krishna,
Kankipati Subbarao), submitted to the Journal of Molecular Liquids.

**Zenodo DOI for this release (v1.5.0):** https://doi.org/10.5281/zenodo.22834618

**Zenodo concept DOI (all versions, resolves to the latest):** https://doi.org/10.5281/zenodo.22070673

This release (v1.5.0) is the first release since v1.4.0 and accompanies the Journal of
Molecular Liquids submission. Compared with v1.4.0 it:

- adds `interface_bounds.py`, which checks the series-resistance bounds on the wall heat
  flux, repeats the dispersion sweeps at fixed interfacial (Kapitza) resistance, calibrates
  the thermal slip parameter against measured Kapitza lengths, verifies the exact
  asymptotic-suction solution used as a solver benchmark, computes the high-Prandtl
  saturation defect against the proved bound, and draws the two interfacial figures;
- extends `dispersion_sensitivity.py` with the dilute 3 vol% state, a thermal-slip-length
  sweep and a check that A3 is insensitive to the particle electrical conductivities;
- extends `additional_verification.py` with the f_inf and Qc sweep at M = 1.5;
- corrects the slip schematic so that both temperature profiles decrease monotonically from
  the wall, with both inner-layer scalings marked, and sets the apparent-Kapitza-length band
  of the regime map to 4-150 um to match the values tabulated in the manuscript;
- regenerates all figures and verification outputs from a single clean run of the six
  scripts, and updates the docstrings and this README to the current manuscript title, with
  a table mapping the internal code numbering to the manuscript numbering.

Two numbers quoted in an earlier draft of the manuscript text (0.4938 and 0.9586 for the
high-Prandtl defect and its bound at Pr = 2000) were wrong; the values produced by this
release are 0.464454 and 0.901602.

This archive contains only the reproducibility code and data. The
manuscript LaTeX source is submitted separately through the journal's
own submission system.

## Contents

```
cone_dynamics.py            hand-rolled RK4 shooting/Newton solver +
                             pseudo-arclength continuation
make_figures.py              regenerates the remaining figures from computed data
residual_diagnostics.py      independent scipy.integrate.solve_bvp
                              collocation cross-check and residual table
interface_bounds.py          series-resistance bounds, fixed-interface sweeps,
                              Kapitza calibration, Figures 1-2
dispersion_sensitivity.py    shape-factor, unequal-loading, mixing-order,
                              dilute-loading, slip-length and sigma checks
additional_verification.py   verifies the generality dichotomy (Sec. 2.5),
                              the projected far-field condition and resolved
                              lambda_c (Sec. 5.3), the M=0 centre-manifold
                              reduction (Sec. 3.2), and the variational-
                              sensitivity check (Supplementary Proposition S1)
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

**Note on `branch_multiplicity_scan.csv`:** the momentum multiplicity search itself always covers the full 2001-point grid on $a\in[-30,30]$ quoted in the manuscript (Section 7.2). The `bounded_range_lo`/`bounded_range_hi` columns record the narrower sub-interval of that grid on which the RK4 shooting trajectory stays numerically finite rather than overflowing; a sign change (and hence a candidate second root) can only be detected inside that finite sub-interval, which is why it is reported alongside the crossing list.

## Reproducing the numerics

```
python3 cone_dynamics.py             # shooting/continuation solver
python3 residual_diagnostics.py      # independent solve_bvp cross-check
python3 make_figures.py              # regenerates figures/*.pdf and *.png
python3 dispersion_sensitivity.py    # fixed-slip dispersion and loading sweeps
python3 interface_bounds.py          # series-resistance bounds, Figures 1-2
python3 additional_verification.py   # generality, projected BC/lambda_c,
                                      # M=0 centre manifold, variational check
```

Requires Python 3 with numpy, scipy and matplotlib. Tested with Python 3.12.3, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.8; no version-specific features are used, so nearby versions should also work.

**Quick verification** (checks the paper's key numerical claims without regenerating figures): run `additional_verification.py` alone (~2 minutes on a single core) -- it independently verifies the generality dichotomy, the projected far-field condition and resolved $\lambda_c=-0.46252\pm0.00002$, the $M=0$ centre-manifold coefficient, and the variational-sensitivity positivity check, printing a pass/fail-style log to `verification_output/additional_verification_log.txt`.

**Full reproduction** (regenerates every figure and diagnostic file from scratch): run all six scripts in the order listed above. Measured single-core runtimes: `cone_dynamics.py` ~5 minutes (the continuation branch is the slow part), `residual_diagnostics.py` under 1 minute, `make_figures.py` ~4 minutes (re-solves several parameter sweeps to regenerate the remaining main and supplementary figure files; internal file names map to the submitted manuscript numbering in the table below), `dispersion_sensitivity.py` under 2 minutes, `interface_bounds.py` a few minutes, `additional_verification.py` ~2 minutes. Total expected runtime for a full reproduction is roughly a quarter of an hour on a single core; none of the scripts require a GPU or parallel hardware.

## Citing this archive

[dataset] S.V.D.S. Madhyannapu, L. Appidi, K.G.R. Deepthi, T. Prasanna Kumar,
D.N.V.R. Krishna, K. Subbarao, cone-boundary-layer-dynamics: reproducibility
code and numerical verification data, Zenodo, v1.5.0, 2026.
https://doi.org/10.5281/zenodo.22834618

## Companion manuscript

A companion manuscript with an overlapping set of authors, covering the parametric
heat-transfer survey and entropy-generation analysis for the same
physical model, is unpublished and not currently under review at any journal. No numerical results,
tables, or figures in this archive are shared with that manuscript.

## License

MIT. See LICENSE.


## Numbering used in code comments and logs

Table, section and proposition numbers printed by the scripts and stored in
`verification_output/` follow the internal numbering used while the code was
written. They correspond to the submitted manuscript as follows:

| Code / log label | Manuscript |
|---|---|
| Table 2 (property ratios) | Table 2 |
| Table 3 (reference-state classification) | Supplementary Table S1 |
| Table 6 (f_inf and Qc against S) | Table 8 |
| Table 7 (nanoparticle loadings) | Table 4 |
| Table 8 (residual diagnostics) | Supplementary Table S3 |
| Table 9 (pseudo-arclength continuation) | Table 9 |
| Table 10 (direct march in lambda) | Supplementary Table S4 |
| Table 11 (lambda_c sensitivity) | Supplementary Table S5 |
| Table 12 in cone_dynamics.py (approach to the ceiling) | Table 3 |
| Table 12 in additional_verification.py (projected condition) | Table 10 |
| Section 2.4 / Proposition 1 (generality dichotomy) | Section 2.6 / Proposition 1 |
| Section 3.2 / Proposition 4 (M=0 reduction) | Supplementary Proposition S2 (summary in Section 5.2) |
| Section 5.2 (methods, multiplicity scan) | Sections 7.1-7.2 |
| Section 5.6 / Proposition 7 (variational sensitivity) | Supplementary Proposition S1 |
| Section 5.7 / Proposition 8 (projected condition) | Section 7.3 / Proposition 7 |

| Figure file | Manuscript |
|---|---|
| fig7_nusselt_ceiling | Figure 1 |
| fig8_regime_map | Figure 2 |
| fig1_thermal_eigenloci | Figure 3 |
| fig3_spectral_admissibility | Figure 4 |
| fig5_continuation_branch | Figure 5 |
| fig2_local_phase_portraits | Supplementary Fig. S1 |
| fig4_loading_robustness | Supplementary Fig. S2 |
| fig6_slip_schematic | Supplementary Fig. S3 |

## interface_bounds.py

Checks the series-resistance bounds of Theorem 4 and hypothesis Q <= Q_min along every
computed profile (Pr sweep, slip-length sweep, loadings, shape factors, compositions and
mixing orders), repeats the dispersion sweeps at fixed interfacial resistance Bi_K
(Tables 3-5), converts published Kapitza lengths into the scaled slip length (Table 6),
draws Figures 1 and 2, checks the exact asymptotic-suction solution used as a benchmark
(Section 3.3), compares the computed Nusselt number with that series estimate (Table 3), and
records the truncation check of the reference state (Section 7.1), and compares the high-Prandtl
saturation defect with the bound proved in Theorem 4 (Section 3.4 of the manuscript).
Output: verification_output/interface_bounds.txt.
Run time is a few minutes on a single core.

## dispersion_sensitivity.py

Reproduces the fixed-slip columns of Tables 4 and 5 of the manuscript (equal loadings including the dilute
3 vol% state; shape factor, unequal loadings and mixing order), the thermal-slip-length
sweep quoted in Section 4.1, and the insensitivity of A3 to the particle electrical
conductivities quoted in Appendix A. Output: verification_output/dispersion_sensitivity.txt.
Run time is under two minutes on a single core.
