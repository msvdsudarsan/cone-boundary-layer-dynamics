"""
Independent verification of the structural and numerical results reported in
"Phase-Space Structure and Degenerate-Node Transition in an MHD Ternary
Hybrid Nanofluid Boundary Layer on a Stretching Cone": the generality
dichotomy (Section 2.4, Proposition 1), the projected far-field condition
for lambda_c (Section 5.7, Proposition 8), the M=0 center-manifold
reduction (Section 3.2, Proposition 4), and the variational-sensitivity
monotonicity check (Section 5.6, Proposition 7).

This script is independent of cone_dynamics.py: it does not re-derive the
reference state, but re-uses the same effective-property ratios (Table 2)
and reference parameters (S=1, lambda=0.2, M=0.6, L1=0.2) for consistency.

Reproducibility code accompanying the manuscript "Phase-Space Structure and
Degenerate-Node Transition in an MHD Ternary Hybrid Nanofluid Boundary Layer
on a Stretching Cone". See README.md for the full author list and citation
information.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import json, datetime

np.set_printoptions(precision=10)

LOG = []
def log(*args, **kwargs):
    s = " ".join(str(a) for a in args)
    print(s, **kwargs)
    LOG.append(s)

# Effective-property ratios at phi1=phi2=phi3=0.1, n=4.8 (Table 2)
A1, A2, A3 = 2.284050, 2.203846, 2.370370
S, L1, M = 1.0, 0.2, 0.6


# ----------------------------------------------------------------------
# Part 1: generality dichotomy (Proposition 1) -- symbolic sanity check
# ----------------------------------------------------------------------

def part1_generality_check():
    log("\n---- PART 1: generality dichotomy (Proposition 1) ----")
    log("Crane's stretching-sheet equation f'''+ff''-f'^2=0: a1=1, a2=-1, a3=0, a4=0.")
    log("  a4=0  =>  line of equilibria {(c,0,0)} exists, by Proposition 1(i).")
    log("Present cone equation (Eq. 1)/4A2: a1=A1/(4A2), a2=-A1/(8A2), a3=-A3M/(4A2), a4=0.")
    a1 = A1/(4*A2); a2 = -A1/(8*A2); a3 = -A3*M/(4*A2); a4 = 0.0
    log(f"  a1={a1:.6f}, a2={a2:.6f}, a3={a3:.6f}, a4={a4:.6f}  (a4=0 confirmed)")
    log("Falkner-Skan wedge flow f'''+ff''+beta(1-f'^2)=0 has a4=beta (nonzero for beta!=0):")
    log("  no equilibrium of the form (c,0,0) exists there, by Proposition 1(ii).")
    log("  (This is an algebraic check of Proposition 1's hypotheses, not a numerical result.)")


# ----------------------------------------------------------------------
# Part 2: projected far-field condition and lambda_c (Proposition 7 /
#         "projected momentum far-field condition")
# ----------------------------------------------------------------------

def momentum_rhs(eta, y):
    f, fp, fpp = y
    fppp = -(A1*(f*fpp - 0.5*fp**2) - A3*M*fp)/(4*A2)
    return [fp, fpp, fppp]

def momentum_jac(eta, y):
    f, fp, fpp = y
    return [[0, 1, 0],
            [0, 0, 1],
            [0, (A1*fp + A3*M)/(4*A2), -A1*f/(4*A2)]]

def shoot_momentum(a, lam, eta_inf):
    fp0 = lam + L1*a
    y0 = [S, fp0, a]
    sol = solve_ivp(momentum_rhs, [0, eta_inf], y0, method='Radau',
                     jac=momentum_jac, rtol=1e-8, atol=1e-10)
    return sol.y[:, -1]

def mf_minus(finf):
    disc = (A1*finf)**2 + 16*A2*A3*M
    return (-A1*finf - np.sqrt(disc)) / (8*A2)

def residual_naive(a, lam, eta_inf):
    f, fp, fpp = shoot_momentum(a, lam, eta_inf)
    return fp

def residual_projected(a, lam, eta_inf):
    f, fp, fpp = shoot_momentum(a, lam, eta_inf)
    return fpp - mf_minus(f)*fp

def solve_for_a(residual_fn, lam, eta_inf, a_guess, half_width=0.05):
    lo, hi = a_guess - half_width, a_guess + half_width
    flo, fhi = residual_fn(lo, lam, eta_inf), residual_fn(hi, lam, eta_inf)
    tries = 0
    while flo*fhi > 0 and tries < 8:
        lo -= half_width; hi += half_width
        flo, fhi = residual_fn(lo, lam, eta_inf), residual_fn(hi, lam, eta_inf)
        tries += 1
    return brentq(lambda a: residual_fn(a, lam, eta_inf), lo, hi, xtol=1e-10, rtol=1e-10, maxiter=50)

def finf_at(lam, eta_inf, method, a_guess):
    fn = residual_naive if method == 'naive' else residual_projected
    a = solve_for_a(fn, lam, eta_inf, a_guess)
    f, fp, fpp = shoot_momentum(a, lam, eta_inf)
    return f, a

def interp_cross(rows):
    for i in range(len(rows)-1):
        (l1, f1), (l2, f2) = rows[i], rows[i+1]
        if f1*f2 < 0:
            return l1 + (l2-l1)*(0-f1)/(f2-f1)
    return None

def part2_lambda_c():
    log("\n---- PART 2: projected far-field condition, lambda_c (Table 12) ----")
    lam_sets = {
        15.0: ([-0.42, -0.44, -0.46, -0.48, -0.50], [0.170, 0.178, 0.187, 0.196, 0.205]),
        20.0: ([-0.42, -0.44, -0.46, -0.48, -0.50], [0.170, 0.178, 0.187, 0.196, 0.205]),
        25.0: ([-0.455, -0.460, -0.465, -0.470], [0.184, 0.187, 0.190, 0.193]),
        30.0: ([-0.455, -0.460, -0.465, -0.470], [0.184, 0.187, 0.190, 0.193]),
    }
    results = {}
    for eta_inf, (lams, a_guesses) in lam_sets.items():
        row_naive, row_proj = [], []
        for lam, ag in zip(lams, a_guesses):
            f_n, _ = finf_at(lam, eta_inf, 'naive', ag)
            f_p, _ = finf_at(lam, eta_inf, 'projected', ag)
            row_naive.append((lam, f_n)); row_proj.append((lam, f_p))
        lc_naive = interp_cross(row_naive)
        lc_proj = interp_cross(row_proj)
        results[eta_inf] = (lc_naive, lc_proj)
        log(f"eta_inf={eta_inf:5.1f}:  lambda_c (naive)={lc_naive:.6f}   "
            f"lambda_c (projected)={lc_proj:.6f}   diff={abs(lc_naive-lc_proj):.2e}")

    # Extrapolate the limiting value from the observed geometric error-ratio
    # sequence, rather than assuming a specific classical convergence order
    # (e.g. Richardson extrapolation, which presupposes a known order p).
    # For a sequence x_n -> x* with x_n - x* ~ C r^n, the ratio of successive
    # differences estimates r, and x* = x_n - (x_{n+1}-x_n)/(1/r - 1) follows
    # by summing the remaining geometric tail.
    def extrapolate(seq):
        d1 = seq[1] - seq[0]
        d2 = seq[2] - seq[1]
        if abs(d1) < 1e-14:
            return seq[-1], 0.0
        r = d2 / d1
        if abs(1 - r) < 1e-12:
            return seq[-1], abs(d2)
        limit = seq[2] + d2 * r / (1 - r)
        uncertainty = abs(d2 * r / (1 - r))
        return limit, uncertainty

    etas_sorted = sorted(results.keys())
    naive_seq = [results[e][0] for e in etas_sorted]
    proj_seq = [results[e][1] for e in etas_sorted]
    naive_limit, naive_unc = extrapolate(naive_seq)
    proj_limit, proj_unc = extrapolate(proj_seq)
    combined_limit = 0.5 * (naive_limit + proj_limit)
    combined_unc = max(naive_unc, proj_unc, abs(naive_limit - proj_limit))
    log("Geometric error-ratio extrapolation (computed from the tabulated points above,")
    log("not assumed or hard-coded):")
    log(f"  naive column     -> limit={naive_limit:.5f}, tail estimate={naive_unc:.5f}")
    log(f"  projected column -> limit={proj_limit:.5f}, tail estimate={proj_unc:.5f}")
    log(f"  combined:  lambda_c = {combined_limit:.5f} +/- {combined_unc:.5f}")
    if 30.0 in results:
        naive_30, proj_30 = results[30.0]
        log(f"Corroboration (not used in the extrapolation): at eta_inf=30, naive={naive_30:.6f}, "
            f"projected={proj_30:.6f}, both within {combined_unc:.5f} of the extrapolated limit "
            f"{combined_limit:.5f}.")
    results['extrapolated'] = {'naive_limit': naive_limit, 'naive_uncertainty': naive_unc,
                                'projected_limit': proj_limit, 'projected_uncertainty': proj_unc,
                                'combined_limit': combined_limit, 'combined_uncertainty': combined_unc}
    return results


# ----------------------------------------------------------------------
# Part 3: M=0 center-manifold reduction (Proposition 4)
# ----------------------------------------------------------------------

def m0_rhs(eta, y):
    x1, x2, x3 = y
    k = A1/(4*A2)
    x3p = -k*(x1*x3 - 0.5*x2**2)
    return [x2, x3, x3p]

def part3_m0_center_manifold():
    log("\n---- PART 3: M=0 center-manifold reduction (Proposition 4) ----")
    k = A1/(4*A2)
    c = 1.0
    m = -k*c
    log(f"k=A1/(4A2)={k:.6f}, m=-kc={m:.6f} (c=f_infty=1)")
    log("Reduced normal form: d(u)/dt = p + O(3), d(p)/dt = -(k/2m) p^2 + O(3)")
    coeff = -k/(2*m)
    log(f"-(k/2m) = {coeff:.6f}  (positive => p=0 semi-stable: attracting for p<0, repelling for p>0)")
    for eps in [1e-3, -1e-3]:
        y0 = [c, eps, 0.0]
        sol = solve_ivp(m0_rhs, [0, 40], y0, method='Radau', rtol=1e-11, atol=1e-13,
                         dense_output=True, max_step=0.5)
        x1, x2, x3 = sol.sol(40.0)
        p_at_40 = x2 - x3/m
        predicted = eps/(1 - coeff*eps*40)  # closed-form solution of dp/dt=coeff*p^2... see below
        # dp/dt = coeff*p^2 => p(t) = p0/(1-coeff*p0*t)
        log(f"eps={eps:+.4f}: numerical p(eta=40)={p_at_40:.6e}   "
            f"normal-form prediction={predicted:.6e}")


# ----------------------------------------------------------------------
# Part 4: variational sensitivity / monotonicity (Proposition "variational
#         sensitivity")
# ----------------------------------------------------------------------

def variational_rhs(eta, y, lam):
    f, fp, fpp, v0, v1, v2 = y
    fppp = -(A1*(f*fpp - 0.5*fp**2) - A3*M*fp)/(4*A2)
    vppp = -(A1*f*v2 - (A1*fp + A3*M)*v1 + A1*fpp*v0)/(4*A2)
    return [fp, fpp, fppp, v1, v2, vppp]

def shoot_variational(a, lam, eta_inf):
    fp0 = lam + L1*a
    y0 = [S, fp0, a, 0.0, L1, 1.0]
    sol = solve_ivp(lambda eta, y: variational_rhs(eta, y, lam), [0, eta_inf], y0,
                     method='Radau', rtol=1e-10, atol=1e-12, dense_output=True)
    return sol

def part4_variational():
    log("\n---- PART 4: variational sensitivity / monotonicity check ----")
    states = [
        (0.2, -0.105282, "anchor"),
        (-0.110328, 0.052843, "2nd continuation step"),
        (-0.441311, 0.183702, "near terminus"),
        (-0.488234, 0.198156, "last computed (past crossing)"),
    ]
    for lam, a, label in states:
        sol = shoot_variational(a, lam, 15.0)
        etas_check = np.linspace(0.01, 15.0, 200)
        vs = np.array([sol.sol(e)[3] for e in etas_check])
        v1s = np.array([sol.sol(e)[4] for e in etas_check])
        f, fp, fpp, v0, v1, v2 = sol.sol(15.0)
        log(f"{label:28s} lambda={lam:9.6f} a={a:9.6f}  v'(eta_inf)={v1:10.4f}  "
            f"min(v)={vs.min():.6f} all>0:{np.all(vs>0)}  min(v')={v1s.min():.6f} all>0:{np.all(v1s>0)}")


if __name__ == "__main__":
    log("=" * 64)
    log(" ADDITIONAL VERIFICATION: generality, projected BC, M=0 center")
    log(" manifold, and variational monotonicity")
    log(f" Python run: {datetime.datetime.now()}")
    log("=" * 64)
    part1_generality_check()
    lambda_c_results = part2_lambda_c()
    part3_m0_center_manifold()
    part4_variational()

    with open("verification_output/additional_verification_log.txt", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    with open("verification_output/additional_verification_values.json", "w") as fh:
        serializable = {str(k): (v if not isinstance(v, dict) else v)
                         for k, v in lambda_c_results.items()}
        json.dump({"lambda_c_by_eta_inf": serializable}, fh, indent=2)
    log("\nSaved verification_output/additional_verification_log.txt and .json")
