"""
Residual and convergence diagnostics for five representative states of the
cone boundary-layer problem (manuscript Table 8 / Section 5.3).

For each state this script reports:
  - the shooting solver's far-field residual components f'(eta_inf), theta(eta_inf)
  - the Newton residual norm used as the convergence criterion
  - the maximum pointwise collocation residual from scipy.integrate.solve_bvp

Uses the staged eta_inf homotopy solver (solve_state) already defined in
cone_dynamics.py, rather than a single-shot Newton call from a fixed guess,
so that states away from the reference solution (e.g. near the continuation
terminus) converge reliably rather than diverging from a poor initial guess.

Run: python3 residual_diagnostics.py
Writes: verification_output/residual_diagnostics.txt
"""
import numpy as np
from scipy.integrate import solve_bvp
from cone_dynamics import baseline, ratios, integ, solve_state

P = ratios(baseline())

def bvp_maxres(Pc, einf, seed):
    def odes(x, y):
        dy = np.empty_like(y)
        dy[0] = y[1]; dy[1] = y[2]
        dy[2] = (-Pc['A1']*(y[0]*y[2]-0.5*y[1]**2)+Pc['A3']*Pc['M']*y[1])/(4*Pc['A2'])
        dy[3] = y[4]
        dy[4] = -(Pc['Pr']*Pc['A4']*y[0]*y[4]+Pc['Pr']*Pc['Q']*y[3])/(4*Pc['A5Rd'])
        return dy
    def bcs(ya, yb):
        return np.array([ya[0]-Pc['S'], ya[1]-Pc['lam']-Pc['L1']*ya[2],
                          ya[3]-1-Pc['L2']*ya[4], yb[1], yb[3]])
    # Seed the collocation mesh with the already-converged RK4 shooting
    # trajectory rather than a generic exponential guess: at Pr=2000 the
    # thermal layer is thin enough that the generic guess forces solve_bvp
    # to refine its mesh from scratch, which is slow; a close starting
    # point cuts this sharply.
    a, b = seed['a'], seed['b']
    N_dense = 4000
    h_dense = einf / N_dense
    y = np.array([Pc['S'], Pc['lam'] + Pc['L1']*a, a, 1 + Pc['L2']*b, b])
    traj = np.empty((5, N_dense + 1))
    traj[:, 0] = y
    for i in range(N_dense):
        k1 = odes(0, y); k2 = odes(0, y + 0.5*h_dense*k1)
        k3 = odes(0, y + 0.5*h_dense*k2); k4 = odes(0, y + h_dense*k3)
        y = y + (h_dense/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        traj[:, i+1] = y
    x_dense = np.linspace(0, einf, N_dense + 1)
    x = np.linspace(0, einf, 400)
    y0 = np.vstack([np.interp(x, x_dense, traj[j]) for j in range(5)])
    sol = solve_bvp(odes, bcs, x, y0, tol=1e-10, max_nodes=200000, verbose=0)
    return (np.max(sol.rms_residuals) if sol.status == 0 else float('nan')), sol.status == 0

cases = [
    ("Reference (Q=0.05, node)",          dict(P, Q=0.05),      20.0, 0.005),
    ("Degenerate node, Q=Qc=0.190342",    dict(P, Q=0.190342),  20.0, 0.005),
    ("Stable spiral, Q=0.25>Qc",          dict(P, Q=0.25),      20.0, 0.005),
    ("High Prandtl, Pr=2000",             dict(P, Pr=2000.0),   20.0, 0.0005),
    ("Approaching terminus, lam=-0.45",   dict(P, S=1.0, lam=-0.45), 15.0, 0.005),
]

lines = []
def out(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)

out(f"{'Case':32s}{'einf':>6s}{'h':>8s}{'|f_prime(einf)|':>16s}{'|theta(einf)|':>14s}{'Newton res norm':>16s}{'max BVP resid':>14s}")
for name, Pc, einf, h in cases:
    r = solve_state(Pc, einf, h)
    if not r['ok']:
        out(f"{name:32s}  FAILED TO CONVERGE")
        continue
    o = integ(r['a'], r['b'], Pc, einf, h)
    fprime, theta = o['y'][1], o['y'][3]
    nres = np.linalg.norm([fprime, theta])
    maxres, ok = bvp_maxres(Pc, einf, r)
    out(f"{name:32s}{einf:6.1f}{h:8.4f}{abs(fprime):16.3e}{abs(theta):14.3e}{nres:16.3e}{maxres:14.3e}")

with open("verification_output/residual_diagnostics.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
