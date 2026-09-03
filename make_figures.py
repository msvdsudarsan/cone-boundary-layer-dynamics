"""
Generate publication-quality figures (vector PDF + 1000 DPI PNG) for the
cone-boundary-layer manuscript, from the verified Python computations in
cone_dynamics.py. Matches the 7 figures of the manuscript:

  fig1_thermal_eigenloci      fig2_local_phase_portraits
  fig3_spectral_admissibility fig4_loading_robustness
  fig5_continuation_branch    fig6_slip_schematic
  fig7_nusselt_ceiling
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cone_dynamics import (baseline, ratios, solve_state, table3_qcS,
                            table4_branch)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)

DPI = 1000
plt.rcParams.update({
    "font.size": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def save(fig, name):
    fig.savefig(f"{OUTDIR}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/{name}.png", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")

P = ratios(baseline())
s15 = solve_state(P, 15, 0.005)
finf = s15['finf']
Qc = P['Pr']*P['A4']**2*finf**2 / (16*P['A5Rd'])

# ---------------------------------------------------------------- Figure 1
Qv = np.linspace(0, 0.30, 800)
re = np.empty_like(Qv); im = np.empty_like(Qv); rp = np.empty_like(Qv)
for i, Q in enumerate(Qv):
    D = (P['Pr']*P['A4']*finf)**2 - 16*P['A5Rd']*P['Pr']*Q
    if D >= 0:
        re[i] = (-P['Pr']*P['A4']*finf - np.sqrt(D)) / (8*P['A5Rd'])
        rp[i] = (-P['Pr']*P['A4']*finf + np.sqrt(D)) / (8*P['A5Rd'])
        im[i] = 0
    else:
        re[i] = -P['Pr']*P['A4']*finf / (8*P['A5Rd'])
        rp[i] = re[i]
        im[i] = np.sqrt(-D) / (8*P['A5Rd'])

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axs[0]
ax.plot(re, im, 'o', color='tab:blue', ms=2, markevery=6)
ax.plot(re, -im, 'o', color='tab:blue', ms=2, markevery=6)
ax.plot(rp, im, 'x', color='tab:red', ms=3, markevery=6)
ax.plot(rp, -im, 'x', color='tab:red', ms=3, markevery=6)
ax.plot(-P['Pr']*P['A4']*finf/(8*P['A5Rd']), 0, '*', color='k', ms=12,
        label='collision at $Q=Q_c$')
ax.set_xlabel(r'Re $m_\theta$'); ax.set_ylabel(r'Im $m_\theta$')
ax.set_title('(a) eigenvalue loci as Q increases'); ax.legend(fontsize=7, loc='upper left')
ax.grid(alpha=0.3)

ax = axs[1]
ax.plot(Qv, re, color='tab:blue', lw=1.3, label=r'Re $m_\theta^-$')
ax.plot(Qv, rp, color='tab:red', lw=1.3, label=r'Re $m_\theta^+$')
ax.plot(Qv, im, '--', color='0.45', lw=1.1, label=r'Im $m_\theta^+$')
ax.axvline(Qc, ls='-.', color='0.3', lw=0.8)
ax.set_xlabel('Q'); ax.set_ylabel('eigenvalue')
ax.set_title('(b) real and imaginary parts'); ax.legend(fontsize=7, loc='lower left')
ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "fig1_thermal_eigenloci")

# ---------------------------------------------------------------- Figure 2
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for pane, (Q, ttl) in enumerate([(0.05, r'(a) Q=0.05 $<Q_c$: stable node'),
                                  (0.25, r'(b) Q=0.25 $>Q_c$: stable spiral')]):
    A = np.array([[0, 1], [-P['Pr']*Q/(4*P['A5Rd']), -P['Pr']*P['A4']*finf/(4*P['A5Rd'])]])
    ax = axs[pane]
    for th0 in [0.2, 0.5, 0.8, 1.0]:
        z = np.array([th0, -0.30])
        traj = [z.copy()]
        for _ in range(4000):
            z = z + 0.01*(A @ z)
            traj.append(z.copy())
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], color='tab:red', lw=0.9)
    if pane == 0:
        w, V = np.linalg.eig(A)
        for j in range(2):
            v = V[:, j].real
            v = v / np.max(np.abs(v))
            ax.plot([-0.7*v[0], 1.0*v[0]], [-0.7*v[1], 1.0*v[1]], color='tab:blue', lw=0.8)
    ax.plot(0, 0, 'o', color='k', ms=5)
    ax.axhline(0, ls=':', color='0.3', lw=0.7)
    ax.axvline(0, ls='-.', color='0.3', lw=0.7)
    ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$d\theta/d\eta$')
    ax.set_title(ttl); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "fig2_local_phase_portraits")

# ---------------------------------------------------------------- Figure 3
t3 = table3_qcS(P)
Sv = np.array([r[0] for r in t3])
fv = np.array([r[1] if r[1] is not None else np.nan for r in t3])
qv = np.array([r[2] if r[2] is not None else np.nan for r in t3])
g = ~np.isnan(qv)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.fill_between(Sv[g], 0, qv[g], color=(.83, .88, .95))
ax.plot(Sv[g], qv[g], color='tab:red', lw=1.8, label=r'$Q_c(S)$: node$\to$spiral')
ax.plot(1.0, 0.05, 'o', color='tab:blue', ms=7, label='reference state (1.0, 0.05)')
ax.plot(-0.5, 0.10, 'x', color='k', ms=10, mew=2.5, label='original case (-0.5, 0.10)')
ax.axvline(0, ls='-.', color='0.3', lw=0.8)
ax.set_xlabel('suction parameter S'); ax.set_ylabel('heat-source parameter Q')
ax.set_title('Spectral admissibility map in the (S, Q) plane')
ax.legend(fontsize=7, loc='upper left')
ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "fig3_spectral_admissibility")

# ---------------------------------------------------------------- Figure 4 (loading robustness)
phis = np.linspace(0.02, 0.15, 40)
fvals, qcvals = [], []
for phi in phis:
    Pp = ratios(P, phi, phi, phi)
    s = solve_state(Pp, 15, 0.005)
    fvals.append(s['finf'])
    qcvals.append(Pp['Pr']*Pp['A4']**2*s['finf']**2 / (16*Pp['A5Rd']))
fvals, qcvals = np.array(fvals), np.array(qcvals)

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axs[0]
ax.plot(phis, qcvals, color='tab:red', lw=1.6)
ax.plot(0.10, Qc, 'o', color='tab:blue', ms=7, label=f'reference (0.10, {Qc:.6f})')
ax.set_xlabel(r'$\varphi_1=\varphi_2=\varphi_3$'); ax.set_ylabel(r'$Q_c$')
ax.set_title(r'(a) $Q_c$ against equal nanoparticle loading')
ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax = axs[1]
ax.plot(phis, fvals, color='tab:blue', lw=1.6)
ax.plot(0.10, finf, 'o', color='tab:red', ms=7)
ax.set_xlabel(r'$\varphi_1=\varphi_2=\varphi_3$'); ax.set_ylabel(r'$f_\infty$')
ax.set_title('(b) far-field entrainment against loading')
ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "fig4_loading_robustness")

# ---------------------------------------------------------------- Figure 5 (continuation branch)
branch, _continuation_diag = table4_branch(P)
# Interpolated crossing f_inf=0 between the last two accepted points, matching the
# manuscript's own treatment (Section 5.2): the raw last computed point already lies
# past the crossing (f_inf<0), so the terminus marker plotted below is this
# interpolated point, not branch[-1] itself.
lam1, fpp1, finf1 = branch[-2]
lam2, fpp2, finf2 = branch[-1]
t_cross = (0.0 - finf1) / (finf2 - finf1)
lam_c = lam1 + t_cross * (lam2 - lam1)
fpp_c = fpp1 + t_cross * (fpp2 - fpp1)
fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2))
ax = axs[0]
ax.plot(branch[:, 0], branch[:, 1], color='tab:red', lw=1.8, label='branch')
ax.plot(branch[0, 0], branch[0, 1], 'o', color='tab:blue', ms=7, label=r'baseline $\lambda=0.2$')
ax.plot(lam_c, fpp_c, 's', color='k', ms=7,
        label=fr'interpolated crossing $\lambda_c\approx{lam_c:.3f}$')
ax.set_xlabel(r'$\lambda$'); ax.set_ylabel(r"$f''(0)$")
ax.set_title(r"(a) solution branch in the $(\lambda, f''(0))$ plane")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax = axs[1]
ax.plot(branch[:, 0], branch[:, 2], color='tab:red', lw=1.8)
ax.plot(lam_c, 0.0, 's', color='k', ms=7, zorder=5)
ax.axhline(0, ls=':', color='0.3', lw=0.8)
ax.set_xlabel(r'$\lambda$'); ax.set_ylabel(r'$f_\infty$')
ax.set_title(r'(b) entrainment $f_\infty$ along branch')
ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "fig5_continuation_branch")

# ---------------------------------------------------------------- Figure 6 (schematic, no data)
fig, ax = plt.subplots(figsize=(5.0, 3.4))
eta = np.linspace(0, 6, 400)
theta = np.where(eta < 0.6, eta/0.6*0.85, 0.85*np.exp(-(eta-0.6)/1.4))
ax.plot(eta, theta, color='tab:blue', lw=1.6)
ax.axhline(1.0, ls='--', color='0.5', lw=0.8)
ax.annotate('wall', xy=(0.05, 1.0), xytext=(0.3, 1.05), fontsize=8)
ax.annotate('slip-controlled drop', xy=(0.0, 0.85), xytext=(-0.05, 0.55),
            fontsize=7, ha='right',
            arrowprops=dict(arrowstyle='-', lw=0.6))
ax.plot([0], [0.85], 'o', color='k', ms=4)
ax.axvspan(0, 0.6, color=(.9, .93, .98))
ax.text(0.28, -0.08, r'inner layer, $O(\varepsilon^{1/2})$', fontsize=7, ha='center')
ax.text(3.2, -0.08, 'outer region, nearly isothermal', fontsize=7, ha='center')
ax.text(3.2, 0.5, r'$\theta(\eta)$', color='tab:blue', fontsize=9)
ax.set_xlabel(r'$\eta$'); ax.set_ylabel(r'$\theta$')
ax.set_title('Schematic of the thermal boundary layer at high Pr (not to scale)')
ax.set_ylim(-0.15, 1.15)
ax.set_xticks([]); ax.set_yticks([0, 1])
fig.tight_layout()
save(fig, "fig6_slip_schematic")

# ---------------------------------------------------------------- Figure 7 (Nusselt ceiling)
Nuinf = P['A5']*(1+P['Rd']) / P['L2']
Prl = [6.2, 10, 20, 50, 100, 200, 500, 1000, 2000]
Nv = []
for Pr in Prl:
    Pp = dict(P); Pp['Pr'] = Pr
    e, hh = (10, 0.0005) if Pr >= 200 else (15, 0.002)
    s = solve_state(Pp, e, hh)
    Nv.append(s['Nu'])
Nv = np.array(Nv); Prl = np.array(Prl)

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axs[0]
ax.semilogx(Prl, Nv, 'o-', color='tab:red', lw=1.6, ms=5)
ax.axhline(Nuinf, ls='--', color='tab:blue', lw=1.6, label=f'$A_5(1+R_d)/L_2$={Nuinf:.4f}')
ax.set_xlabel('Pr'); ax.set_ylabel('Nu')
ax.set_title(r'(a) slip-induced ceiling as Pr $\to\infty$')
ax.legend(fontsize=7, loc='lower right'); ax.grid(alpha=0.3)
ax = axs[1]
d = Nuinf - Nv
ax.loglog(Prl, d, 'o-', color='tab:red', lw=1.6, ms=5, label=r'$Nu_\infty - Nu$')
ax.loglog(Prl, d[0]*np.sqrt(Prl[0]/Prl), '--', color='0.45', lw=1.3, label=r'$Pr^{-1/2}$ reference')
ax.set_xlabel('Pr'); ax.set_ylabel('defect from ceiling')
ax.set_title('(b) approach to the ceiling')
ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both')
fig.tight_layout()
save(fig, "fig7_nusselt_ceiling")

print("\nAll 7 figures written to", OUTDIR, f"at {DPI} DPI (PNG) + vector PDF.")
