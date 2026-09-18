"""
Interfacial (Kapitza) resistance bounds on the wall heat flux.

For a wall temperature jump of Kapitza type, T(x,0) = T_w + R_K q_w with
q_w the total wall flux, the scaled thermal slip length of the similarity model is
L2 = Bi_K (A5 + Rd), where Bi_K = R_K k_f (8c/nu_f)^(1/2) is the interfacial Biot
number. Under hypothesis (iii) of Theorem 1 (S >= 0, Q <= Q_min) the manuscript
proves the series-resistance bounds

    1 / (8/(Pr A4 S) + Bi_K)  <=  Nu  <  1 / Bi_K ,

equivalently, in dimensional form,

    dT / (2/((rho c_p)_thnf |v_w|) + R_K)  <=  q_w  <  dT / R_K .

This script (i) checks the bounds and hypothesis Q <= Q_min on every computed state,
(ii) repeats the loading, shape and mixing-order sweeps at FIXED interfacial
resistance (fixed Bi_K) instead of fixed L2, (iii) converts published Kapitza
lengths into Bi_K, and (iv) draws the two interfacial figures (fig7_nusselt_ceiling, fig8_regime_map; Figures 1
and 2 of the manuscript), and (v) checks the exact
asymptotic-suction solution, the series estimate and the truncation of the reference state.
Output: verification_output/interface_bounds.txt and figures/.

Section, table, proposition and figure numbers used in this file and in the saved
logs follow the internal numbering of the code; README.md gives the mapping to the
numbering of the submitted manuscript.
"""
import sys, itertools
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cone_dynamics import baseline, ratios, solve_state, rhs

OUT = open('verification_output/interface_bounds.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s); OUT.write(s + '\n')

def qmin(P, a, b, einf, h):
    """inf over [0, einf] of Pr A4^2 f^2/(16(A5+Rd)) + A4 f'/2 along the computed profile."""
    N = max(50, round(einf/h)); hh = einf/N
    y = np.array([P['S'], P['lam'] + P['L1']*a, a, 1 + P['L2']*b, b])
    k = P['A5'] + P['Rd']
    q = P['Pr']*P['A4']**2*y[0]**2/(16*k) + P['A4']*y[1]/2
    for i in range(N):
        k1 = rhs(y, P); k2 = rhs(y+0.5*hh*k1, P); k3 = rhs(y+0.5*hh*k2, P); k4 = rhs(y+hh*k3, P)
        y = y + (hh/6)*(k1+2*k2+2*k3+k4)
        q = min(q, P['Pr']*P['A4']**2*y[0]**2/(16*k) + P['A4']*y[1]/2)
    return q

def state(P, einf=None, h=None):
    einf = P['einf'] if einf is None else einf
    h = P['h'] if h is None else h
    s = solve_state(P, einf, h)
    k = P['A5'] + P['Rd']
    Bi = P['L2']/k
    up = 1/Bi
    conv = P['Pr']*P['A4']*P['S']/8
    lo = 1/(1/conv + Bi)
    Qm = qmin(P, s['a'], s['b'], einf, h)
    return dict(Nu=s['Nu'], finf=s['finf'], Bi=Bi, upper=up, lower=lo, Lam=conv*Bi,
                Qmin=Qm, hyp=(P['Q'] <= Qm and P['S'] >= 0),
                inside=(lo <= s['Nu'] < up), th0=1+P['L2']*s['b'])

def spec(p1, p2, p3, n=4.8, order=(1, 2, 3)):
    P = baseline(); P['n'] = n
    if order != (1, 2, 3):
        keys = ['r', 'c', 'k', 's']
        src = {i: {kk: P[f'{kk}{i}'] for kk in keys} for i in (1, 2, 3)}
        loads = {1: p1, 2: p2, 3: p3}
        for slot, sp in enumerate(order, start=1):
            for kk in keys: P[f'{kk}{slot}'] = src[sp][kk]
        p1, p2, p3 = loads[order[0]], loads[order[1]], loads[order[2]]
    return ratios(P, p1, p2, p3)

P0 = ratios(baseline())
BI_REF = P0['L2']/(P0['A5']+P0['Rd'])
log(f"Reference: A4={P0['A4']:.6f} A5+Rd={P0['A5']+P0['Rd']:.6f} L2={P0['L2']} Bi_K={BI_REF:.6f}")

# ---------------------------------------------------------------- 1. Prandtl sweep
log("\n=== 1. Bounds against Pr at the reference state (settings of the ceiling table) ===")
log("   Pr        lower          Nu        upper     Lambda      Q_min   hyp  inside")
prrows = []
for Pr in [6.2, 10, 20, 50, 100, 200, 500, 1000, 2000]:
    P = dict(P0); P['Pr'] = Pr
    e, hh = (10, 0.0005) if Pr >= 200 else (15, 0.002)
    r = state(P, e, hh)
    prrows.append((Pr, r['lower'], r['Nu'], r['upper'], r['Lam']))
    log(f"{Pr:6g} {r['lower']:11.6f} {r['Nu']:11.6f} {r['upper']:11.6f} {r['Lam']:10.5f} {r['Qmin']:10.6f}  {r['hyp']!s:5} {r['inside']!s}")

# ---------------------------------------------------------------- 2. slip-length sweep
log("\n=== 2. Bounds against L2 at Pr=6.2 ===")
log("   L2     Bi_K       lower          Nu        upper   hyp inside")
for L2 in [0.05, 0.1, 0.2, 0.5, 1.0]:
    P = dict(P0); P['L2'] = L2
    r = state(P)
    log(f"{L2:5.2f} {r['Bi']:8.5f} {r['lower']:11.6f} {r['Nu']:11.6f} {r['upper']:11.6f}  {r['hyp']!s:5} {r['inside']!s}")

# ---------------------------------------------------------------- 3. loading at fixed L2 vs fixed Bi_K
log("\n=== 3. Equal loadings: fixed scaled slip L2=0.2 versus fixed interfacial resistance Bi_K ===")
log(" phi  | fixed L2:   Nu      Nu_inf  | fixed Bi_K:  L2       Nu      Nu_inf   lower   hyp inside")
loadrows = []
for phi in [0.01, 0.02, 0.06, 0.10, 0.15]:
    P = ratios(baseline(), phi, phi, phi)
    rL = state(P)
    Pb = dict(P); Pb['L2'] = BI_REF*(P['A5']+P['Rd'])
    rB = state(Pb)
    loadrows.append((phi, rL['Nu'], rL['upper'], Pb['L2'], rB['Nu'], rB['upper'], rB['lower']))
    log(f"{phi:4.2f} | {rL['Nu']:9.6f} {rL['upper']:9.4f} | {Pb['L2']:8.5f} {rB['Nu']:9.6f} {rB['upper']:9.4f} {rB['lower']:8.5f}  {rB['hyp']!s:5} {rB['inside']!s}")

# ---------------------------------------------------------------- 4. shape, composition, order at fixed Bi_K
log("\n=== 4. Dispersion specifications at fixed interfacial resistance Bi_K ===")
log(" specification                      A4        A5     Nu(fixed Bi)   upper    lower   hyp inside")
specs = [(f"n={n}", dict(p=(0.1, 0.1, 0.1), n=n)) for n in [3.0, 3.7, 4.8, 5.7, 6.0]]
specs += [(f"phi={c}", dict(p=c, n=4.8)) for c in [(0.15, 0.05, 0.05), (0.05, 0.15, 0.05), (0.05, 0.05, 0.15),
                                                   (0.02, 0.10, 0.10), (0.10, 0.02, 0.10), (0.10, 0.10, 0.02)]]
specs += [(f"order={o}", dict(p=(0.1, 0.1, 0.1), n=4.8, order=o)) for o in itertools.permutations((1, 2, 3)) if o != (1, 2, 3)]
nus = []
for name, d in specs:
    P = spec(*d['p'], n=d['n'], order=d.get('order', (1, 2, 3)))
    P['L2'] = BI_REF*(P['A5']+P['Rd'])
    r = state(P)
    nus.append(r['Nu'])
    log(f" {name:32s} {P['A4']:.6f} {P['A5']:.6f} {r['Nu']:11.6f} {r['upper']:9.4f} {r['lower']:8.5f}  {r['hyp']!s:5} {r['inside']!s}")
nus.append(state(dict(P0))['Nu'])
log(f" spread of Nu over all specifications at fixed Bi_K: {min(nus):.6f} to {max(nus):.6f} "
    f"({100*(max(nus)-min(nus))/min(nus):.2f}% of the minimum); upper bound 1/Bi_K = {1/BI_REF:.6f} for every one")

# ---------------------------------------------------------------- 5. molecular calibration
log("\n=== 5. Kapitza lengths expressed as Bi_K = l_K (8c/nu_f)^(1/2) (water side) ===")
kf, rhof, cpf = P0['kf'], P0['rf'], P0['cf']
mu_f = P0['Pr']*kf/cpf; nu_f = mu_f/rhof
rc_thnf = P0['A4']*rhof*cpf
log(f" nu_f = Pr k_f/(rho_f c_pf) = {nu_f:.4e} m^2/s ; (rho c_p)_thnf = {rc_thnf:.4e} J/m^3K")
for lab, lK in [("hydrophilic TDTR 3 nm", 3e-9), ("hydrophobic TDTR 12 nm", 12e-9),
                ("apparent microscale 4.2 um", 4.225e-6), ("apparent microchannel 150 um", 150e-6)]:
    for c in [0.1, 1.0, 10.0]:
        Bi = lK*np.sqrt(8*c/nu_f)
        log(f" {lab:30s} c={c:5.1f} 1/s: Bi_K={Bi:.3e}  L2={Bi*(P0['A5']+P0['Rd']):.3e}")
log(" Scaled slip L2=0.2 at the reference state corresponds to:")
for c in [0.1, 1.0, 10.0]:
    lK = BI_REF/np.sqrt(8*c/nu_f)
    vw = P0['S']*np.sqrt(nu_f*c/2)
    log(f"   c={c:5.1f} 1/s: l_K = {lK*1e6:.2f} um, suction |v_w| = {vw*1e3:.4f} mm/s")
log(" Interface-limited regime Lambda = l_K (rho c_p)_thnf |v_w| / (2 k_f) = 1 requires |v_w| = 2 k_f/(l_K (rho c_p)_thnf):")
for lab, lK in [("3 nm", 3e-9), ("12 nm", 12e-9), ("150 um", 150e-6)]:
    log(f"   l_K = {lab:7s}: |v_w| = {2*kf/(lK*rc_thnf):.4e} m/s")

# ---------------------------------------------------------------- 6. exact asymptotic-suction solution
log("\n=== 6. Exact asymptotic-suction solution (lambda=0, Q=0: f = S, theta = theta(0) exp(-beta eta)) ===")
log(" Nu_exact = 1/(4/(Pr A4 S) + Bi_K); solver run at eta_inf = 60, h = 0.002")
for S in [0.5, 1.0, 2.0]:
    for Pr in [6.2, 50.0]:
        P = dict(P0); P['lam'] = 0.0; P['Q'] = 0.0; P['S'] = S; P['Pr'] = Pr
        sx = solve_state(P, 60, 0.002)
        ex = 1/(4/(Pr*P['A4']*S) + BI_REF)
        log(f"   S={S:3.1f} Pr={Pr:5.1f}: f_inf={sx['finf']:.10f} Nu={sx['Nu']:.10f} exact={ex:.10f} diff={sx['Nu']-ex:.1e}")

log("\n=== 7. Series estimate 1/(4/(Pr A4 S)+Bi_K) against computed Nu at the reference state ===")
for Pr, lo, nu, up, _lam in prrows:
    est = 1/(4/(Pr*P0['A4']*P0['S']) + BI_REF)
    log(f"   Pr={Pr:6g}: estimate={est:.6f} Nu={nu:.6f} difference={(nu-est)/est*100:+.3f}%")

log("\n=== 8. Truncation check of the reference state (h = 0.005) ===")
for e in [15, 20, 30]:
    sx = solve_state(dict(P0), e, 0.005)
    log(f"   eta_inf={e:3d}: f_inf={sx['finf']:.8f} Nu={sx['Nu']:.8f}")

# ---------------------------------------------------------------- figures
plt.rcParams.update({'font.size': 9})
# ---------------------------------------------------------------- 9. saturation defect vs proved bound
log("\n=== 9. Saturation defect Nu_inf - Nu against the proved bound Nu_inf/(1+Lambda) ===")
log(" Theorem 4 gives 0 < Nu_inf - Nu <= Nu_inf/(1 + gamma L2) with gamma L2 = Lambda.")
log("   Pr      Nu_inf          Nu      defect       bound   bound/defect")
_d, _b = [], []
for Pr, lo, nu, up, lam in prrows:
    dfc = up - nu
    bnd = up/(1.0 + lam)
    _d.append(dfc); _b.append(bnd)
    log(f"{Pr:6g} {up:11.6f} {nu:11.6f} {dfc:11.6f} {bnd:11.6f} {bnd/dfc:12.3f}")
_P = np.array([r[0] for r in prrows], float); _d = np.array(_d); _b = np.array(_b)
_c, _res, _, _, _ = np.polyfit(np.log(_P), np.log(_d), 1, full=True)
_yh = np.polyval(_c, np.log(_P))
_sd = np.sqrt(((np.log(_d) - _yh)**2).sum()/(len(_P) - 2))
_se = _sd/np.sqrt(((np.log(_P) - np.log(_P).mean())**2).sum())
log(f" log-log defect exponent, all {len(_P)} points: {_c[0]:.4f}  (standard error {_se:.4f};"
    f" 95% CI +/- {1.96*_se:.4f} normal, +/- {2.365*_se:.4f} with t(7))")
_m = _P >= 200
log(f" log-log defect exponent, Pr in [200,2000]: {np.polyfit(np.log(_P[_m]), np.log(_d[_m]), 1)[0]:.4f}")
_i = list(_P).index(2000.0)
log(f" at Pr=2000: computed defect {_d[_i]:.6f}, proved bound {_b[_i]:.6f}, ratio {_b[_i]/_d[_i]:.3f}")
_j = list(_P).index(500.0)
log(f" apparent Nu exponent between Pr=500 and 2000: "
    f"{np.log(prrows[_i][2]/prrows[_j][2])/np.log(2000/500):.3f}")

pr = np.array(prrows)
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
ax[0].plot(pr[:, 0], pr[:, 2], 'o-', color='tab:red', ms=4, label='computed Nu')
ax[0].plot(pr[:, 0], pr[:, 3], '--', color='tab:blue', label=r'upper bound $1/\mathrm{Bi}_K$')
ax[0].plot(pr[:, 0], pr[:, 1], ':', color='k', lw=1.6, label=r'lower bound $[8/(\mathrm{Pr}A_4S)+\mathrm{Bi}_K]^{-1}$')
ax[0].set_xscale('log'); ax[0].set_xlabel('Pr'); ax[0].set_ylabel('Nu')
ax[0].set_title('(a) series-resistance bounds against Pr', fontsize=9)
ax[0].legend(fontsize=7, loc='lower right', frameon=False)
Prs = np.logspace(np.log10(6.2), np.log10(2000), 200)
Bi = BI_REF
ax[1].loglog(pr[:, 0], (pr[:, 3]-pr[:, 2]), 'o-', color='tab:red', ms=4, label=r'$1/\mathrm{Bi}_K-\mathrm{Nu}$')
conv = Prs*P0['A4']*P0['S']/8
ax[1].loglog(Prs, 1/Bi - 1/(1/conv+Bi), '--', color='tab:blue', label='proved bound on the defect')
ax[1].set_xlabel('Pr'); ax[1].set_ylabel('distance from the interfacial ceiling')
ax[1].set_title('(b) defect against the proved bound', fontsize=9)
ax[1].legend(fontsize=7, frameon=False)
for a_ in ax: a_.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig('figures/fig7_nusselt_ceiling.pdf'); fig.savefig('figures/fig7_nusselt_ceiling.png', dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
vw = np.logspace(-6, 2, 300)
for Lam, ls in [(0.01, ':'), (1.0, '-'), (100.0, ':')]:
    lK = 2*Lam*kf/(vw*rc_thnf)
    ax.loglog(vw, lK, ls, color='k', lw=1.3 if Lam == 1 else 0.9)
    yl = 3e-4
    ax.text(2*Lam*kf/(yl*rc_thnf)*1.3, yl*1.3, rf'$\Lambda={Lam:g}$', fontsize=7.5, rotation=-38,
            ha='left', va='bottom')
ax.axhspan(3e-9, 12e-9, color='tab:green', alpha=0.25, lw=0)
ax.text(1.5e-6, 1.6e-8, 'molecular Kapitza lengths, TDTR (3-12 nm)', fontsize=7, color='darkgreen')
ax.axhspan(4.2e-6, 1.5e-4, color='tab:orange', alpha=0.18, lw=0)
ax.text(1.5e-6, 1.0e-5, 'apparent lengths from\nmicroscale experiments (4-150 $\\mu$m)', fontsize=7, color='saddlebrown')
for c, mk in [(0.1, 's'), (1.0, 'o'), (10.0, '^')]:
    lK = BI_REF/np.sqrt(8*c/nu_f); v = P0['S']*np.sqrt(nu_f*c/2)
    ax.loglog(v, lK, mk, color='tab:red', ms=6, label=rf'$c={c:g}\,\mathrm{{s}}^{{-1}}$')
ax.fill_between(vw, 2*1.0*kf/(vw*rc_thnf), 1e-2, color='tab:blue', alpha=0.07, lw=0)
ax.text(2e0, 3e-3, 'interface-limited ' + r'($\Lambda>1$)', fontsize=7.5, color='navy', ha='center')
ax.text(1e-4, 3e-8, 'convection-limited ' + r'($\Lambda<1$)', fontsize=7.5, color='navy', ha='center')
ax.set_xlim(1e-6, 1e2); ax.set_ylim(1e-9, 1e-2)
ax.set_xlabel(r'wall suction velocity $|v_w|$ (m s$^{-1}$)')
ax.set_ylabel(r'wall Kapitza length $\ell_K$ (m)')
leg = ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False,
                title='$L_2=0.2$, $S=1$', title_fontsize=7)
ax.set_title(r'Regime map, $\Lambda=\ell_K(\rho c_p)_{thnf}|v_w|/(2k_f)$', fontsize=9)
fig.tight_layout(); fig.savefig('figures/fig8_regime_map.pdf'); fig.savefig('figures/fig8_regime_map.png', dpi=200)
plt.close(fig)
log("\nwrote figures/fig7_nusselt_ceiling.pdf and figures/fig8_regime_map.pdf")
OUT.close()
