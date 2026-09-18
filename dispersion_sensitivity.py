import sys, itertools
sys.path.insert(0,'.')
import numpy as np
from cone_dynamics import baseline, ratios, solve_state

def qc(P, finf):
    return P['Pr']*P['A4']**2*finf**2/(16*(P['A5']+P['Rd']))

def run(p1,p2,p3,n=4.8,order=(1,2,3)):
    P = baseline(); P['n']=n
    # reorder particle species if requested
    if order!=(1,2,3):
        keys=['r','c','k','s']
        src={i:{k:P[f'{k}{i}'] for k in keys} for i in (1,2,3)}
        loads={1:p1,2:p2,3:p3}
        for slot,spec in enumerate(order,start=1):
            for k in keys: P[f'{k}{slot}']=src[spec][k]
        p1,p2,p3 = loads[order[0]],loads[order[1]],loads[order[2]]
    P = ratios(P,p1,p2,p3)
    s = solve_state(P, P['einf'], P['h'])
    finf = s['finf']
    return dict(A1=P['A1'],A2=P['A2'],A3=P['A3'],A4=P['A4'],A5=P['A5'],
                finf=finf, Qc=qc(P,finf), Nuinf=(P['A5']+P['Rd'])/P['L2'])

print("=== shape factor n sweep (phi=0.10 each) ===")
for n in [3.0,3.7,4.8,5.7,6.0]:
    r=run(0.10,0.10,0.10,n=n)
    print(f"n={n:4.1f} A5={r['A5']:.6f} Nuinf={r['Nuinf']:.6f} finf={r['finf']:.6f} Qc={r['Qc']:.6f}")

print("\n=== unequal loadings (n=4.8) ===")
cases=[(0.10,0.10,0.10),(0.15,0.05,0.05),(0.05,0.15,0.05),(0.05,0.05,0.15),
       (0.02,0.10,0.10),(0.10,0.02,0.10),(0.10,0.10,0.02)]
for c in cases:
    r=run(*c)
    print(f"phi={c} A4={r['A4']:.6f} A5={r['A5']:.6f} finf={r['finf']:.6f} Qc={r['Qc']:.6f} Nuinf={r['Nuinf']:.6f}")

print("\n=== mixing-order sensitivity at phi=(0.10,0.10,0.10) ===")
for o in itertools.permutations((1,2,3)):
    r=run(0.10,0.10,0.10,order=o)
    print(f"order={o} A5={r['A5']:.6f} A4={r['A4']:.6f} finf={r['finf']:.6f} Qc={r['Qc']:.6f} Nuinf={r['Nuinf']:.6f}")

print("\n=== equal loadings including a dilute state (n=4.8), with Nu at Pr=6.2 ===")
for phi in [0.01,0.02,0.06,0.10,0.15]:
    P = ratios(baseline(),phi,phi,phi)
    s = solve_state(P, P['einf'], P['h'])
    Nuinf=(P['A5']+P['Rd'])/P['L2']
    print(f"phi_i={phi:.2f} total={3*phi:.2f} A5={P['A5']:.6f} finf={s['finf']:.6f} Qc={qc(P,s['finf']):.6f} Nuinf={Nuinf:.6f} Nu={s['Nu']:.6f} Nu/Nuinf={s['Nu']/Nuinf:.4f}")

print("\n=== thermal slip length at the reference state, Pr=6.2 ===")
for L2 in [0.05,0.1,0.2,0.5,1.0]:
    P = ratios(baseline()); P['L2']=L2
    s = solve_state(P, P['einf'], P['h'])
    Nuinf=(P['A5']+P['Rd'])/L2
    print(f"L2={L2:.2f} Nu={s['Nu']:.6f} Nuinf={Nuinf:.6f} Nu/Nuinf={s['Nu']/Nuinf:.3f}")

print("\n=== insensitivity of A3 to the particle electrical conductivities ===")
for fac in [1e-3,1e-2,1e-1,1.0]:
    P=baseline()
    for i in (1,2,3): P[f's{i}']*=fac
    P=ratios(P)
    print(f"particle sigma x {fac:g}: A3={P['A3']:.6f}")
