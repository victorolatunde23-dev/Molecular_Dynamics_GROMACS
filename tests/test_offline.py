"""Offline tests for md_advanced: everything that does not need GROMACS."""
import json, math, sys, tempfile, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import md_pipeline as mp
import md_advanced as ma

tmp = Path(tempfile.mkdtemp())
ok = lambda m: print("PASS", m)

# ---------- gro reading, PBC helpers ----------
def gro_line(resnr, resname, name, idx, x, y, z):
    return f"{resnr:5d}{resname:<5}{name:>5}{idx:5d}{x:8.3f}{y:8.3f}{z:8.3f}"

atoms = []
i = 1
for r in range(1, 6):           # 5 residues, CA spread along x, one split across box
    for nm, dx in (("N", -0.1), ("CA", 0.0), ("C", 0.1)):
        x = (r * 1.0 + dx) % 6.0
        atoms.append(gro_line(r, "ALA", nm, i, x, 3.0, 3.0)); i += 1
for k in range(6):
    atoms.append(gro_line(6, "LIG", f"C{k+1}", i, 5.9 + 0.01 * k, 3.0, 3.0)); i += 1
atoms.append(gro_line(7, "SOL", "OW", i, 1.0, 1.0, 1.0)); i += 1
gro = tmp / "t.gro"
gro.write_text("t\n%5d\n" % len(atoms) + "\n".join(atoms) + "\n   6.00000   6.00000   6.00000\n")
A, box = ma.read_gro(gro)
assert len(A) == 22 and A[0]["name"] == "N" and A[15]["resname"] == "LIG" and box == [6, 6, 6]
assert len(ma.protein_atoms(A)) == 15
v = ma.min_image([5.0, 0, 0], box); assert abs(v[0] + 1.0) < 1e-9
c = ma.centre_pbc([(5.9, 0, 0), (0.1, 0, 0)], box); assert abs(c[0] - 6.0) < 1e-9 or abs(c[0]) < 1e-9
c = ma.centre_pbc([(x % 6.0, 0, 0) for x in np.arange(1.0, 5.6, 0.38)], box)
assert abs(c[0] - np.arange(1.0, 5.6, 0.38).mean()) < 1e-9   # chain longer than half the box
# triclinic dodecahedron box
dod = [6.0, 6.0, 4.2426, 0, 0, 0, 3.0, 3.0, -3.0]
M = ma.box_matrix(dod); assert M[2] == [3.0, -3.0, 4.2426]
vv = ma.min_image([M[2][0] + 0.1, M[2][1], M[2][2]], dod); assert abs(vv[0] - 0.1) < 1e-9 and abs(vv[2]) < 1e-9
ok("gro / minimum image / triclinic")

g = ma.pull_geometry(gro)
assert g["ref"] == [2, 5, 8, 11, 14] and g["lig"] == list(range(16, 22))
assert abs(math.hypot(*g["vec"]) - 1) < 1e-9
assert abs(g["vec"][0] - 1) < 1e-9 and abs(g["start_nm"] - 2.925) < 1e-6, g
print("   pull vec", [round(x, 3) for x in g["vec"]], "start", round(g["start_nm"], 3), "pbcatom", g["pbcatom"], "half box", g["half_box_nm"])
ok("pull geometry")

# ---------- ndx ----------
ma.write_ndx(tmp / "a.ndx", {"System": list(range(1, 40)), "Pull_lig": [16, 17]})
G = ma.read_ndx(tmp / "a.ndx"); assert G["System"] == list(range(1, 40)) and G["Pull_lig"] == [16, 17]
ok("ndx round trip")

# ---------- mdp helpers ----------
cfg = mp.Config(output_root=str(tmp / "res"))
m = mp._dyn(cfg, 1000, 500, True, True, False)
m2 = ma.set_define(m, "-DPOSRES_CA"); assert "define              = -DPOSRES_CA" in m2 and "-DPOSRES_LIG" not in m2
m3 = ma.set_mdp(m2, "gen_seed", 2001); assert re.search(r"^gen_seed\s+= 2001$", m3, re.M) and m3.count("gen_seed") == 1
pm = ma._pull_mdp(cfg, 1000, g, 0.005, 1000, 2001)
for key in ("pull                    = yes", "pull-coord1-geometry    = direction", "gen_vel             = yes",
            "continuation        = no", "pull-coord1-rate        = 0.005", "refcoord_scaling    = com"):
    assert key in pm, key
sd = ma._sd_mdp(cfg, 1000, True, True)
assert "integrator          = sd" in sd and "tcoupl              = no" in sd and "tau_t               = 1.0" in sd
assert sd.count("integrator") == 1 and sd.count("tcoupl") == 1
ok("mdp editing, pull and sd mdp")

c, vdw = ma.lambda_schedule()
assert len(c) == len(vdw) == 19 and c[:5] == [0, .25, .5, .75, 1] and vdw[:5] == [0] * 5 and vdw[-1] == 1 and c[-1] == 1
fb = ma.fe_block(3, c, vdw, "LIG")
assert "init-lambda-state   = 3" in fb and "couple-moltype      = LIG" in fb and len(fb.split("coul-lambdas")[1].split("\n")[0].split()) == 20
ok("lambda schedule + FE block")
print(ma.hydration_topology(cfg, "LIG"))

# ---------- CA position restraints on a pdb2gmx-style topology ----------
sysd = tmp / "sys"; sysd.mkdir()
(sysd / "topol.top").write_text("""; Include forcefield parameters
#include "amber99sb-ildn.ff/forcefield.itp"

[ moleculetype ]
; Name            nrexcl
Protein_chain_A     3

[ atoms ]
;   nr       type  resnr residue  atom   cgnr     charge       mass  typeB    chargeB      massB
     1         N3      1    MET      N      1     0.1592      14.01
     2          H      1    MET     H1      2     0.1984      1.008
     3         CT      1    MET     CA      3     0.0221      12.01
     4          N      2    ASN      N      4    -0.4157      14.01
     5         CT      2    ASN     CA      5     0.0143      12.01

[ bonds ]
    1     2     1

; Include Position restraint file
#ifdef POSRES
#include "posre.itp"
#endif

; Include ligand topology
#include "LIG.itp"
#ifdef POSRES_LIG
#include "posre_LIG.itp"
#endif

; Include water topology
#include "amber99sb-ildn.ff/tip3p.itp"

[ system ]
Protein

[ molecules ]
Protein_chain_A     1
LIG                 1
""")
w = ma.add_ca_posres(sysd, 500)
t = (sysd / "topol.top").read_text()
assert '#ifdef POSRES_CA\n#include "posre_CA.itp"\n#endif' in t
assert t.index("POSRES_CA") < t.index("Include ligand topology")
assert (sysd / "posre_CA.itp").read_text().count("  500  500  500") == 2
ma.add_ca_posres(sysd, 500); assert (sysd / "topol.top").read_text().count("POSRES_CA") == 1
ok("CA restraints (single chain, idempotent)")
# multi-chain style
sys2 = tmp / "sys2"; sys2.mkdir()
(sys2 / "topol.top").write_text('#include "amber99sb-ildn.ff/forcefield.itp"\n#include "topol_Protein_chain_A.itp"\n#include "topol_Protein_chain_B.itp"\n[ system ]\nx\n[ molecules ]\nProtein_chain_A 1\nProtein_chain_B 1\n')
for ch in "AB":
    (sys2 / f"topol_Protein_chain_{ch}.itp").write_text(f"[ moleculetype ]\nProtein_chain_{ch} 3\n\n[ atoms ]\n     1 N3 1 MET N 1 0.1 14\n     2 CT 1 MET CA 2 0.1 12\n\n#ifdef POSRES\n#include \"posre_Protein_chain_{ch}.itp\"\n#endif\n")
w2 = ma.add_ca_posres(sys2)
assert sorted(p.name for p in w2) == ["posre_Protein_chain_A_CA.itp", "posre_Protein_chain_B_CA.itp"]
assert '#ifdef POSRES_CA' in (sys2 / "topol_Protein_chain_B.itp").read_text()
ok("CA restraints (multi chain)")

# ---------- statistics ----------
rng = np.random.default_rng(0)
x = np.zeros(4096)
for k in range(1, len(x)):
    x[k] = 0.95 * x[k - 1] + rng.normal()
sizes, sems = ma.block_average(x)
tau = ma.autocorr_time(x)
naive = x.std(ddof=1) / math.sqrt(len(x))
print(f"   AR(1) phi=.95: tau={tau:.1f} (theory 39), naive SEM {naive:.3f}, block plateau {sems[-3]:.3f}")
assert 25 < tau < 55 and sems[-3] > 3 * naive
ok("block averaging + autocorrelation")

# Kabsch
P0 = rng.normal(size=(30, 3))
th = 0.7
Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
Q = P0 @ Rz.T + np.array([1, 2, 3])
assert np.allclose(ma._kabsch(Q, P0), P0, atol=1e-8)
Qm = -Q  # reflection must not be "fixed" by a reflection
assert not np.allclose(ma._kabsch(Qm, P0), P0, atol=1e-3)
ok("Kabsch alignment (proper rotation only)")

# Jarzynski: Gaussian work -> exact answer mean - beta var/2
T = 300; beta = 1 / (ma.KB_KJ * T)
W = rng.normal(50, 3, size=(200000, 1))
jar, cum = ma.jarzynski(W, T)
exact = 50 - beta * 9 / 2
print(f"   Jarzynski {jar[0]:.3f}, cumulant {cum[0]:.3f}, exact {exact:.3f}")
assert abs(jar[0] - exact) < 0.2 and abs(cum[0] - exact) < 0.05
ok("Jarzynski")

# window picking
t = np.arange(0, 400.01, 0.1)
xs = 1.3 + 0.005 * t + rng.normal(0, 0.01, len(t))
picks = ma.choose_windows(t, xs, 0.1, 1.0)
assert all(abs(p[0] - round(p[0])) < 1e-6 for p in picks)
d = np.diff([p[1] for p in picks]); assert (d > 0.05).all() and len(picks) >= 18
print("   windows:", len(picks), "first", picks[0], "last", picks[-1])
ok("umbrella window choice")

# ---------- PLUMED text ----------
pl = ma.dipeptide_plumed([5, 7, 9, 15], [7, 9, 15, 17])
assert "phi: TORSION ATOMS=5,7,9,15" in pl and "metad: METAD ARG=phi,psi" in pl and "PRINT ARG=phi,psi,metad.bias" in pl
assert "METAD" not in ma.dipeptide_plumed([1, 2, 3, 4], [2, 3, 4, 5], metad=False)
lp = ma.ligand_plumed(2600, [2601, 2602, 2603], [10, 20, 30])
assert "WHOLEMOLECULES ENTITY0=1-2600 ENTITY1=2601-2603" in lp and "GRID_BIN=350" in lp
print(lp)
ok("PLUMED inputs")

# dipeptide torsions from a gro
dl = []
names = [(1, "ACE", ["CH3", "C", "O"]), (2, "ALA", ["N", "H", "CA", "HA", "CB", "C", "O"]), (3, "NME", ["N", "H", "CH3"])]
k = 1
for r, rn, ns in names:
    for n in ns:
        dl.append(gro_line(r, rn, n, k, 0.1 * k, 0, 0)); k += 1
(tmp / "d.gro").write_text("d\n%5d\n" % len(dl) + "\n".join(dl) + "\n 3 3 3\n")
phi, psi = ma.dipeptide_torsions(tmp / "d.gro")
assert phi == [2, 4, 6, 9] and psi == [4, 6, 9, 11], (phi, psi)
ok("dipeptide torsion atoms")

# colvar + fes parsing + basin dF
(tmp / "COLVAR").write_text("#! FIELDS time phi psi metad.bias\n#! SET min_phi -pi\n0 -1.5 2.5 0\n1 -1.4 2.6 0.5\n")
cv = ma.read_colvar(tmp / "COLVAR"); assert list(cv) == ["time", "phi", "psi", "metad.bias"] and cv["phi"][1] == -1.4
gx = np.linspace(-math.pi, math.pi, 41)
rows = []
for yy in gx:
    for xx in gx:
        F = 0.0 if xx < 0 else 5.0
        rows.append((xx, yy, F, 0, 0))
np.savetxt(tmp / "fes.dat", rows, header="FIELDS phi psi file.free der_phi der_psi")
X, Y, F = ma.read_fes2d(tmp / "fes.dat")
assert F.shape == (41, 41) and not np.isnan(F).any()
dF = ma.basin_dF(X, Y, F, 300)
kT = ma.KB_KJ * 300
wpos = (X > 0).sum() * math.exp(-5 / kT)
wneg = (X < 0).sum() + (X == 0).sum() * math.exp(-5 / kT)
expect = -kT * math.log(wpos / wneg)
print(f"   basin dF {dF:.3f} expected {expect:.3f}")
assert abs(dF - expect) < 1e-6
ok("COLVAR / FES parsing / basin free energy")

# ---------- MM/PBSA output parsing ----------
(tmp / "FINAL_RESULTS_MMPBSA.dat").write_text("""| Run on Mon
GENERALIZED BORN:

Complex:
Energy Component       Average     SD(Prop.)         SD   SEM(Prop.)        SEM
-------------------------------------------------------------------------------
ΔBOND                    0.00         0.00       0.00         0.00       0.00
Delta (Complex - Receptor - Ligand):
Energy Component       Average     SD(Prop.)         SD   SEM(Prop.)        SEM
ΔVDWAALS               -20.37         1.75       1.75         0.18       0.18
ΔTOTAL                 -15.12         2.30       1.90         0.23       0.19

POISSON BOLTZMANN:
DELTA TOTAL            -10.50         3.10       2.80         0.30       0.28
""")
r = ma.parse_mmpbsa(tmp / "FINAL_RESULTS_MMPBSA.dat")
assert r["dG_bind_GB"]["mean"] == -15.12 and r["dG_bind_GB"]["sd"] == 1.90 and r["dG_bind_PB"]["mean"] == -10.50
ok("MM/PBSA result parsing")
(tmp / "dec.dat").write_text("""Energy Decomposition Analysis (All units kcal/mol): Generalized Born solvent

DELTAS:
Total Energy Decomposition:
Residue  |  Location  |  Internal  |  van der Waals  |  Electrostatic  |  Polar Solvation  |  Non-Polar Solv.  |  TOTAL
R:A:LEU:84 | R LEU 84 | 0.000 +/- 0.000 | -1.20 +/- 0.30 | -0.10 +/- 0.05 | 0.20 +/- 0.05 | -0.10 +/- 0.01 | -1.20 +/- 0.31
R:A:LEU:99 | R ALA 99 | 0.000 +/- 0.000 | -0.80 +/- 0.20 | 0.00 +/- 0.05 | 0.10 +/- 0.05 | -0.05 +/- 0.01 | -0.75 +/- 0.21

""")
dec = ma.parse_decomp(tmp / "dec.dat")
assert [d["residue"] for d in dec] == ["R:A:LEU:84", "R:A:LEU:99"] and dec[0]["total"] == -1.20 and dec[1]["sd"] == 0.21
ma.plot_decomp(dec, tmp / "dec.png"); assert (tmp / "dec.png").exists()
ok("decomposition parsing + plot")

# ---------- reuse_inputs / replica config ----------
base = mp.Config(name="sysA", output_root=str(tmp / "res"))
Pa = mp.Paths(base); Pa.make()
(Pa.inp / "input.pdb").write_text("X"); (Pa.lig / "molname.txt").write_text("LIG"); (Pa.sys / "LIG.itp").write_text("x")
big = ma.smd_config(base)
Pb = mp.Paths(big)
assert big.name == "sysA_bigbox" and big.box_distance == 2.0 and (Pb.inp / "input.pdb").exists() and (Pb.sys / "LIG.itp").exists() and (Pb.lig / "molname.txt").exists()
rc = ma.replica_config(base, 2)
assert mp.Paths(rc).root == mp.Paths(base).root / "L2_replicas" / "rep2" and rc.seed == 1002
dc = ma.dipeptide_config(base)
assert dc.ligand_resname is None and dc.pdb_id is None and not dc.has_ligand and dc.fix_protein is False
ok("configs for replicas / big box / dipeptide")

# ---------- compare_replicas end-to-end on fake analysis outputs ----------
cfgs = [ma.replica_config(base, i) for i in (1, 2, 3)]
for j, cc in enumerate(cfgs):
    P = mp.Paths(cc); P.make(); (P.ana / "csv").mkdir(exist_ok=True)
    s = {}
    for key, _ in ma.METRICS:
        tt = np.arange(0, 10.01, 0.01)
        yy = 0.1 + 0.01 * j + 0.02 * rng.normal(size=len(tt))
        np.savetxt(P.ana / "csv" / f"{key}.csv", np.c_[tt, yy], delimiter=",", header="Time (ns),y", comments="")
        s[key] = {"mean_second_half": float(yy[len(yy)//2:].mean())}
    (P.ana / "summary.json").write_text(json.dumps(s))
table, acf = ma.compare_replicas(cfgs)
print(table.round(4).to_string()); print(acf.to_string())
assert len(table) == len(ma.METRICS) and (table["n"] == 3).all()
out = mp.Paths(base).root / "L2_replicas" / "comparison"
assert (out / "replica_summary.csv").exists() and (out / "block_average_rmsd_ligand.png").exists()
ok("compare_replicas")

# ---------- notebook-facing: xvg reading via mp.read_xvg for pullx-like file ----------
(tmp / "pullx.xvg").write_text('# x\n@    title "Pull COM"\n@    xaxis  label "Time (ps)"\n@ s0 legend "1"\n0.0 1.30\n0.1 1.31\n')
a = ma.xvg_array(tmp / "pullx.xvg"); assert a.shape == (2, 2)
ok("xvg reading")
print("\nALL TESTS PASSED")
