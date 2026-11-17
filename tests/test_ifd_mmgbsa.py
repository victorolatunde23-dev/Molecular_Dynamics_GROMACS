"""Offline tests for ifd_mmgbsa (receptor cleaning, charges, estimates, tables)."""
import json, sys, tempfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import md_pipeline as mp
import ifd_mmgbsa as im

tmp = Path(tempfile.mkdtemp())


def atom(rec, i, name, resn, ch, resi, x, y, z, el):
    nm = f" {name:<3}" if len(name) < 4 else name
    return f"{rec:<6}{i:5d} {nm} {resn:>3} {ch}{resi:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}"


# Maestro-like receptor: ACE cap, residues 1-3 continuous, gap, residues 10-11, NMA cap,
# a HIE, hydrogens, a water, a Zn ion, a cofactor, and an alt-loc B atom.
L = []
i = 1
def add(rec, name, resn, ch, resi, x, y=0.0, z=0.0, el=None):
    global i
    L.append(atom(rec, i, name, resn, ch, resi, x, y, z, el or name[0])); i += 1
add("HETATM", "CH3", "ACE", "A", 0, -2.0); add("HETATM", "C", "ACE", "A", 0, -1.5); add("HETATM", "O", "ACE", "A", 0, -1.5, 1.2)
add("HETATM", "H1", "ACE", "A", 0, -2.5, el="H")
for r, x0, rn in ((1, 0.0, "ALA"), (2, 3.8, "HIE"), (3, 7.6, "GLY"), (10, 30.0, "LEU"), (11, 33.8, "ARN")):
    add("ATOM", "N", rn, "A", r, x0); add("ATOM", "H", rn, "A", r, x0, 1.0, el="H")
    add("ATOM", "CA", rn, "A", r, x0 + 1.4); add("ATOM", "C", rn, "A", r, x0 + 2.5); add("ATOM", "O", rn, "A", r, x0 + 2.5, 1.2)
add("HETATM", "N", "NMA", "A", 12, 37.3); add("HETATM", "CA", "NMA", "A", 12, 38.7); add("HETATM", "HA1", "NMA", "A", 12, 39.0, el="H")
add("HETATM", "O", "HOH", "A", 500, 50.0); add("HETATM", "ZN", "ZN", "A", 501, 60.0, el="ZN")
add("HETATM", "C1", "NAD", "A", 502, 70.0)
L.append(atom("ATOM", i, "CB", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")[:16] + "B" + atom("ATOM", i, "CB", "ALA", "A", 1, 0, 0, 0, "C")[17:])
# second chain
add("ATOM", "N", "GLY", "B", 1, 0.0, 20.0); add("ATOM", "CA", "GLY", "B", 1, 1.4, 20.0)
src = tmp / "rec.pdb"
src.write_text("\n".join(L) + "\nEND\n")
rep = im.clean_receptor_pdb(src, tmp / "clean.pdb")
out = (tmp / "clean.pdb").read_text().splitlines()
print("\n".join(out)); print(rep)
body = [l for l in out if l.startswith("ATOM")]
assert not any(l[76:78].strip() == "H" for l in body), "hydrogens must be removed"
assert {l[17:20] for l in body} == {"ACE", "ALA", "HIE", "GLY", "LEU", "ARG", "NME"}, {l[17:20] for l in body}
assert any(l[17:20] == "NME" and l[12:16].strip() == "CH3" for l in body), "NMA CA -> NME CH3"
assert rep["dropped"] == {"HOH": 1, "ZN": 1, "NAD": 1}, rep["dropped"]
assert len(rep["gaps"]) == 1 and rep["gaps"][0].startswith("A3-A10"), rep["gaps"]
# TER between residue 3 and 10, between chain A and B, and at the end
ters = [k for k, l in enumerate(out) if l == "TER"]
assert len(ters) == 3, ters
assert out[ters[0] - 1][22:26].strip() == "3" and out[ters[0] + 1][22:26].strip() == "10"
assert out[ters[1] + 1][21] == "B"
assert not any(l[12:16].strip() == "CB" for l in body), "alt-loc B atom dropped"
assert all(len(l) >= 78 for l in body)
# columns parse like a PDB reader would
for l in body:
    float(l[30:38]); float(l[38:46]); float(l[46:54]); int(l[22:26])
print("PASS receptor cleaning")

# SDF charges
sdf_v2 = """lig
  RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
M  CHG  2   1   1   3  -1
M  END
$$$$
"""
(tmp / "a.sdf").write_text(sdf_v2)
assert im.sdf_formal_charge(tmp / "a.sdf") == 0
(tmp / "b.sdf").write_text(sdf_v2.replace("M  CHG  2   1   1   3  -1", "M  CHG  1   1   1"))
assert im.sdf_formal_charge(tmp / "b.sdf") == 1
(tmp / "c.sdf").write_text(sdf_v2.replace("M  CHG  2   1   1   3  -1\n", "").replace(
    "2.0000    0.0000    0.0000 O   0  0", "2.0000    0.0000    0.0000 O   0  5"))
assert im.sdf_formal_charge(tmp / "c.sdf") == -1
print("PASS SDF formal charge")

# manual folder route
for p in ("p1", "p2"):
    d = tmp / "manual" / p; d.mkdir(parents=True)
    (d / "receptor.pdb").write_text("END\n"); (d / "ligand.sdf").write_text(sdf_v2)
(tmp / "manual" / "junk").mkdir()
poses = im.poses_from_folder(tmp / "manual")
assert [p["pose"] for p in poses] == ["p1", "p2"] and poses[0]["ligand_charge"] == 0
cfg = im.pose_config(mp.Config(output_root=str(tmp / "res")), "proj", poses[0])
assert cfg.name == "proj/p1" and cfg.ligand_file.endswith("ligand.sdf") and cfg.ligand_resname is None
assert cfg.pdb_id is None and cfg.has_ligand and cfg.fix_protein is False
print("PASS manual poses + config")

assert im.safe_name("lig A (pose #3)") == "lig_A_pose_3"

# estimates
q = im.estimate_hours(10, "quick", 0)
m = im.estimate_hours(10, "md", 5, ns_per_day=100)
print(f"   10 poses: quick {q:.1f} h, md 5 ns {m:.1f} h")
assert 2 < q < 4 and 15 < m < 25
print("PASS estimates")

# summary + plot
res = [{"pose": "pose01_a", "dG_bind_kcal": -30.1, "dG_sd_kcal": 3.0, "ifd_score": -650.2, "minutes": 80},
       {"pose": "pose02_a", "dG_bind_kcal": -42.0, "dG_sd_kcal": 2.0, "ifd_score": -651.0, "minutes": 75},
       {"pose": "pose03_a", "error": "boom", "ifd_score": -649.0}]
df = im.summary_table(res)
print(df.to_string())
assert list(df["pose"]) == ["pose02_a", "pose01_a", "pose03_a"] and df["rank"].tolist() == [1, 2, 3]
im.plot_vs_scores(df, tmp / "p.png"); assert (tmp / "p.png").exists()
print("PASS summary table + plot")
# PBC fix: the commands sent to GROMACS
class FakeR:
    def __init__(self): self.calls = []
    def gmx(self, args, step, stdin=None, **kw): self.calls.append(([str(a) for a in args], stdin))
groups = ["System", "Protein", "Protein-H", "C-alpha", "Backbone", "LIG", "Complex"]
fr = FakeR()
mp.cluster_complex(fr, "md.tpr", "md.xtc", "index.ndx", tmp / "c.xtc", groups, "t")
(a1, s1), (a2, s2) = fr.calls
assert a1[:1] == ["trjconv"] and a1[a1.index("-pbc") + 1] == "whole" and s1 == "0\n"
assert a2[a2.index("-pbc") + 1] == "cluster" and "-center" in a2 and "mol" not in a2
assert a2[a2.index("-f") + 1] == a1[a1.index("-o") + 1], "cluster step reads the whole-molecule output"
assert s2 == "6\n6\n0\n", s2          # cluster Complex, centre Complex, output System
assert im.RESULT_TAG == "_v3" and im.MMGBSA_TAG == "_v2"
print("PASS PBC clustering commands")
# analyse(): fit and RMSD use the clustered first-frame reference, not md.tpr
class FakeR2(FakeR):
    def gmx(self, args, step, stdin=None, check=True, cwd=None, **kw):
        self.calls.append(([str(a) for a in args], stdin, str(cwd) if cwd else None)); return 0
acfg = mp.Config(name="ana", output_root=str(tmp / "res"))
AP = mp.Paths(acfg); AP.make()
orig_make_index = mp.make_index
mp.make_index = lambda cfg, P, R, tpr: groups
try:
    fr2 = FakeR2(); mp.analyse(acfg, AP, fr2)
finally:
    mp.make_index = orig_make_index
cmds = [c[0] for c in fr2.calls]
ref = str(AP.ana / "md_ref.tpr")
i_gpp = next(i for i, c in enumerate(cmds) if c[0] == "grompp")
assert cmds[i_gpp][cmds[i_gpp].index("-o") + 1] == ref and fr2.calls[i_gpp][2] == str(AP.sys)
fit = next(c for c in cmds if "-fit" in c)
assert fit[fit.index("-s") + 1] == ref, "fit must use the clustered reference"
for c in cmds:
    if c[0] in ("rms", "rmsf", "gyrate", "mindist", "hbond"):
        assert c[c.index("-s") + 1] == ref, c
assert (AP.ana / "md_fit.ref_v2").exists()
print("PASS analysis uses clustered reference")
print("\nALL TESTS PASSED")
