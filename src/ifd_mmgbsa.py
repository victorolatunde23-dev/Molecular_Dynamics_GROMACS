"""
ifd_mmgbsa.py  -  MM/GBSA in GROMACS (gmx_MMPBSA) for Schrodinger Induced Fit Docking poses
==========================================================================================

Input  : the IFD output file (e.g. `ifd-out.maegz`), read with RDKit, or per-pose files
         exported from Maestro (receptor .pdb + ligand .sdf with hydrogens).
Per pose:
  1. split receptor / ligand; keep the ligand's bond orders, hydrogens and formal charge
  2. clean the receptor for AMBER/GROMACS (caps, protonation names, gaps, drop waters/ions)
  3. build the solvated system with md_pipeline (AMBER99SB-ILDN, GAFF2/AM1-BCC, TIP3P)
  4. mode "quick": energy minimisation only, MM/GBSA on that one structure
                   (closest to Prime MM-GBSA in Maestro)
     mode "md"   : minimisation, NVT/NPT, short production MD, MM/GBSA over frames
                   (+ ligand RMSD to check the docked pose stays put)
  5. collect dG(bind) with the IFD / Glide scores in one table

Results: <output_root>/<project>/<pose>/... and <output_root>/<project>/mmgbsa_summary.csv
"""
from __future__ import annotations

import gzip
import json
import math
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path

import md_pipeline as mp
from md_pipeline import LIG, Config, Paths, Runner, banner

AMINO = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
         "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
         # protonation / disulfide variants that AMBER force fields in GROMACS understand
         "HID", "HIE", "HIP", "ASH", "GLH", "LYN", "CYX", "CYM"}
RENAME = {"ARN": "ARG", "HSD": "HID", "HSE": "HIE", "HSP": "HIP", "NMA": "NME"}
CAPS = {"ACE", "NME"}
WATER_ION = {"HOH", "WAT", "SOL", "TIP3", "T3P", "SPC", "NA", "CL", "K", "MG", "CA", "ZN", "MN",
             "Na+", "Cl-", "NA+", "CL-"}
SCORE_KEYS = {"r_psp_IFDScore": "ifd_score", "r_i_docking_score": "docking_score",
              "r_i_glide_gscore": "glide_gscore", "r_psp_Prime_Energy": "prime_energy",
              "r_i_glide_emodel": "glide_emodel"}


# =========================================================================== #
# Receptor cleaning
# =========================================================================== #
def _fmt_atom(rec, serial, name, resn, chain, resi, icode, x, y, z, elem=""):
    name = name.strip()
    nm = f" {name:<3}" if len(name) < 4 else name[:4]
    return (f"{rec:<6}{serial % 100000:5d} {nm} {resn:>3} {chain or 'A'}{resi % 10000:4d}{icode or ' '}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}")


def clean_receptor_pdb(src: Path, dst: Path, gap_cutoff: float = 2.0) -> dict:
    """Make a Maestro/IFD receptor PDB digestible for pdb2gmx (AMBER):
    - keeps standard amino acids (incl. HID/HIE/HIP, ASH, GLH, LYN, CYX) and ACE/NME caps
    - renames Maestro names (NMA -> NME, ARN -> ARG, ...) and the NME methyl carbon to CH3
    - drops hydrogens (pdb2gmx rebuilds them), waters, ions and other HETATM residues
    - puts TER records at chain changes and at gaps (C...N > gap_cutoff A), so pdb2gmx does not
      bond across missing loops
    Returns a report of what was dropped and where gaps were found."""
    residues, order = {}, []
    dropped = {}
    for line in Path(src).read_text(errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) > 16 and line[16] not in (" ", "A"):          # alternate locations
            continue
        resn = line[17:21].strip().upper()
        resn = RENAME.get(resn, resn)
        name = line[12:16].strip()
        elem = line[76:78].strip().upper() if len(line) >= 78 else ""
        if (elem == "H") or (not elem and re.match(r"^\d*H", name)):
            continue
        if resn not in AMINO and resn not in CAPS:
            dropped[resn] = dropped.get(resn, 0) + 1
            continue
        if resn == "NME" and name in ("CA", "C", "CT", "CM"):
            name = "CH3"
        key = (line[21], int(line[22:26]), line[26])
        if key not in residues:
            residues[key] = {"resn": resn, "atoms": []}
            order.append(key)
        residues[key]["atoms"].append((name, float(line[30:38]), float(line[38:46]), float(line[46:54]),
                                       elem or name[0]))
    out, gaps, serial = [], [], 1
    prev = None
    for key in order:
        r = residues[key]
        if prev is not None:
            pr = residues[prev]
            brk = key[0] != prev[0]
            if not brk and r["resn"] not in CAPS and pr["resn"] not in CAPS:
                c = next((a for a in pr["atoms"] if a[0] == "C"), None)
                n = next((a for a in r["atoms"] if a[0] == "N"), None)
                if c and n and math.dist(c[1:4], n[1:4]) > gap_cutoff:
                    brk = True
                    gaps.append(f"{prev[0]}{prev[1]}-{key[0]}{key[1]} ({math.dist(c[1:4], n[1:4]):.1f} A)")
            if brk:
                out.append("TER")
        for name, x, y, z, el in r["atoms"]:
            out.append(_fmt_atom("ATOM", serial, name, r["resn"], key[0], key[1], key[2], x, y, z, el))
            serial += 1
        prev = key
    out += ["TER", "END"]
    Path(dst).write_text("\n".join(out) + "\n")
    rep = {"residues": len(order), "heavy_atoms": serial - 1, "gaps": gaps, "dropped": dropped,
           "caps": sorted({residues[k]["resn"] for k in order} & CAPS)}
    return rep


# =========================================================================== #
# Reading IFD output
# =========================================================================== #
def _resinfo(atom):
    """(resname, resnum, chain) from RDKit residue info or the Maestro atom properties."""
    ri = atom.GetPDBResidueInfo()
    if ri is not None and ri.GetResidueName().strip():
        return ri.GetResidueName().strip().upper(), ri.GetResidueNumber(), ri.GetChainId()
    g = lambda k, d: atom.GetProp(k) if atom.HasProp(k) else d
    return (str(g("s_m_pdb_residue_name", "UNK")).strip().upper(), int(g("i_m_residue_number", 0)),
            str(g("s_m_chain_name", " ")))


def _ensure_pdb_info(mol):
    from rdkit import Chem
    for a in mol.GetAtoms():
        if a.GetPDBResidueInfo() is None:
            g = lambda k, d: a.GetProp(k) if a.HasProp(k) else d
            info = Chem.AtomPDBResidueInfo()
            info.SetName(str(g("s_m_pdb_atom_name", a.GetSymbol())).rjust(4)[:4])
            info.SetResidueName(str(g("s_m_pdb_residue_name", "UNK")).strip()[:3])
            info.SetResidueNumber(int(g("i_m_residue_number", 0)))
            info.SetChainId(str(g("s_m_chain_name", "A"))[:1] or "A")
            info.SetInsertionCode(str(g("s_m_insertion_code", " "))[:1] or " ")
            info.SetIsHeteroAtom(False)
            a.SetMonomerInfo(info)


def find_ligand_fragment(mol, ligand_resname: str | None = None):
    """Atom indices of the docked ligand: the fragment whose residue name is `ligand_resname`
    or, if not given, the largest fragment that is not protein, water or ion."""
    from rdkit import Chem
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    best, candidates = None, []
    for f in frags:
        names = {_resinfo(mol.GetAtomWithIdx(i))[0] for i in f}
        if ligand_resname:
            if ligand_resname.upper() in names:
                return list(f)
            continue
        if names & (AMINO | CAPS | set(RENAME)) or names <= WATER_ION:
            continue
        heavy = sum(1 for i in f if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
        if heavy >= 4:
            candidates.append("/".join(sorted(names)))
            if best is None or len(f) > len(best):
                best = f
    if len(candidates) > 1:
        print(f"    WARNING: several non-protein molecules {candidates}; using the largest. "
              "If that is a cofactor, pass ligand_resname=... to pick the docked ligand")
    return list(best) if best else None


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")[:40] or "pose"


def extract_ifd_poses(mae_path: str, outdir: str, ligand_resname: str | None = None,
                      max_poses: int | None = None) -> list[dict]:
    """Split every IFD pose (one complex per Maestro entry) into receptor.pdb + ligand.sdf.
    Returns one dict per pose with file paths, ligand charge and IFD/Glide scores."""
    from rdkit import Chem
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fh = gzip.open(mae_path) if str(mae_path).endswith("gz") else open(mae_path, "rb")
    sup = Chem.MaeMolSupplier(fh, sanitize=False, removeHs=False)
    poses = []
    for k, mol in enumerate(sup, start=1):
        if mol is None:
            print(f"    entry {k}: could not be read - skipped")
            continue
        mol.UpdatePropertyCache(strict=False)
        title = mol.GetProp("s_m_title") if mol.HasProp("s_m_title") else f"entry{k}"
        lig_idx = find_ligand_fragment(mol, ligand_resname)
        if not lig_idx:
            print(f"    entry {k} ({title}): no ligand found - skipped (receptor-only entry?)")
            continue
        name = f"pose{len(poses) + 1:02d}_{safe_name(title)}"
        d = outdir / name
        d.mkdir(exist_ok=True)
        # ligand: keep bond orders, hydrogens, formal charges
        lig = Chem.RWMol(mol)
        for i in sorted(set(range(mol.GetNumAtoms())) - set(lig_idx), reverse=True):
            lig.RemoveAtom(i)
        lig = lig.GetMol()
        try:
            Chem.SanitizeMol(lig)
        except Exception as e:
            print(f"    {name}: ligand sanitisation warning ({e}); writing anyway")
        if not any(a.GetAtomicNum() == 1 for a in lig.GetAtoms()):
            print(f"    {name}: WARNING ligand has no hydrogens - adding them (check protonation)")
            lig = Chem.AddHs(lig, addCoords=True)
        charge = Chem.GetFormalCharge(lig)
        lig.SetProp("_Name", name)
        w = Chem.SDWriter(str(d / "ligand.sdf"))
        w.write(lig)
        w.close()
        # receptor: everything except ligand; waters/ions removed later by clean_receptor_pdb
        rec = Chem.RWMol(mol)
        for i in sorted(lig_idx, reverse=True):
            rec.RemoveAtom(i)
        rec = rec.GetMol()
        _ensure_pdb_info(rec)
        (d / "receptor_raw.pdb").write_text(Chem.MolToPDBBlock(rec, flavor=2))   # 2 = no CONECT records
        scores = {v: float(mol.GetProp(k)) for k, v in SCORE_KEYS.items() if mol.HasProp(k)}
        lig_res = _resinfo(mol.GetAtomWithIdx(lig_idx[0]))[0]
        poses.append({"pose": name, "title": title, "entry": k, "dir": str(d),
                      "receptor": str(d / "receptor_raw.pdb"), "ligand": str(d / "ligand.sdf"),
                      "ligand_resname_in_file": lig_res, "ligand_atoms": len(lig_idx),
                      "ligand_charge": charge, **scores})
        print(f"    {name}: ligand {lig_res} ({len(lig_idx)} atoms, charge {charge:+d})"
              + "".join(f", {k} {v:.2f}" for k, v in scores.items()))
        if max_poses and len(poses) >= max_poses:
            break
    fh.close()
    (outdir / "poses.json").write_text(json.dumps(poses, indent=2))
    if not poses:
        raise RuntimeError("No poses with a ligand found. Check the file, or pass ligand_resname=...")
    return poses


def poses_from_folder(folder: str) -> list[dict]:
    """Manual route: one sub-folder per pose with receptor.pdb (or *receptor*.pdb) and
    ligand.sdf (with hydrogens, docked coordinates). Ligand charge read from the SDF."""
    poses = []
    for d in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        rec = next(iter(sorted(d.glob("*receptor*.pdb"))), None) or next(iter(sorted(d.glob("*.pdb"))), None)
        lig = next(iter(sorted(d.glob("*.sdf"))), None) or next(iter(sorted(d.glob("*.mol2"))), None)
        if not (rec and lig):
            print(f"    {d.name}: needs a receptor .pdb and a ligand .sdf/.mol2 - skipped")
            continue
        poses.append({"pose": safe_name(d.name), "title": d.name, "dir": str(d), "receptor": str(rec),
                      "ligand": str(lig), "ligand_charge": sdf_formal_charge(lig)})
    return poses


def sdf_formal_charge(path: Path) -> int | None:
    """Net formal charge from an SDF (M  CHG lines, or atom-block charge codes)."""
    text = Path(path).read_text(errors="replace")
    if Path(path).suffix.lower() == ".mol2":
        return None                                    # acpype will guess
    chg = re.findall(r"^M  CHG\s+\d+((?:\s+-?\d+\s+-?\d+)+)", text, flags=re.M)
    if chg:
        vals = [int(x) for x in " ".join(chg).split()]
        return sum(vals[1::2])
    lines = text.splitlines()
    try:
        na = int(lines[3][:3])
    except (IndexError, ValueError):
        return None
    code = {1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}
    tot = 0
    for l in lines[4:4 + na]:
        try:
            tot += code.get(int(l[36:39]), 0)
        except ValueError:
            pass
    return tot


# =========================================================================== #
# Per-pose run
# =========================================================================== #
MMPBSA_IN = """&general
sys_name="{name}", startframe={start}, endframe={end}, interval={interval},
temperature={temp}, verbose=2,
/
&gb
igb={igb}, saltcon={salt},
/
"""
DECOMP = """&decomp
idecomp=2, dec_verbose=0, print_res="within 4",
/
"""


def pose_config(base: Config, project: str, pose: dict) -> Config:
    return replace(base, name=f"{project}/{pose['pose']}", pdb_id=None, pdb_file=pose["receptor"],
                   ligand_resname=None, ligand_file=pose["ligand"],
                   ligand_charge=pose.get("ligand_charge"), fix_protein=False, launch_chimerax=False)


def prepare_receptor(cfg: Config) -> dict:
    """Write the cleaned receptor where md_pipeline expects it (02_protein/protein_clean.pdb),
    so the pipeline uses it as-is (gaps keep their TER records)."""
    P = Paths(cfg)
    P.make()
    clean = P.prot / "protein_clean.pdb"
    rep_file = P.prot / "clean_report.json"
    if clean.exists() and rep_file.exists() and not cfg.overwrite:
        return json.loads(rep_file.read_text())
    rep = clean_receptor_pdb(Path(cfg.pdb_file), clean)
    rep_file.write_text(json.dumps(rep, indent=2))
    print(f"    receptor: {rep['residues']} residues; caps {rep['caps'] or 'none'}; "
          f"gaps {len(rep['gaps'])}; dropped {rep['dropped'] or 'nothing'}")
    return rep


RESULT_TAG = "_v2"     # v2: complex kept in one periodic image; earlier results are not reused


def gmx_mmpbsa(cfg: Config, R: Runner, workdir: Path, tpr: Path, xtc: Path, ndx: Path,
               interval: int = 1, start: int = 1, end: int = 9999999, igb: int = 5,
               decomp: bool = False) -> dict:
    import md_advanced as adv
    workdir.mkdir(parents=True, exist_ok=True)
    P = Paths(cfg)
    for f in list(P.sys.glob("*.itp")) + [P.sys / "topol.top"]:
        shutil.copy(f, workdir / f.name)
    groups = mp.read_ndx_names(ndx)
    (workdir / "mmpbsa.in").write_text(
        MMPBSA_IN.format(name=cfg.name.replace("/", "_"), start=start, end=end, interval=interval,
                         temp=cfg.temperature, igb=igb, salt=cfg.salt_conc) + (DECOMP if decomp else ""))
    res = workdir / "FINAL_RESULTS_MMPBSA.dat"
    if not res.exists() or cfg.overwrite:
        cmd = ["gmx_MMPBSA", "-O", "-i", "mmpbsa.in", "-cs", tpr, "-ci", ndx,
               "-cg", groups.index("Protein"), groups.index(LIG), "-ct", xtc, "-cp", "topol.top",
               "-o", "FINAL_RESULTS_MMPBSA.dat", "-eo", "FINAL_RESULTS_MMPBSA.csv", "-nogui"]
        if decomp:
            cmd += ["-do", "FINAL_DECOMP_MMPBSA.dat", "-deo", "FINAL_DECOMP_MMPBSA.csv"]
        R.run(cmd, f"mmgbsa_{workdir.name}", cwd=workdir)
    out = adv.parse_mmpbsa(res)
    if decomp and (workdir / "FINAL_DECOMP_MMPBSA.dat").exists():
        out["decomp"] = adv.parse_decomp(workdir / "FINAL_DECOMP_MMPBSA.dat")
    (workdir / "mmgbsa_summary.json").write_text(json.dumps(out, indent=2))
    return out


def run_pose(cfg: Config, mode: str = "md", interval: int = 5, decomp: bool = False, igb: int = 5) -> dict:
    """mode 'quick': EM only, MM/GBSA on the minimised complex (1 frame).
       mode 'md'   : EM + NVT + NPT + cfg.prod_ns production, MM/GBSA every `interval` frames."""
    t0 = time.time()
    banner(f"{cfg.name}  [{mode}]")
    rep = prepare_receptor(cfg)
    out = {"pose": cfg.name.split("/")[-1], "mode": mode, "receptor_gaps": len(rep["gaps"])}
    if mode == "quick":
        P = Paths(cfg)
        P.make()
        R = Runner(cfg, P)
        mp.check_environment(cfg)
        src = mp.get_structure(cfg, P)
        mp.prepare_protein(cfg, P, src)
        molname = mp.prepare_ligand(cfg, P, R, src)
        mp.build_system(cfg, P, R, P.prot / "protein_clean.pdb", molname)
        if not (P.em / "em.gro").exists():
            mp.grompp(cfg, R, P, mp.mdp_em(cfg), P.em, "em", "04_em", P.sys / "system.gro")
            mp.mdrun(cfg, R, P.em, "em", "04_em", "em")
        tpr = P.em / "em.tpr"
        groups = mp.make_index(cfg, P, R, tpr)
        ndx = P.ana / "index.ndx"
        whole = P.em / "em_cluster.xtc"
        if not whole.exists() or cfg.overwrite:
            mp.cluster_complex(R, tpr, P.em / "em.gro", ndx, whole, groups, "mmgbsa_quick_pbc")
        res = gmx_mmpbsa(cfg, R, P.root / f"mmgbsa_quick{RESULT_TAG}", tpr, whole, ndx, 1, 1, 1, igb, decomp)
    else:
        result = mp.run_pipeline(cfg)              # build, equilibrate, MD, analysis, viewer files
        P = Paths(cfg)
        R = Runner(cfg, P)
        res = gmx_mmpbsa(cfg, R, P.root / f"mmgbsa_md{RESULT_TAG}", P.md / "md.tpr", P.ana / "md_fit.xtc",
                         P.ana / "index.ndx", interval, 1, 9999999, igb, decomp)
        s = result["summary"]
        for k in ("rmsd_ligand", "rmsd_backbone", "contacts_protein_ligand"):
            if k in s:
                out[f"{k}_mean"] = round(s[k]["mean_second_half"], 3)
        out["ns_per_day"] = result["info"].get("production_ns_per_day")
    gb = res.get("dG_bind_GB", {})
    out.update({"dG_bind_kcal": gb.get("mean"), "dG_sd_kcal": gb.get("sd"),
                "minutes": round((time.time() - t0) / 60, 1)})
    if "decomp" in res:
        top = sorted(res["decomp"], key=lambda r: r["total"])[:5]
        out["top_residues"] = ", ".join(f"{r['residue']} {r['total']:.1f}" for r in top)
    (Paths(cfg).root / f"pose_result_{mode}{RESULT_TAG}.json").write_text(json.dumps(out, indent=2))
    print(f"    dG(bind, GB) = {out['dG_bind_kcal']} ± {out['dG_sd_kcal']} kcal/mol   "
          f"({out['minutes']} min)")
    return out


def estimate_hours(n_poses: int, mode: str, prod_ns: float, ns_per_day: float = 100.0,
                   setup_min: float = 15.0, mmgbsa_min_per_100_frames: float = 15.0,
                   frames: int = 100) -> float:
    """Rough GPU-session time. setup = ligand charges (AM1-BCC) + build + EM."""
    per = setup_min
    if mode == "md":
        per += (0.2 + prod_ns) / ns_per_day * 24 * 60 + 5        # NVT+NPT 0.2 ns, analysis ~5 min
        per += mmgbsa_min_per_100_frames * frames / 100
    else:
        per += 2
    return n_poses * per / 60


def run_all(base: Config, project: str, poses: list[dict], mode: str = "md", interval: int = 5,
            decomp: bool = False, igb: int = 5) -> list[dict]:
    """Run every pose (finished poses are skipped / resumed). After the first pose, prints a
    projection of the remaining time from the measured speed."""
    import pandas as pd
    results = []
    root = Paths(replace(base, name=project)).root
    root.mkdir(parents=True, exist_ok=True)
    for i, pose in enumerate(poses):
        cfg = pose_config(base, project, pose)
        done_file = Paths(cfg).root / f"pose_result_{mode}{RESULT_TAG}.json"
        if done_file.exists() and not base.overwrite:
            r = json.loads(done_file.read_text())
            print(f"{pose['pose']}: already done - dG {r.get('dG_bind_kcal')} kcal/mol")
        else:
            try:
                r = run_pose(cfg, mode, interval, decomp, igb)
            except Exception as e:                      # one bad pose must not stop the rest
                print(f"!!! {pose['pose']} failed: {str(e)[:2000]}")
                r = {"pose": pose["pose"], "mode": mode, "error": str(e)[:300]}
            if i == 0 and len(poses) > 1 and "minutes" in r:
                left = r["minutes"] * (len(poses) - 1) / 60
                print(f"\n>>> First pose took {r['minutes']} min; remaining {len(poses) - 1} poses "
                      f"≈ {left:.1f} h at the same speed\n")
        r.update({k: pose[k] for k in ("title", "ligand_charge", *SCORE_KEYS.values()) if k in pose})
        results.append(r)
        pd.DataFrame(results).to_csv(root / f"mmgbsa_summary_{mode}.csv", index=False)
    return results


def summary_table(results: list[dict]):
    import pandas as pd
    df = pd.DataFrame(results)
    cols = [c for c in ("pose", "title", "dG_bind_kcal", "dG_sd_kcal", "ifd_score", "docking_score",
                        "glide_gscore", "rmsd_ligand_mean", "contacts_protein_ligand_mean",
                        "ligand_charge", "receptor_gaps", "minutes", "top_residues", "error") if c in df]
    df = df[cols]
    if "dG_bind_kcal" in df:
        df = df.sort_values("dG_bind_kcal", na_position="last").reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
    return df


def plot_vs_scores(df, path: Path | None = None):
    import matplotlib.pyplot as plt
    score = next((c for c in ("ifd_score", "docking_score", "glide_gscore") if c in df and df[c].notna().any()), None)
    fig, ax = plt.subplots(figsize=(6, 4))
    d = df.dropna(subset=["dG_bind_kcal"])
    if score:
        ax.errorbar(d[score], d["dG_bind_kcal"], yerr=d.get("dG_sd_kcal"), fmt="o", capsize=3)
        for _, r in d.iterrows():
            ax.annotate(str(r["pose"])[:12], (r[score], r["dG_bind_kcal"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel=score.replace("_", " "), ylabel="MM/GBSA ΔG bind (kcal/mol)",
               title="MM/GBSA vs docking score")
    else:
        ax.barh(d["pose"], d["dG_bind_kcal"], xerr=d.get("dG_sd_kcal"))
        ax.set(xlabel="MM/GBSA ΔG bind (kcal/mol)", title="MM/GBSA per pose")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.show()
