"""
md_advanced.py  -  Levels 2-5 of the MD learning path, built on md_pipeline.py
==============================================================================

Every function reuses the system that md_pipeline builds (same Config), and writes
into a sub-folder of that system's results folder:

  Level 2  run_replicas / compare_replicas / pca_replicas / run_apo / compare_rmsf
           -> <name>/L2_replicas/rep1..N, <name>/L2_replicas/comparison, <name>/L2_apo
  Level 3  run_mmpbsa                 -> <name>/L3_mmpbsa          (gmx_MMPBSA, MM/GBSA)
           run_hydration_fe / analyse_hydration_fe
                                       -> <name>/L3_hydration       (alchemical, MBAR)
  Level 4  smd_config / run_smd / analyse_smd
                                       -> <name>_bigbox/L4_smd      (steered MD, Jarzynski)
  Level 5  run_umbrella / run_wham     -> <name>_bigbox/L5_umbrella (umbrella sampling, WHAM)
           dipeptide_config / build_dipeptide / run_dipeptide_metad / analyse_dipeptide_metad
                                       -> ala_dipeptide/L5_metad    (PLUMED metadynamics)
           run_ligand_metad / analyse_ligand_metad
                                       -> <name>/L5_metad_ligand    (optional)

All long runs checkpoint and resume: re-run the same cell after a Colab disconnect.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import md_pipeline as mp
from md_pipeline import LIG, Config, Paths, Runner, banner

KB_KJ = 0.0083144626      # kJ/mol/K
KB_KCAL = 0.0019872043    # kcal/mol/K
KJ_PER_KCAL = 4.184


# =========================================================================== #
# Shared helpers
# =========================================================================== #
def stage_dir(cfg: Config, name: str) -> Path:
    d = Paths(cfg).root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def runner_for(cfg: Config) -> tuple[Paths, Runner]:
    P = Paths(cfg)
    P.make()
    return P, Runner(cfg, P)


def read_gro(path: Path):
    """Return (atoms, box) from a .gro file. atoms: list of dicts with
    resnr, resname, name, index (1-based), xyz (nm)."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[1])
    atoms = []
    for i, l in enumerate(lines[2:2 + n]):
        atoms.append({"resnr": int(l[0:5]), "resname": l[5:10].strip(), "name": l[10:15].strip(),
                      "index": i + 1,
                      "xyz": (float(l[20:28]), float(l[28:36]), float(l[36:44]))})
    box = [float(x) for x in lines[2 + n].split()]
    return atoms, box


def box_matrix(box):
    """3x3 box (rows = box vectors) from a .gro box line."""
    if len(box) == 3:
        return [[box[0], 0, 0], [0, box[1], 0], [0, 0, box[2]]]
    v1x, v2y, v3z, v1y, v1z, v2x, v2z, v3x, v3y = box[:9]
    return [[v1x, v1y, v1z], [v2x, v2y, v2z], [v3x, v3y, v3z]]


def min_image(vec, box):
    """Shortest periodic image of a difference vector (works for triclinic boxes)."""
    M = box_matrix(box)
    best, bestd = vec, float("inf")
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                v = [vec[a] + i * M[0][a] + j * M[1][a] + k * M[2][a] for a in range(3)]
                d = sum(x * x for x in v)
                if d < bestd:
                    best, bestd = v, d
    return best


def centre(xyzs):
    n = len(xyzs)
    return [sum(p[a] for p in xyzs) / n for a in range(3)]


def centre_pbc(xyzs, box):
    """Geometric centre of a group that may be split across the periodic box.
    Each point is unwrapped next to the PREVIOUS one (consecutive atoms of a chain are
    always close), so groups larger than half the box are handled correctly."""
    pts = [list(xyzs[0])]
    for p in xyzs[1:]:
        prev = pts[-1]
        d = min_image([p[a] - prev[a] for a in range(3)], box)
        pts.append([prev[a] + d[a] for a in range(3)])
    return centre(pts)


def protein_atoms(atoms):
    """Atoms before the first ligand / water / ion residue."""
    stop = {LIG, "SOL", "HOH", "WAT", "NA", "CL", "K", "MG", "CA2", "ZN"}
    out = []
    for a in atoms:
        if a["resname"] in stop:
            break
        out.append(a)
    return out


def write_ndx(path: Path, groups: dict):
    with open(path, "w") as fh:
        for name, idx in groups.items():
            fh.write(f"[ {name} ]\n")
            for i in range(0, len(idx), 15):
                fh.write(" ".join(f"{x:5d}" for x in idx[i:i + 15]) + "\n")
            fh.write("\n")


def read_ndx(path: Path) -> dict:
    groups, cur = {}, None
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\s*\[\s*(.+?)\s*\]", line)
        if m:
            cur = m.group(1)
            groups[cur] = []
        elif cur and line.strip():
            groups[cur] += [int(x) for x in line.split()]
    return groups


def xvg_array(path: Path):
    import numpy as np
    rows, *_ = mp.read_xvg(Path(path))
    w = min(len(r) for r in rows)
    return np.array([r[:w] for r in rows])


def set_define(mdp: str, define: str) -> str:
    return re.sub(r"^define\s*=.*$", f"define              = {define}", mdp, flags=re.M)


def set_mdp(mdp: str, key: str, value) -> str:
    """Replace `key = ...` if present, else append it."""
    pat = rf"^{re.escape(key)}\s*=.*$"
    line = f"{key:<20}= {value}"
    if re.search(pat, mdp, flags=re.M):
        return re.sub(pat, line, mdp, flags=re.M)
    return mdp.rstrip("\n") + "\n" + line + "\n"


def run_md(cfg: Config, P: Paths, R: Runner, workdir: Path, name: str, mdp: str, coord: Path,
           cpt: Path | None = None, ref: Path | None = None, ndx: Path | None = None,
           extra: list | None = None, plumed: str | None = None, step: str | None = None,
           stage: str = "md") -> bool:
    """grompp + mdrun with resume. Finished = <name>.gro exists. Returns True if it ran."""
    workdir.mkdir(parents=True, exist_ok=True)
    step = step or f"{workdir.parent.name}_{workdir.name}_{name}"
    if (workdir / f"{name}.gro").exists() and not cfg.overwrite:
        print(f"    {workdir.name}/{name}: already done - skipping")
        return False
    tpr, cp = workdir / f"{name}.tpr", workdir / f"{name}.cpt"
    resume = tpr.exists() and cp.exists() and not cfg.overwrite
    if not resume:
        (workdir / f"{name}.mdp").write_text(mdp)
        cmd = ["grompp", "-f", workdir / f"{name}.mdp", "-c", coord, "-p", "topol.top", "-o", tpr,
               "-po", workdir / f"{name}_out.mdp", "-maxwarn", cfg.maxwarn]
        if ref:
            cmd += ["-r", ref]
        if cpt:
            cmd += ["-t", cpt]
        if ndx:
            cmd += ["-n", ndx]
        R.gmx(cmd, step, cwd=P.sys)
    else:
        print(f"    {workdir.name}/{name}: checkpoint found - resuming")
    extra = list(extra or [])
    if plumed is not None:
        (workdir / "plumed.dat").write_text(("RESTART\n" if resume else "") + plumed)
        extra += ["-plumed", "plumed.dat"]
        if not cfg.use_gpu:                      # PLUMED cannot run with >1 thread-MPI rank
            extra += ["-ntmpi", "1"]
    mp.mdrun(cfg, R, workdir, name, step, stage, cpi=cp if resume else None, extra=extra)
    return True


def reuse_inputs(src: Config, dst: Config):
    """Copy downloaded structure, ligand parameters and cleaned protein from one system
    folder to another, so a rebuilt system (e.g. a bigger box) skips those steps."""
    a, b = Paths(src), Paths(dst)
    b.make()
    for d in ("inp", "lig", "prot"):
        s, t = getattr(a, d), getattr(b, d)
        if s.exists() and not any(t.iterdir()):
            shutil.copytree(s, t, dirs_exist_ok=True)
    for f in (f"{LIG}.itp", f"{LIG}_atomtypes.itp", f"posre_{LIG}.itp", f"{LIG}.gro"):
        if (a.sys / f).exists() and not (b.sys / f).exists():
            shutil.copy(a.sys / f, b.sys / f)


def plot_save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        pass
    import matplotlib.pyplot as plt
    plt.close(fig)


# =========================================================================== #
# Level 2 - replicas, statistics, PCA, apo vs holo
# =========================================================================== #
def replica_config(base: Config, i: int, prod_ns: float | None = None) -> Config:
    return replace(base, name=f"{base.name}/L2_replicas/rep{i}", seed=1000 + i,
                   prod_ns=prod_ns if prod_ns is not None else base.prod_ns, launch_chimerax=False)


def run_replicas(base: Config, n: int = 3, prod_ns: float | None = None) -> list[Config]:
    """n independent replicas: same minimised structure, new random velocities
    (seed 1001, 1002, ...), then NVT, NPT, production and the standard analysis."""
    banner(f"Level 2: {n} replicas")
    P0, _, _ = mp.prepare_equilibrated(base)           # makes sure the base system exists
    cfgs = []
    for i in range(1, n + 1):
        rc = replica_config(base, i, prod_ns)
        Pi = Paths(rc)
        Pi.make()
        if not (Pi.em / "em.gro").exists():
            for d in ("inp", "lig", "prot", "sys", "em"):
                shutil.copytree(getattr(P0, d), getattr(Pi, d), dirs_exist_ok=True)
        banner(f"Replica {i} of {n} (seed {rc.seed})")
        mp.run_pipeline(rc)
        cfgs.append(rc)
    return cfgs


def block_average(x, max_blocks: int = 64):
    """Standard error of the mean vs block length (Flyvbjerg-Petersen style).
    Returns (block_lengths, sem). The plateau is the honest error bar."""
    import numpy as np
    x = np.asarray(x, float)
    sizes, sems = [], []
    b = 1
    while len(x) // b >= 4:
        nb = len(x) // b
        means = x[:nb * b].reshape(nb, b).mean(1)
        sizes.append(b)
        sems.append(means.std(ddof=1) / math.sqrt(nb))
        b *= 2
        if len(sizes) > max_blocks:
            break
    return np.array(sizes), np.array(sems)


def autocorr_time(x):
    """Integrated autocorrelation time (in samples), summed until the ACF first drops below 0."""
    import numpy as np
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 or not x.any():
        return 1.0
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n):
        if acf[k] < 0:
            break
        tau += 2 * acf[k]
    return float(tau)


METRICS = [("rmsd_backbone", "Backbone RMSD (nm)"), ("rmsd_ligand", "Ligand RMSD (nm)"),
           ("contacts_protein_ligand", "Protein-ligand contacts"),
           ("hbonds_protein_ligand", "Protein-ligand H-bonds"), ("gyrate_protein", "Radius of gyration (nm)")]


def compare_replicas(cfgs: list[Config], out_cfg: Config | None = None):
    """Mean +/- SEM across replicas (second half of each run), overlay plots,
    block averaging and autocorrelation of the ligand RMSD."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    out_cfg = out_cfg or cfgs[0]
    base_name = out_cfg.name.split("/L2_replicas")[0]
    out = Paths(replace(out_cfg, name=base_name)).root / "L2_replicas" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    labels = [c.name.split("/")[-1] for c in cfgs]

    rows = []
    for key, label in METRICS:
        vals = []
        for c in cfgs:
            s = json.loads((Paths(c).ana / "summary.json").read_text())
            if key in s:
                vals.append(s[key]["mean_second_half"])
        if len(vals) >= 2:
            v = np.array(vals)
            rows.append({"quantity": label, **{l: round(x, 4) for l, x in zip(labels, vals)},
                         "mean": v.mean(), "sem": v.std(ddof=1) / math.sqrt(len(v)), "n": len(v)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "replica_summary.csv", index=False)

    for key, label in METRICS:
        files = [(l, Paths(c).ana / "csv" / f"{key}.csv") for l, c in zip(labels, cfgs)]
        files = [(l, f) for l, f in files if f.exists()]
        if not files:
            continue
        fig, ax = plt.subplots(figsize=(7, 3.4))
        for l, f in files:
            a = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
            ax.plot(a[:, 0], a[:, 1], lw=0.8, label=l)
        ax.set(title=label, xlabel=f.read_text().splitlines()[0].split(",")[0], ylabel=label)
        ax.legend(fontsize=8)
        plot_save(fig, out / f"overlay_{key}.png")

    # block averaging + autocorrelation on ligand RMSD (or backbone RMSD for apo)
    key = "rmsd_ligand" if (Paths(cfgs[0]).ana / "csv" / "rmsd_ligand.csv").exists() else "rmsd_backbone"
    fig, ax = plt.subplots(figsize=(6, 3.4))
    acf_rows = []
    for l, c in zip(labels, cfgs):
        a = np.loadtxt(Paths(c).ana / "csv" / f"{key}.csv", delimiter=",", skiprows=1, ndmin=2)
        x = a[len(a) // 2:, 1]
        sizes, sems = block_average(x)
        dt = (a[1, 0] - a[0, 0]) if len(a) > 1 else 1
        ax.plot(sizes * dt, sems, "o-", ms=3, label=l)
        tau = autocorr_time(x)
        acf_rows.append({"replica": l, "tau_samples": round(tau, 1), "tau_time": round(tau * dt, 4),
                         "n_samples": len(x), "n_effective": round(len(x) / tau, 1)})
    ax.set(xscale="log", xlabel="Block length (same unit as time axis)", ylabel="SEM",
           title=f"Block averaging, {key} (2nd half)")
    ax.legend(fontsize=8)
    plot_save(fig, out / f"block_average_{key}.png")
    acf = pd.DataFrame(acf_rows)
    acf.to_csv(out / f"autocorrelation_{key}.csv", index=False)
    print(f"Saved to {out}")
    return table, acf


def _kabsch(X, ref):
    import numpy as np
    Xc, Rc = X - X.mean(0), ref - ref.mean(0)
    U, S, Vt = np.linalg.svd(Xc.T @ Rc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return Xc @ Rm.T + ref.mean(0)


def pca_replicas(cfgs: list[Config], selection: str = f"name CA and not resname {LIG}", n_pc: int = 5):
    """PCA of the combined C-alpha trajectories (all frames aligned to replica 1, frame 0).
    Uses the protein+ligand files in 09_visualization."""
    import warnings
    warnings.filterwarnings("ignore", module="MDAnalysis")
    import MDAnalysis as mda
    import numpy as np
    import matplotlib.pyplot as plt
    out = Paths(replace(cfgs[0], name=cfgs[0].name.split("/L2_replicas")[0])).root / "L2_replicas" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    blocks, labels, ref = [], [], None
    for c in cfgs:
        v = Paths(c).viz
        u = mda.Universe(str(v / "complex_start.pdb"), str(v / "complex_traj.xtc"))
        sel = u.select_atoms(selection)
        frames = []
        for _ in u.trajectory:
            X = sel.positions.astype(float)
            if ref is None:
                ref = X.copy()
            frames.append(_kabsch(X, ref))
        blocks.append(np.array(frames))
        labels.append(c.name.split("/")[-1])
    allX = np.concatenate([b.reshape(len(b), -1) for b in blocks])
    mean = allX.mean(0)
    _, S, Vt = np.linalg.svd(allX - mean, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    start = 0
    for lab, b in zip(labels, blocks):
        proj = (b.reshape(len(b), -1) - mean) @ Vt[:2].T / 10      # Å -> nm
        a1.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.6, label=lab)
        np.savetxt(out / f"pca_projection_{lab}.csv", proj, delimiter=",", header="PC1_nm,PC2_nm", comments="")
        start += len(b)
    a1.set(xlabel=f"PC1 ({var[0]:.0%})", ylabel=f"PC2 ({var[1]:.0%})", title="Replicas in PC1/PC2 space")
    a1.legend(fontsize=8, markerscale=3)
    a2.bar(range(1, n_pc + 1), var[:n_pc] * 100)
    a2.set(xlabel="Principal component", ylabel="Variance explained (%)", title="Variance per PC")
    plot_save(fig, out / "pca.png")
    return var[:n_pc]


def run_apo(base: Config, prod_ns: float | None = None) -> Config:
    """Same protein without the ligand, for apo vs holo comparison."""
    apo = replace(base, name=f"{base.name}/L2_apo", ligand_resname=None, ligand_file=None,
                  prod_ns=prod_ns if prod_ns is not None else base.prod_ns, launch_chimerax=False)
    Pa, Pb = Paths(apo), Paths(base)
    Pa.make()
    for d in ("inp", "prot"):
        if not any(getattr(Pa, d).iterdir()):
            shutil.copytree(getattr(Pb, d), getattr(Pa, d), dirs_exist_ok=True)
    mp.run_pipeline(apo)
    return apo


def compare_rmsf(holo_cfgs: list[Config], apo_cfg: Config, highlight=(107, 115)):
    import numpy as np
    import matplotlib.pyplot as plt
    out = Paths(replace(holo_cfgs[0], name=holo_cfgs[0].name.split("/L2_replicas")[0])).root / "L2_replicas" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    holo = [np.loadtxt(Paths(c).ana / "csv" / "rmsf_calpha.csv", delimiter=",", skiprows=1, ndmin=2)
            for c in holo_cfgs]
    apo = np.loadtxt(Paths(apo_cfg).ana / "csv" / "rmsf_calpha.csv", delimiter=",", skiprows=1, ndmin=2)
    n = min(len(h) for h in holo)
    H = np.array([h[:n, 1] for h in holo])
    res = holo[0][:n, 0]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.fill_between(res, H.mean(0) - H.std(0), H.mean(0) + H.std(0), alpha=0.3, label="holo, mean ± SD")
    ax.plot(res, H.mean(0), lw=1)
    ax.plot(apo[:, 0], apo[:, 1], lw=1, label="apo")
    if highlight:
        ax.axvspan(*highlight, color="grey", alpha=0.15, label=f"residues {highlight[0]}-{highlight[1]}")
    ax.set(xlabel="Residue", ylabel="C-alpha RMSF (nm)", title="Flexibility with and without benzene")
    ax.legend(fontsize=8)
    plot_save(fig, out / "rmsf_apo_vs_holo.png")


# =========================================================================== #
# Level 3A - MM/GBSA with gmx_MMPBSA
# =========================================================================== #
MMPBSA_IN = """&general
sys_name="{name}", startframe={start}, endframe={end}, interval={interval},
temperature={temp}, verbose=2,
/
&gb
igb=5, saltcon={salt},
/
"""
MMPBSA_PB = """&pb
istrng={salt}, fillratio=4.0,
/
"""
MMPBSA_DECOMP = """&decomp
idecomp=2, dec_verbose=0, print_res="within 4",
/
"""


def run_mmpbsa(cfg: Config, interval: int = 10, start: int = 1, end: int = 9999999,
               pb: bool = False, decomp: bool = True, workname: str = "L3_mmpbsa") -> dict:
    """MM/GBSA (and optionally MM/PBSA) on the production trajectory of `cfg`.
    Needs gmx_MMPBSA installed in the GROMACS environment (see the notebook)."""
    banner("Level 3A: MM/GBSA with gmx_MMPBSA")
    P, R = runner_for(cfg)
    tpr, xtc, ndx = P.md / "md.tpr", P.ana / "md_fit.xtc", P.ana / "index.ndx"
    for f in (tpr, xtc, ndx):
        if not f.exists():
            raise FileNotFoundError(f"{f} missing - run the Level 1 pipeline for '{cfg.name}' first")
    if not shutil.which("gmx_MMPBSA"):
        raise RuntimeError("gmx_MMPBSA not found - run the install cell of the Level 3 notebook")
    w = stage_dir(cfg, workname)
    for f in list(P.sys.glob("*.itp")) + [P.sys / "topol.top"]:
        shutil.copy(f, w / f.name)
    groups = mp.read_ndx_names(ndx)
    rec, lig = groups.index("Protein"), groups.index(LIG)
    text = MMPBSA_IN.format(name=cfg.name.replace("/", "_"), start=start, end=end, interval=interval,
                            temp=cfg.temperature, salt=cfg.salt_conc)
    if pb:
        text += MMPBSA_PB.format(salt=cfg.salt_conc)
    if decomp:
        text += MMPBSA_DECOMP
    (w / "mmpbsa.in").write_text(text)
    res = w / "FINAL_RESULTS_MMPBSA.dat"
    if res.exists() and not cfg.overwrite:
        print("    already done - skipping (set overwrite=True to redo)")
    else:
        cmd = ["gmx_MMPBSA", "-O", "-i", "mmpbsa.in", "-cs", tpr, "-ci", ndx, "-cg", rec, lig,
               "-ct", xtc, "-cp", "topol.top", "-o", "FINAL_RESULTS_MMPBSA.dat",
               "-eo", "FINAL_RESULTS_MMPBSA.csv", "-nogui"]
        if decomp:
            cmd += ["-do", "FINAL_DECOMP_MMPBSA.dat", "-deo", "FINAL_DECOMP_MMPBSA.csv"]
        R.run(cmd, "L3_mmpbsa", cwd=w)
    out = parse_mmpbsa(res)
    if decomp and (w / "FINAL_DECOMP_MMPBSA.dat").exists():
        out["decomp"] = parse_decomp(w / "FINAL_DECOMP_MMPBSA.dat")
        plot_decomp(out["decomp"], w / "decomposition_top_residues.png")
    (w / "mmpbsa_summary.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        if k != "decomp":
            print(f"    {k}: {v['mean']:.2f} ± {v['sd']:.2f} kcal/mol (SD over frames)")
    return out


def parse_mmpbsa(path: Path) -> dict:
    """Pull the Delta TOTAL lines (one per method) out of FINAL_RESULTS_MMPBSA.dat."""
    text = Path(path).read_text(errors="replace")
    out, method = {}, "GB"
    for line in text.splitlines():
        if "GENERALIZED BORN" in line.upper():
            method = "GB"
        elif "POISSON BOLTZMANN" in line.upper():
            method = "PB"
        m = re.match(r"\s*(?:Δ|DELTA\s*)TOTAL\s+(-?\d+\.\d+)\s+(\d+\.\d+)(?:\s+(\d+\.\d+))?", line)
        if m:
            sd = float(m.group(3) or m.group(2))
            out[f"dG_bind_{method}"] = {"mean": float(m.group(1)), "sd": sd, "units": "kcal/mol"}
    if not out:
        raise RuntimeError(f"No ΔTOTAL line found in {path}; open the file to see what went wrong")
    return out


def parse_decomp(path: Path) -> list:
    """Per-residue TOTAL contributions from the DELTAS block of FINAL_DECOMP_MMPBSA.dat."""
    lines = Path(path).read_text(errors="replace").splitlines()
    try:
        i = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("DELTAS"))
    except StopIteration:
        return []
    out = []
    for l in lines[i + 1:]:
        pairs = re.findall(r"(-?\d+\.\d+)\s*\+/-\s*(\d+\.\d+)", l)
        if not pairs:
            if out and not l.strip():
                break
            continue
        label = re.split(r"[|,]", l)[0].strip()
        if label:
            out.append({"residue": label, "total": float(pairs[-1][0]), "sd": float(pairs[-1][1])})
    return out


def plot_decomp(rows, path: Path, top: int = 12):
    if not rows:
        return
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["total"])[:top]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(rows) + 1.2))
    ax.barh([r["residue"] for r in rows][::-1], [r["total"] for r in rows][::-1],
            xerr=[r["sd"] for r in rows][::-1], color="tab:blue")
    ax.set(xlabel="Contribution (kcal/mol)", title="Most favourable residues (MM/GBSA decomposition)")
    plot_save(fig, path)


# =========================================================================== #
# Level 3B - alchemical hydration free energy
# =========================================================================== #
COUL = (0.0, 0.25, 0.5, 0.75, 1.0)
VDW = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)


def lambda_schedule(coul=COUL, vdw=VDW):
    """Charges are switched off first (vdW kept on), then van der Waals.
    0 = fully coupled, 1 = fully decoupled."""
    c = list(coul) + [1.0] * len(vdw)
    v = [0.0] * len(coul) + list(vdw)
    return c, v


def fe_block(state: int, coul, vdw, molname: str, nstdhdl: int = 100) -> str:
    fmt = lambda xs: " ".join(f"{x:.2f}" for x in xs)
    return f"""; ---- free energy ----
free-energy         = yes
init-lambda-state   = {state}
coul-lambdas        = {fmt(coul)}
vdw-lambdas         = {fmt(vdw)}
calc-lambda-neighbors = -1
couple-moltype      = {molname}
couple-lambda0      = vdw-q
couple-lambda1      = none
couple-intramol     = no
sc-alpha            = 0.5
sc-power            = 1
sc-sigma            = 0.3
nstdhdl             = {nstdhdl}
separate-dhdl-file  = yes
dhdl-derivatives    = yes
"""


def _sd_mdp(cfg: Config, nsteps: int, pcoupl: bool, continuation: bool) -> str:
    """Langevin (sd) dynamics, recommended for free energy windows."""
    m = mp._dyn(cfg, nsteps, max(1, mp._steps(cfg, 1.0)), False, pcoupl, continuation)
    m = set_mdp(m, "integrator", "sd")
    m = set_mdp(m, "tcoupl", "no")              # sd has its own thermostat
    m = set_mdp(m, "tau_t", "1.0")
    m = set_mdp(m, "nstxout-compressed", 0)
    return m


def hydration_topology(cfg: Config, molname: str) -> str:
    return (f'#include "{cfg.forcefield}.ff/forcefield.itp"\n'
            f'#include "{LIG}_atomtypes.itp"\n'
            f'#include "{LIG}.itp"\n'
            f'#include "{cfg.forcefield}.ff/{cfg.water}.itp"\n\n'
            f"[ system ]\nLigand in water\n\n[ molecules ]\n{molname:<20}1\n")


def run_hydration_fe(cfg: Config, coul=COUL, vdw=VDW, equil_ps: float = 100, prod_ns: float = 1.0,
                     box_nm: float = 1.2, windows: list | None = None) -> Path:
    """Decouple the ligand from water window by window. Uses the ligand parameters
    built in Level 1. Each window: EM -> NPT -> production (sd integrator)."""
    banner("Level 3B: alchemical hydration free energy")
    Pb = Paths(cfg)
    for f in (f"{LIG}.itp", f"{LIG}_atomtypes.itp", f"{LIG}.gro"):
        if not (Pb.sys / f).exists():
            raise FileNotFoundError(f"{Pb.sys / f} missing - run Level 1 for '{cfg.name}' first")
    molname = (Pb.lig / "molname.txt").read_text().strip()
    hcfg = replace(cfg, name=f"{cfg.name}/L3_hydration")
    P, R = runner_for(hcfg)
    setup = P.sys                    # L3_hydration/03_system holds the solvated ligand
    for f in (f"{LIG}.itp", f"{LIG}_atomtypes.itp", f"{LIG}.gro"):
        shutil.copy(Pb.sys / f, setup / f)
    if not (setup / "solvated.gro").exists() or cfg.overwrite:
        (setup / "topol.top").write_text(hydration_topology(cfg, molname))
        R.gmx(["editconf", "-f", f"{LIG}.gro", "-o", "boxed.gro", "-bt", "cubic", "-d", box_nm, "-c"],
              "L3_hyd_setup", cwd=setup)
        R.gmx(["solvate", "-cp", "boxed.gro", "-cs", "spc216.gro", "-o", "solvated.gro", "-p", "topol.top"],
              "L3_hyd_setup", cwd=setup)
    c, v = lambda_schedule(coul, vdw)
    (P.root / "lambdas.json").write_text(json.dumps({"coul": c, "vdw": v}))
    todo = windows if windows is not None else range(len(c))
    for i in todo:
        wd = P.root / f"lambda_{i:02d}"
        banner(f"window {i} of {len(c) - 1}: coul-lambda {c[i]:.2f}, vdw-lambda {v[i]:.2f}")
        em = mp.mdp_em(cfg) + fe_block(i, c, v, molname, nstdhdl=0)
        run_md(hcfg, P, R, wd, "em", em, setup / "solvated.gro", stage="em")
        npt = _sd_mdp(cfg, mp._steps(cfg, equil_ps), True, False) + fe_block(i, c, v, molname, nstdhdl=0)
        run_md(hcfg, P, R, wd, "npt", npt, wd / "em.gro")
        prod = _sd_mdp(cfg, mp._steps(cfg, prod_ns * 1000), True, True) + fe_block(i, c, v, molname)
        run_md(hcfg, P, R, wd, "prod", prod, wd / "npt.gro", cpt=wd / "npt.cpt",
               extra=["-dhdl", "dhdl.xvg"])
    return P.root


def analyse_hydration_fe(cfg: Config, skip_ps: float = 100, experiment_kcal: float | None = -0.86):
    """MBAR with alchemlyb. Returns dG_hydration (kcal/mol) and its error."""
    import alchemlyb
    import matplotlib.pyplot as plt
    from alchemlyb.estimators import MBAR
    from alchemlyb.parsing.gmx import extract_u_nk
    from alchemlyb.preprocessing.subsampling import decorrelate_u_nk, slicing
    root = Paths(replace(cfg, name=f"{cfg.name}/L3_hydration")).root
    files = sorted(root.glob("lambda_*/dhdl.xvg"))
    if not files:
        raise FileNotFoundError(f"No dhdl.xvg files under {root} - run run_hydration_fe first")
    T = cfg.temperature
    u_nk = [decorrelate_u_nk(slicing(extract_u_nk(str(f), T=T), lower=skip_ps)) for f in files]
    mbar = MBAR().fit(alchemlyb.concat(u_nk))
    dg_dec_kT = float(mbar.delta_f_.iloc[0, -1])
    err_kT = float(mbar.d_delta_f_.iloc[0, -1])
    kT = KB_KCAL * T
    dg_hyd, err = -dg_dec_kT * kT, err_kT * kT
    out = {"dG_decouple_kcal": dg_dec_kT * kT, "dG_hydration_kcal": dg_hyd, "error_kcal": err,
           "n_windows": len(files), "experiment_kcal": experiment_kcal}
    (root / "hydration_result.json").write_text(json.dumps(out, indent=2))
    try:
        from alchemlyb.visualisation import plot_mbar_overlap_matrix
        ax = plot_mbar_overlap_matrix(mbar.overlap_matrix)
        ax.figure.savefig(root / "overlap_matrix.png", dpi=150)
        plt.show()
    except Exception as e:                     # plotting is a bonus, never fatal
        print("(overlap plot skipped:", e, ")")
    print(f"Hydration free energy: {dg_hyd:.2f} ± {err:.2f} kcal/mol"
          + (f"   (experiment {experiment_kcal} kcal/mol)" if experiment_kcal is not None else ""))
    return out


# =========================================================================== #
# Level 4 - steered MD
# =========================================================================== #
def smd_config(base: Config, box_distance: float = 2.0) -> Config:
    """Same system in a larger box so the ligand can be pulled into bulk water."""
    big = replace(base, name=f"{base.name}_bigbox", box_distance=box_distance, prod_ns=0,
                  launch_chimerax=False)
    reuse_inputs(base, big)          # skip download and ligand parameterisation
    return big


def _moleculetype_files(sysdir: Path):
    """(file, posre include name) for every protein moleculetype with a POSRES block."""
    out = []
    for f in [sysdir / "topol.top", *sorted(sysdir.glob("topol_*.itp"))]:
        if not f.exists():
            continue
        text = f.read_text()
        m = re.search(r'#ifdef POSRES\s*\n\s*#include\s+"([^"]+)"\s*\n\s*#endif', text)
        if m and "[ atoms ]" in text.replace("[atoms]", "[ atoms ]"):
            out.append((f, m.group(1)))
    return out


def add_ca_posres(sysdir: Path, k: float = 500.0) -> list[Path]:
    """Write posre_CA files (C-alpha only, force constant k kJ/mol/nm^2) and add
    `#ifdef POSRES_CA` blocks next to the existing POSRES blocks. Idempotent."""
    written = []
    for f, posre in _moleculetype_files(sysdir):
        text = f.read_text()
        ca_name = posre.replace(".itp", "_CA.itp")
        if f'#include "{ca_name}"' in text:
            written.append(sysdir / ca_name)
            continue
        # C-alpha atom numbers of this moleculetype
        sec, ca = None, []
        for line in text.splitlines():
            m = re.match(r"\s*\[\s*(\w+)\s*\]", line)
            if m:
                sec = m.group(1).lower()
                continue
            body = line.split(";")[0].split()
            if sec == "atoms" and len(body) >= 5 and body[4] == "CA":
                ca.append(int(body[0]))
        if not ca:
            continue
        lines = ["; C-alpha position restraints", "[ position_restraints ]", ";  ai  funct  fcx  fcy  fcz"]
        lines += [f"{i:6d}     1  {k:g}  {k:g}  {k:g}" for i in ca]
        (sysdir / ca_name).write_text("\n".join(lines) + "\n")
        block = f'#ifdef POSRES\n#include "{posre}"\n#endif'
        new = block + f'\n\n; C-alpha restraints for pulling\n#ifdef POSRES_CA\n#include "{ca_name}"\n#endif'
        f.write_text(re.sub(r'#ifdef POSRES\s*\n\s*#include\s+"' + re.escape(posre) + r'"\s*\n\s*#endif',
                            lambda _: new, text, count=1))
        written.append(sysdir / ca_name)
    if not written:
        raise RuntimeError(f"No protein position-restraint block found in {sysdir}")
    return written


def pull_geometry(gro: Path):
    """Reference group (protein C-alpha), ligand group, pulling direction (protein centre ->
    ligand centre) and a pbc atom for the reference group, all from one .gro."""
    atoms, box = read_gro(gro)
    prot = protein_atoms(atoms)
    ca = [a for a in prot if a["name"] == "CA"]
    lig = [a for a in atoms if a["resname"] == LIG]
    if not ca or not lig:
        raise RuntimeError("Need protein C-alpha atoms and a ligand (resname LIG) in " + str(gro))
    c_ca = centre_pbc([a["xyz"] for a in ca], box)
    c_lig = centre_pbc([a["xyz"] for a in lig], box)
    vec = min_image([c_lig[i] - c_ca[i] for i in range(3)], box)
    dist = math.sqrt(sum(x * x for x in vec))
    unit = [x / dist for x in vec]
    pbcatom = min(ca, key=lambda a: sum(d * d for d in min_image(
        [a["xyz"][i] - c_ca[i] for i in range(3)], box)))["index"]
    half_box = 0.5 * min(math.sqrt(sum(x * x for x in row)) for row in box_matrix(box))
    return {"ref": [a["index"] for a in ca], "lig": [a["index"] for a in lig], "vec": unit,
            "start_nm": dist, "pbcatom": pbcatom, "half_box_nm": half_box}


def pull_block(g: dict, rate: float, k: float, nst: int = 50, start: bool = True, init: float = 0.0) -> str:
    v = " ".join(f"{x:.4f}" for x in g["vec"])
    return f"""; ---- COM pulling ----
pull                    = yes
pull-ncoords            = 1
pull-ngroups            = 2
pull-group1-name        = Pull_ref
pull-group2-name        = Pull_lig
pull-group1-pbcatom     = {g['pbcatom']}
pull-pbc-ref-prev-step-com = yes
pull-coord1-groups      = 1 2
pull-coord1-type        = umbrella
pull-coord1-geometry    = direction
pull-coord1-vec         = {v}
pull-coord1-start       = {'yes' if start else 'no'}
pull-coord1-init        = {init}
pull-coord1-rate        = {rate}
pull-coord1-k           = {k}
pull-nstxout            = {nst}
pull-nstfout            = {nst}
"""


def _pull_mdp(cfg: Config, nsteps: int, g: dict, rate: float, k: float, seed: int | None,
              xtc_ps: float = 1.0) -> str:
    m = mp._dyn(cfg, nsteps, max(1, mp._steps(cfg, xtc_ps)), False, True, seed is None)
    m = set_define(m, "-DPOSRES_CA")
    if seed is not None:
        m = set_mdp(m, "gen_seed", seed)
    return m + pull_block(g, rate, k)


def run_smd(cfg_big: Config, n_pulls: int = 5, rate: float = 0.005, k: float = 1000.0,
            pull_nm: float = 2.0, ca_k: float = 500.0) -> Path:
    """Build/equilibrate the big-box system, then pull the ligand out n_pulls times,
    each with new random velocities. rate in nm/ps, k in kJ/mol/nm^2."""
    banner("Level 4: steered MD")
    P, R, _ = mp.prepare_equilibrated(cfg_big)
    add_ca_posres(P.sys, ca_k)
    w = stage_dir(cfg_big, "L4_smd")
    g = pull_geometry(P.npt / "npt.gro")
    if g["start_nm"] + pull_nm > 0.98 * g["half_box_nm"]:
        new = max(0.5, 0.98 * g["half_box_nm"] - g["start_nm"])
        print(f"    pull distance reduced from {pull_nm} to {new:.2f} nm to stay within half the box")
        pull_nm = new
    ndx = w / "pull.ndx"
    write_ndx(ndx, {"System": list(range(1, len(read_gro(P.npt / 'npt.gro')[0]) + 1)),
                    "Pull_ref": g["ref"], "Pull_lig": g["lig"]})
    info = {**{k_: v for k_, v in g.items() if k_ not in ("ref", "lig")}, "rate_nm_per_ps": rate,
            "k_kJ_mol_nm2": k, "pull_nm": pull_nm, "n_pulls": n_pulls, "ca_restraint_k": ca_k}
    (w / "smd_settings.json").write_text(json.dumps(info, indent=2))
    print(f"    start distance {g['start_nm']:.2f} nm, pulling {pull_nm:.2f} nm at {rate} nm/ps "
          f"({pull_nm / rate:.0f} ps per pull)")
    nsteps = mp._steps(cfg_big, pull_nm / rate)
    for i in range(1, n_pulls + 1):
        banner(f"pull {i} of {n_pulls}")
        run_md(cfg_big, P, R, w / f"pull{i}", "pull", _pull_mdp(cfg_big, nsteps, g, rate, k, 2000 + i),
               P.npt / "npt.gro", ref=P.npt / "npt.gro", ndx=ndx,
               extra=["-px", "pullx.xvg", "-pf", "pullf.xvg"])
    return w


def jarzynski(works, T: float):
    """Jarzynski free energy (same units as works, kJ/mol) for each time point.
    works: array (n_pulls, n_points)."""
    import numpy as np
    beta = 1.0 / (KB_KJ * T)
    W = np.asarray(works, float)
    a = -beta * W
    amax = a.max(0)
    lse = amax + np.log(np.exp(a - amax).mean(0))
    jar = -lse / beta
    cum2 = W.mean(0) - beta * W.var(0) / 2           # second-order cumulant estimate
    return jar, cum2


def analyse_smd(cfg_big: Config):
    """Force vs displacement, work vs pulling distance, Jarzynski estimate."""
    import numpy as np
    import matplotlib.pyplot as plt
    w = Paths(cfg_big).root / "L4_smd"
    s = json.loads((w / "smd_settings.json").read_text())
    rate = s["rate_nm_per_ps"]
    pulls = sorted(w.glob("pull*/pullf.xvg"), key=lambda p: int(p.parent.name[4:]))
    if not pulls:
        raise FileNotFoundError("No pulls found - run run_smd first")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    works, xs, rows = [], None, []
    for f in pulls:
        F = xvg_array(f)
        X = xvg_array(f.parent / "pullx.xvg")
        n = min(len(F), len(X))
        t, force, x = F[:n, 0], F[:n, 1], X[:n, 1]
        disp = rate * (t - t[0])                        # spring position relative to start (nm)
        dt = np.diff(t, prepend=t[0])
        W = np.cumsum(force * rate * dt)                # kJ/mol
        a1.plot(x - x[0], force, lw=0.7, label=f.parent.name)
        a2.plot(disp, W, lw=0.8, label=f.parent.name)
        works.append(W)
        xs = disp if xs is None or len(disp) < len(xs) else xs
        rows.append({"pull": f.parent.name, "peak_force_kJ_mol_nm": float(force.max()),
                     "final_work_kJ_mol": float(W[-1])})
    m = min(len(x) for x in works)
    W = np.array([x[:m] for x in works])
    jar, cum2 = jarzynski(W, cfg_big.temperature)
    a2.plot(xs[:m], jar, "k-", lw=2, label="Jarzynski")
    a2.plot(xs[:m], cum2, "k--", lw=1.2, label="2nd-order cumulant")
    a1.set(xlabel="Ligand displacement along pull (nm)", ylabel="Force (kJ/mol/nm)", title="Pulling force")
    a2.set(xlabel="Spring displacement (nm)", ylabel="Work (kJ/mol)", title="Work and Jarzynski estimate")
    a1.legend(fontsize=7)
    a2.legend(fontsize=7)
    plot_save(fig, w / "smd_force_work.png")
    out = {"pulls": rows, "jarzynski_final_kJ_mol": float(jar[-1]),
           "cumulant_final_kJ_mol": float(cum2[-1]), "mean_work_final_kJ_mol": float(W[:, -1].mean())}
    (w / "smd_summary.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"    {r['pull']}: peak force {r['peak_force_kJ_mol_nm']:.0f} kJ/mol/nm, "
              f"work {r['final_work_kJ_mol']:.1f} kJ/mol")
    print(f"    mean work {out['mean_work_final_kJ_mol']:.1f}, Jarzynski {out['jarzynski_final_kJ_mol']:.1f} "
          f"kJ/mol (divide by 4.184 for kcal/mol)")
    return out


# =========================================================================== #
# Level 5A - umbrella sampling + WHAM
# =========================================================================== #
def choose_windows(times, values, spacing: float = 0.1, frame_dt: float = 1.0):
    """Pick (time, value) pairs about `spacing` nm apart along the pull coordinate, using
    only times at which a trajectory frame exists (multiples of frame_dt ps)."""
    import numpy as np
    t, x = np.asarray(times, float), np.asarray(values, float)
    ok = np.isclose((t - t[0]) / frame_dt, np.round((t - t[0]) / frame_dt), atol=1e-3)
    t, x = t[ok], x[ok]
    targets = np.arange(x[0], x.max() + 1e-9, spacing)
    picks = []
    for tg in targets:
        i = int(np.argmin(np.abs(x - tg)))
        if not picks or abs(x[i] - picks[-1][1]) > spacing / 2:
            picks.append((float(t[i]), float(x[i])))
    return picks


def run_umbrella(cfg_big: Config, pull: int = 1, spacing: float = 0.1, k: float = 1000.0,
                 equil_ps: float = 100, prod_ns: float = 1.0, windows: list | None = None) -> Path:
    """Umbrella windows from snapshots of one Level 4 pull."""
    banner("Level 5A: umbrella sampling")
    P, R = runner_for(cfg_big)
    smd = P.root / "L4_smd" / f"pull{pull}"
    if not (smd / "pull.gro").exists():
        raise FileNotFoundError(f"{smd} incomplete - finish Level 4 first")
    w = stage_dir(cfg_big, "L5_umbrella")
    g = pull_geometry(P.npt / "npt.gro")
    ndx = P.root / "L4_smd" / "pull.ndx"
    X = xvg_array(smd / "pullx.xvg")
    picks = choose_windows(X[:, 0], X[:, 1], spacing)
    (w / "windows.json").write_text(json.dumps([{"window": i, "time_ps": t, "coord_nm": x}
                                                for i, (t, x) in enumerate(picks)], indent=2))
    print(f"    {len(picks)} windows from {picks[0][1]:.2f} to {picks[-1][1]:.2f} nm")
    todo = windows if windows is not None else range(len(picks))
    for i in todo:
        t, x = picks[i]
        wd = w / f"win_{i:02d}"
        wd.mkdir(exist_ok=True)
        conf = wd / "start.gro"
        if not conf.exists():
            R.gmx(["trjconv", "-s", smd / "pull.tpr", "-f", smd / "pull.xtc", "-o", conf, "-dump", t],
                  "L5_umbrella_frames", stdin="0\n")
        banner(f"window {i} of {len(picks) - 1}: {x:.2f} nm")
        eq = mp._dyn(cfg_big, mp._steps(cfg_big, equil_ps), max(1, mp._steps(cfg_big, 1.0)), False, True, False)
        eq = set_define(eq, "-DPOSRES_CA") + pull_block(g, 0.0, k)
        run_md(cfg_big, P, R, wd, "npt", eq, conf, ref=conf, ndx=ndx)
        pr = mp._dyn(cfg_big, mp._steps(cfg_big, prod_ns * 1000), max(1, mp._steps(cfg_big, 10.0)),
                     False, True, True)
        pr = set_define(pr, "-DPOSRES_CA") + pull_block(g, 0.0, k)
        run_md(cfg_big, P, R, wd, "umbrella", pr, wd / "npt.gro", cpt=wd / "npt.cpt", ref=conf, ndx=ndx,
               extra=["-px", "pullx.xvg", "-pf", "pullf.xvg"])
    return w


def run_wham(cfg_big: Config, skip_ps: float = 100, n_bootstrap: int = 50, units: str = "kCal"):
    """gmx wham over all finished windows; plots PMF (with bootstrap error) and histograms."""
    import numpy as np
    import matplotlib.pyplot as plt
    P, R = runner_for(cfg_big)
    w = Paths(cfg_big).root / "L5_umbrella"
    wins = sorted(d for d in w.glob("win_*") if (d / "umbrella.gro").exists())
    if len(wins) < 3:
        raise RuntimeError(f"Only {len(wins)} finished windows - need at least 3")
    (w / "tpr-files.dat").write_text("\n".join(str(d / "umbrella.tpr") for d in wins) + "\n")
    (w / "pullf-files.dat").write_text("\n".join(str(d / "pullf.xvg") for d in wins) + "\n")
    R.gmx(["wham", "-it", "tpr-files.dat", "-if", "pullf-files.dat", "-o", "profile.xvg",
           "-hist", "histo.xvg", "-unit", units, "-b", skip_ps, "-temp", cfg_big.temperature,
           "-nBootstrap", n_bootstrap, "-bsres", "bsResult.xvg", "-bsprof", "bsProfs.xvg"],
          "L5_wham", cwd=w)
    prof = xvg_array(w / "profile.xvg")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    unit = "kcal/mol" if units.lower() == "kcal" else "kJ/mol" if units.lower() == "kj" else "kT"
    a1.plot(prof[:, 0], prof[:, 1], "k-", lw=1.5, label="PMF")
    if (w / "bsResult.xvg").exists():
        bs = xvg_array(w / "bsResult.xvg")
        if bs.shape[1] >= 3:
            a1.fill_between(bs[:, 0], bs[:, 1] - bs[:, 2], bs[:, 1] + bs[:, 2], alpha=0.3,
                            label="bootstrap ± SD")
    a1.set(xlabel="Pull coordinate (nm)", ylabel=f"PMF ({unit})", title="Potential of mean force")
    a1.legend(fontsize=8)
    h = xvg_array(w / "histo.xvg")
    for j in range(1, h.shape[1]):
        a2.plot(h[:, 0], h[:, j], lw=0.8)
    a2.set(xlabel="Pull coordinate (nm)", ylabel="Count", title="Window histograms (must overlap)")
    plot_save(fig, w / "pmf.png")
    depth = float(prof[:, 1].max() - prof[:, 1].min())
    print(f"    {len(wins)} windows; PMF range {depth:.2f} {unit} (max minus min)")
    return {"windows": len(wins), "pmf_range": depth, "units": unit}


# =========================================================================== #
# Level 5B - metadynamics with PLUMED
# =========================================================================== #
def plumed_available(cfg: Config) -> bool:
    import os
    kern = os.environ.get("PLUMED_KERNEL", "")
    h = subprocess.run([cfg.gmx, "mdrun", "-h"], capture_output=True, text=True)
    ok_flag = "-plumed" in (h.stdout + h.stderr)
    ok_kern = bool(kern) and Path(kern).exists()
    print(f"    gmx mdrun -plumed option: {'yes' if ok_flag else 'NO'};  PLUMED_KERNEL: {kern or 'not set'}"
          f" ({'found' if ok_kern else 'MISSING'})")
    return ok_flag and ok_kern


def dipeptide_config(base: Config | None = None, **kw) -> Config:
    base = base or Config()
    d = dict(name="ala_dipeptide", pdb_id=None, pdb_file=None, ligand_resname=None, ligand_file=None,
             fix_protein=False, box_distance=1.0, prod_ns=0, launch_chimerax=False)
    d.update(kw)
    return replace(base, **d)


LEAP_DIPEPTIDE = """source leaprc.protein.ff14SB
m = sequence { ACE ALA NME }
savepdb m ala2_leap.pdb
quit
"""


def build_dipeptide(cfg: Config):
    """Make ACE-ALA-NME with tleap (AmberTools), then build/equilibrate it with the base pipeline."""
    banner("Level 5B: alanine dipeptide")
    root = Paths(cfg).root
    root.mkdir(parents=True, exist_ok=True)
    pdb = root / "ala2_leap.pdb"
    if not pdb.exists():
        (root / "leap.in").write_text(LEAP_DIPEPTIDE)
        r = subprocess.run(["tleap", "-f", "leap.in"], cwd=root, capture_output=True, text=True)
        (root / "leap.log.txt").write_text(r.stdout + r.stderr)
        if not pdb.exists():
            raise RuntimeError(f"tleap failed - see {root / 'leap.log.txt'}")
    cfg.pdb_file = str(pdb)
    return mp.prepare_equilibrated(cfg)


def dipeptide_torsions(gro: Path):
    """1-based atom numbers for phi (C-N-CA-C) and psi (N-CA-C-N) of the alanine."""
    atoms, _ = read_gro(gro)
    resnrs = sorted({a["resnr"] for a in atoms if a["resname"] in ("ACE", "ALA", "NME")})
    ace, ala, nme = resnrs[:3]
    pick = lambda r, n: next(a["index"] for a in atoms if a["resnr"] == r and a["name"] == n)
    phi = [pick(ace, "C"), pick(ala, "N"), pick(ala, "CA"), pick(ala, "C")]
    psi = [pick(ala, "N"), pick(ala, "CA"), pick(ala, "C"), pick(nme, "N")]
    return phi, psi


def dipeptide_plumed(phi, psi, metad: bool = True, pace: int = 500, height: float = 1.2,
                     sigma: float = 0.35, biasfactor: float = 6, temp: float = 300) -> str:
    j = lambda xs: ",".join(map(str, xs))
    s = f"phi: TORSION ATOMS={j(phi)}\npsi: TORSION ATOMS={j(psi)}\n"
    if metad:
        s += (f"metad: METAD ARG=phi,psi PACE={pace} HEIGHT={height} SIGMA={sigma},{sigma} "
              f"BIASFACTOR={biasfactor} TEMP={temp} FILE=HILLS "
              f"GRID_MIN=-pi,-pi GRID_MAX=pi,pi GRID_BIN=150,150\n")
        s += "PRINT ARG=phi,psi,metad.bias STRIDE=500 FILE=COLVAR\n"
    else:
        s += "PRINT ARG=phi,psi STRIDE=500 FILE=COLVAR\n"
    return s


def run_dipeptide_metad(cfg: Config, ns: float = 10.0, unbiased: bool = True, **metad_kw) -> Path:
    if not plumed_available(cfg):
        raise RuntimeError("GROMACS cannot use PLUMED here - see the install cell of the notebook")
    P, R, _ = build_dipeptide(cfg)
    phi, psi = dipeptide_torsions(P.npt / "npt.gro")
    w = stage_dir(cfg, "L5_metad")
    (w / "torsions.json").write_text(json.dumps({"phi": phi, "psi": psi}))
    nsteps = mp._steps(cfg, ns * 1000)
    mdp = mp._dyn(cfg, nsteps, max(1, mp._steps(cfg, 1.0)), False, True, True)
    runs = (["unbiased"] if unbiased else []) + ["metad"]
    for r in runs:
        banner(f"{r} run, {ns} ns")
        run_md(cfg, P, R, w / r, "md", mdp, P.npt / "npt.gro", cpt=P.npt / "npt.cpt",
               plumed=dipeptide_plumed(phi, psi, metad=(r == "metad"), temp=cfg.temperature, **metad_kw))
    return w


def read_colvar(path: Path):
    import numpy as np
    names, rows = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#! FIELDS"):
            names = line.split()[2:]
        elif line.strip() and not line.startswith("#"):
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                pass
    a = np.array(rows)
    return {n: a[:, i] for i, n in enumerate(names) if i < a.shape[1]}


def sum_hills(workdir: Path, ndim: int = 2, stride: int | None = None, extra: list | None = None):
    cmd = ["plumed", "sum_hills", "--hills", "HILLS", "--mintozero"]
    if stride:
        cmd += ["--stride", str(stride), "--outfile", "fes_t_"]
    else:
        cmd += ["--outfile", "fes.dat"]
    cmd += list(extra or [])
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    (workdir / "sum_hills.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"plumed sum_hills failed - see {workdir / 'sum_hills.log'}")


def read_fes2d(path: Path):
    import numpy as np
    d = np.loadtxt(path, comments="#")
    xs, ys = np.unique(d[:, 0]), np.unique(d[:, 1])
    F = np.full((len(ys), len(xs)), np.nan)
    ix = np.searchsorted(xs, d[:, 0])
    iy = np.searchsorted(ys, d[:, 1])
    F[iy, ix] = d[:, 2]
    return xs, ys, F


def basin_dF(xs, ys, F, T: float):
    """Free energy of phi>0 relative to phi<0 (kJ/mol), by Boltzmann-summing the FES."""
    import numpy as np
    kT = KB_KJ * T
    p = np.exp(-(F - np.nanmin(F)) / kT)
    p = np.nan_to_num(p)
    pos = p[:, xs > 0].sum()
    neg = p[:, xs <= 0].sum()
    return float(-kT * math.log(pos / neg)) if pos > 0 and neg > 0 else float("nan")


def analyse_dipeptide_metad(cfg: Config, n_snapshots: int = 10):
    import numpy as np
    import matplotlib.pyplot as plt
    w = Paths(cfg).root / "L5_metad"
    m = w / "metad"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    if (w / "unbiased" / "COLVAR").exists():
        cv = read_colvar(w / "unbiased" / "COLVAR")
        axes[0].scatter(cv["phi"], cv["psi"], s=2, alpha=0.5)
        axes[0].set(xlim=(-math.pi, math.pi), ylim=(-math.pi, math.pi), xlabel="φ (rad)", ylabel="ψ (rad)",
                    title="Unbiased MD: where it went")
    cvm = read_colvar(m / "COLVAR")
    axes[1].scatter(cvm["phi"], cvm["psi"], s=2, alpha=0.5, c=cvm["time"], cmap="viridis")
    axes[1].set(xlim=(-math.pi, math.pi), ylim=(-math.pi, math.pi), xlabel="φ (rad)", ylabel="ψ (rad)",
                title="Metadynamics: where it went")
    sum_hills(m)
    xs, ys, F = read_fes2d(m / "fes.dat")
    cs = axes[2].contourf(xs, ys, np.clip(F, 0, 50), levels=25, cmap="viridis")
    fig.colorbar(cs, ax=axes[2], label="Free energy (kJ/mol)")
    axes[2].set(xlabel="φ (rad)", ylabel="ψ (rad)", title="Free-energy surface")
    plot_save(fig, w / "dipeptide_fes.png")

    # convergence: dF(phi>0 vs phi<0) as hills accumulate
    nh = sum(1 for l in (m / "HILLS").read_text().splitlines() if l.strip() and not l.startswith("#"))
    stride = max(1, nh // n_snapshots)
    sum_hills(m, stride=stride)
    snaps = sorted(m.glob("fes_t*.dat"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    pace_ps = cfg.dt * 500
    dfs = []
    for i, f in enumerate(snaps):
        xs_, ys_, F_ = read_fes2d(f)
        dfs.append(((i + 1) * stride * pace_ps / 1000, basin_dF(xs_, ys_, F_, cfg.temperature)))
    if dfs:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot([d[0] for d in dfs], [d[1] for d in dfs], "o-")
        ax.set(xlabel="Simulation time (ns)", ylabel="ΔF(φ>0 − φ<0) (kJ/mol)",
               title="Convergence: should level off")
        plot_save(fig, w / "dipeptide_convergence.png")
    out = {"dF_phi_pos_minus_neg_kJ": dfs[-1][1] if dfs else None, "convergence": dfs}
    (w / "metad_summary.json").write_text(json.dumps(out, indent=2))
    return out


def ligand_plumed(nprot: int, lig: list, pocket: list, wall_nm: float = 2.5, pace: int = 500,
                  height: float = 1.0, sigma: float = 0.05, biasfactor: float = 10, temp: float = 300) -> str:
    j = lambda xs: ",".join(map(str, xs))
    return (f"WHOLEMOLECULES ENTITY0=1-{nprot} ENTITY1={lig[0]}-{lig[-1]}\n"
            f"lig: COM ATOMS={lig[0]}-{lig[-1]}\n"
            f"pocket: COM ATOMS={j(pocket)}\n"
            f"d: DISTANCE ATOMS=lig,pocket\n"
            f"uwall: UPPER_WALLS ARG=d AT={wall_nm} KAPPA=2000.0\n"
            f"metad: METAD ARG=d PACE={pace} HEIGHT={height} SIGMA={sigma} BIASFACTOR={biasfactor} "
            f"TEMP={temp} FILE=HILLS GRID_MIN=0 GRID_MAX={wall_nm + 1.0} GRID_BIN={int((wall_nm + 1.0) / 0.01)}\n"
            f"PRINT ARG=d,metad.bias,uwall.bias STRIDE=500 FILE=COLVAR\n")


def run_ligand_metad(cfg: Config, ns: float = 20.0, pocket_cutoff_nm: float = 0.8,
                     wall_nm: float = 2.5, **metad_kw) -> Path:
    """Well-tempered metadynamics on the ligand-pocket distance (Level 1 system)."""
    banner("Level 5B (optional): ligand metadynamics")
    if not plumed_available(cfg):
        raise RuntimeError("GROMACS cannot use PLUMED here - see the install cell of the notebook")
    P, R, _ = mp.prepare_equilibrated(cfg)
    atoms, box = read_gro(P.npt / "npt.gro")
    prot = protein_atoms(atoms)
    lig = [a["index"] for a in atoms if a["resname"] == LIG]
    lc = centre_pbc([atoms[i - 1]["xyz"] for i in lig], box)
    pocket = [a["index"] for a in prot if a["name"] == "CA" and
              math.dist([0, 0, 0], min_image([a["xyz"][i] - lc[i] for i in range(3)], box)) < pocket_cutoff_nm]
    if len(pocket) < 3:
        raise RuntimeError("Fewer than 3 C-alpha atoms near the ligand - increase pocket_cutoff_nm")
    if wall_nm > 0.95 * 0.5 * min(math.sqrt(sum(x * x for x in r)) for r in box_matrix(box)):
        print("    WARNING: wall is close to half the box; distances may wrap through the periodic box")
    w = stage_dir(cfg, "L5_metad_ligand")
    (w / "groups.json").write_text(json.dumps({"ligand": lig, "pocket_CA": pocket, "n_protein_atoms": len(prot)}))
    mdp = mp._dyn(cfg, mp._steps(cfg, ns * 1000), max(1, mp._steps(cfg, 10.0)), False, True, True)
    run_md(cfg, P, R, w, "metad", mdp, P.npt / "npt.gro", cpt=P.npt / "npt.cpt",
           plumed=ligand_plumed(len(prot), lig, pocket, wall_nm, temp=cfg.temperature, **metad_kw))
    return w


def analyse_ligand_metad(cfg: Config):
    import numpy as np
    import matplotlib.pyplot as plt
    w = Paths(cfg).root / "L5_metad_ligand"
    cv = read_colvar(w / "COLVAR")
    sum_hills(w)
    d = np.loadtxt(w / "fes.dat", comments="#")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(cv["time"] / 1000, cv["d"], lw=0.6)
    a1.set(xlabel="Time (ns)", ylabel="Ligand-pocket distance (nm)", title="Does the ligand leave?")
    a2.plot(d[:, 0], d[:, 1])
    a2.set(xlabel="Ligand-pocket distance (nm)", ylabel="Free energy (kJ/mol)",
           title="1D free-energy profile (not converged unless it rebinds)")
    plot_save(fig, w / "ligand_metad.png")
    first_exit = next((t for t, x in zip(cv["time"], cv["d"]) if x > 1.5), None)
    print(f"    max distance {cv['d'].max():.2f} nm; first time beyond 1.5 nm: "
          f"{'never' if first_exit is None else f'{first_exit / 1000:.1f} ns'}")
    return {"max_distance_nm": float(cv["d"].max()),
            "first_exit_ns": None if first_exit is None else first_exit / 1000}
