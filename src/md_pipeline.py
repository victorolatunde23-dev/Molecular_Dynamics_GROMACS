#!/usr/bin/env python3
"""
md_pipeline.py  -  Reusable protein-ligand MD pipeline (GROMACS + local GPU)
==============================================================================

Designed to run from a Google Colab notebook connected to a LOCAL runtime, so
every GROMACS call executes on your own machine and uses your own NVIDIA GPU.
It also works as a plain command-line script.

Default example: T4 lysozyme L99A with benzene (PDB 181L, ligand resname BNZ),
a small, well-studied protein-ligand complex (~2.6k protein atoms).

Pipeline
--------
 00_input           downloaded / copied structure
 01_ligand          ligand extraction, protonation (Open Babel), GAFF2 + AM1-BCC (acpype)
 02_protein         protein cleaning (PDBFixer, optional)
 03_system          pdb2gmx (AMBER99SB-ILDN / TIP3P), complex, box, water, ions
 04_em              energy minimisation
 05_nvt             NVT equilibration (protein + ligand heavy atoms restrained)
 06_npt             NPT equilibration (restrained)
 07_production      unrestrained production MD (resumable from checkpoint)
 08_analysis        PBC-corrected/fitted trajectory, RMSD, RMSF, Rg, contacts,
                    H-bonds, energies -> .xvg + .csv + .png + summary.json
 09_visualization   complex-only trajectory + PDBs + ChimeraX scripts (.cxc)
 logs               full stdout/stderr of every command

Re-running is safe: finished stages are skipped (set overwrite=True to redo),
and an interrupted production run resumes from md.cpt.

One-time setup on your LOCAL machine (Linux, or Windows via WSL2; NVIDIA driver installed)
-------------------------------------------------------------------------------
  conda create -n md -c conda-forge python=3.11 "gromacs=*=nompi_cuda*" \
        acpype ambertools openbabel pdbfixer numpy matplotlib py3dmol jupyter
  conda activate md
  gmx --version | grep "GPU support"        # should print: CUDA

  # start Jupyter so Colab can connect to it
  jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' \
        --port=8888 --NotebookApp.port_retries=0
  #   (Notebook >= 7:  --ServerApp.allow_origin=... --ServerApp.port_retries=0)
  # In Colab: Connect (top right) > "Connect to a local runtime" > paste the
  # http://localhost:8888/?token=... URL printed by Jupyter.

Use in a Colab cell
-------------------
  import sys; sys.path.append("/path/to/folder/with/this/script")
  from md_pipeline import Config, run_pipeline, open_in_chimerax, show_in_notebook

  cfg = Config(name="t4l_benzene", pdb_id="181L", ligand_resname="BNZ",
               ligand_charge=0, prod_ns=10)
  run_pipeline(cfg)
  open_in_chimerax(cfg)          # launches ChimeraX on your desktop
  show_in_notebook(cfg)          # quick 3D view inside the notebook (py3Dmol)

Your own system
---------------
  Config(name="my_complex", pdb_file="complex.pdb", ligand_resname="XYZ",
         ligand_charge=-1)
  # Recommended for drug-like ligands (PDB files carry no bond orders):
  Config(name="my_complex", pdb_file="protein.pdb",
         ligand_file="ligand_docked_withH.sdf", ligand_charge=0)
  # the ligand file must already be posed in the protein's coordinate frame
  # and contain all hydrogens. Apo protein: leave ligand_resname/ligand_file None.

Command line
------------
  python md_pipeline.py --pdb-id 181L --ligand-resname BNZ --ligand-charge 0 \
         --name t4l_benzene --prod-ns 10 --chimerax
  python md_pipeline.py --config my_system.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, fields
from pathlib import Path

LIG = "LIG"  # residue / molecule name used for the ligand everywhere


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- system ---
    name: str = "t4l_benzene"
    pdb_id: str | None = "181L"          # downloaded from RCSB if pdb_file is None
    pdb_file: str | None = None          # local PDB (protein, or protein+ligand)
    ligand_resname: str | None = "BNZ"   # ligand residue name inside the PDB
    ligand_file: str | None = None       # optional SDF/MOL2 with H (overrides resname)
    ligand_chain: str | None = None      # pick ligand copy from this chain
    ligand_charge: int | None = 0        # net charge; None lets acpype guess
    protein_chains: list[str] | None = None  # e.g. ["A"]; None = all chains
    ph: float = 7.4                      # used when protonating a PDB ligand
    fix_protein: bool = True             # add missing heavy atoms with PDBFixer

    # --- force field ---
    forcefield: str = "amber99sb-ildn"
    water: str = "tip3p"
    ligand_ff: str = "gaff2"
    charge_method: str = "bcc"           # AM1-BCC
    positive_ion: str = "NA"
    negative_ion: str = "CL"

    # --- box / conditions ---
    box_type: str = "dodecahedron"
    box_distance: float = 1.0            # nm, solute-box distance
    salt_conc: float = 0.15              # M
    temperature: float = 300.0           # K
    pressure: float = 1.0                # bar

    # --- lengths ---
    dt: float = 0.002                    # ps
    em_steps: int = 50000
    nvt_ps: float = 100.0
    npt_ps: float = 100.0
    prod_ns: float = 10.0
    traj_interval_ps: float = 10.0       # production frame spacing

    # --- execution ---
    output_root: str = "md_results"
    gmx: str = "gmx"
    use_gpu: bool = True
    gpu_id: str | None = None            # e.g. "0"
    ntomp: int | None = None             # OpenMP threads; None = GROMACS decides
    maxwarn: int = 0
    overwrite: bool = False
    checkpoint_min: float = 5.0          # mdrun checkpoint interval (minutes); short = safe resume
    seed: int = -1                       # velocity seed for NVT (-1 = random); set for replicas
    seed: int = -1                       # velocity + thermostat seed; -1 = random each run

    # --- visualisation ---
    viewer_stride: int = 1               # keep every n-th frame in the viewer trajectory
    chimera_pdb_stride: int = 5          # every n-th frame in the Chimera multi-model PDB (keeps it small)
    chimerax_path: str | None = None     # auto-detected if None
    launch_chimerax: bool = False        # open ChimeraX automatically at the end

    @property
    def has_ligand(self) -> bool:
        return bool(self.ligand_file or self.ligand_resname)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class Paths:
    def __init__(self, cfg: Config):
        self.root = Path(cfg.output_root).expanduser().resolve() / cfg.name
        self.inp = self.root / "00_input"
        self.lig = self.root / "01_ligand"
        self.prot = self.root / "02_protein"
        self.sys = self.root / "03_system"
        self.em = self.root / "04_em"
        self.nvt = self.root / "05_nvt"
        self.npt = self.root / "06_npt"
        self.md = self.root / "07_production"
        self.ana = self.root / "08_analysis"
        self.viz = self.root / "09_visualization"
        self.logs = self.root / "logs"

    def make(self):
        for p in vars(self).values():
            Path(p).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Command runner
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, cfg: Config, paths: Paths):
        self.cfg, self.p = cfg, paths
        self.env = os.environ.copy()
        self.env["GMX_MAXBACKUP"] = "-1"  # no #backup# files cluttering the folders

    def run(self, cmd, step, cwd=None, stdin=None, check=True) -> int:
        cmd = [str(c) for c in cmd]
        log = self.p.logs / f"{step}.log"
        print(f"    $ {' '.join(cmd)}")
        with open(log, "a") as fh:
            fh.write(f"\n### {time.ctime()}  cwd={cwd}\n$ {' '.join(cmd)}\n")
            if stdin:
                fh.write(f"(stdin: {stdin!r})\n")
            fh.flush()
            r = subprocess.run(cmd, cwd=cwd, input=stdin, text=True, env=self.env,
                               stdout=fh, stderr=subprocess.STDOUT)
        if check and r.returncode != 0:
            tail = "".join(log.read_text(errors="replace").splitlines(True)[-40:])
            raise RuntimeError(f"Step '{step}' failed (exit {r.returncode}). "
                               f"Full log: {log}\n----- last lines -----\n{tail}")
        return r.returncode

    def gmx(self, args, step, **kw) -> int:
        return self.run([self.cfg.gmx, *args], step, **kw)


def banner(msg):
    print(f"\n=== {msg} " + "=" * max(0, 70 - len(msg)))


def done(path: Path, cfg: Config) -> bool:
    if path.exists() and not cfg.overwrite:
        print(f"    already done ({path.name}) - skipping")
        return True
    return False


# --------------------------------------------------------------------------- #
# Environment checks
# --------------------------------------------------------------------------- #
def check_environment(cfg: Config) -> dict:
    banner("Checking environment")
    info = {}
    if not shutil.which(cfg.gmx):
        raise RuntimeError(f"'{cfg.gmx}' not found on PATH. Activate the conda env "
                           "that has GROMACS (see the header of this script).")
    out = subprocess.run([cfg.gmx, "--version"], capture_output=True, text=True).stdout
    ver = re.search(r"GROMACS version:\s*(\S+)", out)
    gpu = re.search(r"GPU support:\s*(.+)", out)
    info["gromacs_version"] = ver.group(1) if ver else "unknown"
    info["gromacs_gpu_support"] = gpu.group(1).strip() if gpu else "unknown"
    print(f"    GROMACS {info['gromacs_version']}  (GPU support: {info['gromacs_gpu_support']})")

    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True)
        info["gpus"] = [l.strip() for l in smi.stdout.splitlines() if l.strip()]
    except FileNotFoundError:
        info["gpus"] = []
    for g in info["gpus"]:
        print(f"    GPU: {g}")

    if cfg.use_gpu:
        if "disabled" in info["gromacs_gpu_support"].lower():
            print("    WARNING: this GROMACS build has no GPU support -> running on CPU.\n"
                  "             Install a CUDA build: conda install -c conda-forge 'gromacs=*=nompi_cuda*'")
            cfg.use_gpu = False
        elif not info["gpus"]:
            print("    WARNING: nvidia-smi found no GPU -> running on CPU. Local runtime: is Colab "
                  "connected to it? Colab cloud: Runtime > Change runtime type > GPU.")
            cfg.use_gpu = False

    if cfg.has_ligand:
        for tool in ("acpype", "obabel"):
            if not shutil.which(tool):
                raise RuntimeError(f"'{tool}' not found. conda install -c conda-forge acpype ambertools openbabel")
    return info


# --------------------------------------------------------------------------- #
# Structure preparation
# --------------------------------------------------------------------------- #
def get_structure(cfg: Config, P: Paths) -> Path:
    banner("Input structure")
    dst = P.inp / "input.pdb"
    if done(dst, cfg):
        return dst
    if cfg.pdb_file:
        shutil.copy(Path(cfg.pdb_file).expanduser(), dst)
        print(f"    copied {cfg.pdb_file}")
    elif cfg.pdb_id:
        url = f"https://files.rcsb.org/download/{cfg.pdb_id.upper()}.pdb"
        print(f"    downloading {url}")
        urllib.request.urlretrieve(url, dst)
    else:
        raise ValueError("Give either pdb_id or pdb_file")
    return dst


def _first_model(lines):
    out = []
    for l in lines:
        if l.startswith("ENDMDL"):
            break
        out.append(l)
    return out


def _altloc_ok(line):
    return len(line) > 16 and line[16] in (" ", "A")


def _no_altloc(line):
    return line[:16] + " " + line[17:]


def prepare_protein(cfg: Config, P: Paths, src: Path) -> Path:
    banner("Protein")
    raw = P.prot / "protein_raw.pdb"
    clean = P.prot / "protein_clean.pdb"
    if done(clean, cfg):
        return clean
    keep = []
    for l in _first_model(src.read_text().splitlines()):
        if l.startswith("ATOM  ") and _altloc_ok(l):
            if cfg.protein_chains and l[21] not in cfg.protein_chains:
                continue
            keep.append(_no_altloc(l))
    if not keep:
        raise RuntimeError("No protein ATOM records found in the input structure")
    raw.write_text("\n".join(keep) + "\nEND\n")

    if cfg.fix_protein:
        try:
            from openmm.app import PDBFile
            from pdbfixer import PDBFixer
            fx = PDBFixer(filename=str(raw))
            fx.findMissingResidues()
            fx.missingResidues = {}           # do not model missing loops/termini
            fx.findNonstandardResidues()
            fx.replaceNonstandardResidues()
            fx.removeHeterogens(keepWater=False)
            fx.findMissingAtoms()
            fx.addMissingAtoms()
            with open(clean, "w") as fh:
                PDBFile.writeFile(fx.topology, fx.positions, fh, keepIds=True)
            print("    PDBFixer: added missing heavy atoms, replaced non-standard residues")
            return clean
        except ImportError:
            print("    PDBFixer not installed - using the raw protein (conda install pdbfixer)")
    shutil.copy(raw, clean)
    return clean


def _extract_ligand_pdb(cfg: Config, src: Path, out: Path):
    lines = [l for l in _first_model(src.read_text().splitlines())
             if l.startswith(("HETATM", "ATOM  ")) and l[17:20].strip() == cfg.ligand_resname
             and _altloc_ok(l) and (cfg.ligand_chain is None or l[21] == cfg.ligand_chain)]
    if not lines:
        raise RuntimeError(f"Ligand '{cfg.ligand_resname}' not found in {src}")
    first = (lines[0][21], lines[0][22:27])          # first copy only (chain, resSeq)
    lines = [l for l in lines if (l[21], l[22:27]) == first]
    lines = [_no_altloc(l)[:17] + f"{LIG:>3}" + l[20:] for l in lines]
    out.write_text("\n".join(lines) + "\nEND\n")
    print(f"    extracted {len(lines)} atoms of {cfg.ligand_resname} "
          f"(chain {first[0]}, residue {first[1].strip()})")


def _split_section(text, section):
    """Split an .itp into (lines inside [ section ], all other lines)."""
    inside, other, cur = [], [], None
    for line in text.splitlines():
        m = re.match(r"\s*\[\s*(\w+)\s*\]", line)
        if m:
            cur = m.group(1).lower()
        (inside if cur == section else other).append(line)
    return inside, other


def process_ligand_itp(itp_text: str):
    """Return (atomtypes_itp, molecule_itp, posre_itp, molname) from an acpype itp."""
    atomtypes, rest = _split_section(itp_text, "atomtypes")
    out, cur, molname, heavy = [], None, None, []
    for line in rest:
        m = re.match(r"\s*\[\s*(\w+)\s*\]", line)
        if m:
            cur = m.group(1).lower()
            out.append(line)
            continue
        body = line.split(";")[0].split()
        if cur == "moleculetype" and body and molname is None:
            molname = body[0]
        if cur == "atoms" and len(body) >= 8:
            body[3] = LIG                                   # residue name
            if float(body[7]) > 1.5:                        # heavy atom
                heavy.append(int(body[0]))
            comment = line.split(";", 1)[1] if ";" in line else ""
            line = ("{:>6} {:>10} {:>6} {:>6} {:>6} {:>6} {:>12} {:>10}".format(*body[:8])
                    + ("".join(f" {x}" for x in body[8:]))
                    + (f" ;{comment}" if comment else ""))
        out.append(line)
    posre = ["; position restraints on ligand heavy atoms",
             "[ position_restraints ]", ";  ai  funct  fcx  fcy  fcz"]
    posre += [f"{i:6d}     1  1000  1000  1000" for i in heavy]
    return "\n".join(atomtypes) + "\n", "\n".join(out) + "\n", "\n".join(posre) + "\n", molname


def prepare_ligand(cfg: Config, P: Paths, R: Runner, src: Path) -> str:
    banner("Ligand parameters (GAFF2 / AM1-BCC via acpype)")
    itp_out = P.sys / f"{LIG}.itp"
    molname_file = P.lig / "molname.txt"
    if itp_out.exists() and molname_file.exists() and not cfg.overwrite:
        print("    already done - skipping")
        return molname_file.read_text().strip()

    mol2 = P.lig / "ligand.mol2"
    if cfg.ligand_file:
        lf = Path(cfg.ligand_file).expanduser()
        ext = lf.suffix.lower().lstrip(".")
        shutil.copy(lf, P.lig / f"ligand_input.{ext}")
        R.run(["obabel", f"ligand_input.{ext}", "-O", "ligand.mol2"], "01_ligand", cwd=P.lig)
        print("    using the supplied ligand file as-is (it must contain hydrogens)")
    else:
        _extract_ligand_pdb(cfg, src, P.lig / "ligand_raw.pdb")
        R.run(["obabel", "-ipdb", "ligand_raw.pdb", "-omol2", "-O", "ligand.mol2", "-p", str(cfg.ph)],
              "01_ligand", cwd=P.lig)
        print(f"    protonated at pH {cfg.ph} with Open Babel - CHECK ligand.mol2 (bond orders are "
              "guessed from PDB geometry; supply an SDF via ligand_file for drug-like ligands)")
    if not mol2.exists():
        raise RuntimeError("Open Babel did not produce ligand.mol2")

    cmd = ["acpype", "-i", "ligand.mol2", "-b", LIG, "-c", cfg.charge_method,
           "-a", cfg.ligand_ff, "-o", "gmx"]
    if cfg.ligand_charge is not None:
        cmd += ["-n", str(cfg.ligand_charge)]
    R.run(cmd, "01_ligand", cwd=P.lig)

    adir = P.lig / f"{LIG}.acpype"
    itp = adir / f"{LIG}_GMX.itp"
    gro = adir / f"{LIG}_GMX.gro"
    if not itp.exists():
        raise RuntimeError(f"acpype failed - see {P.logs / '01_ligand.log'}")

    types, mol, posre, molname = process_ligand_itp(itp.read_text())
    (P.sys / f"{LIG}_atomtypes.itp").write_text(types)
    itp_out.write_text(mol)
    (P.sys / f"posre_{LIG}.itp").write_text(posre)
    # ligand coordinates with residue name LIG
    g = gro.read_text().splitlines()
    n = int(g[1])
    atoms = [l[:5] + f"{LIG:<5}" + l[10:] for l in g[2:2 + n]]
    (P.sys / f"{LIG}.gro").write_text("\n".join([g[0], g[1], *atoms, g[2 + n]]) + "\n")
    molname_file.write_text(molname)
    print(f"    ligand topology ready (moleculetype '{molname}', {n} atoms)")
    return molname


def merge_gro(protein_gro: Path, ligand_gro: Path, out: Path):
    p = protein_gro.read_text().splitlines()
    l = ligand_gro.read_text().splitlines()
    pn, ln = int(p[1]), int(l[1])
    atoms = p[2:2 + pn] + l[2:2 + ln]
    out.write_text("\n".join(["Protein-ligand complex", f"{len(atoms):5d}", *atoms, p[2 + pn]]) + "\n")


def add_ligand_to_top(top: Path, molname: str):
    text = top.read_text()
    if f'#include "{LIG}.itp"' in text:
        return
    lines = text.splitlines()
    i = next(i for i, l in enumerate(lines) if re.search(r'#include\s+".*forcefield\.itp"', l))
    lines[i + 1:i + 1] = ["", "; Ligand atom types", f'#include "{LIG}_atomtypes.itp"']
    j = next((i for i, l in enumerate(lines) if "Include water topology" in l), None)
    if j is None:
        j = next(i for i, l in enumerate(lines) if re.match(r"\s*\[\s*system\s*\]", l))
    lines[j:j] = ["; Include ligand topology", f'#include "{LIG}.itp"',
                  f"#ifdef POSRES_{LIG}", f'#include "posre_{LIG}.itp"', "#endif", ""]
    lines.append(f"{molname:<20}1")
    top.write_text("\n".join(lines) + "\n")


def build_system(cfg: Config, P: Paths, R: Runner, protein_pdb: Path, molname: str | None):
    banner("Topology, box, solvent and ions")
    final = P.sys / "system.gro"
    if done(final, cfg):
        return final
    shutil.copy(protein_pdb, P.sys / "protein_clean.pdb")
    R.gmx(["pdb2gmx", "-f", "protein_clean.pdb", "-o", "protein.gro", "-p", "topol.top",
           "-i", "posre.itp", "-ff", cfg.forcefield, "-water", cfg.water, "-ignh"],
          "03_pdb2gmx", cwd=P.sys)
    if molname:
        merge_gro(P.sys / "protein.gro", P.sys / f"{LIG}.gro", P.sys / "complex.gro")
        add_ligand_to_top(P.sys / "topol.top", molname)
    else:
        shutil.copy(P.sys / "protein.gro", P.sys / "complex.gro")
    R.gmx(["editconf", "-f", "complex.gro", "-o", "boxed.gro", "-c",
           "-d", cfg.box_distance, "-bt", cfg.box_type], "03_box", cwd=P.sys)
    R.gmx(["solvate", "-cp", "boxed.gro", "-cs", "spc216.gro", "-o", "solvated.gro",
           "-p", "topol.top"], "03_solvate", cwd=P.sys)
    (P.sys / "ions.mdp").write_text(mdp_em(cfg))
    R.gmx(["grompp", "-f", "ions.mdp", "-c", "solvated.gro", "-p", "topol.top", "-o", "ions.tpr",
           "-po", "ions_out.mdp", "-maxwarn", max(cfg.maxwarn, 2)], "03_ions", cwd=P.sys)
    R.gmx(["genion", "-s", "ions.tpr", "-o", "system.gro", "-p", "topol.top",
           "-pname", cfg.positive_ion, "-nname", cfg.negative_ion,
           "-neutral", "-conc", cfg.salt_conc], "03_ions", cwd=P.sys, stdin="SOL\n")
    n = int((P.sys / "system.gro").read_text().splitlines()[1])
    print(f"    system built: {n:,} atoms")
    return final


# --------------------------------------------------------------------------- #
# MDP templates
# --------------------------------------------------------------------------- #
_COMMON = """cutoff-scheme       = Verlet
nstlist             = 10
pbc                 = xyz
coulombtype         = PME
rcoulomb            = 1.0
rvdw                = 1.0
fourierspacing      = 0.16
pme_order           = 4
DispCorr            = EnerPres
"""


def mdp_em(cfg):
    return f"""; energy minimisation
integrator          = steep
emtol               = 1000.0
emstep              = 0.01
nsteps              = {cfg.em_steps}
{_COMMON}"""


def _dyn(cfg, nsteps, nout, posres, pcoupl, continuation):
    define = f"-DPOSRES -DPOSRES_{LIG}" if posres else ""
    s = f"""define              = {define}
integrator          = md
dt                  = {cfg.dt}
nsteps              = {nsteps}
nstxout-compressed  = {nout}
nstenergy           = {nout}
nstlog              = {nout}
continuation        = {"yes" if continuation else "no"}
constraint_algorithm = lincs
constraints         = h-bonds
lincs_iter          = 1
lincs_order         = 4
{_COMMON}tcoupl              = V-rescale
tc-grps             = System
tau_t               = 0.1
ref_t               = {cfg.temperature}
ld_seed             = {cfg.seed}
"""
    if pcoupl:
        s += f"""pcoupl              = C-rescale
pcoupltype          = isotropic
tau_p               = 2.0
ref_p               = {cfg.pressure}
compressibility     = 4.5e-5
refcoord_scaling    = com
"""
    else:
        s += "pcoupl              = no\n"
    s += (f"gen_vel             = yes\ngen_temp            = {cfg.temperature}\ngen_seed            = {cfg.seed}\n"
          if not continuation else "gen_vel             = no\n")
    return s


def _steps(cfg, ps):
    return int(round(ps / cfg.dt))


# --------------------------------------------------------------------------- #
# Simulation stages
# --------------------------------------------------------------------------- #
def _gpu_variants(cfg, stage):
    if not cfg.use_gpu:
        return [[]]
    if stage == "em":
        return [["-nb", "gpu"], []]
    return [["-nb", "gpu", "-pme", "gpu", "-bonded", "gpu", "-update", "gpu"],
            ["-nb", "gpu", "-pme", "gpu", "-bonded", "gpu"],
            ["-nb", "gpu"],
            []]


def mdrun(cfg, R, workdir, deffnm, step, stage, cpi=None, extra=None):
    base = ["mdrun", "-deffnm", deffnm, "-cpt", cfg.checkpoint_min, *(extra or [])]
    if cfg.ntomp:
        base += ["-ntomp", cfg.ntomp]
    if cfg.use_gpu and "mpi" not in Path(cfg.gmx).name:
        base += ["-ntmpi", "1"]
    if cfg.use_gpu and cfg.gpu_id is not None:
        base += ["-gpu_id", cfg.gpu_id]
    if cpi:
        base += ["-cpi", cpi]
    for flags in _gpu_variants(cfg, stage):
        if R.gmx(base + flags, step, cwd=workdir, check=False) == 0:
            print(f"    mdrun OK  (offload: {' '.join(flags) if flags else 'CPU only'})")
            return flags
        print(f"    mdrun failed with [{' '.join(flags) or 'CPU'}] - retrying with less GPU offload")
    raise RuntimeError(f"mdrun failed for {deffnm}; see {R.p.logs / (step + '.log')} "
                       f"and {Path(workdir) / (deffnm + '.log')}")


def grompp(cfg, R, P, mdp_text, workdir: Path, name, step, coord, ref=None, cpt=None):
    (workdir / f"{name}.mdp").write_text(mdp_text)
    cmd = ["grompp", "-f", workdir / f"{name}.mdp", "-c", coord, "-p", "topol.top",
           "-o", workdir / f"{name}.tpr", "-po", workdir / f"{name}_out.mdp", "-maxwarn", cfg.maxwarn]
    if ref:
        cmd += ["-r", ref]
    if cpt:
        cmd += ["-t", cpt]
    R.gmx(cmd, step, cwd=P.sys)   # run in 03_system so relative #includes resolve


def performance(log: Path):
    if log.exists():
        m = re.search(r"Performance:\s+([\d.]+)", log.read_text(errors="replace"))
        return float(m.group(1)) if m else None
    return None


def run_simulations(cfg: Config, P: Paths, R: Runner) -> dict:
    used = run_equilibration(cfg, P, R)
    return run_production(cfg, P, R, used)


def run_equilibration(cfg: Config, P: Paths, R: Runner) -> dict:
    used = {}
    banner("Energy minimisation")
    if not done(P.em / "em.gro", cfg):
        grompp(cfg, R, P, mdp_em(cfg), P.em, "em", "04_em", P.sys / "system.gro")
        used["em"] = mdrun(cfg, R, P.em, "em", "04_em", "em")
        txt = (P.em / "em.log").read_text(errors="replace")
        for key in ("Potential Energy", "Maximum force"):
            m = re.search(rf"{key}\s*=\s*(\S+)", txt)
            if m:
                print(f"    {key}: {m.group(1)}")

    nout_eq = max(1, _steps(cfg, 1.0))  # every 1 ps during equilibration
    banner("NVT equilibration")
    if not done(P.nvt / "nvt.gro", cfg):
        grompp(cfg, R, P, _dyn(cfg, _steps(cfg, cfg.nvt_ps), nout_eq, True, False, False),
               P.nvt, "nvt", "05_nvt", P.em / "em.gro", ref=P.em / "em.gro")
        used["nvt"] = mdrun(cfg, R, P.nvt, "nvt", "05_nvt", "md")

    banner("NPT equilibration")
    if not done(P.npt / "npt.gro", cfg):
        grompp(cfg, R, P, _dyn(cfg, _steps(cfg, cfg.npt_ps), nout_eq, True, True, True),
               P.npt, "npt", "06_npt", P.nvt / "nvt.gro", ref=P.nvt / "nvt.gro", cpt=P.nvt / "nvt.cpt")
        used["npt"] = mdrun(cfg, R, P.npt, "npt", "06_npt", "md")
    return used


def run_production(cfg: Config, P: Paths, R: Runner, used: dict | None = None) -> dict:
    used = dict(used or {})
    banner(f"Production MD ({cfg.prod_ns} ns)")
    if not done(P.md / "md.gro", cfg):
        cpt = P.md / "md.cpt"
        resume = cpt.exists() and (P.md / "md.tpr").exists() and not cfg.overwrite
        if resume:
            print("    checkpoint found - resuming production run")
        else:
            grompp(cfg, R, P, _dyn(cfg, _steps(cfg, cfg.prod_ns * 1000),
                                   max(1, _steps(cfg, cfg.traj_interval_ps)), False, True, True),
                   P.md, "md", "07_md", P.npt / "npt.gro", cpt=P.npt / "npt.cpt")
        used["md"] = mdrun(cfg, R, P.md, "md", "07_md", "md", cpi=cpt if resume else None)
    perf = performance(P.md / "md.log")
    if perf:
        print(f"    production performance: {perf} ns/day")
    return {"mdrun_offload": used, "production_ns_per_day": perf}


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def read_ndx_names(ndx: Path):
    return re.findall(r"^\s*\[\s*(.+?)\s*\]", ndx.read_text(), flags=re.M)


def make_index(cfg, P, R, tpr) -> list[str]:
    ndx = P.ana / "index.ndx"
    tmp = P.ana / "index_default.ndx"
    R.gmx(["make_ndx", "-f", tpr, "-o", tmp], "08_index", stdin="q\n")
    groups = read_ndx_names(tmp)
    cmds = []
    if cfg.has_ligand:
        if LIG not in groups:
            cmds += [f"r {LIG}", f"name {len(groups)} {LIG}"]
            groups.append(LIG)
        cmds += [f"{groups.index('Protein')} | {groups.index(LIG)}", f"name {len(groups)} Complex"]
    else:
        cmds += [f"{groups.index('Protein')}", f"name {len(groups)} Complex"]
    groups.append("Complex")
    R.gmx(["make_ndx", "-f", tpr, "-n", tmp, "-o", ndx], "08_index",
          stdin="\n".join(cmds) + "\nq\n")
    return read_ndx_names(ndx)


def read_xvg(path: Path):
    rows, legends, xl, yl, title = [], [], "", "", path.stem
    clean = lambda s: re.sub(r"\\[A-Za-z]", "", s)
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("@"):
            if m := re.search(r'xaxis\s+label\s+"(.*)"', line):
                xl = clean(m.group(1))
            elif m := re.search(r'yaxis\s+label\s+"(.*)"', line):
                yl = clean(m.group(1))
            elif m := re.search(r'^@\s+title\s+"(.*)"', line):
                title = clean(m.group(1))
            elif m := re.search(r's\d+\s+legend\s+"(.*)"', line):
                legends.append(clean(m.group(1)))
            continue
        try:
            rows.append([float(x) for x in line.split()])
        except ValueError:
            pass
    return rows, xl, yl, title, legends


def xvg_to_outputs(xvg: Path, csv_dir: Path, png_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    rows, xl, yl, title, legends = read_xvg(xvg)
    if not rows:
        return None
    width = min(len(r) for r in rows)
    a = np.array([r[:width] for r in rows])
    names = [xl or "x"] + (legends if len(legends) == width - 1 else
                           [yl or f"y{i}" for i in range(1, width)])
    np.savetxt(csv_dir / f"{xvg.stem}.csv", a, delimiter=",",
               header=",".join(n.replace(",", ";") for n in names), comments="")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i in range(1, width):
        ax.plot(a[:, 0], a[:, i], lw=0.9, label=names[i] if width > 2 else None)
    ax.set(xlabel=xl, ylabel=yl, title=title)
    if width > 2:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_dir / f"{xvg.stem}.png", dpi=150)
    plt.close(fig)
    y = a[:, 1]
    half = y[len(y) // 2:]
    return {"label": yl, "mean": float(y.mean()), "std": float(y.std()),
            "min": float(y.min()), "max": float(y.max()), "final": float(y[-1]),
            "mean_second_half": float(half.mean()), "std_second_half": float(half.std())}


def analyse(cfg: Config, P: Paths, R: Runner) -> dict:
    banner("Analysis")
    xd, cd, pd = P.ana / "xvg", P.ana / "csv", P.ana / "plots"
    for d in (xd, cd, pd):
        d.mkdir(exist_ok=True)
    tpr, xtc = P.md / "md.tpr", P.md / "md.xtc"
    groups = make_index(cfg, P, R, tpr)
    ndx = P.ana / "index.ndx"
    g = lambda name: str(groups.index(name))
    lig = cfg.has_ligand
    fit = P.ana / "md_fit.xtc"

    if not done(fit, cfg):
        center = P.ana / "md_center.xtc"
        R.gmx(["trjconv", "-s", tpr, "-f", xtc, "-n", ndx, "-o", center, "-center",
               "-pbc", "mol", "-ur", "compact"], "08_trjconv",
              stdin=f"{g('Complex')}\n{g('System')}\n")
        R.gmx(["trjconv", "-s", tpr, "-f", center, "-n", ndx, "-o", fit, "-fit", "rot+trans"],
              "08_trjconv", stdin=f"{g('Backbone')}\n{g('System')}\n")
        center.unlink(missing_ok=True)

    common = ["-s", tpr, "-f", fit, "-n", ndx]

    def gmx_try(step, *variants):
        for args, stdin in variants:
            if R.gmx(args, step, stdin=stdin, check=False) == 0:
                return True
        print(f"    (skipped {step}: not supported by this GROMACS version - see logs)")
        return False

    gmx_try("08_rmsd", (["rms", *common, "-o", xd / "rmsd_backbone.xvg", "-tu", "ns"],
                        f"{g('Backbone')}\n{g('Backbone')}\n"))
    gmx_try("08_rmsf", (["rmsf", *common, "-o", xd / "rmsf_calpha.xvg", "-res"], f"{g('C-alpha')}\n"))
    gmx_try("08_gyrate",
            (["gyrate", *common, "-o", xd / "gyrate_protein.xvg"], f"{g('Protein')}\n"),
            (["gyrate", *common, "-o", xd / "gyrate_protein.xvg", "-sel", 'group "Protein"'], None))
    if lig:
        gmx_try("08_rmsd", (["rms", *common, "-o", xd / "rmsd_ligand.xvg", "-tu", "ns"],
                            f"{g('Backbone')}\n{g(LIG)}\n"))
        gmx_try("08_contacts", (["mindist", *common, "-od", xd / "mindist_protein_ligand.xvg",
                                 "-on", xd / "contacts_protein_ligand.xvg", "-d", "0.4", "-tu", "ns"],
                                f"{g('Protein')}\n{g(LIG)}\n"))
        gmx_try("08_hbond",
                (["hbond", *common, "-r", 'group "Protein"', "-t", f'group "{LIG}"',
                  "-num", xd / "hbonds_protein_ligand.xvg", "-tu", "ns"], None),
                (["hbond", *common, "-num", xd / "hbonds_protein_ligand.xvg", "-tu", "ns"],
                 f"{g('Protein')}\n{g(LIG)}\n"))

    energy_terms = [("em", P.em / "em.edr", "Potential"),
                    ("nvt", P.nvt / "nvt.edr", "Temperature"),
                    ("npt", P.npt / "npt.edr", "Pressure"), ("npt", P.npt / "npt.edr", "Density"),
                    ("md", P.md / "md.edr", "Potential"), ("md", P.md / "md.edr", "Temperature"),
                    ("md", P.md / "md.edr", "Pressure"), ("md", P.md / "md.edr", "Density")]
    for stage, edr, term in energy_terms:
        if edr.exists():
            R.gmx(["energy", "-f", edr, "-o", xd / f"energy_{stage}_{term.lower()}.xvg"],
                  "08_energy", stdin=f"{term}\n0\n", check=False)

    summary = {}
    for x in sorted(xd.glob("*.xvg")):
        s = xvg_to_outputs(x, cd, pd)
        if s:
            summary[x.stem] = s
    (P.ana / "summary.json").write_text(json.dumps(summary, indent=2))
    dashboard(P, summary)
    for k in ("rmsd_backbone", "rmsd_ligand", "contacts_protein_ligand", "hbonds_protein_ligand"):
        if k in summary:
            s = summary[k]
            print(f"    {k:<26} mean {s['mean']:.3f} ± {s['std']:.3f}  (2nd half {s['mean_second_half']:.3f})")
    return summary


def dashboard(P: Paths, summary: dict):
    import matplotlib.pyplot as plt
    import numpy as np
    panels = [("rmsd_backbone", "Backbone RMSD (nm)"), ("rmsd_ligand", "Ligand RMSD, fit on backbone (nm)"),
              ("rmsf_calpha", "C-alpha RMSF (nm)"), ("gyrate_protein", "Radius of gyration (nm)"),
              ("contacts_protein_ligand", "Protein-ligand contacts < 0.4 nm"),
              ("hbonds_protein_ligand", "Protein-ligand H-bonds")]
    panels = [p for p in panels if p[0] in summary]
    if not panels:
        return
    cols = 2
    rows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows), squeeze=False)
    for ax, (key, title) in zip(axes.flat, panels):
        a = np.loadtxt(P.ana / "csv" / f"{key}.csv", delimiter=",", skiprows=1, ndmin=2)
        xl = (P.ana / "csv" / f"{key}.csv").read_text().splitlines()[0].split(",")[0]
        ax.plot(a[:, 0], a[:, 1], lw=0.9)
        ax.set(title=title, xlabel=xl)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(P.root.name)
    fig.tight_layout()
    fig.savefig(P.ana / "dashboard.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Visualisation (ChimeraX + notebook)
# --------------------------------------------------------------------------- #
def export_viewer_files(cfg: Config, P: Paths, R: Runner):
    banner("Visualisation files")
    tpr, fit, ndx = P.md / "md.tpr", P.ana / "md_fit.xtc", P.ana / "index.ndx"
    groups = read_ndx_names(ndx)
    cpx = str(groups.index("Complex"))
    common = ["-s", tpr, "-f", fit, "-n", ndx]
    R.gmx(["trjconv", *common, "-o", P.viz / "complex_start.pdb", "-dump", "0"], "09_viz", stdin=cpx + "\n")
    R.gmx(["trjconv", *common, "-o", P.viz / "complex_final.pdb", "-dump", cfg.prod_ns * 1000],
          "09_viz", stdin=cpx + "\n")
    R.gmx(["trjconv", *common, "-o", P.viz / "complex_traj.xtc", "-skip", cfg.viewer_stride],
          "09_viz", stdin=cpx + "\n")
    write_chimerax_scripts(cfg, P)
    print(f"    ChimeraX scripts: {P.viz / 'view_trajectory.cxc'}\n"
          f"                      {P.viz / 'view_start_vs_final.cxc'}")
    export_chimera_files(cfg, P, R)


def export_chimera_files(cfg: Config, P: Paths, R: Runner):
    """Files for UCSF Chimera (the older program, not ChimeraX): a multi-model PDB
    trajectory for Tools > MD/Ensemble Analysis > MD Movie, plus .cmd style scripts."""
    tpr, fit, ndx = P.md / "md.tpr", P.ana / "md_fit.xtc", P.ana / "index.ndx"
    groups = read_ndx_names(ndx)
    cpx = str(groups.index("Complex"))
    raw = P.viz / "complex_traj_multimodel_raw.pdb"
    R.gmx(["trjconv", "-s", tpr, "-f", fit, "-n", ndx, "-o", raw,
           "-skip", cfg.chimera_pdb_stride], "09_viz", stdin=cpx + "\n")
    n = clean_multimodel_pdb(raw, P.viz / "complex_traj_multimodel.pdb")
    raw.unlink(missing_ok=True)
    print(f"    Chimera trajectory: {n} frames")
    write_chimera_scripts(cfg, P)
    print(f"    Chimera files:    {P.viz / 'complex_traj_multimodel.pdb'} (MD Movie, PDB format)\n"
          f"                      {P.viz / 'chimera_style.cmd'}, {P.viz / 'chimera_start_vs_final.cmd'}")


def clean_multimodel_pdb(src: Path, dst: Path) -> int:
    """Rewrite a GROMACS multi-frame PDB in the strict form Chimera's MD Movie expects:
    MODEL records numbered 1, 2, 3, ... in columns 11-14, only ATOM/HETATM/TER inside
    each model, ENDMDL after each, and a single END at the very end."""
    out, n, inside, frame = [], 0, False, []

    def flush():
        nonlocal n
        if frame:
            n += 1
            out.append(f"MODEL     {n:4d}")
            out.extend(frame)
            out.append("ENDMDL")

    for line in Path(src).read_text(errors="replace").splitlines():
        rec = line[:6]
        if rec.startswith("MODEL"):
            flush()
            frame, inside = [], True
        elif rec.startswith("ENDMDL"):
            flush()
            frame, inside = [], False
        elif rec in ("ATOM  ", "HETATM"):
            line = line.ljust(80)
            # fill a blank chain ID so every field has a value
            if line[21] == " ":
                line = line[:21] + "A" + line[22:]
            frame.append(line.rstrip())
        elif rec.startswith("TER"):
            frame.append("TER")
    flush()  # file without a final ENDMDL
    out.append("END")
    Path(dst).write_text("\n".join(out) + "\n")
    return n


def _chimera_style(model, lig):
    s = [f"~display {model}", f"ribbon {model}", f"ribcolor lightgray {model}"]
    if lig:
        s += [f"display {model}:{LIG}", f"represent bs {model}:{LIG}",
              f"color orange {model}:{LIG}", f"color byhet {model}:{LIG}",
              f"display {model}:{LIG} z<4.5 & protein & ~@H*",
              f"represent stick {model}:{LIG} z<4.5 & protein"]
    return s


def write_chimera_scripts(cfg: Config, P: Paths):
    lig = cfg.has_ligand
    head = ["# UCSF Chimera command script generated by md_pipeline.py",
            "background solid white", "set silhouette"]
    style = head + ["# Run AFTER loading the trajectory with MD Movie (it becomes model #0):",
                    "#   Tools > MD/Ensemble Analysis > MD Movie > format PDB > complex_traj_multimodel.pdb",
                    "# then File > Open > chimera_style.cmd"] + _chimera_style("#0", lig)
    if lig:
        style += [f"select #0:{LIG}", "findhbond selRestrict cross reveal true", "~select",
                  f"focus #0:{LIG}",
                  "# To update H-bonds every frame: MD Movie > Per-Frame > Define script... (Chimera commands)",
                  f"#   select #0:{LIG}; findhbond selRestrict cross reveal true; ~select"]
    else:
        style += ["focus"]
    (P.viz / "chimera_style.cmd").write_text("\n".join(style) + "\n")

    cmp_ = head + ["# Open from the 09_visualization folder (or run: cd <that folder>)",
                   "open #0 complex_start.pdb", "open #1 complex_final.pdb"]
    cmp_ += _chimera_style("#0", lig) + _chimera_style("#1", lig)
    cmp_ += ["ribcolor cornflower blue #0", "ribcolor salmon #1"]
    if lig:
        cmp_ += [f"color cornflower blue #0:{LIG}", f"color byhet #0:{LIG}",
                 f"focus #0:{LIG}", f"rmsd #0:{LIG}&~@H* #1:{LIG}&~@H*"]
    else:
        cmp_ += ["focus"]
    (P.viz / "chimera_start_vs_final.cmd").write_text("\n".join(cmp_) + "\n")


def _style(model, lig):
    s = [f"hide {model} atoms", f"cartoon {model}", f"color {model} & protein lightgray target c"]
    if lig:
        s += [f"show {model} & :{LIG} atoms", f"style {model} & :{LIG} ball",
              f"color {model} & :{LIG} & C orange", f"color {model} & :{LIG} byhetero",
              f"show {model} & (:{LIG} :<4.5) & protein & ~HC atoms",
              f"style {model} & (:{LIG} :<4.5) & protein stick"]
    return s


def write_chimerax_scripts(cfg: Config, P: Paths):
    lig = cfg.has_ligand
    head = ["# Generated by md_pipeline.py. Paths are relative to this folder:",
            "# run with `chimerax <this file>` from this folder, or File > Open in ChimeraX.",
            "set bgColor white", "lighting soft", "graphics silhouettes true"]
    traj = head + ["open complex_start.pdb", "open complex_traj.xtc structureModel #1"] + _style("#1", lig)
    if lig:
        traj += [f"hbonds #1 & :{LIG} restrict cross reveal true name ligand_hbonds",
                 f"view #1 & :{LIG} pad 0.8",
                 "# recompute H-bonds on every frame while playing (slower):",
                 f'# perframe "hbonds #1 & :{LIG} restrict cross reveal true name ligand_hbonds"']
    else:
        traj += ["view"]
    traj += ["coordset slider #1"]
    (P.viz / "view_trajectory.cxc").write_text("\n".join(traj) + "\n")

    cmp_ = head + ["open complex_start.pdb", "open complex_final.pdb"] + _style("#1", lig) + _style("#2", lig)
    cmp_ += ["color #1 & protein lightblue target c", "color #2 & protein salmon target c"]
    if lig:
        cmp_ += [f"color #1 & :{LIG} & C cornflowerblue", f"color #2 & :{LIG} & C orange",
                 f"view :{LIG} pad 0.8",
                 f"rmsd #1 & :{LIG} & ~H to #2 & :{LIG} & ~H"]
    else:
        cmp_ += ["view"]
    cmp_ += ["2dlabels text 'start (blue)  vs  final (salmon/orange)' xpos 0.03 ypos 0.95 size 18"]
    (P.viz / "view_start_vs_final.cxc").write_text("\n".join(cmp_) + "\n")


def find_chimerax(cfg: Config | None = None) -> str | None:
    if cfg and cfg.chimerax_path:
        return cfg.chimerax_path
    for name in ("chimerax", "ChimeraX", "chimerax-daily"):
        if p := shutil.which(name):
            return p
    patterns = ["/Applications/ChimeraX*.app/Contents/MacOS/ChimeraX",
                "/usr/bin/chimerax*", "/opt/UCSF/ChimeraX*/bin/ChimeraX",
                "/usr/lib/ucsf-chimerax*/bin/ChimeraX",
                r"C:\Program Files\ChimeraX*\bin\ChimeraX.exe",
                "/mnt/c/Program Files/ChimeraX*/bin/ChimeraX.exe"]   # Windows ChimeraX from WSL
    for pat in patterns:
        hits = sorted(glob.glob(pat), reverse=True)
        if hits:
            return hits[0]
    return None


def open_in_chimerax(cfg: Config, view: str = "trajectory"):
    """Launch ChimeraX on your desktop. view = 'trajectory' or 'start_vs_final'."""
    P = Paths(cfg)
    cxc = P.viz / f"view_{view}.cxc"
    if not cxc.exists():
        raise FileNotFoundError(f"{cxc} not found - run the pipeline first")
    exe = find_chimerax(cfg)
    if not exe:
        print("ChimeraX not found. Set Config(chimerax_path='/path/to/ChimeraX') or open this file "
              f"manually in ChimeraX:\n  {cxc}")
        return
    arg = str(cxc)
    if exe.lower().endswith(".exe") and shutil.which("wslpath"):
        arg = subprocess.check_output(["wslpath", "-w", arg], text=True).strip()
    kw = {"start_new_session": True} if os.name == "posix" else {}
    subprocess.Popen([exe, arg], cwd=P.viz, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, **kw)
    print(f"Opened {cxc.name} in ChimeraX ({exe})")


def package_for_download(cfg: Config, include_full_trajectory: bool = False) -> Path:
    """Zip the visualisation + analysis folders (and optionally the full trajectory) for
    download, e.g. from Colab to open the ChimeraX scripts on your own computer."""
    import zipfile
    P = Paths(cfg)
    out = P.root / f"{cfg.name}_results.zip"
    keep = [P.viz, P.ana / "plots", P.ana / "csv", P.ana / "xvg"]
    extra = [P.ana / "summary.json", P.ana / "dashboard.png", P.ana / "index.ndx",
             P.root / "config.json", P.root / "run_info.json", P.root / "README.txt",
             P.sys / "topol.top", P.md / "md.tpr"]
    if include_full_trajectory:
        extra.append(P.ana / "md_fit.xtc")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in keep:
            for f in d.rglob("*") if d.exists() else []:
                if f.is_file():
                    z.write(f, f.relative_to(P.root))
        for f in extra:
            if f.exists():
                z.write(f, f.relative_to(P.root))
    print(f"Packed {out} ({out.stat().st_size / 1e6:.1f} MB). Unzip it and open "
          "09_visualization/view_trajectory.cxc in ChimeraX.")
    return out


def show_in_notebook(cfg: Config, which: str = "final", width=700, height=500):
    """Quick inline 3D view in Colab/Jupyter with py3Dmol (which = 'start' or 'final')."""
    import py3Dmol
    pdb = Paths(cfg).viz / f"complex_{which}.pdb"
    v = py3Dmol.view(width=width, height=height)
    v.addModel(pdb.read_text(), "pdb")
    v.setStyle({"cartoon": {"color": "spectrum"}})
    if cfg.has_ligand:
        v.setStyle({"resn": LIG}, {"stick": {"colorscheme": "orangeCarbon"}})
        v.addStyle({"within": {"distance": 4.5, "sel": {"resn": LIG}}, "not": {"resn": LIG}},
                   {"stick": {"radius": 0.15}})
        v.zoomTo({"resn": LIG})
    else:
        v.zoomTo()
    return v.show()


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
README = """Results layout
==============
00_input/          input structure
01_ligand/         ligand extraction, protonation, acpype output (GAFF2/AM1-BCC)
02_protein/        cleaned protein
03_system/         topol.top, *.itp, system.gro  (full solvated, neutralised system)
04_em/ 05_nvt/ 06_npt/   minimisation and equilibration (.tpr .gro .edr .log .cpt)
07_production/     md.tpr, md.xtc (raw trajectory), md.edr, md.log, md.cpt
08_analysis/       md_fit.xtc  - PBC-corrected, fitted full trajectory (use with md.tpr)
                   index.ndx   - groups incl. LIG and Complex (protein+ligand)
                   xvg/ csv/ plots/, dashboard.png, summary.json
09_visualization/  complex_start.pdb, complex_final.pdb, complex_traj.xtc (protein+ligand only)
                   view_trajectory.cxc, view_start_vs_final.cxc (ChimeraX)
                   complex_traj_multimodel.pdb, chimera_style.cmd,
                   chimera_start_vs_final.cmd (UCSF Chimera, via MD Movie)
logs/              every command and its output
config.json        exact settings used; run_info.json - versions, GPU, performance

Python follow-up, e.g. with MDAnalysis:
  import MDAnalysis as mda
  u = mda.Universe("09_visualization/complex_start.pdb", "09_visualization/complex_traj.xtc")
"""


def prepare_equilibrated(cfg: Config):
    """Build the system and run EM, NVT and NPT (finished steps are skipped).
    Returns (Paths, Runner, info). Used by run_pipeline and by md_advanced."""
    P = Paths(cfg)
    P.make()
    R = Runner(cfg, P)
    print(f"Results folder: {P.root}")
    info = check_environment(cfg)
    (P.root / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (P.root / "README.txt").write_text(README)

    src = get_structure(cfg, P)
    protein = prepare_protein(cfg, P, src)
    molname = prepare_ligand(cfg, P, R, src) if cfg.has_ligand else None
    build_system(cfg, P, R, protein, molname)
    info["mdrun_offload_equil"] = run_equilibration(cfg, P, R)
    return P, R, info


def run_pipeline(cfg: Config) -> dict:
    t0 = time.time()
    P, R, info = prepare_equilibrated(cfg)
    info.update(run_production(cfg, P, R))
    summary = analyse(cfg, P, R)
    export_viewer_files(cfg, P, R)

    info["wall_time_min"] = round((time.time() - t0) / 60, 1)
    (P.root / "run_info.json").write_text(json.dumps(info, indent=2))
    banner("Done")
    print(f"All results in: {P.root}\nDashboard:      {P.ana / 'dashboard.png'}")
    if "ipykernel" in sys.modules and (P.ana / "dashboard.png").exists():
        try:
            from IPython.display import Image, display
            display(Image(str(P.ana / "dashboard.png")))
        except ImportError:
            pass
    if cfg.launch_chimerax:
        open_in_chimerax(cfg)
    return {"paths": P, "summary": summary, "info": info}


def _cli():
    ap = argparse.ArgumentParser(description="Protein-ligand MD with GROMACS on a local GPU")
    ap.add_argument("--config", help="JSON file with Config fields (CLI flags override it)")
    ap.add_argument("--name")
    ap.add_argument("--pdb-id")
    ap.add_argument("--pdb-file")
    ap.add_argument("--ligand-resname")
    ap.add_argument("--ligand-file")
    ap.add_argument("--ligand-charge", type=int)
    ap.add_argument("--apo", action="store_true", help="no ligand (protein only)")
    ap.add_argument("--prod-ns", type=float)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--output-root")
    ap.add_argument("--gmx")
    ap.add_argument("--gpu-id")
    ap.add_argument("--ntomp", type=int)
    ap.add_argument("--cpu", action="store_true", help="do not use the GPU")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--chimerax", action="store_true", help="open ChimeraX at the end")
    a = ap.parse_args()

    cfg = Config.from_json(a.config) if a.config else Config()
    for k in ("name", "pdb_id", "pdb_file", "ligand_resname", "ligand_file", "ligand_charge",
              "prod_ns", "temperature", "output_root", "gmx", "gpu_id", "ntomp"):
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)
    if a.pdb_file and not a.pdb_id:
        cfg.pdb_id = None
    if a.ligand_file and not a.ligand_resname:
        cfg.ligand_resname = None
    if a.apo:
        cfg.ligand_resname = cfg.ligand_file = None
    cfg.use_gpu = not a.cpu
    cfg.overwrite = a.overwrite
    cfg.launch_chimerax = a.chimerax
    run_pipeline(cfg)


if __name__ == "__main__":
    _cli()
