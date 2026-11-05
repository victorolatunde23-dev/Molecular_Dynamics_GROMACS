# Molecular Dynamics with GROMACS on Google Colab

A learning path in molecular dynamics, from a first protein–ligand simulation to free-energy methods and enhanced sampling. Every notebook runs GROMACS on Colab's GPU and uses the same reference system, **T4 lysozyme L99A with benzene (PDB 181L)**, so results can be compared with published work.

## Notebooks

| Level | Notebook | What it runs | Open |
| --- | --- | --- | --- |
| 1 | `md_pipeline_colab_gpu.ipynb` | Build, equilibrate and run 10 ns of the protein–ligand complex; RMSD, RMSF, contacts, H-bonds; files for Chimera / ChimeraX | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victorolatunde23-dev/Molecular_Dynamics_GROMACS/blob/main/notebooks/md_pipeline_colab_gpu.ipynb) |
| 1 | `md_pipeline.ipynb` | Same pipeline, for a Colab *local runtime* on your own Linux / WSL2 GPU machine | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victorolatunde23-dev/Molecular_Dynamics_GROMACS/blob/main/notebooks/md_pipeline.ipynb) |
| 2 | `L2_replicas_analysis.ipynb` | Independent replicas, mean ± standard error, block averaging, autocorrelation, PCA, apo vs holo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victorolatunde23-dev/Molecular_Dynamics_GROMACS/blob/main/notebooks/L2_replicas_analysis.ipynb) |
| 3 | `L3_free_energy.ipynb` | MM/GBSA with gmx_MMPBSA (with per-residue decomposition); alchemical hydration free energy of benzene with MBAR (alchemlyb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victorolatunde23-dev/Molecular_Dynamics_GROMACS/blob/main/notebooks/L3_free_energy.ipynb) |
| 4 | `L4_steered_md.ipynb` | Steered MD: pulling benzene out of the cavity, force and work curves, Jarzynski estimate | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victorolatunde23-dev/Molecular_Dynamics_GROMACS/blob/main/notebooks/L4_steered_md.ipynb) |
| 5a | `L5a_umbrella_sampling.ipynb` | Umbrella sampling from a Level 4 pull, potential of mean force with WHAM and bootstrap errors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victorolatunde23-dev/Molecular_Dynamics_GROMACS/blob/main/notebooks/L5a_umbrella_sampling.ipynb) |
| 5b | `L5b_metadynamics_plumed.ipynb` | Well-tempered metadynamics with PLUMED: alanine dipeptide φ/ψ free-energy surface; optional benzene unbinding | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victorolatunde23-dev/Molecular_Dynamics_GROMACS/blob/main/notebooks/L5b_metadynamics_plumed.ipynb) |

**Order:** run Level 1 first; Levels 2–5 reuse its system. Level 5a needs Level 4 finished.

## How to run

1. Open a notebook with its Colab badge.
2. **Runtime → Change runtime type → GPU.**
3. Run the cells from the top. The first cells install GROMACS (CUDA build), AmberTools, acpype, Open Babel and, where needed, gmx_MMPBSA or PLUMED into a conda environment on the Colab machine. This takes a few minutes per session.
4. Results are written to Google Drive under `MyDrive/md_results/<system name>/`, so they survive disconnects. Long runs checkpoint every 5 minutes: after a disconnect, run the setup cells again, then the interrupted cell, and it continues where it stopped.

Each notebook is self-contained: it writes the pipeline modules from `src/` to the Colab machine itself, so nothing else needs uploading.

## Repository layout

```
notebooks/   Colab notebooks, one per level
src/         md_pipeline.py  - base pipeline (build, equilibrate, produce, analyse, visualise)
             md_advanced.py  - Levels 2-5 (replicas, MM/GBSA, alchemical FE, SMD, umbrella, metadynamics)
tests/       test_offline.py - tests for everything that does not need GROMACS
```

`src/md_pipeline.py` also works as a command-line script on a machine with GROMACS installed:

```bash
python src/md_pipeline.py --pdb-id 181L --ligand-resname BNZ --ligand-charge 0 --name t4l_benzene --prod-ns 10
```

## Using your own system

Change the `Config(...)` cell: `pdb_id="XXXX"` or `pdb_file="complex.pdb"`, plus `ligand_resname="ABC"` or (better for drug-like ligands) `ligand_file="ligand_withH.sdf"` posed in the protein's frame. Always set `ligand_charge` to the ligand's net charge.

## Methods and defaults

- Protein: AMBER99SB-ILDN, TIP3P water, 0.15 M NaCl, dodecahedral box
- Ligand: GAFF2 with AM1-BCC charges (acpype / AmberTools)
- 300 K (V-rescale), 1 bar (C-rescale), PME, 1.0 nm cut-offs, 2 fs time step with H-bond constraints

## Testing

```bash
python tests/test_offline.py
```

The tests cover topology editing, periodic-box geometry, PLUMED inputs, statistics (block averaging, autocorrelation, Jarzynski), umbrella window selection and output parsing. The simulations themselves need GROMACS and a GPU, so they are not part of the tests.

## Visualisation

- **ChimeraX:** open `09_visualization/view_trajectory.cxc` or `view_start_vs_final.cxc`.
- **UCSF Chimera:** Tools → MD/Ensemble Analysis → MD Movie → format PDB → `complex_traj_multimodel.pdb`, then open `chimera_style.cmd`.
- **In the notebook:** `show_in_notebook(cfg)` (py3Dmol).
