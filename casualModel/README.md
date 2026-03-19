# Vectorized Causal Computing

This repository contains the reference code accompanying the paper on vectorized causal computing. The code is organized as a small, script-first research package: the core models live in standalone Python modules, and the evaluation scripts can be run directly from the repository root.

## Repository layout

- `caxm.py` — CAxM for binary-treatment causal-axis inference.
- `caxmpp.py` — CAxM++ with gated nonlinear readout and low-rank compression.
- `vcg.py` — vectorized causal graph prototype.
- `vcg_forest.py` — forest-decomposed vectorized causal graph.
- `klein_vc.py`, `mobius_vc.py` — archived experimental variants kept for completeness.
- `t_learner.py`, `psm_iptw.py`, `dr_aipw.py` — baseline estimators used by the IHDP scripts.
- `ihdp_run.py` — single-seed IHDP experiment.
- `ihdp_multiseed.py` — repeated IHDP experiment over multiple seeds.
- `acic_run.py` — ACIC experiment script.
- `tests_smoke.py` — lightweight import and execution checks.

## Environment

Create a fresh environment and install the minimal dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

### IHDP
Place `ihdp_data.csv` in the repository root.
The script expects the standard columns:
`
`
- `treatment`
- `y_factual`
- `y_cfactual`
- `x1` ... `x25`

### ACIC
Place the following files in the repository root:

- `training_sample.csv`
- `testing_sample.csv`
- `predictions.csv`

## Quick checks

Run the smoke tests first:

```bash
python tests_smoke.py
```

This verifies that the core models import correctly and that their minimal inference paths execute without shape errors.

## Running IHDP

Single run:

```bash
python ihdp_run.py
```

Repeated runs across seeds:

```bash
python ihdp_multiseed.py
```

The scripts write CSV outputs into the repository root.

## Running ACIC

A minimal ACIC run:

```bash
python acic_run.py --sample_n 256 --seed 42
```

## Notes on scope

The code in this repository is intended to match the experiments reported in the manuscript and to remain easy to inspect. Some modules, especially `vcg.py` and `vcg_forest.py`, are research prototypes rather than general-purpose libraries. The archived variants `klein_vc.py` and `mobius_vc.py` are included for completeness but are not required for the main paper results.

## Reproducibility checklist

- fixed random seeds are exposed in the experiment scripts
- outputs are written to CSV rather than only printed to stdout
- smoke tests are included for import and inference sanity checks
- dependencies are intentionally minimal

## Validation performed before packaging

Before creating the release archive, the repository was checked with:

```bash
python -m py_compile *.py
python tests_smoke.py
python ihdp_run.py
```

The ACIC script was also exercised on the provided sample files to confirm that the precheck and model evaluation paths start correctly.
