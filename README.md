# DeepMedAlign

Repository scaffold for the R2 infra setup branch.

## Branch rules

- `r1/data-pipeline` for R1 work
- `r2/infra-setup` for R2 work
- `r3/preprocess-viz` for R3 work
- `r4/research-docs` for R4 work

Rule: never commit directly to `main`. Open a PR to `main` at the end of each day, and keep `main` runnable.

## Current state

- Notebook moved to [notebooks/01_explore.ipynb](notebooks/01_explore.ipynb)
- Core config and helper utilities live in [src/config.py](src/config.py) and [src/utils.py](src/utils.py)
- Tests pass with `pytest tests/ -v`
- Windows users can run Day 5 with [scripts/setup_env.ps1](scripts/setup_env.ps1) instead of the bash script
- `make` and `conda` are still required for the original shell workflow, but the Windows path now uses `venv` + pip