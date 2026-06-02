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
- `make` and `conda` are required for the Day 3 and Day 5 shell commands, but they are not installed in this Windows environment