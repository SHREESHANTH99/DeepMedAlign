.PHONY: help env manifest validate preprocess classical train view test lint clean

help:
	@echo "Available commands:"
	@echo "  make env          Create conda environment"
	@echo "  make manifest     Build subject manifest CSV"
	@echo "  make validate     Run shape + HU QC on all subjects"
	@echo "  make preprocess   Run full preprocessing pipeline"
	@echo "  make classical    Run rigid + affine + B-spline baseline"
	@echo "  make train        Train VoxelMorph"
	@echo "  make view         Run viewer on first subject"
	@echo "  make test         Run all tests"
	@echo "  make lint         Check code style"
	@echo "  make clean        Remove .pyc and cache files"

env:
	@conda env create -f environment.yml

manifest:
	@echo "manifest generation is not implemented yet"

validate:
	@echo "validation is not implemented yet"

preprocess:
	@echo "preprocessing is not implemented yet"

classical:
	@echo "classical baseline is not implemented yet"

train:
	@echo "training is not implemented yet"

view:
	@echo "viewer is not implemented yet"

test:
	@pytest tests/ -v

lint:
	@python -m compileall src tests

clean:
	@python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]; [p.unlink() for p in Path('.').rglob('*.pyc')]"
