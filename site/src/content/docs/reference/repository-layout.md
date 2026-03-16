---
title: Repository layout
description: A quick map of the most important files and directories in the repository.
sidebar:
  order: 1
---

## Core Python package

- `desolve/DESolver.py` contains the main solver driver and method registration
- `desolve/methods_*.py` contains method tables grouped by family
- `desolve/problems_ODEs.py` and `desolve/problems_PDEs.py` contain reference
  problems and semi-discretizations

## Research material

- `notebooks/` contains the curated example notebooks kept in the distribution
- `reproducibility/Constantinescu_2021/` contains the IMEX-MRK paper notebooks
  and generated figure assets
- `notebooks/research/` is a local-only ignored area for exploratory notebooks
  and artifacts
- `tests/` is the place for regression tests as the automated suite grows

## Packaging and docs

- `pyproject.toml` defines Python packaging metadata and dependencies
- `site/` contains the Astro + Starlight documentation site
- `.github/workflows/deploy-docs.yml` deploys the docs site to GitHub Pages

## Local setup recap

For Python work:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For docs work:

```bash
cd site
npm install
npm run dev
```
