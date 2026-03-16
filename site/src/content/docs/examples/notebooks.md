---
title: Notebook experiments
description: How the repository separates curated notebooks, reproducibility material, and local-only research work.
sidebar:
  order: 2
---

## Repository notebook workflow

The repository now splits notebook material into three buckets:

- `notebooks/` contains the small curated example set kept in the distribution
- `reproducibility/Constantinescu_2021/` contains the IMEX-MRK paper notebooks
  and their generated figure assets
- `notebooks/research/` is a local-only scratch area for exploratory notebooks
  and generated artifacts and is ignored by Git

For the IMEX multirate paper specifically, see the dedicated
[advection-diffusion paper example](/desolve/examples/advection-diffusion-paper/).

## Recommended practice

If you are extending the notebooks:

- keep `notebooks/` limited to small, reusable examples
- put paper-specific workflows under `reproducibility/`
- keep exploratory studies and generated artifacts under local-only
  `notebooks/research/`
- move reusable logic back into the Python package when it stops being
  notebook-specific

## When to use notebooks vs. tests

Use notebooks when you are:

- exploring a new method idea
- tuning a numerical experiment
- generating plots for analysis

Use tests when you are:

- locking in a bug fix
- validating an invariant
- checking a stable reference behavior
