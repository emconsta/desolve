---
title: Notebook experiments
description: Where the repository keeps exploratory notebooks, plots, and numerical studies.
sidebar:
  order: 2
---

## Repository notebook workflow

The `notebooks/` directory contains the exploratory side of the project:

- method comparisons
- convergence studies
- advection, reaction, and diffusion experiments
- Lorenz and other dynamical-system examples
- generated figures used in talks or paper drafts

## Recommended practice

If you are extending the notebooks:

- keep outputs minimal when possible
- avoid committing temporary checkpoints or cache artifacts
- move reusable logic back into the Python package when it stops being notebook-specific

## When to use notebooks vs. tests

Use notebooks when you are:

- exploring a new method idea
- tuning a numerical experiment
- generating plots for analysis

Use tests when you are:

- locking in a bug fix
- validating an invariant
- checking a stable reference behavior
