---
title: Advection-diffusion paper example
description: How DESolve reproduces the 1D advection-diffusion experiment from the IMEX-MRK paper.
sidebar:
  order: 3
---

## Scope

This page documents the PDE example used in:

- Emil M. Constantinescu, *Implicit extensions of an explicit multirate
  Runge-Kutta scheme*, Applied Mathematics Letters 128 (2022), 107871

and shows how the example is represented in this repository.

The method construction itself is documented separately on the
[IMEX-MRK implementation notes](/desolve/methods/imex-mrk/). Here the focus is
the semi-discrete PDE problem and how it is wired into `DESolver`.

## The model problem

Section 4 of the paper studies the periodic one-dimensional advection-diffusion
equation

```text
u_t + (omega(x) u)_x = delta u_xx
```

with:

- periodic boundary conditions
- explicit treatment of advection
- implicit treatment of diffusion
- a spatially varying wave speed `omega(x)` that creates fast and slow regions

In the repository code, the diffusion coefficient is named `kappa` instead of
`delta`, but it plays the same role as the paper's stiffness parameter.

## Where the implementation lives

The main implementation is the `AdvectionDiffusion1D` class in
`desolve/problems_PDEs.py`.

Its responsibilities are:

- assemble the periodic advection operator
- assemble the periodic diffusion operator and Jacobian
- provide vectorization and partitioning helpers
- expose full-rate and multirate right-hand sides in the signatures expected by
  `DESolver`

The paper-oriented notebook workflows live in:

- `notebooks/TestMRIMEXAdvectionDiffusionPaper.ipynb`
- `notebooks/TestMRIMEXAdvectionDiffusionPaperPlots.ipynb`

## The explicit-implicit split

The code splits the semi-discrete operator into:

```text
F(u) = f(u) + g(u)
```

with:

- `f(u)` = advection, handled explicitly
- `g(u)` = diffusion, handled implicitly

In repository terms:

- `rhs_e_fast(...)` computes the explicit advection update on the fast spatial block
- `rhs_e_slow(...)` computes the explicit advection update on the slow spatial block
- `rhs_e(...)` computes the explicit advection update on the full domain
- `rhs_mr_implicit(...)` computes the full-domain diffusion update and returns
  its Jacobian

That mapping is exactly what the IMEX-MRK stepper in `DESolver` expects.

## Spatial discretization

### Advection

The paper describes:

- a conservative third-order upwind-biased finite-difference discretization for
  the advective term

In the repository, the paper notebooks use:

```python
'Flux_name': 'FVStagVanLeer-k=1/3'
```

which is the third-order Van Leer style staggered flux option implemented in
`LinearAdvectionSemiDiscretization1D(...)`.

The wave speed is supplied as `Flux_cv`, a cellwise array that determines where
the explicit CFL restriction is tightest.

### Diffusion

The implicit part uses a standard second-order periodic finite-difference
Laplacian. In `rhs_mr_implicit(...)`, the code builds the tridiagonal periodic
matrix

```text
(1 / dx^2) * tridiag(1, -2, 1)
```

with wraparound entries in the first and last rows, then scales it by `kappa`.

Because the operator is linear, the method returns both:

- `A u`
- the Jacobian `A`

which makes the implicit stage solve in `DESolver` straightforward.

## Fast and slow regions

The multirate paper example creates regions with different local explicit
stability limits by varying `omega(x)`.

In the repository, the domain is split contiguously:

```text
[ fast block | slow block ]
```

through:

- `split_solution(...)`
- `merge_solution(...)`

For the default setup:

- the first half of the 1D grid is the fast partition
- the second half is the slow partition

The fast and slow explicit operators reconstruct ghost values from the opposite
partition so the interface remains consistent during the multirate stages.

## Notebook parameters used for the paper runs

The paper notebooks use a setup of the form:

```python
pde_problem_setup = {
    'mx': 81,
    'n': 1,
    'Flux': 'linear-advection',
    'Flux_cv': Flux_cv,
    'BC': 'periodic',
    'x_min': 0.0,
    'x_max': 1.0,
    'kappa': 0.05,
    'Flux_name': 'FVStagVanLeer-k=1/3',
}
```

and then register the multirate explicit and implicit right-hand sides with:

```python
solver.set_rhs({
    'mr_explicit_fast': problem.rhs_e_fast,
    'mr_explicit_slow': problem.rhs_e_slow,
    'mr_implicit': problem.rhs_mr_implicit,
})
```

Typical method choices in the notebook are:

- `MPRK2-IMEX2` for the A-stable multirate method
- `MPRK2-IMEX` for the L-stable multirate method
- `MPRK2-m4-IMEX` for the `m = 4` fast/slow ratio experiment

## Why this example matters for the method

This PDE example is a good match for the IMEX-MRK design because it has both:

- local explicit stiffness induced by the varying advection speed `omega(x)`
- a globally stiff diffusion term

That is exactly the setting targeted by the paper:

- multirate explicit stepping on the nonstiff part
- a single-rate implicit treatment for the stiff part
- conservation inherited from the explicit MPRK completion weights

## Practical reading order

If you want to follow the example from top to bottom in the code base:

1. Read `desolve/methods_imex_mrk.py` for the coefficient tables.
2. Read the IMEX-MRK branch in `desolve/DESolver.py`.
3. Read `AdvectionDiffusion1D` in `desolve/problems_PDEs.py`.
4. Open `notebooks/TestMRIMEXAdvectionDiffusionPaper.ipynb`.

That sequence moves from method design to solver implementation to the PDE
experiment itself.
