---
title: IMEX-MRK implementation notes
description: Mathematical and implementation notes for the IMEX multirate Runge-Kutta tables in desolve/methods_imex_mrk.py.
sidebar:
  order: 2
---

## Scope

This page documents the method tables implemented in
`desolve/methods_imex_mrk.py` and relates them to:

- Emil M. Constantinescu, *Implicit extensions of an explicit multirate
  Runge-Kutta scheme*, Applied Mathematics Letters 128 (2022), 107871
- the local reference PDF at `reference_materials/MRIMEX_Paper.pdf`
- the IMEX-MRK stepping logic in `desolve/DESolver.py`

This page focuses on the time-integration method implementation. The PDE example
from the paper can be documented separately.

## Mathematical structure

The paper starts from a conservative explicit multirate partitioned Runge-Kutta
(MPRK) scheme with a fast partition and a slow partition:

```text
Y_i^F = y_n^F + dt * sum_j A^F_{ij} f^F(Y_j^F, Y_j^S)
Y_i^S = y_n^S + dt * sum_j A^S_{ij} f^S(Y_j^F, Y_j^S)
```

The implicit extension adds a stiff contribution `g(y)` through an additional
tableau `A_tilde`. In the published construction, only the last stage has
nonzero implicit coefficients:

```text
Y_s = y_n
    + dt * sum_j A^F_{s,j} f^F(Y_j)
    + dt * sum_j A^S_{s,j} f^S(Y_j)
    + dt * sum_j A_tilde_{s,j} g(Y_j)

y_{n+1} = y_n
        + dt * sum_i b_i ( f(Y_i) + g(Y_i) )
```

The implementation mirrors that decomposition directly:

- `AB`, `bB`, `cB` store the base Runge-Kutta tableau
- `AF`, `bF`, `cF` store the explicit fast tableau
- `AS`, `bS`, `cS` store the explicit slow or buffer tableau
- `AT`, `bT`, `cT` store the implicit augmentation

For the conservative methods in the paper, the completion weights match:

```text
bF = bS = bT
```

That equality is what lets the implementation combine explicit and implicit
increments without losing the conservative structure of the underlying MPRK
scheme.

## How DESolve uses these tables

Inside `DESolver.step()` for `METHOD['type'] == 'IMEX-MRK'`, the solver does the
following:

1. Split the solution into fast and slow spatial partitions with
   `SplitSolution`.
2. Build explicit stage states with `AF` and `AS`.
3. Merge the partitions into a full stage state `YstageG`.
4. Apply previously computed implicit contributions through `AT`.
5. If `AT[i, i] != 0`, solve the implicit stage with `fsolve` by calling
   `rhs_mr_implicit`.
6. Re-split the merged stage and evaluate `rhs_fast` and `rhs_slow`.
7. Finish the step by combining the explicit updates with `bF` and `bS`, then
   the implicit updates with `bT`.

That is why the dictionaries in `methods_imex_mrk.py` need all four tableaux,
not just a single Butcher table.

## Published method families

### `MPRK2-IMEX2`

This is the paper's A-stable second-order implicit extension from Section 3.1
and Eq. (5).

Its defining feature is the last implicit row

```text
AT[-1, :] = [1/2, 1/2, ..., 1/2]
```

In the paper, this produces an implicit stability function

```text
R(z) = (2 + z) / (2 - z)
```

and therefore an A-stable implicit part with second-order coupling.

### `MPRK2-IMEX`

This is the paper's L-stable first-order implicit extension from Section 3.2
and Eq. (6).

Its defining last implicit row is

```text
AT[-1, :] = [1, 1, ..., 1]
```

which corresponds to the implicit stability function

```text
R(z) = 1 / (1 - z)
```

The explicit side remains second order, while the implicit side is first order
but L-stable.

## Reference and exploratory variants in the file

The registry contains more than the two published MPRK2 tables.

| Registry name | Role in the repository | Relation to the paper |
| --- | --- | --- |
| `PRK2-IMEX` | Single-rate partitioned reference scheme | Same one-row implicit idea, but without the conservative MPRK slow tableau |
| `PRK2-IMEX2` | Single-rate A-stable reference scheme | Same comment as above |
| `MPRK2-IMEX-BE` | Conservative comparison variant | Repository-specific experiment, not one of the published Eq. (5)/(6) tables |
| `MPRK2-IMEX-BE-NC` | Nonconservative comparison variant | Same as above, but `bT` no longer matches `bF` and `bS` |
| `MPRK2-IMEXb` | Custom implicit coupling experiment | Repository-specific |
| `MPRK2-RK3SSPHig-IMEX` | Same packaging applied to the SSP RK3 explicit base | Extension beyond the published Heun-based examples |
| `MPRK2-m4-IMEX` | Heun-based method with multirate ratio `m = 4` | Same implementation pattern, larger explicit fast/slow hierarchy |

The important distinction is that the paper's main construction uses a single
implicit stage attached to the last row, while some repository variants explore
alternative implicit weightings for comparison purposes.

## Reading the file

The current implementation is organized around helper constructors:

- `_build_fast_tableau(...)` subcycles the base method to produce `AF`
- `_build_slow_tableau(...)` repeats the base tableau on the block diagonal to
  produce `AS`
- `_build_last_stage_only_tableau(...)` creates the paper's one-row implicit
  augmentation
- `_append_imex_method(...)` packages the arrays into the dictionary format
  expected by `DESolver`

That organization is deliberate: it keeps the mathematical correspondence to
the paper visible and reduces the risk of transcription mistakes between the
published tableaux and the Python registry.

## Implementation details worth noticing

### The published conservative methods differ mainly in `AT`

For the Heun-based MPRK2 methods, the explicit base, fast, and slow tables are
the same between `MPRK2-IMEX` and `MPRK2-IMEX2`. The practical difference is the
implicit last row:

- all ones for the L-stable variant
- all one-halves for the A-stable variant

That mirrors the paper's design goal: keep the explicit multirate machinery
unchanged and vary only the implicit coupling in the common final stage.

### `bT` controls whether the method stays conservative

The conservative methods use:

```text
bT = bF = bS
```

The `MPRK2-IMEX-BE-NC` method breaks that relation intentionally:

```text
bT = [0, 0, 0, 1]
```

This makes it a useful comparison case when studying the role of conservative
completion weights.

### The metadata fields are solver-facing

Each dictionary also carries:

- `s*` for stage counts
- `c*` for stage abscissae
- `p*` for historical order metadata

Those are not decoration. `DESolver` prints and inspects them when reporting a
selected method.

## Practical takeaway

If you want to study the published method from the paper inside this code base,
start with:

- `MPRK2-IMEX2` for the A-stable second-order implicit extension
- `MPRK2-IMEX` for the L-stable first-order implicit extension

If you want to understand how those tables are actually consumed during a step,
read the IMEX-MRK branch in `desolve/DESolver.py` next to this file.
