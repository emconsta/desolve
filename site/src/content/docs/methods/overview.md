---
title: Method overview
description: Survey the built-in method families currently registered by DESolve.
sidebar:
  order: 1
---

## Built-in families

DESolve organizes time integrators by method-table modules. The main families in
the repository are:

- `methods_rk.py` for explicit Runge-Kutta schemes
- `methods_esdirk.py` for singly diagonally implicit RK variants
- `methods_ark.py` for additive Runge-Kutta methods
- `methods_glee.py` and `methods_glee_eimex.py` for GLEE-related methods
- `methods_mrk.py` and `methods_imex_mrk.py` for multirate formulations
- `methods_etrs.py` and `methods_ide.py` for additional research-specific tables

## How registration works

During `DESolver.setup()`, the solver calls `_RegisterDefaultMethods()`, which
collects method dictionaries from each module and stores them in the internal
method registry.

From a user perspective, that means two things:

- adding a new method is mostly a data-entry problem in the appropriate method table
- a method is not available until it is registered inside `DESolver`

## Choosing a first method

Use this rough rule of thumb:

- start with `RK4` for simple explicit baselines
- move to ESDIRK or ARK variants when stiffness or splitting matters
- use IMEX and multirate methods when you want explicit/implicit separation or
  multiple time scales

## Extending the library

When adding a method:

1. Put the coefficient data in the relevant `desolve/methods_*.py` file.
2. Keep the naming consistent with the existing method registry.
3. Register it in `_RegisterDefaultMethods()`.
4. Add a small regression test or notebook reproduction.

## Detailed IMEX-MRK notes

The IMEX multirate methods have their own dedicated documentation page because
their implementation uses four coupled tableaux (`AB`, `AF`, `AS`, `AT`) and
maps directly onto the 2022 Applied Mathematics Letters paper that introduced
the one-implicit-stage extension.

See the [IMEX-MRK implementation notes](/desolve/methods/imex-mrk/).
