---
title: Symplectic methods
description: Explicit splitting methods for separable Hamiltonian systems and a shared harmonic-oscillator example.
sidebar:
  order: 2
---

## What DESolve implements

The symplectic family in `desolve/methods_symplectic.py` stores coefficient tables
in the paper notation

\[
p_i = p_{i-1} + b_i h F(q_{i-1}), \qquad
q_i = q_{i-1} + a_i h P(p_i),
\]

with

\[
F(q) = -\nabla_q V(q), \qquad P(p) = \nabla_p T(p).
\]

This matches the explicit composition form used in Candy-Rozmus and in
McLachlan's summary table. The solver reads only the stored `a` and `b`
coefficients and advances every symplectic method through the same kick-then-drift
loop.

The built-in methods are:

- `Verlet-DKD`
- `Verlet-KDK`
- `McLachlan-Optimal2`
- `Ruth3`
- `McLachlan-Optimal3`
- `Candy-Rozmus4`
- `McLachlan-RKN4`
- `McLachlan-RKN5`

`McLachlan-RKN4` and `McLachlan-RKN5` are the quadratic-kinetic optimized
methods from Table 2 of McLachlan and Atela, so their published order statements
apply when \(T(p)\) is quadratic.

## Shared example problem

`desolve/problems_ODEs.py` now contains a small separable Hamiltonian example:

\[
H(q, p) = \frac{1}{2}p^2 + \frac{1}{2}\omega^2 q^2.
\]

The important design point is that the problem exposes both interfaces:

- `rhs_e(t, u, ctx)` for full-vector methods like `RK4`
- `get_symplectic_rhs()` for split drift/kick methods

That means you can solve the *same* physical problem with either a standard RK
integrator or a symplectic composition method without rewriting the model.

## Solve it with RK4

```python
from desolve.DESolver import DESolver
from desolve.problems_ODEs import ProblemsODE

rhs_e, rhs_i, u_ini, problem_setup, problem = ProblemsODE("HarmonicOscillator")

solver = DESolver()
solver.set_function_context(problem_setup["context"])
solver.setup(keep_history=True)

solver.set_rhs(rhs_e)
solver.set_initial_solution(u_ini)
solver.set_duration(
    t_start=problem_setup["T_DURATION"]["start"],
    t_end=problem_setup["T_DURATION"]["end"],
    dt=problem_setup["DT"],
)
solver.set_method("RK4")
solver.solve()
```

## Solve it with a symplectic method

```python
from desolve.DESolver import DESolver
from desolve.problems_ODEs import ProblemsODE

rhs_e, rhs_i, u_ini, problem_setup, problem = ProblemsODE("HarmonicOscillator")

solver = DESolver()
solver.set_function_context(problem_setup["context"])
solver.setup(keep_history=True)

solver.set_rhs(problem.get_symplectic_rhs())
solver.set_initial_solution(u_ini)
solver.set_duration(
    t_start=problem_setup["T_DURATION"]["start"],
    t_end=problem_setup["T_DURATION"]["end"],
    dt=problem_setup["DT"],
)
solver.set_method("Verlet-KDK")
solver.solve()
```

## Why the example is structured this way

- the state is stored as one vector `u = [q, p]`, so full-vector methods can use it directly
- the context also provides `SplitSolution` and `MergeSolution`, so the symplectic solver can work with canonical blocks cleanly
- the same `problem_setup["context"]` carries `omega`, dtype information, and the split/merge helpers
- this keeps the example easy to reuse in convergence tests, notebooks, and side-by-side method comparisons
