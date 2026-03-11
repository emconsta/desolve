---
title: Solver workflow
description: Understand the core DESolve setup sequence and what each step configures.
sidebar:
  order: 2
---

## Typical call sequence

Most DESolve runs follow the same shape:

1. Construct `DESolver()`.
2. Set the function context, including the numerical dtype.
3. Call `setup()` to register methods and initialize solver state.
4. Provide the right-hand side function.
5. Set the initial solution, time interval, time step, and method.
6. Call `solve()`.
7. Read the trajectory or final state.

## Core ideas

### Function signatures stay explicit

DESolve expects right-hand sides in the form:

```python
(t, u, ctx) -> (f, jac)
```

That keeps numerical data flow visible:

- `t` is the current time
- `u` is the state vector
- `ctx` holds user-supplied context such as dtype or coefficients
- `jac` may be `None` when you do not have a Jacobian

### Methods are table-driven

The solver registers built-in methods during `setup()`. Those method tables live
in the `desolve/methods_*.py` modules and are selected later with
`set_method("...")`.

### History is optional but useful

Use `setup(keep_history=True)` when you want trajectories for diagnostics,
plotting, or convergence studies. If you only care about the terminal state,
you can turn history off to reduce storage.

## Practical advice

- Prefer `set_initial_solution(...)` over `solve(u_ini=...)` for NumPy arrays.
- Keep Jacobians consistent with the dtype stored in `ctx`.
- Start with simple methods like `RK4` before switching to implicit or split
  formulations.
- When experimenting with new schemes, add the method table first and then wire
  it into `DESolver._RegisterDefaultMethods()`.
