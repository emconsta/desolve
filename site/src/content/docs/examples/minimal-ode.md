---
title: Minimal ODE example
description: A small complete example you can use as a template for new problems.
sidebar:
  order: 1
---

## Example

```python
import numpy as np
from desolve.DESolver import DESolver

def rhs(t, u, ctx):
    rate = ctx["rate"]
    f = -rate * u
    jac = -rate * np.eye(u.size, dtype=ctx["data-type"])
    return f, jac

solver = DESolver()
solver.set_function_context({
    "data-type": np.float64,
    "rate": 1.5,
})
solver.setup(keep_history=True)

solver.set_rhs(rhs)
solver.set_initial_solution(np.array([1.0], dtype=np.float64))
solver.set_duration(t_start=0.0, t_end=2.0, dt=0.02)
solver.set_method("RK4")

solver.solve()
t, u = solver.get_trajectory()
```

## Why this is a good template

- it uses the standard RHS signature
- it keeps coefficients inside `ctx`
- it returns a Jacobian consistent with the selected dtype
- it leaves the solver workflow readable enough to copy into a notebook or test
