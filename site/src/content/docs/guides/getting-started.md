---
title: Getting started
description: Install DESolve locally and run a first solver example.
sidebar:
  order: 1
---

## Requirements

- Python 3.8 or newer
- `pip`
- `numpy`
- `scipy`

## Install the package

Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you also want the notebook tooling:

```bash
python -m pip install -e ".[dev,notebooks]"
```

## Smoke test

Confirm the solver imports cleanly:

```bash
python -c "from desolve.DESolver import DESolver; DESolver(); print('ok')"
```

## First solve

The example below advances the scalar ODE `u' = -u` using `RK4`.

```python
import numpy as np
from desolve.DESolver import DESolver

def rhs(t, u, ctx):
    f = -u
    jac = -np.eye(u.size, dtype=ctx["data-type"])
    return f, jac

solver = DESolver()
solver.set_function_context({"data-type": np.float64})
solver.setup(keep_history=True)

solver.set_rhs(rhs)
solver.set_initial_solution(np.array([1.0], dtype=np.float64))
solver.set_duration(t_start=0.0, t_end=1.0, dt=1e-2)
solver.set_method("RK4")

solver.solve()
t, u = solver.get_trajectory()
print(t[-1], u[:, -1])
```

## Next step

Once the basic example works, move to the [solver workflow](/desolve/guides/solver-workflow/)
page to understand the typical call sequence and where method selection and
problem setup fit together.
