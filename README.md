# DESolve

DESolve is a small, research-oriented Python library for experimenting with
time-integration methods for ODEs and semi-discrete PDEs. The core entry point
is `desolve.DESolver.DESolver`, which advances a state vector `u(t)` using a
selected method table (RK/ESDIRK/ARK/GLEE/IMEX variants).

## Install (editable)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Runtime dependencies are not pinned in this repo; you will typically need
`numpy` and `scipy`.

## Quickstart: solve u' = -u

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

Notes:
- Prefer `set_initial_solution(...)` over `solve(u_ini=...)` for NumPy arrays.
- RHS functions follow `(t, u, ctx) -> (f, jac)`; return `None` for `jac` when
  not available.

## Where things live
- `desolve/DESolver.py`: solver driver and method registration.
- `desolve/methods_*.py`: method coefficient tables.
- `desolve/problems_ODEs.py`, `desolve/problems_PDEs.py`: reference problems and
  semi-discretizations used by the notebooks.
- `notebooks/`: experiments, examples, and figures.

## License
MIT (see `LICENSE`).
