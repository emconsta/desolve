# DESolve

DESolve is a research-oriented Python package for experimenting with
time-integration methods for ODEs and semi-discrete PDEs. The main entry point
is `desolve.DESolver.DESolver`, which advances a state vector `u(t)` using a
selected method table such as RK, ESDIRK, ARK, GLEE, or IMEX variants.

## Setup

### Requirements

- Python 3.8 or newer
- `pip`
- A virtual environment is strongly recommended

### Create an environment and install

For day-to-day development, install the package in editable mode together with
the lightweight development extras:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you only want the core library, use:

```bash
python -m pip install -e .
```

If you plan to run or extend the notebooks, install the notebook extras too:

```bash
python -m pip install -e ".[dev,notebooks]"
```

### Smoke test

After installation, verify that the solver imports cleanly:

```bash
python -c "from desolve.DESolver import DESolver; DESolver(); print('ok')"
```

### Build a distribution

The `dev` extra includes `build`, so you can create source and wheel artifacts
with:

```bash
python -m build
```

## Website and docs

This repository now includes a GitHub Pages documentation site under `site/`,
built with Astro and Starlight.

Run it locally with:

```bash
cd site
npm install
npm run dev
```

Create a production build with:

```bash
cd site
npm run build
```

The site is configured for deployment from this repository to
`https://emconsta.github.io/desolve/` using the workflow in
`.github/workflows/deploy-docs.yml`. In the GitHub repository settings, set
Pages to use `GitHub Actions` as the source.

## Quickstart

The example below solves `u' = -u` with `RK4`.

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
  a Jacobian is not available.

## Project layout

- `desolve/DESolver.py`: solver driver and method registration.
- `desolve/methods_*.py`: method coefficient tables.
- `desolve/problems_ODEs.py` and `desolve/problems_PDEs.py`: reference
  problems and semi-discretizations used by the notebooks.
- `notebooks/`: experiments, examples, and generated figures.

## License

MIT. See `LICENSE`.
