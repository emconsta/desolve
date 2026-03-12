import math

import numpy as np

from desolve.DESolver import DESolver
from desolve.problems_ODEs import ProblemsODE


def _harmonic_drift_rhs(t, q_in, p_in, ctx=None):
    return np.array(p_in, copy=True), None


def _harmonic_kick_rhs(t, q_in, p_in, ctx=None):
    return -np.array(q_in, copy=True), None


def _make_solver(method_name, dt, t_end, use_split_helpers):
    solver = DESolver()

    ctx = {"data-type": np.float64}
    if use_split_helpers:
        ctx["SplitSolution"] = lambda u_in: (u_in[0:1], u_in[1:2])
        ctx["MergeSolution"] = lambda q_in, p_in: np.concatenate((q_in, p_in))

    solver.set_function_context(ctx)
    solver.setup()
    solver.set_rhs(
        {
            "symplectic_drift": _harmonic_drift_rhs,
            "symplectic_kick": _harmonic_kick_rhs,
        }
    )
    solver.set_method(method_name)
    solver.set_duration(t_start=0.0, t_end=t_end, dt=dt)
    solver.set_initial_solution(np.array([1.0, 0.0], dtype=np.float64))
    return solver


def _harmonic_exact_solution(t_end):
    return np.array([math.cos(t_end), -math.sin(t_end)], dtype=np.float64)


def _endpoint_error(method_name, dt, t_end=1.0, use_split_helpers=True):
    solver = _make_solver(method_name, dt=dt, t_end=t_end, use_split_helpers=use_split_helpers)
    solver.solve()
    return np.linalg.norm(solver.get_solution() - _harmonic_exact_solution(solver._t))


def test_symplectic_methods_are_registered():
    solver = DESolver()
    solver.set_function_context({"data-type": np.float64})
    solver.setup()

    methods = set(solver.view_registered_methods())
    expected = {
        "Verlet-DKD",
        "Verlet-KDK",
        "McLachlan-Optimal2",
        "Ruth3",
        "McLachlan-Optimal3",
        "Candy-Rozmus4",
        "McLachlan-RKN4",
        "McLachlan-RKN5",
    }
    assert expected.issubset(methods)


def test_verlet_kdk_works_with_default_canonical_split():
    solver = _make_solver("Verlet-KDK", dt=0.05, t_end=1.0, use_split_helpers=False)
    solver.solve()

    error = np.linalg.norm(solver.get_solution() - _harmonic_exact_solution(solver._t))
    assert error < 5.0e-3


def test_verlet_kdk_keeps_harmonic_oscillator_energy_bounded():
    solver = _make_solver("Verlet-KDK", dt=0.1, t_end=100.0, use_split_helpers=True)
    solver.solve()

    _, trajectory = solver.get_trajectory()
    energy = 0.5 * (trajectory[0, :] ** 2 + trajectory[1, :] ** 2)
    assert np.max(np.abs(energy - energy[0])) < 5.0e-3


def test_symplectic_methods_match_nominal_order_on_harmonic_oscillator():
    thresholds = {
        "Verlet-KDK": 3.5,
        "Ruth3": 6.0,
        "Candy-Rozmus4": 12.0,
        "McLachlan-RKN5": 20.0,
    }

    for method_name, min_ratio in thresholds.items():
        coarse_error = _endpoint_error(method_name, dt=0.2)
        fine_error = _endpoint_error(method_name, dt=0.1)
        assert coarse_error / fine_error > min_ratio


def test_harmonic_oscillator_problem_supports_rk4_and_symplectic_rhs():
    rhs_e, rhs_i, u_ini, problem_setup, problem = ProblemsODE("HarmonicOscillator")
    assert rhs_i is None

    rk_solver = DESolver()
    rk_solver.set_function_context(problem_setup["context"])
    rk_solver.setup()
    rk_solver.set_rhs(rhs_e)
    rk_solver.set_initial_solution(u_ini)
    rk_solver.set_duration(t_start=0.0, t_end=1.0, dt=0.05)
    rk_solver.set_method("RK4")
    rk_solver.solve()
    rk_error = np.linalg.norm(
        rk_solver.get_solution() - problem.exact_solution(rk_solver._t, problem_setup["context"])
    )
    assert rk_error < 1.0e-5

    sym_solver = DESolver()
    sym_solver.set_function_context(problem_setup["context"])
    sym_solver.setup()
    sym_solver.set_rhs(problem.get_symplectic_rhs())
    sym_solver.set_initial_solution(u_ini)
    sym_solver.set_duration(t_start=0.0, t_end=1.0, dt=0.05)
    sym_solver.set_method("Verlet-KDK")
    sym_solver.solve()
    sym_error = np.linalg.norm(
        sym_solver.get_solution() - problem.exact_solution(sym_solver._t, problem_setup["context"])
    )
    assert sym_error < 5.0e-3
