"""Explicit symplectic splitting methods for separable Hamiltonians.

This module stores coefficient tables for methods written in the same form as
the explicit symplectic algorithms in Candy-Rozmus (1991) and McLachlan-Atela
(1992).  For a separable Hamiltonian

    H(q, p) = T(p) + V(q),

the solver applies the stages

    p_i = p_{i-1} + b_i h F(q_{i-1}),
    q_i = q_{i-1} + a_i h P(p_i),

where

    F(q) = -dV/dq,
    P(p) = dT/dp.

This is the notation used in Candy-Rozmus Table III.  Different familiar variants
such as drift-kick-drift and kick-drift-kick are represented only through the
coefficient tables, typically by setting one of the end coefficients to zero.

References
----------
- J. Candy and W. Rozmus, "A Symplectic Integration Algorithm for Separable
  Hamiltonian Functions", Journal of Computational Physics 92 (1991), 230-256.
- R. I. McLachlan and P. Atela, "The accuracy of symplectic integrators",
  Nonlinearity 5 (1992), 541-562.
"""

import math

import numpy as np


def _append_symplectic_method(
    all_methods,
    name,
    a_coeffs,
    b_coeffs,
    order,
    paper_name=None,
    reference=None,
    quadratic_kinetic_only=False,
):
    """Store one coefficient table in the common format used by ``DESolver``."""
    a = np.asarray(a_coeffs, dtype=float)
    b = np.asarray(b_coeffs, dtype=float)

    if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
        raise ValueError("Symplectic methods require one-dimensional coefficient tables of equal length.")

    method = {
        "type": "SYMPLECTIC",
        "name": name,
        "a": a,
        "b": b,
        "s": a.size,
        "p": order,
        "paper_name": paper_name or name,
        "reference": reference,
        "quadratic_kinetic_only": quadratic_kinetic_only,
    }
    all_methods.append(method)


def _mclachlan_optimal_order3_coefficients():
    """Return the optimal third-order coefficients from McLachlan Table 2."""
    z = -(2.0 / 27.0 - 1.0 / (9.0 * math.sqrt(3.0))) ** (1.0 / 3.0)
    w = -2.0 / 3.0 + 1.0 / (9.0 * z) + z
    y = 0.25 * (1.0 + w**2)

    a1 = math.sqrt(1.0 / (9.0 * y) - 0.5 * w + math.sqrt(y)) - 1.0 / (3.0 * math.sqrt(y))
    a2 = 1.0 / (4.0 * a1) - 0.5 * a1
    a3 = 1.0 - a1 - a2

    a = np.array([a1, a2, a3], dtype=float)
    b = np.array([a3, a2, a1], dtype=float)
    return a, b


def _candy_rozmus_order4_coefficients():
    """Return the fourth-order Candy-Rozmus / Forest-Ruth coefficients."""
    cbrt2 = 2.0 ** (1.0 / 3.0)
    denom = 2.0 - cbrt2

    a1 = 1.0 / (2.0 * denom)
    a2 = (1.0 - cbrt2) / (2.0 * denom)
    b2 = 1.0 / denom
    b3 = -cbrt2 / denom

    a = np.array([a1, a2, a2, a1], dtype=float)
    b = np.array([0.0, b2, b3, b2], dtype=float)
    return a, b


def Default_Symplectic_Methods():
    """Return the default explicit symplectic coefficient tables.

    The methods are grouped exactly as in the source papers:

    - second-order Verlet/leapfrog variants
    - classical Ruth third-order
    - McLachlan optimal third-order
    - Candy-Rozmus / Forest-Ruth fourth-order
    - McLachlan optimal fourth- and fifth-order methods for quadratic kinetic
      energy (Runge-Kutta-Nystrom setting)
    """
    all_methods = []

    _append_symplectic_method(
        all_methods,
        "Verlet-DKD",
        [0.5, 0.5],
        [0.0, 1.0],
        order=2,
        paper_name="Leapfrog",
        reference="McLachlan and Atela (1992), Table 2; Candy and Rozmus (1991), Table II",
    )

    _append_symplectic_method(
        all_methods,
        "Verlet-KDK",
        [1.0, 0.0],
        [0.5, 0.5],
        order=2,
        paper_name="Pseudo-leapfrog",
        reference="McLachlan and Atela (1992), Table 2; Candy and Rozmus (1991), Table II",
    )

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    _append_symplectic_method(
        all_methods,
        "McLachlan-Optimal2",
        [inv_sqrt2, 1.0 - inv_sqrt2],
        [inv_sqrt2, 1.0 - inv_sqrt2],
        order=2,
        paper_name="Optimal order-2",
        reference="McLachlan and Atela (1992), Table 2",
    )

    _append_symplectic_method(
        all_methods,
        "Ruth3",
        [2.0 / 3.0, -2.0 / 3.0, 1.0],
        [7.0 / 24.0, 3.0 / 4.0, -1.0 / 24.0],
        order=3,
        paper_name="Ruth (1983)",
        reference="McLachlan and Atela (1992), Table 2; Candy and Rozmus (1991), Section 3.4",
    )

    a3_opt, b3_opt = _mclachlan_optimal_order3_coefficients()
    _append_symplectic_method(
        all_methods,
        "McLachlan-Optimal3",
        a3_opt,
        b3_opt,
        order=3,
        paper_name="Optimal order-3",
        reference="McLachlan and Atela (1992), Table 2",
    )

    a4_cr, b4_cr = _candy_rozmus_order4_coefficients()
    _append_symplectic_method(
        all_methods,
        "Candy-Rozmus4",
        a4_cr,
        b4_cr,
        order=4,
        paper_name="Candy and Rozmus (1991) / Forest and Ruth (1990)",
        reference="McLachlan and Atela (1992), Table 2; Candy and Rozmus (1991), Table II",
    )

    _append_symplectic_method(
        all_methods,
        "McLachlan-RKN4",
        [
            0.5153528374311229364,
            -0.0857820194129736460,
            0.4415830236164665242,
            0.1288461583653841854,
        ],
        [
            0.1344961992774310892,
            -0.2248198030794208058,
            0.7563200005156682911,
            0.3340036032863214255,
        ],
        order=4,
        paper_name="Optimal order-4 for quadratic kinetic energy",
        reference="McLachlan and Atela (1992), Table 2",
        quadratic_kinetic_only=True,
    )

    _append_symplectic_method(
        all_methods,
        "McLachlan-RKN5",
        [
            0.3398396258391100000,
            -0.0886013369030273290,
            0.5858564768259621188,
            -0.6030393565364918880,
            0.3235807965546976394,
            0.4423637942197494587,
        ],
        [
            0.1193900292875672758,
            0.6989273703824752308,
            -0.1713123582716007754,
            0.4012695022513534480,
            0.0107050818482359840,
            -0.0589796254980311632,
        ],
        order=5,
        paper_name="Optimal order-5 for quadratic kinetic energy",
        reference="McLachlan and Atela (1992), Table 2",
        quadratic_kinetic_only=True,
    )

    return all_methods
