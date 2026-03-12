"""IMEX multirate Runge-Kutta method tables.

This module stores the coefficient dictionaries consumed by
``DESolver`` when ``METHOD['type'] == 'IMEX-MRK'``.

The layout follows the notation used in

    Emil M. Constantinescu,
    "Implicit extensions of an explicit multirate Runge-Kutta scheme",
    Applied Mathematics Letters 128 (2022), 107871.
    DOI: 10.1016/j.aml.2021.107871

In the paper's notation:

- ``AB, bB, cB`` encode the base RK tableau ``(A, b, c)``.
- ``AF, bF, cF`` encode the explicit fast tableau ``A^F`` obtained by
  subcycling the base method.
- ``AS, bS, cS`` encode the explicit slow/buffer tableau ``A^S``.
- ``AT, bT, cT`` encode the implicit augmentation ``A_tilde`` from
  Eq. (4), which in the published methods has only the last row nonzero.

The conservative schemes described in Sections 3.1 and 3.2 keep
``bF = bS = bT`` so that the explicit and implicit contributions are
combined with the same completion weights.  The implementation here also
includes single-rate comparison schemes and a few repository-specific
exploratory variants that were used in notebook experiments.

The ``p*`` entries are historical metadata used elsewhere in the code
base for method reporting.  They are preserved here to avoid changing
solver behavior.
"""

import numpy as np


def _heun_base_method():
    """Return the two-stage Heun / explicit trapezoidal base RK tableau."""
    ab = np.zeros((2, 2))
    ab[1, 0] = 1.0
    bb = np.array([0.5, 0.5], dtype=float)
    return ab, bb


def _rk3_ssp_high_base_method():
    """Return the three-stage SSP RK3 base tableau used in the MRK file."""
    ab = np.zeros((3, 3))
    ab[1, 0] = 1.0
    ab[2, 0] = 0.25
    ab[2, 1] = 0.25
    bb = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], dtype=float)
    return ab, bb


def _build_fast_tableau(ab, bb, m):
    """Subcycle the base tableau ``m`` times to obtain the fast explicit part."""
    sb = bb.size
    af = np.zeros((sb * m, sb * m))

    for k in range(m):
        row = slice(k * sb, (k + 1) * sb)
        col = slice(k * sb, (k + 1) * sb)
        af[row, col] = ab / m

        # Later fast blocks see the already completed earlier substeps through
        # the base weights divided by the multirate ratio.
        for ell in range(k):
            af[row, ell * sb:(ell + 1) * sb] = bb[np.newaxis, :] / m

    bf = np.tile(bb / m, m)
    return af, bf


def _build_slow_tableau(ab, bb, m):
    """Repeat the base tableau on the diagonal to obtain the slow/buffer part."""
    sb = bb.size
    a_slow = np.zeros((sb * m, sb * m))

    for k in range(m):
        row = slice(k * sb, (k + 1) * sb)
        col = slice(k * sb, (k + 1) * sb)
        a_slow[row, col] = ab

    b_slow = np.tile(bb / m, m)
    return a_slow, b_slow


def _build_last_stage_only_tableau(nstages, last_row):
    """Create the paper's one-implicit-stage tableau with a single nonzero row."""
    at = np.zeros((nstages, nstages))
    at[-1, :] = np.asarray(last_row, dtype=float)
    return at


def _append_imex_method(
    all_methods,
    name,
    ab,
    bb,
    af,
    bf,
    a_slow,
    b_slow,
    at,
    bt,
    p_t,
    p_explicit=2,
):
    """Package a method dictionary in the format expected by ``DESolver``."""
    method = {"type": "IMEX-MRK", "name": name}

    for prefix, a_mat, b_vec, order in (
        ("B", ab, bb, p_explicit),
        ("F", af, bf, p_explicit),
        ("S", a_slow, b_slow, p_explicit),
        ("T", at, bt, p_t),
    ):
        a_copy = np.array(a_mat, copy=True)
        b_copy = np.array(b_vec, copy=True)

        method[f"A{prefix}"] = a_copy
        method[f"b{prefix}"] = b_copy
        method[f"c{prefix}"] = np.sum(a_copy, axis=1)
        method[f"s{prefix}"] = np.size(b_copy)
        method[f"p{prefix}"] = order

    all_methods.append(method)


def Default_IMEX_MRK_Methods():
    """Return the IMEX-MRK registry used by ``DESolver._RegisterDefaultMethods()``.

    Published methods from Constantinescu (2022):

    - ``MPRK2-IMEX2`` implements the A-stable second-order implicit extension
      from Section 3.1 / Eq. (5) by using ``AT[-1, :] = 1/2``.
    - ``MPRK2-IMEX`` implements the L-stable first-order implicit extension
      from Section 3.2 / Eq. (6) by using ``AT[-1, :] = 1``.

    The remaining tables are useful reference or exploratory variants:

    - ``PRK2-IMEX`` and ``PRK2-IMEX2`` are single-rate partitioned references.
    - ``MPRK2-IMEX-BE``, ``MPRK2-IMEX-BE-NC``, and ``MPRK2-IMEXb`` are
      repository-specific experiments.
    - ``MPRK2-RK3SSPHig-IMEX`` and ``MPRK2-m4-IMEX`` extend the same registry
      structure to a different explicit base or a different multirate ratio.
    """
    all_methods_imex_mrk = []

    heun_ab, heun_bb = _heun_base_method()
    heun_fast_a, heun_fast_b = _build_fast_tableau(heun_ab, heun_bb, m=2)
    heun_fast_aslow = np.array(heun_fast_a, copy=True)
    heun_fast_bslow = np.array(heun_fast_b, copy=True)
    heun_mprk_a_slow, heun_mprk_b_slow = _build_slow_tableau(heun_ab, heun_bb, m=2)

    # Single-rate partitioned reference schemes.  These reuse the same explicit
    # 4-stage Heun subcycling on both partitions, so they are useful baselines
    # when comparing the multirate conservative variants.
    _append_imex_method(
        all_methods_imex_mrk,
        "PRK2-IMEX",
        heun_ab,
        heun_bb,
        heun_fast_a,
        heun_fast_b,
        heun_fast_aslow,
        heun_fast_bslow,
        _build_last_stage_only_tableau(4, np.ones(4)),
        heun_fast_b,
        p_t=1,
    )
    _append_imex_method(
        all_methods_imex_mrk,
        "PRK2-IMEX2",
        heun_ab,
        heun_bb,
        heun_fast_a,
        heun_fast_b,
        heun_fast_aslow,
        heun_fast_bslow,
        _build_last_stage_only_tableau(4, 0.5 * np.ones(4)),
        heun_fast_b,
        p_t=2,
    )

    # Conservative Heun-based MPRK2 variants.  These use the multirate slow
    # tableau from Eq. (3), where the buffer/slow partition keeps the base
    # method on the diagonal and shares the explicit completion weights.
    #
    # These two Backward-Euler-like variants are repository comparison methods.
    # The conservative one keeps bT = bF, while the NC ("nonconservative")
    # version uses a one-hot implicit completion weight.
    _append_imex_method(
        all_methods_imex_mrk,
        "MPRK2-IMEX-BE",
        heun_ab,
        heun_bb,
        heun_fast_a,
        heun_fast_b,
        heun_mprk_a_slow,
        heun_mprk_b_slow,
        _build_last_stage_only_tableau(4, np.array([0.0, 0.0, 0.0, 1.0])),
        heun_fast_b,
        p_t=2,
    )
    _append_imex_method(
        all_methods_imex_mrk,
        "MPRK2-IMEX-BE-NC",
        heun_ab,
        heun_bb,
        heun_fast_a,
        heun_fast_b,
        heun_mprk_a_slow,
        heun_mprk_b_slow,
        _build_last_stage_only_tableau(4, np.array([0.0, 0.0, 0.0, 1.0])),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        p_t=2,
    )

    # Paper Eq. (6): second-order explicit / first-order implicit L-stable
    # extension obtained by setting a_tilde_{s,j} = 1 for every explicit stage.
    _append_imex_method(
        all_methods_imex_mrk,
        "MPRK2-IMEX",
        heun_ab,
        heun_bb,
        heun_fast_a,
        heun_fast_b,
        heun_mprk_a_slow,
        heun_mprk_b_slow,
        _build_last_stage_only_tableau(4, np.ones(4)),
        heun_fast_b,
        p_t=1,
    )

    # Paper Eq. (5): second-order explicit / second-order implicit A-stable
    # extension obtained by setting a_tilde_{s,j} = 1/2 for every stage.
    _append_imex_method(
        all_methods_imex_mrk,
        "MPRK2-IMEX2",
        heun_ab,
        heun_bb,
        heun_fast_a,
        heun_fast_b,
        heun_mprk_a_slow,
        heun_mprk_b_slow,
        _build_last_stage_only_tableau(4, 0.5 * np.ones(4)),
        heun_fast_b,
        p_t=2,
    )

    # Repository experiment: a custom implicit coupling that still ends in a
    # fully implicit last stage but injects part of g into an earlier explicit
    # stage as well.
    mprk2_imexb_at = np.zeros((4, 4))
    mprk2_imexb_at[2, 0] = 0.5
    mprk2_imexb_at[3, 0] = 1.0
    mprk2_imexb_at[3, 1] = 0.5
    mprk2_imexb_at[3, 2] = 1.0
    mprk2_imexb_at[3, 3] = 1.0
    _append_imex_method(
        all_methods_imex_mrk,
        "MPRK2-IMEXb",
        heun_ab,
        heun_bb,
        heun_fast_a,
        heun_fast_b,
        heun_mprk_a_slow,
        heun_mprk_b_slow,
        mprk2_imexb_at,
        heun_fast_b,
        p_t=1,
    )

    # Same IMEX-MRK packaging idea applied to the alternate SSP RK3 explicit
    # base already used in methods_mrk.py.
    rk3_ab, rk3_bb = _rk3_ssp_high_base_method()
    rk3_fast_a, rk3_fast_b = _build_fast_tableau(rk3_ab, rk3_bb, m=2)
    rk3_slow_a, rk3_slow_b = _build_slow_tableau(rk3_ab, rk3_bb, m=2)
    _append_imex_method(
        all_methods_imex_mrk,
        "MPRK2-RK3SSPHig-IMEX",
        rk3_ab,
        rk3_bb,
        rk3_fast_a,
        rk3_fast_b,
        rk3_slow_a,
        rk3_slow_b,
        _build_last_stage_only_tableau(
            6,
            np.array([0.25, 0.25, 1.0, 0.25, 0.25, 1.0]),
        ),
        rk3_fast_b,
        p_t=1,
    )

    # Heun-based MPRK2 with multirate ratio m = 4.  The explicit part is the
    # same repeated/subcycled construction as in the paper, only with four fast
    # substeps instead of two.
    heun_m4_fast_a, heun_m4_fast_b = _build_fast_tableau(heun_ab, heun_bb, m=4)
    heun_m4_slow_a, heun_m4_slow_b = _build_slow_tableau(heun_ab, heun_bb, m=4)
    _append_imex_method(
        all_methods_imex_mrk,
        "MPRK2-m4-IMEX",
        heun_ab,
        heun_bb,
        heun_m4_fast_a,
        heun_m4_fast_b,
        heun_m4_slow_a,
        heun_m4_slow_b,
        _build_last_stage_only_tableau(8, np.ones(8)),
        heun_m4_fast_b,
        p_t=1,
    )

    return all_methods_imex_mrk
