#!/usr/bin/env python3
"""
Programmatic verification of the Kurdjumov-Sachs (K-S) variant
misorientation table used by `gb_generator.py`.

The table in `gb_generator.PAIRS` is taken verbatim from Table 2 of:

    Morito, S., Tanaka, H., Konishi, R., Furuhara, T., & Maki, T. (2003).
    The morphology and crystallography of lath martensite in Fe-C alloys.
    Acta Materialia, 51(6), 1789-1799.
    https://doi.org/10.1016/S1359-6454(02)00577-3

(axis/angle of V1->Vn expressed in V1's cubic alpha frame). This script
verifies that table from first principles by:

    1. Building the K-S rotation R_KS from the parallelism conditions
           (1 1 1)_gamma || (0 1 1)_alpha,
           [-1 0 1]_gamma || [-1 -1 1]_alpha .
    2. Generating the 24 variant rotations as the right-coset orbit
           {R_KS . S_gamma : S_gamma in O}
       (the 24 cubic-gamma symmetry operations realise the 24 distinct
       parent-frame embeddings consistent with the K-S parallelism;
       see `variant_orbit()` for why right-multiplication is required
       and left-multiplication by S_alpha collapses trivially under
       the disorientation reduction).
    3. For each Pair p in `gb_generator.PAIRS`, finding the variant
       index n in {2..24} whose disorientation with V1 best matches
       (angle, axis) given in p, and reporting any mismatch.

Both axis and angle are reduced into the cubic disorientation
fundamental zone (smallest positive angle over S_a . R . S_b^T,
S_a, S_b in O). Axis equivalence is checked modulo the 24 cubic
symmetry operations and modulo sign.

A pass means: for every row in `gb_generator.PAIRS` there exists a
variant in the orbit whose (axis, angle) reproduces the table to
within ANGLE_TOL_DEG and AXIS_TOL_DEG.

Run:
    python3 ks_misorientations.py [-v]

Exit code 0 means every row in the table is consistent with the
first-principles K-S construction.
"""
from __future__ import annotations

import argparse
import logging
import sys
from itertools import permutations, product
from pathlib import Path

import numpy as np

# Allow running this script from inside the verification/ subfolder by
# putting the project root (which holds gb_generator.py) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gb_generator import PAIRS, DEFAULT_V1_DIRS  # noqa: E402

log = logging.getLogger("ks_misorientations")

ANGLE_TOL_DEG = 0.05      # match tolerance on the rotation angle
AXIS_TOL_DEG = 0.5        # max angular deviation of axis directions


# =====================================================================
# Cubic symmetry (24 proper rotations of point group O)
# =====================================================================

def cubic_rotation_group() -> list[np.ndarray]:
    """Return the 24 proper rotations of the cubic point group."""
    mats: list[np.ndarray] = []
    seen: set[bytes] = set()
    for perm in permutations(range(3)):
        for signs in product((+1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, j in enumerate(perm):
                M[i, j] = signs[i]
            if np.linalg.det(M) <= 0:
                continue        # drop improper rotations (mirrors)
            key = M.astype(np.int8).tobytes()
            if key in seen:
                continue        # de-duplicate (shouldn't trigger, defensive)
            seen.add(key)
            mats.append(M)
    assert len(mats) == 24, f"expected 24 cubic rotations, got {len(mats)}"
    return mats


CUBIC_24 = cubic_rotation_group()


# =====================================================================
# Rotation utilities
# =====================================================================

def rot_from_two_pairs(a1: np.ndarray, b1: np.ndarray,
                       a2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Rotation R such that R a1 || b1 and R a2 || b2."""
    def basis(u, v):
        # Build a right-handed orthonormal triad (u, t, w) with u along
        # `u` and w perpendicular to the (u, v) plane.
        u = u / np.linalg.norm(u)
        w = np.cross(u, v)
        w /= np.linalg.norm(w)
        t = np.cross(w, u)
        return np.column_stack([u, t, w])
    A = basis(a1, a2)
    B = basis(b1, b2)
    return B @ A.T


def axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (unit axis, angle in degrees) for rotation matrix R.
    Angle is in [0, 180]."""
    cos_th = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if np.pi - th < 1e-9:
        # 180 deg rotation: axis is the +1 eigenvector of (R + I)/2.
        M = (R + np.eye(3)) / 2.0
        col = int(np.argmax(np.linalg.norm(M, axis=0)))
        axis = M[:, col]
        axis /= np.linalg.norm(axis)
        return axis, np.degrees(th)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])
    axis /= (2.0 * np.sin(th))
    return axis, np.degrees(th)


def disorientation(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Reduce R into the cubic disorientation fundamental zone.
    Returns (axis, angle_deg) with smallest positive angle."""
    best_ang = 360.0
    best_R = R
    for S1 in CUBIC_24:
        for S2 in CUBIC_24:
            R2 = S1 @ R @ S2
            _, ang = axis_angle(R2)
            if ang < best_ang:
                best_ang = ang
                best_R = R2
    ax, ang = axis_angle(best_R)
    return ax, ang


def axes_equivalent_under_cubic(ax_a: np.ndarray, ax_b: np.ndarray,
                                tol_deg: float) -> tuple[bool, float]:
    """Check if two axes are parallel (modulo sign and cubic symmetry)."""
    a = ax_a / np.linalg.norm(ax_a)
    b = ax_b / np.linalg.norm(ax_b)
    best = 180.0
    for S in CUBIC_24:
        Sa = S @ a
        c = abs(float(np.dot(Sa, b)))
        d = float(np.degrees(np.arccos(min(1.0, c))))
        if d < best:
            best = d
    return best <= tol_deg, best


# =====================================================================
# K-S construction
# =====================================================================

def build_ks_base() -> np.ndarray:
    """Reference K-S rotation: gamma -> V1 alpha frame from the two
    parallelism conditions (111)_g || (011)_a, [-101]_g || [-1-11]_a."""
    n_g = np.array([1.0, 1.0, 1.0])
    n_a = np.array([0.0, 1.0, 1.0])
    d_g = np.array([-1.0, 0.0, 1.0])
    d_a = np.array([-1.0, -1.0, 1.0])
    return rot_from_two_pairs(n_g, n_a, d_g, d_a)


def variant_orbit() -> list[np.ndarray]:
    """The 24 K-S variants as the orbit {R_KS . S : S in O_gamma}."""
    R_KS = build_ks_base()
    return [R_KS @ S for S in CUBIC_24]


# =====================================================================
# K-S parallelism (Miller indices)
# =====================================================================

def _signed_family(abs_pattern: tuple[int, int, int]) -> list[np.ndarray]:
    """All distinct signed integer triplets sharing the given multiset
    of absolute values."""
    seen: set[tuple[int, int, int]] = set()
    out: list[np.ndarray] = []
    for perm in permutations(abs_pattern):
        for signs in product((1, -1), repeat=3):
            v = tuple(int(s * x) for s, x in zip(signs, perm))
            if v in seen:
                continue
            seen.add(v)
            out.append(np.array(v, dtype=int))
    return out


# {1 1 1}_gamma close-packed planes (8 signed triplets) and the
# {0 1 1}_alpha close-packed planes they map to (12 signed triplets).
GAMMA_PLANE_FAMILY = _signed_family((1, 1, 1))
ALPHA_PLANE_FAMILY = _signed_family((1, 1, 0))

# <1 0 -1>_gamma close-packed directions (12 signed triplets) and the
# <1 1 -1>_alpha close-packed directions they map to (8 signed triplets).
GAMMA_DIR_FAMILY = _signed_family((1, 1, 0))
ALPHA_DIR_FAMILY = _signed_family((1, 1, 1))


def _best_parallel(
    R: np.ndarray,
    gamma_family: list[np.ndarray],
    alpha_family: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find (v_gamma, v_alpha) integer triplets such that R @ unit(v_g)
    is most closely parallel (same direction) to unit(v_a). Returns
    (v_gamma, v_alpha, dot)."""
    best_dot = -2.0
    best_pair = (gamma_family[0], alpha_family[0])
    for vg in gamma_family:
        ug = vg / np.linalg.norm(vg)
        img = R @ ug
        for va in alpha_family:
            ua = va / np.linalg.norm(va)
            dot = float(np.dot(img, ua))
            if dot > best_dot:
                best_dot = dot
                best_pair = (vg, va)
    return best_pair[0], best_pair[1], best_dot


def ks_parallelism(R: np.ndarray) -> dict[str, np.ndarray]:
    """Return the K-S close-packed plane and direction parallelisms
    realised by a single orbit member R: gamma -> alpha."""
    pg, pa, _ = _best_parallel(R, GAMMA_PLANE_FAMILY, ALPHA_PLANE_FAMILY)
    dg, da, _ = _best_parallel(R, GAMMA_DIR_FAMILY, ALPHA_DIR_FAMILY)
    return {"plane_gamma": pg, "plane_alpha": pa,
            "dir_gamma": dg, "dir_alpha": da}


def _match_orbit_for_pair(p, orbit: list[np.ndarray],
                          R1: np.ndarray) -> np.ndarray:
    """Return one orbit member reproducing PAIR p's (axis, angle)
    disorientation with V1, modulo cubic symmetry."""
    ax_tab = np.asarray(p.axis_alpha, float)
    best = orbit[0]
    best_score = (np.inf, np.inf)
    for Rn in orbit:
        ax, ang = disorientation(Rn @ R1.T)
        if ang < 1e-3:
            continue
        ang_dev = abs(ang - p.angle_deg)
        _, axis_dev = axes_equivalent_under_cubic(ax, ax_tab, tol_deg=180.0)
        score = (ang_dev if ang_dev > ANGLE_TOL_DEG else 0.0, axis_dev)
        if score < best_score:
            best_score = score
            best = Rn
    return best


def _fmt_miller(v: np.ndarray) -> str:
    return "[" + " ".join(f"{int(round(x)):2d}" for x in v) + "]"


def print_miller_table() -> None:
    """Print, for V1 and every PAIRS row, the K-S Miller-index
    parallelism (h k l)_gamma || (h k l)_alpha and
    [u v w]_gamma || [u v w]_alpha.
    """
    R1 = build_ks_base()
    orbit = variant_orbit()

    log.info("")
    log.info("Per-variant K-S Miller-index parallelism")
    log.info("(close-packed plane and close-packed direction in each crystal)")
    log.info("%-5s   %-10s || %-10s    %-10s || %-10s",
             "var", "(hkl)_g", "(hkl)_a", "[uvw]_g", "[uvw]_a")

    par = ks_parallelism(R1)
    log.info("%-5s   %-10s || %-10s    %-10s || %-10s",
             "V1",
             _fmt_miller(par["plane_gamma"]),
             _fmt_miller(par["plane_alpha"]),
             _fmt_miller(par["dir_gamma"]),
             _fmt_miller(par["dir_alpha"]))

    for p in PAIRS:
        Rn = _match_orbit_for_pair(p, orbit, R1)
        par = ks_parallelism(Rn)
        log.info("%-5s   %-10s || %-10s    %-10s || %-10s",
                 f"V{p.i}",
                 _fmt_miller(par["plane_gamma"]),
                 _fmt_miller(par["plane_alpha"]),
                 _fmt_miller(par["dir_gamma"]),
                 _fmt_miller(par["dir_alpha"]))


# =====================================================================
# Per-variant Miller directions along the GLOBAL BOX axes
# =====================================================================

V1_BOX_AXES_CUBIC = np.array(DEFAULT_V1_DIRS, dtype=float)  # rows: X, Y, Z
V1_BOX_AXIS_LENGTHS = np.linalg.norm(V1_BOX_AXES_CUBIC, axis=1)


def _rodrigues(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix for a rotation by `angle_deg` about `axis`."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) * c + s * K + (1.0 - c) * np.outer(axis, axis)


def miller_along_box(R_cubic: np.ndarray) -> np.ndarray:
    """Cubic-alpha Miller directions of grain 2's crystal along the
    GLOBAL box X, Y, Z axes, given the V1->Vn rotation R_cubic in the
    cubic-alpha frame. Returns a (3, 3) array whose rows are the
    box-X, box-Y, box-Z Miller triplets, scaled to V1's integer
    Miller magnitudes (sqrt(3), sqrt(6), sqrt(2)) so that R_cubic = I
    reproduces V1_DIRS exactly.
    """
    out = np.zeros((3, 3))
    for i, (axis_cubic, length) in enumerate(
            zip(V1_BOX_AXES_CUBIC, V1_BOX_AXIS_LENGTHS)):
        unit_in_v1 = axis_cubic / length
        # The cubic Miller direction in grain 2 that lies along the same
        # box axis is the V1 cubic direction rotated *backwards* by the
        # V1->Vn rotation.
        out[i] = R_cubic.T @ unit_in_v1 * length
    return out


def _fmt_float_miller(v: np.ndarray) -> str:
    return "[" + " ".join(f"{x:+7.4f}" for x in v) + "]"


def _fmt_int_miller(v: np.ndarray) -> str:
    return "[" + " ".join(f"{int(round(x)):+3d}" for x in v) + "]"


def print_box_axes_table() -> None:
    """Print, for V1 and every PAIRS row, the cubic-alpha Miller
    directions of the grain's crystal that lie along the global box
    X, Y, Z axes.
    """
    log.info("")
    log.info("Per-variant Miller directions of the grain crystal along the global box axes")
    log.info("(cubic-alpha frame; V1 row reproduces V1_DIRS exactly, others are irrational)")
    log.info("%-5s   %-25s   %-25s   %-25s",
             "var", "X_box (V1: [1 1 1])",
             "Y_box (V1: [1 1 -2])", "Z_box (V1: [-1 1 0])")

    v1_axes = miller_along_box(np.eye(3))
    log.info("%-5s   %-25s   %-25s   %-25s",
             "V1",
             _fmt_int_miller(v1_axes[0]),
             _fmt_int_miller(v1_axes[1]),
             _fmt_int_miller(v1_axes[2]))

    for p in PAIRS:
        R = _rodrigues(np.asarray(p.axis_alpha, float), p.angle_deg)
        axes = miller_along_box(R)
        log.info("%-5s   %-25s   %-25s   %-25s",
                 f"V{p.i}",
                 _fmt_float_miller(axes[0]),
                 _fmt_float_miller(axes[1]),
                 _fmt_float_miller(axes[2]))


# =====================================================================
# Verification
# =====================================================================

def match_pairs() -> tuple[int, int]:
    """For each Pair p in PAIRS, find an orbit member whose disorientation
    with V1 reproduces (p.angle_deg, p.axis_alpha). Returns (n_ok, n_total)."""
    orbit = variant_orbit()
    # V1 itself is the orbit member whose disorientation with R_KS is 0.
    R1 = build_ks_base()

    # Pre-compute disorientations of each orbit member with V1.
    # Skip orbit members whose disorientation is ~0 deg (V1 itself, i.e.
    # any orbit member whose stabiliser includes the identity element of
    # O_alpha relative to R_KS). Those are not "Vn != V1" pairs.
    disos: list[tuple[np.ndarray, float]] = []
    for Rn in orbit:
        ax, ang = disorientation(Rn @ R1.T)
        if ang < 1e-3:
            continue
        disos.append((ax, ang))

    log.info("%-9s %8s %26s   %8s %26s   %s",
             "pair", "ang_tab", "axis_tab",
             "ang_calc", "axis_calc", "status")

    n_ok = 0
    for p in PAIRS:
        ax_tab = np.asarray(p.axis_alpha, float)
        # Find best-matching orbit member: first restrict to those with
        # the right angle (within ANGLE_TOL_DEG); among those, pick the
        # one with smallest axis deviation. This avoids cross-matching
        # two distinct variants that happen to share the same angle.
        candidates = []
        for ax_calc, ang_calc in disos:
            ang_dev = abs(ang_calc - p.angle_deg)
            _, axis_dev = axes_equivalent_under_cubic(ax_calc, ax_tab,
                                                     tol_deg=180.0)
            candidates.append((ang_dev, axis_dev, ax_calc, ang_calc))
        # Prefer angle-matching candidates; otherwise fall back to the
        # closest by combined deviation.
        ang_matched = [c for c in candidates if c[0] <= ANGLE_TOL_DEG]
        pool = ang_matched if ang_matched else candidates
        ang_dev, axis_dev, ax_calc, ang_calc = min(pool, key=lambda c: c[1])

        ang_ok = ang_dev <= ANGLE_TOL_DEG
        ax_ok = axis_dev <= AXIS_TOL_DEG
        ok = ang_ok and ax_ok
        if ok:
            n_ok += 1
            status = "OK"
        else:
            bits = []
            if not ang_ok:
                bits.append(f"ANGLE off by {ang_calc - p.angle_deg:+.3f}")
            if not ax_ok:
                bits.append(f"AXIS off by {axis_dev:.2f} deg")
            status = "FAIL: " + "; ".join(bits)

        log.info("%-9s %8.2f %26s   %8.3f %26s   %s",
                 p.name, p.angle_deg,
                 np.array2string(ax_tab, precision=4, suppress_small=True),
                 ang_calc,
                 np.array2string(ax_calc, precision=4, suppress_small=True),
                 status)
    return n_ok, len(PAIRS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify gb_generator.PAIRS against the K-S construction.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    n_ok, n_total = match_pairs()
    print_miller_table()
    print_box_axes_table()
    log.info("")
    if n_ok == n_total:
        log.info("All %d rows consistent with the K-S construction.", n_total)
        sys.exit(0)
    else:
        log.error("%d / %d rows mismatched.", n_total - n_ok, n_total)
        sys.exit(1)


if __name__ == "__main__":
    main()
