#!/usr/bin/env python3
"""
Build BCC Fe K-S martensite variant-pair bicrystals with a single uniform
pipeline:

    1. Generate the V1 reference crystal once.
    2. For every V1-Vn pair, build grain 2 by rotating a copy of V1 by
       the K-S misorientation (axis + angle in the cubic alpha frame)
       and cropping back to V1's box dimensions.
    3. Stack grain 1 (V1) and grain 2 along Y, sharing the V1 Lx and Lz.

All bicrystals are non-periodic (pbc=False); the resulting LAMMPS data
files should be used with `boundary s s s`.

V1 orientation:
    Default X = [1 1 1],  Y = [1 1 -2],  Z = [-1 1 0]   (right-handed).
    The interface normal is always box Y; [1 1 -2] is only its default
    direction in V1's cubic alpha frame. Use --boundary-plane H K L to
    select another V1 interface plane, with X and Z chosen automatically.
    Alternatively, set --v1-x / --v1-y / --v1-z (integer or real vectors
    in V1's cubic alpha frame; must be orthogonal and right-handed).
    The interface normal is separate from each pair's rotation axis;
    changing the plane preserves the K-S misorientation.

The pair table (axes/angles in V1's cubic alpha frame) is taken
verbatim from Morito et al. (2003), Acta Materialia 51(6), 1789-1799,
Table 2 [doi:10.1016/S1359-6454(02)00577-3], and is verified
row-by-row by `ks_misorientations.py`.

Requirements:
    pip install ase scipy numpy

Run:
    python3 gb_generator.py [--help]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.lattice.cubic import BodyCenteredCubic
from scipy.spatial import KDTree

log = logging.getLogger("gb_generator")

# =====================================================================
# Defaults (all overridable from CLI)
# =====================================================================
DEFAULT_A = 2.866                # BCC Fe lattice constant (Angstrom)
DEFAULT_NX, DEFAULT_NY, DEFAULT_NZ = 15, 5, 10  # unit cells per grain
DEFAULT_OVERLAP_TOL = 0.5        # A; merge atoms closer than this at GB
DEFAULT_OVERSIZE = (5, 5, 5)     # oversize factors for rotate+crop
DEFAULT_CROP_EPS = 1e-6          # A; crop face tolerance
DEFAULT_GB_GAP = 0.0             # A; vacuum between grains (0 = touching)
DEFAULT_RATIO_TOL = 0.02         # max |density_ratio - 1| after crop
DEFAULT_GB_SHIFT_X = 0.0         # A; in-plane shift of grain 2 along X
DEFAULT_GB_SHIFT_Z = 0.0         # A; in-plane shift of grain 2 along Z
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"

# ---------------------------------------------------------------------
# V1 reference orientation (default): X=[111], Y=[11-2], Z=[-110]
# Select the interface normal via --boundary-plane, or set all three axes
# via --v1-x / --v1-y / --v1-z (must be orthogonal, right-handed).
DEFAULT_V1_DIRS = ((1, 1, 1), (1, 1, -2), (-1, 1, 0))

# Tolerance on V1 box-axis orthogonality. Off-diagonal entries of the
# normalized Gram matrix (i.e. |cos(angle_ij)| for i != j) must be
# below this value. 1e-3 corresponds to ~0.06 deg out-of-orthogonal,
# which comfortably accommodates 4-decimal rounding of irrational
# axes (e.g. 0.5774 ~ 1/sqrt(3)).
V1_ORTHO_ATOL = 1.0e-3


def validate_v1_dirs(dirs, ortho_atol: float = V1_ORTHO_ATOL):
    """Validate V1 box axes: must be 3 orthogonal right-handed vectors.
    Returns as ints if integer, else as floats."""
    arr = np.asarray(dirs, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"V1 directions must be a 3x3 array, got {arr.shape}")
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("V1 box axes must be non-zero vectors")
    G = arr @ arr.T
    cos_off = (G - np.diag(np.diag(G))) / np.outer(norms, norms)
    max_off = float(np.abs(cos_off).max())
    if max_off > ortho_atol:
        raise ValueError(
            f"V1 box axes are not mutually orthogonal: "
            f"max |cos(angle_ij)| = {max_off:.3e} > {ortho_atol:.0e}")
    if np.linalg.det(arr) <= 0:
        raise ValueError(f"V1 box axes are not right-handed: {dirs!r}")
    if np.allclose(arr, np.round(arr)) and np.max(np.abs(arr)) < 1000:
        return tuple(tuple(int(round(x)) for x in row) for row in arr)
    return tuple(tuple(float(x) for x in row) for row in arr)


def v1_dirs_from_boundary_plane(plane):
    """Build a right-handed V1 box frame from a GB-plane normal.

    The requested normal becomes box Y.  Box X is the projection of the
    default X direction into that plane, and box Z = X cross Y.  Primitive
    integer directions are retained when integer Miller indices are supplied.
    """
    y = np.asarray(plane, dtype=float)
    if y.shape != (3,) or np.linalg.norm(y) == 0.0:
        raise ValueError("boundary plane must be three numbers and non-zero")

    candidates = [np.asarray(DEFAULT_V1_DIRS[0], dtype=float),
                  np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]]
    x = None
    yy = float(y @ y)
    for candidate in candidates:
        projected = candidate * yy - y * float(candidate @ y)
        if np.linalg.norm(projected) > 1.0e-12 * yy:
            x = projected
            break
    assert x is not None
    z = np.cross(x, y)

    if np.allclose(y, np.round(y)):
        def primitive(v):
            values = np.rint(v).astype(int)
            divisor = int(np.gcd.reduce(np.abs(values)))
            return tuple((values // max(divisor, 1)).tolist())
        dirs = (primitive(x), primitive(y), primitive(z))
    else:
        dirs = tuple(tuple((v / np.linalg.norm(v)).tolist())
                     for v in (x, y, z))
    return validate_v1_dirs(dirs)


def _is_integer_miller(dirs) -> bool:
    """Whether `dirs` are integer Miller indices (eligible for ASE's
    fast periodic-tiling path)."""
    arr = np.asarray(dirs, dtype=float)
    return bool(np.allclose(arr, np.round(arr)))


# Validate the default at import so a broken default is caught early.
validate_v1_dirs(DEFAULT_V1_DIRS)


# =====================================================================
# K-S variant pair table (Morito et al. 2003, Acta Mater. 51(6), Table 2)
# =====================================================================

@dataclass(frozen=True)
class Pair:
    """A single V1-Vn bicrystal recipe (name, variant index, angle, axis)."""
    name: str
    i: int
    angle_deg: float
    axis_alpha: tuple[float, float, float]

    def label(self) -> str:
        return f"V1-V{self.i:<2d} {self.angle_deg:5.2f} deg"


# Misorientation table V1 -> Vn (axis + angle in V1's cubic alpha frame).
PAIRS: list[Pair] = [
    Pair("ks_v1v2",  2,  60.00, ( 0.5774, -0.5774,  0.5774)),
    Pair("ks_v1v3",  3,  60.00, ( 0.0000, -0.7071, -0.7071)),
    Pair("ks_v1v4",  4,  10.53, ( 0.0000,  0.7071,  0.7071)),
    Pair("ks_v1v5",  5,  60.00, ( 0.0000,  0.7071,  0.7071)),
    Pair("ks_v1v6",  6,  49.47, ( 0.0000, -0.7071, -0.7071)),
    Pair("ks_v1v7",  7,  49.47, (-0.5774,  0.5774,  0.5774)),
    Pair("ks_v1v8",  8,  10.53, ( 0.5774, -0.5774,  0.5774)),
    Pair("ks_v1v9",  9,  50.51, (-0.1862,  0.7666,  0.6145)),
    Pair("ks_v1v10", 10, 50.51, (-0.4904, -0.4625,  0.7387)),
    Pair("ks_v1v11", 11, 14.88, ( 0.3543, -0.9329, -0.0650)),
    Pair("ks_v1v12", 12, 57.21, ( 0.3568, -0.7136,  0.6029)),
    Pair("ks_v1v13", 13, 14.88, ( 0.9329,  0.3543,  0.0650)),
    Pair("ks_v1v14", 14, 50.51, (-0.7387,  0.4625, -0.4904)),
    Pair("ks_v1v15", 15, 57.21, (-0.2461, -0.6278, -0.7384)),
    Pair("ks_v1v16", 16, 20.61, ( 0.6589,  0.6589,  0.3628)),
    Pair("ks_v1v17", 17, 51.73, (-0.6589,  0.3628, -0.6589)),
    Pair("ks_v1v18", 18, 47.11, (-0.3022, -0.6255, -0.7193)),
    Pair("ks_v1v19", 19, 50.51, (-0.6145,  0.1862, -0.7666)),
    Pair("ks_v1v20", 20, 57.21, (-0.3568, -0.6029, -0.7136)),
    Pair("ks_v1v21", 21, 20.61, ( 0.9551,  0.0000, -0.2962)),
    Pair("ks_v1v22", 22, 47.11, (-0.7193,  0.3022, -0.6255)),
    Pair("ks_v1v23", 23, 57.21, (-0.7384, -0.2461,  0.6278)),
    Pair("ks_v1v24", 24, 21.06, ( 0.9121,  0.4100,  0.0000)),
]


@dataclass(frozen=True)
class Config:
    """Runtime configuration populated from CLI."""
    a: float = DEFAULT_A
    nx: int = DEFAULT_NX
    ny: int = DEFAULT_NY
    nz: int = DEFAULT_NZ
    oversize: tuple[int, int, int] = DEFAULT_OVERSIZE
    overlap_tol: float = DEFAULT_OVERLAP_TOL
    crop_eps: float = DEFAULT_CROP_EPS
    gb_gap: float = DEFAULT_GB_GAP
    ratio_tol: float = DEFAULT_RATIO_TOL
    gb_shift_x: float = DEFAULT_GB_SHIFT_X
    gb_shift_z: float = DEFAULT_GB_SHIFT_Z
    out_dir: Path = DEFAULT_OUT_DIR
    v1_dirs: tuple = DEFAULT_V1_DIRS


def bcc_nn_distance(a: float) -> float:
    """First-neighbour distance of a BCC lattice with lattice constant `a`."""
    return a * np.sqrt(3.0) / 2.0


def gb_gap_warning(gb_gap: float, a: float) -> Optional[str]:
    """Return a warning string if gb_gap is in a risky range, else None."""
    nn_dist = bcc_nn_distance(a)
    if gb_gap < 0.0:
        return f"negative gb_gap={gb_gap:g} A causes interpenetration"
    if 0.0 < gb_gap < nn_dist:
        return (f"gb_gap={gb_gap:g} A is below 1NN distance ({nn_dist:.3f} A); "
                f"atoms may start inside the repulsive wall")
    return None


# =====================================================================
# Core builders
# =====================================================================

def build_oriented(dirs, cfg: Config, size=None) -> Atoms:
    """BCC Fe crystal with the given Miller directions as box axes.
    Integer dirs use ASE tiling; real-valued dirs use rotate-and-crop."""
    if size is None:
        size = (cfg.nx, cfg.ny, cfg.nz)
    if _is_integer_miller(dirs):
        return BodyCenteredCubic(
            symbol="Fe", latticeconstant=cfg.a,
            directions=[[int(round(x)) for x in d] for d in dirs],
            size=size,
        )
    return _build_oriented_general(dirs, cfg, size)


def _build_oriented_general(dirs, cfg: Config, size) -> Atoms:
    """Rotate-and-crop builder for real-valued (non-integer) box axes."""
    nx, ny, nz = size
    a = cfg.a
    norms = np.linalg.norm(np.asarray(dirs, dtype=float), axis=1)
    Lbox = a * norms * np.array([nx, ny, nz], dtype=float)

    diag = float(np.linalg.norm(Lbox))
    n_cubic = int(np.ceil(diag / a)) + 4
    big = BodyCenteredCubic(
        symbol="Fe", latticeconstant=a,
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        size=(n_cubic, n_cubic, n_cubic),
    )

    # QR-orthonormalized rotation: cubic frame -> box frame
    R = _orthonormal_box_basis(dirs)

    centre = n_cubic * a / 2.0
    pos = (big.positions - centre) @ R.T + Lbox / 2.0

    # Half-open crop, matching the convention in rotated_grain().
    eps = cfg.crop_eps
    upper = Lbox - eps
    mask = ((pos >= -eps) & (pos < upper)).all(axis=1)
    n_kept = int(mask.sum())
    log.debug("    general V1 build: kept %d / %d atoms",
              n_kept, len(big))
    return Atoms(symbols=["Fe"] * n_kept, positions=pos[mask],
                 cell=np.diag(Lbox), pbc=False)


def rotation_matrix(axis, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix for a unit-axis rotation."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(axis, axis)


def _orthonormal_box_basis(dirs) -> np.ndarray:
    """3x3 orthonormal matrix whose rows are unit V1 box axes (via QR)."""
    A = np.column_stack([
        np.asarray(d, dtype=float) / np.linalg.norm(d) for d in dirs
    ])
    Q, R = np.linalg.qr(A)
    # QR sign convention: enforce diag(R) > 0 so Q has the same
    # orientation as A (first column of Q = +unit(dirs[0])).
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    return Q.T  # rows = unit box axes in cubic frame


def cubic_to_box_frame(axis_cubic, v1_dirs=DEFAULT_V1_DIRS) -> np.ndarray:
    """Project a cubic-frame [hkl] axis onto V1's box frame. Returns unit vector."""
    e = _orthonormal_box_basis(v1_dirs)
    v = np.array(axis_cubic, dtype=float)
    v /= np.linalg.norm(v)
    return e @ v


def _cell_lengths(atoms: Atoms) -> np.ndarray:
    """Diagonal cell vector lengths (asserts cell is diagonal)."""
    C = atoms.cell.array
    off = C - np.diag(np.diag(C))
    assert np.allclose(off, 0.0, atol=1e-9), \
        f"non-diagonal cell encountered:\n{C}"
    return np.diag(C).copy()


def rotated_grain(angle_deg: float, axis_cubic,
                  target_dims: np.ndarray, cfg: Config) -> Atoms:
    """Build oversized V1, rotate by (axis_cubic, angle), crop to target_dims.
    Returns Atoms with pbc=False."""
    Lx, Ly, Lz = target_dims

    if abs(angle_deg) < 1e-12:
        atoms = build_oriented(cfg.v1_dirs, cfg)
        atoms.set_pbc(False)
        return atoms

    # Step 1: choose a source box large enough to contain the target box after
    # inverse rotation.  Multiplying each dimension by a fixed factor alone is
    # insufficient for strongly anisotropic cells: a long target X dimension
    # can rotate into the much shorter source Y dimension.
    ox, oy, oz = cfg.oversize
    axis_box = cubic_to_box_frame(axis_cubic, cfg.v1_dirs)
    R = rotation_matrix(axis_box, angle_deg)
    required_dims = np.abs(R.T) @ np.asarray(target_dims, dtype=float)
    periods = np.asarray(target_dims, dtype=float) / np.array(
        [cfg.nx, cfg.ny, cfg.nz], dtype=float)
    required_size = np.ceil(required_dims / periods).astype(int) + 2
    requested_size = np.array(
        [cfg.nx * ox, cfg.ny * oy, cfg.nz * oz], dtype=int)
    source_size = tuple(np.maximum(required_size, requested_size).tolist())
    log.debug("    rotated source size: requested=%s required=%s selected=%s",
              tuple(requested_size), tuple(required_size), source_size)
    big = build_oriented(cfg.v1_dirs, cfg, size=source_size)
    big_diag = _cell_lengths(big)

    # Steps 2-4: centre, rotate, recentre.
    pos = big.positions - big_diag / 2.0
    pos = pos @ R.T                              # active rotation of positions
    pos = pos + np.array([Lx, Ly, Lz]) / 2.0     # recentre on target box

    # Crop to half-open box [0, L)
    eps = cfg.crop_eps
    upper = np.array([Lx, Ly, Lz]) - eps
    mask = ((pos >= -eps) & (pos < upper)).all(axis=1)

    n_kept = int(mask.sum())
    # Density check
    expected = len(big) * np.prod(target_dims) / np.prod(big_diag)
    ratio = n_kept / expected
    is_general = not _is_integer_miller(cfg.v1_dirs)
    # Atom counts in very small rotated cells fluctuate because lattice sites
    # cross the crop faces.  Use a 1/sqrt(N) finite-size floor so a difference
    # of a few boundary atoms is not mistaken for an undersized source box.
    finite_size_tol = 1.0 / np.sqrt(expected)
    effective_tol = max(cfg.ratio_tol, finite_size_tol,
                        0.15 if is_general else 0.0)
    if abs(ratio - 1.0) > effective_tol:
        raise RuntimeError(
            f"rotated-crop density ratio = {ratio:.4f} (expected ~1.0; "
            f"tol={effective_tol}); increase OVERSIZE or check CROP_EPS."
        )
    log.debug("    rotated-crop density ratio = %.5f (n=%d)", ratio, n_kept)

    return Atoms(symbols=["Fe"] * n_kept, positions=pos[mask],
                 cell=np.diag([Lx, Ly, Lz]), pbc=False)


def stack_along_y(g1: Atoms, g2: Atoms, cfg: Config) -> Atoms:
    """Stack g2 on top of g1 along Y with optional gap and in-plane shift.
    Removes overlapping atoms at the GB seam."""
    c1, c2 = g1.cell.array, g2.cell.array
    assert np.allclose(c1[0], c2[0]), "Lx mismatch between grains"
    assert np.allclose(c1[2], c2[2]), "Lz mismatch between grains"
    Ly1 = c1[1, 1]
    gap = cfg.gb_gap
    y_shift = Ly1 + gap
    dx, dz = cfg.gb_shift_x, cfg.gb_shift_z
    g2_pos_shifted = g2.positions + np.array([dx, y_shift, dz])
    pos = np.vstack([g1.positions, g2_pos_shifted])

    cell = c1.copy()
    cell[1, 1] = c1[1, 1] + c2[1, 1] + gap
    if dx != 0.0 or dz != 0.0:
        lo = pos.min(axis=0)
        hi = pos.max(axis=0)
        pos[:, 0] -= lo[0]
        pos[:, 2] -= lo[2]
        cell[0, 0] = hi[0] - lo[0]
        cell[2, 2] = hi[2] - lo[2]

    merged = Atoms(symbols=["Fe"] * len(pos), positions=pos,
                   cell=cell, pbc=False)
    y_seam = Ly1 + 0.5 * gap
    return _remove_central_overlaps(merged, y_seam, cfg.overlap_tol)


def _remove_central_overlaps(a: Atoms, y_mid: float, tol: float) -> Atoms:
    """Remove duplicate atoms within `tol` of the GB at y=y_mid.
    Drops the atom farther from y_mid in each pair."""
    pos = a.positions
    cand = np.where(np.abs(pos[:, 1] - y_mid) < 2 * tol)[0]
    if len(cand) == 0:
        return a
    tree = KDTree(pos[cand])
    pairs = tree.query_pairs(r=tol, output_type="ndarray")
    if len(pairs) == 0:
        return a

    drop_global: set[int] = set()
    for i, j in pairs:
        gi, gj = int(cand[i]), int(cand[j])
        if gi in drop_global or gj in drop_global:
            continue
        di = abs(pos[gi, 1] - y_mid)
        dj = abs(pos[gj, 1] - y_mid)
        if di > dj:
            drop_global.add(gi)
        elif dj > di:
            drop_global.add(gj)
        else:
            # Tie: drop the higher global index (deterministic).
            drop_global.add(max(gi, gj))

    keep = np.ones(len(a), dtype=bool)
    keep[list(drop_global)] = False
    log.info("    removed %d overlapping atoms at GB (tol=%g A)",
             len(drop_global), tol)
    return a[keep]


# =====================================================================
# I/O and provenance
# =====================================================================

def _git_sha() -> str:
    """Best-effort short git SHA."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fmt_dir(v):
    """Format a direction vector: integers if close, else 4-decimal reals."""
    v = np.asarray(v, dtype=float)
    rounded = np.round(v)
    if np.max(np.abs(v - rounded)) < 0.05:
        return "[{:+d}, {:+d}, {:+d}]".format(*rounded.astype(int))
    v[np.abs(v) < 0.005] = 0.0
    return "[{:+.4f}, {:+.4f}, {:+.4f}]".format(*v)


def _cubic_symmetry_ops():
    """Generate all 24 proper rotation matrices of the cubic point group."""
    ops = []
    for ax in np.eye(3):
        for angle in [0, 90, 180, 270]:
            if angle == 0 and not np.allclose(ax, [1, 0, 0]):
                continue
            c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            ops.append(np.eye(3) + s * K + (1 - c) * (K @ K))
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            ax = np.array([1, s1, s2], dtype=float)
            ax /= np.linalg.norm(ax)
            for angle in [120, 240]:
                c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
                K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                ops.append(np.eye(3) + s * K + (1 - c) * (K @ K))
    for ax_raw in [[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1]]:
        ax = np.array(ax_raw, dtype=float)
        ax /= np.linalg.norm(ax)
        c, s = np.cos(np.pi), np.sin(np.pi)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        ops.append(np.eye(3) + s * K + (1 - c) * (K @ K))
    return ops


_CUBIC_OPS = _cubic_symmetry_ops()


def _relabel_to_match(dirs, ref_dirs):
    """Find the cubic-equivalent description of `dirs` closest to `ref_dirs`."""
    dirs_unit = np.array([d / np.linalg.norm(d) for d in dirs])
    ref_unit = np.array([d / np.linalg.norm(d) for d in ref_dirs])
    best_score, best_S = -np.inf, None
    for S in _CUBIC_OPS:
        rotated = np.array([S @ d for d in dirs_unit])
        score = sum(np.dot(rotated[j], ref_unit[j]) for j in range(3))
        if score > best_score + 1e-12:
            best_score, best_S = score, S
    return [best_S @ d for d in dirs_unit]


def _v2_dirs(v1_dirs, axis_cubic, angle_deg):
    """Compute grain-2 crystal directions along box axes (R_cubic^T applied
    to V1 dirs, then relabeled to closest cubic equivalent)."""
    R_cubic = rotation_matrix(
        np.asarray(axis_cubic, dtype=float), angle_deg)
    v1_unit = [np.asarray(d, dtype=float) / np.linalg.norm(d) for d in v1_dirs]
    v2_raw = [R_cubic.T @ v for v in v1_unit]
    return _relabel_to_match(v2_raw, v1_unit)


def _provenance_header(p: Optional[Pair], a: Atoms, cfg: Config) -> str:
    """Build a multi-line LAMMPS comment recording how the file was made."""
    L = _cell_lengths(a)
    when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Generated by gb_generator.py",
        f"#   timestamp (UTC) : {when}",
        f"#   git SHA         : {_git_sha()}",
        f"#   lattice (a, A)  : {cfg.a}",
        f"#   NX, NY, NZ      : {cfg.nx}, {cfg.ny}, {cfg.nz}",
        f"#   oversize        : {cfg.oversize}",
        f"#   overlap_tol (A) : {cfg.overlap_tol}",
        f"#   crop_eps  (A)   : {cfg.crop_eps}",
        f"#   gb_gap    (A)   : {cfg.gb_gap}",
        f"#   gb_shift_x (A)  : {cfg.gb_shift_x}",
        f"#   gb_shift_z (A)  : {cfg.gb_shift_z}",
        f"#   V1 orient       : X={list(cfg.v1_dirs[0])} Y={list(cfg.v1_dirs[1])} Z={list(cfg.v1_dirs[2])}",
        # Machine-parseable form of the V1 box axes (cubic alpha frame),
        # used by the verification scripts to reconstruct the box.
        f"#   v1_dirs (cubic) : ({list(cfg.v1_dirs[0])}, {list(cfg.v1_dirs[1])}, {list(cfg.v1_dirs[2])})",
        f"#   box (A)         : {L[0]:.4f} x {L[1]:.4f} x {L[2]:.4f}",
        f"#   n_atoms         : {len(a)}",
    ]
    if p is not None:
        v2 = _v2_dirs(cfg.v1_dirs, p.axis_alpha, p.angle_deg)
        lines += [
            f"#   pair            : {p.name}  ({p.label()})",
            f"#   axis (cubic a)  : {p.axis_alpha}",
            f"#   angle (deg)     : {p.angle_deg}",
            f"#   grain1 (cubic)  : X={_fmt_dir(cfg.v1_dirs[0])} Y={_fmt_dir(cfg.v1_dirs[1])} Z={_fmt_dir(cfg.v1_dirs[2])}",
            f"#   grain2 (cubic)  : X={_fmt_dir(v2[0])} Y={_fmt_dir(v2[1])} Z={_fmt_dir(v2[2])}",
        ]
    return "\n".join(lines) + "\n"


def write_lammps(a: Atoms, path: Path, cfg: Config,
                 pair: Optional[Pair] = None) -> None:
    """Write LAMMPS data file with provenance header replacing ASE's title line."""
    ase_write(str(path), a, format="lammps-data",
              atom_style="atomic", specorder=["Fe"])
    body = path.read_text()
    body = body.split("\n", 1)[1] if "\n" in body else ""
    header = _provenance_header(pair, a, cfg)
    path.write_text(header + body)
    L = _cell_lengths(a)
    log.info("  -> %s: %d atoms, box = %.2f x %.2f x %.2f A "
             "(use boundary s s s)", path.name, len(a), L[0], L[1], L[2])


# =====================================================================
# Per-pair build
# =====================================================================

def build_pair(p: Pair, v1: Atoms, v1_dims: np.ndarray, cfg: Config) -> Atoms:
    """Build the V1-Vn bicrystal for a single Pair via rotate-and-crop."""
    g2 = rotated_grain(p.angle_deg, p.axis_alpha, v1_dims, cfg)
    return stack_along_y(v1, g2, cfg)


# =====================================================================
# CLI and top-level driver
# =====================================================================

def _parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="Generate BCC Fe K-S variant-pair bicrystals.")
    ap.add_argument("-a", "--lattice", type=float, default=DEFAULT_A,
                    help=f"BCC Fe lattice constant in A (default: {DEFAULT_A})")
    ap.add_argument("--nx", type=int, default=DEFAULT_NX,
                    help=f"V1 cells along X (default: {DEFAULT_NX})")
    ap.add_argument("--ny", type=int, default=DEFAULT_NY,
                    help=f"V1 cells along Y per grain (default: {DEFAULT_NY})")
    ap.add_argument("--nz", type=int, default=DEFAULT_NZ,
                    help=f"V1 cells along Z (default: {DEFAULT_NZ})")
    ap.add_argument("--oversize", type=int, nargs=3, default=list(DEFAULT_OVERSIZE),
                    metavar=("OX", "OY", "OZ"),
                    help=f"oversize factors for rotate+crop "
                         f"(default: {DEFAULT_OVERSIZE})")
    ap.add_argument("--overlap-tol", type=float, default=DEFAULT_OVERLAP_TOL,
                    help=f"GB-overlap merge distance, A "
                         f"(default: {DEFAULT_OVERLAP_TOL})")
    ap.add_argument("--crop-eps", type=float, default=DEFAULT_CROP_EPS,
                    help=f"crop face tolerance, A (default: {DEFAULT_CROP_EPS})")
    ap.add_argument("--gb-gap", type=float, default=DEFAULT_GB_GAP,
                    help=f"vacuum gap inserted at the GB, A "
                         f"(default: {DEFAULT_GB_GAP}; use 0 for touching, "
                        f">=2.5 for free surfaces; all values allowed)")
    ap.add_argument("--ratio-tol", type=float, default=DEFAULT_RATIO_TOL,
                    help=f"max allowed |density ratio - 1| from rotate+crop "
                         f"(default: {DEFAULT_RATIO_TOL})")
    ap.add_argument("--gb-shift-x", type=float, default=DEFAULT_GB_SHIFT_X,
                    help=f"rigid displacement of grain 2 along box X "
                         f"(an in-plane GB direction), A "
                         f"(default: {DEFAULT_GB_SHIFT_X})")
    ap.add_argument("--gb-shift-z", type=float, default=DEFAULT_GB_SHIFT_Z,
                    help=f"rigid displacement of grain 2 along box Z "
                         f"(the other in-plane GB direction), A "
                         f"(default: {DEFAULT_GB_SHIFT_Z})")
    ap.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"output directory (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--boundary-plane", type=float, nargs=3, default=None,
                    metavar=("H", "K", "L"),
                    help="V1 grain-boundary plane normal in the cubic alpha "
                         "frame (default: 1 1 -2); X and Z are selected "
                         "automatically. Cannot be combined with --v1-x, "
                         "--v1-y, or --v1-z")
    ap.add_argument("--v1-x", type=float, nargs=3, default=None,
                    metavar=("H", "K", "L"),
                    help=f"V1 box X axis in the cubic alpha frame; integer "
                         f"Miller indices or real values (e.g. an irrational "
                         f"close-packed axis) are both accepted "
                         f"(default: {list(DEFAULT_V1_DIRS[0])})")
    ap.add_argument("--v1-y", type=float, nargs=3, default=None,
                    metavar=("H", "K", "L"),
                    help=f"V1 box Y axis (the GB-plane normal) in the cubic "
                         f"alpha frame (default: {list(DEFAULT_V1_DIRS[1])})")
    ap.add_argument("--v1-z", type=float, nargs=3, default=None,
                    metavar=("H", "K", "L"),
                    help=f"V1 box Z axis in the cubic alpha frame "
                         f"(default: {list(DEFAULT_V1_DIRS[2])}). "
                         f"The three axes must be mutually orthogonal (within "
                         f"~0.06 deg) and form a right-handed triad.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="enable DEBUG-level logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    warning = gb_gap_warning(args.gb_gap, args.lattice)
    if warning:
        log.warning("warning: %s", warning)

    custom_axes = (args.v1_x, args.v1_y, args.v1_z)
    if args.boundary_plane is not None and any(v is not None for v in custom_axes):
        ap.error("--boundary-plane cannot be combined with --v1-x, --v1-y, or --v1-z")

    try:
        if args.boundary_plane is not None:
            v1_dirs = v1_dirs_from_boundary_plane(args.boundary_plane)
        else:
            v1_dirs = validate_v1_dirs(tuple(
                supplied if supplied is not None else default
                for supplied, default in zip(custom_axes, DEFAULT_V1_DIRS)))
    except ValueError as e:
        ap.error(str(e))

    return Config(
        a=args.lattice,
        nx=args.nx, ny=args.ny, nz=args.nz,
        oversize=tuple(args.oversize),
        overlap_tol=args.overlap_tol,
        crop_eps=args.crop_eps,
        gb_gap=args.gb_gap,
        ratio_tol=args.ratio_tol,
        gb_shift_x=args.gb_shift_x,
        gb_shift_z=args.gb_shift_z,
        out_dir=args.out_dir,
        v1_dirs=v1_dirs,
    )


def main() -> None:
    cfg = _parse_args()
    cfg.out_dir.mkdir(exist_ok=True, parents=True)

    log.info("V1 orientation: X=%s Y=%s Z=%s",
             list(cfg.v1_dirs[0]), list(cfg.v1_dirs[1]), list(cfg.v1_dirs[2]))
    v1 = build_oriented(cfg.v1_dirs, cfg)
    v1.set_pbc(False)
    v1_dims = _cell_lengths(v1)
    log.info("V1 reference crystal: %d atoms, box = %.3f x %.3f x %.3f A",
             len(v1), v1_dims[0], v1_dims[1], v1_dims[2])
    write_lammps(v1, cfg.out_dir / "v1_reference.lmp", cfg, pair=None)

    for idx, p in enumerate(PAIRS, 1):
        log.info("[%2d/%d] %s axis=%s angle=%g deg",
                 idx, len(PAIRS), p.label(), p.axis_alpha, p.angle_deg)
        bicrystal = build_pair(p, v1, v1_dims, cfg)
        write_lammps(bicrystal, cfg.out_dir / f"{p.name}.lmp", cfg, pair=p)

    log.info("\nDone. Wrote v1_reference.lmp + %d bicrystals to %s",
             len(PAIRS), cfg.out_dir)


if __name__ == "__main__":
    main()
