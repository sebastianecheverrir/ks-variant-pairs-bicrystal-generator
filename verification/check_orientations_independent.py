#!/usr/bin/env python3
"""Independent orientation check for generated K-S bicrystal `.lmp` files.

This script is intentionally independent of `check_outputs.py`'s
orientation-fit path. It recovers each grain orientation directly from
the atomic coordinates by using a local BCC nearest-neighbour shell:

1. Find an interior atom with a complete 8-atom first-NN shell.
2. Use pairwise differences between those 8 NN positions to recover the
   three orthonormal cube-edge directions of the local BCC lattice.
3. Build one basis for the lower grain and one for the upper grain.
4. Convert the measured grain-to-grain rotation from the V1 box frame to
   the V1 cubic-alpha frame.
5. Compare that measured misorientation to the axis/angle recorded in the
   LAMMPS provenance header, modulo cubic symmetry on both grains.

This does NOT prove that the tabulated Morito K-S misorientations are
correct by itself; `ks_misorientations.py` already covers that. Its role
is narrower: verify that the atoms written to each `.lmp` file really do
carry the recorded grain misorientation.

Run:
    python3 check_orientations_independent.py [out_dir]

Exit code 0 means every bicrystal file passed this independent check.
"""
from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from itertools import permutations, product
from pathlib import Path

import numpy as np
from ase.io import read as ase_read
from scipy.spatial import cKDTree

log = logging.getLogger("check_orientations_independent")

DEFAULT_A = 2.866
MISORIENT_TOL_DEG = 0.5
EDGE_TOL_FRAC = 0.05


def v1_basis_from_dirs(v1_dirs) -> np.ndarray:
    """Box-to-cubic basis matrix for a given V1 box-axis triple."""
    arr = np.asarray(v1_dirs, dtype=float)
    return (arr / np.linalg.norm(arr, axis=1, keepdims=True)).T

HDR_RE = re.compile(r"^#\s+([\w()\s.,]+?)\s*:\s*(.+)$")


def parse_provenance(path: Path) -> dict[str, str]:
    """Parse the leading `# key : value` comment block of a .lmp file."""
    info: dict[str, str] = {}
    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = HDR_RE.match(line.rstrip())
            if m:
                info[m.group(1).strip()] = m.group(2).strip()
    return info


def cubic_rotation_group() -> list[np.ndarray]:
    """Return the 24 proper rotations of the cubic point group."""
    mats: list[np.ndarray] = []
    seen: set[bytes] = set()
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, j in enumerate(perm):
                M[i, j] = signs[i]
            if np.linalg.det(M) <= 0.0:
                continue
            key = M.astype(np.int8).tobytes()
            if key in seen:
                continue
            seen.add(key)
            mats.append(M)
    assert len(mats) == 24, f"expected 24 cubic rotations, got {len(mats)}"
    return mats


CUBIC_24 = cubic_rotation_group()


def axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (unit axis, angle in degrees) for rotation matrix R."""
    cos_th = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(cos_th))
    if th < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if np.pi - th < 1e-9:
        M = (R + np.eye(3)) / 2.0
        col = int(np.argmax(np.linalg.norm(M, axis=0)))
        axis = M[:, col]
        axis /= np.linalg.norm(axis)
        return axis, np.degrees(th)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(th))
    return axis, np.degrees(th)


def disorientation(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Reduce R into the cubic disorientation fundamental zone."""
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


def rodrigues(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix for a rotation by `angle_deg` about `axis`."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) * c + s * K + (1.0 - c) * np.outer(axis, axis)


def equivalence_residual(R_meas: np.ndarray, R_hdr: np.ndarray) -> float:
    """Smallest rotation angle between two misorientations modulo cubic symmetry."""
    residual_deg = 360.0
    for S1 in CUBIC_24:
        for S2 in CUBIC_24:
            _, ang = axis_angle(S1 @ R_meas @ S2 @ R_hdr.T)
            if ang < residual_deg:
                residual_deg = ang
    return residual_deg


def bcc_nn_distance(a: float) -> float:
    """First-neighbour distance in a BCC crystal."""
    return a * np.sqrt(3.0) / 2.0


def recover_edge_basis(pos: np.ndarray, a: float) -> np.ndarray:
    """Recover a grain's orthonormal cubic basis from a local BCC NN shell."""
    nn_dist = bcc_nn_distance(a)
    pad = 1.05 * nn_dist
    lo = pos.min(axis=0) + pad
    hi = pos.max(axis=0) - pad
    interior = np.where(((pos >= lo) & (pos <= hi)).all(axis=1))[0]
    if len(interior) == 0:
        raise RuntimeError("grain too thin: no interior atoms")

    centre = pos[interior].mean(axis=0)
    order = interior[np.argsort(np.linalg.norm(pos[interior] - centre, axis=1))]
    tree = cKDTree(pos)
    corners = None
    for cand in order:
        d, j = tree.query(pos[cand], k=10)  # self + 8 NN + 1 guard
        if d[1:9].max() < 1.05 * nn_dist and d[9] > 1.10 * nn_dist:
            corners = pos[j[1:9]] - pos[cand]
            break
    if corners is None:
        raise RuntimeError("no interior atom with complete BCC NN shell found")

    edge_dirs: list[np.ndarray] = []
    for i in range(8):
        for j in range(i + 1, 8):
            dv = corners[i] - corners[j]
            ln = np.linalg.norm(dv)
            if abs(ln - a) > EDGE_TOL_FRAC * a:
                continue
            u = dv / ln
            if any(abs(float(np.dot(u, v))) > 0.98 for v in edge_dirs):
                continue
            edge_dirs.append(u)

    if len(edge_dirs) != 3:
        raise RuntimeError(
            f"expected 3 unique cube-edge directions, got {len(edge_dirs)}"
        )

    edge_dirs.sort(key=lambda v: tuple(np.round(v, 8)))
    basis = np.column_stack(edge_dirs)
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0

    G = basis.T @ basis
    if not np.allclose(G, np.eye(3), atol=5e-2):
        raise RuntimeError(f"recovered basis is not orthonormal enough:\n{G}")
    return basis


def check_file(path: Path) -> tuple[bool, list[str]]:
    """Run the independent orientation check on one bicrystal `.lmp` file."""
    info = parse_provenance(path)
    msgs: list[str] = []

    if path.name == "v1_reference.lmp":
        return True, ["skipped: single-grain reference file"]

    try:
        a = float(info["lattice (a, A)"])
        axis_hdr = np.array(ast.literal_eval(info["axis (cubic a)"]), float)
        ang_hdr = float(info["angle (deg)"])
    except (KeyError, ValueError, SyntaxError) as exc:
        return False, [f"bad provenance header: {exc}"]

    try:
        v1_dirs = np.asarray(ast.literal_eval(info["v1_dirs (cubic)"]), float)
    except (KeyError, ValueError, SyntaxError) as exc:
        return False, [f"bad v1_dirs (cubic) header: {exc}"]
    V1_BASIS_FILE = v1_basis_from_dirs(v1_dirs)

    atoms = ase_read(str(path), format="lammps-data", atom_style="atomic")
    pos = atoms.positions
    Ly = float(atoms.cell.array[1, 1])
    lower = pos[pos[:, 1] < Ly / 2.0]
    upper = pos[pos[:, 1] >= Ly / 2.0]

    try:
        B1 = recover_edge_basis(lower, a)
        B2 = recover_edge_basis(upper, a)
    except RuntimeError as exc:
        return False, [f"orientation recovery failed: {exc}"]

    R_box = B2 @ B1.T
    R_meas = V1_BASIS_FILE @ R_box @ V1_BASIS_FILE.T
    R_hdr = rodrigues(axis_hdr, ang_hdr)
    residual_deg = equivalence_residual(R_meas, R_hdr)
    ax_meas, ang_meas = disorientation(R_meas)

    msgs.append(
        f"independent edge-based orientation residual = {residual_deg:.2e} deg "
        f"(measured {ang_meas:.2f} deg about {ax_meas.round(4).tolist()})"
    )
    return residual_deg <= MISORIENT_TOL_DEG, msgs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Independently recover bicrystal orientations from .lmp atom coordinates."
    )
    ap.add_argument(
        "out_dir",
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "out",
        type=Path,
        help="directory containing generated .lmp files (default: ./out)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not args.out_dir.is_dir():
        print(f"ERROR: {args.out_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    files = sorted(args.out_dir.glob("*.lmp"))
    if not files:
        print(f"ERROR: no .lmp files in {args.out_dir}", file=sys.stderr)
        sys.exit(2)

    n_ok = 0
    checked = 0
    for path in files:
        ok, msgs = check_file(path)
        tag = "OK  " if ok else "FAIL"
        print(f"  {tag} {path.name}")
        for msg in msgs:
            print(f"         - {msg}")
        if path.name != "v1_reference.lmp":
            checked += 1
        if ok:
            n_ok += 1

    print()
    if n_ok == len(files):
        print(f"All {checked} bicrystal files passed the independent orientation check.")
        sys.exit(0)

    print(f"{len(files) - n_ok} / {len(files)} files failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()