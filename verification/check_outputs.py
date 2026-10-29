#!/usr/bin/env python3
"""
Sanity-check every LAMMPS data file produced by `gb_generator.py`.

For each `out/ks_v1v*.lmp` (and `v1_reference.lmp`) this script:

    1. Loads the file with ASE (parses header, box, atoms).
    2. Verifies the box dimensions match what the provenance comment
       claims (Lx, Ly, Lz; n_atoms).
    3. Splits atoms into two grains across y = Ly / 2 and verifies
       the bicrystal has TWO grains (a thin slab around y_mid is not
       denser than bulk -- rules out a single accidentally-bonded
       crystal).
    4. Reports scientifically unsafe vacuum slits:
         any `0 < gb_gap < a*sqrt(3)/2` for BCC Fe is flagged before
         reconstruction because it drives atoms from opposite grains
         inside the repulsive wall.
     5. RECONSTRUCTS each grain from the recipe (V1 + the rotate-and-
       crop pipeline using the Pair's axis and angle from the
       provenance comment) and verifies that the reconstruction
       reproduces the file's positions atom-for-atom (sorted match,
       max position deviation <= POSITION_TOL Angstrom).
     6. INDEPENDENTLY recovers the orientation of each grain from the
       atomic positions alone (without using the provenance axis or
       angle) and verifies the V1->Vn misorientation thus measured
       reproduces the (axis, angle) recorded in the header within
       ANGLE_TOL_DEG / AXIS_TOL_DEG, modulo cubic symmetry.

Steps 5 and 6 are complementary:
    - 5 verifies "the file came from the recipe in its header" (binary,
    machine-precision).
    - 6 verifies "the orientation in the file equals the K-S table
    value" without trusting the header axis/angle at all.

A check on the *recipe itself* (i.e. that the table of axes/angles
really is the K-S misorientations) lives in `ks_misorientations.py`.

Run:
    python3 check_outputs.py [out_dir]      # default: ./out

Exit code 0 means every file is consistent.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
from ase.io import read as ase_read
from scipy.spatial import cKDTree

# Allow running this script from inside the verification/ subfolder by
# putting the project root (which holds gb_generator.py) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gb_generator import (  # noqa: E402
    Config, _is_integer_miller, bcc_nn_distance,
    build_oriented, gb_gap_warning, rotated_grain, stack_along_y,
    validate_v1_dirs,
)
from ks_misorientations import CUBIC_24, disorientation  # noqa: E402

# Tolerances ----------------------------------------------------------
POSITION_TOL = 1e-3      # A; max per-atom position deviation reproduction
MISORIENT_TOL_DEG = 0.5  # disorientation between measured and header
                         # rotations (full 3D check, not separate
                         # angle/axis checks).


# V1 box-frame basis: columns are the unit vectors of the V1 box axes
# (e.g. [111], [11-2], [-110] for the default V1), expressed in the
# cubic-alpha frame. So
#     v_cubic = V1_BASIS  @ v_box        (box  -> cubic)
#     v_box   = V1_BASIS.T @ v_cubic     (cubic -> box)
# and analogously for rotation matrices:
#     R_cubic = V1_BASIS  @ R_box   @ V1_BASIS.T
#     R_box   = V1_BASIS.T @ R_cubic @ V1_BASIS
# The basis is now per-file (V1 orientation is configurable on the
# generator CLI); use `v1_basis(v1_dirs)` to build it from a triple.
def v1_basis(v1_dirs) -> np.ndarray:
    arr = np.asarray(v1_dirs, dtype=float)
    return (arr / np.linalg.norm(arr, axis=1, keepdims=True)).T


# Header parsing ------------------------------------------------------
_HDR_RE = re.compile(r"^#\s+([\w()\s.,]+?)\s*:\s*(.+)$")


def parse_provenance(path: Path) -> dict:
    """Parse the leading `# key : value` comment block of a .lmp file."""
    info: dict = {}
    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = _HDR_RE.match(line.rstrip())
            if m:
                info[m.group(1).strip()] = m.group(2).strip()
    return info


def cfg_from_provenance(info: dict) -> Config:
    """Reconstruct a Config from the provenance header so we can rebuild."""
    a = float(info["lattice (a, A)"])
    nx, ny, nz = (int(s) for s in info["NX, NY, NZ"].split(","))
    oversize = tuple(int(s) for s in
                     info["oversize"].strip("()").split(","))
    overlap_tol = float(info["overlap_tol (A)"])
    crop_eps = float(info["crop_eps  (A)"])
    gb_gap = float(info["gb_gap    (A)"])
    gb_shift_x = float(info["gb_shift_x (A)"])
    gb_shift_z = float(info["gb_shift_z (A)"])
    v1_dirs = validate_v1_dirs(ast.literal_eval(info["v1_dirs (cubic)"]))
    return Config(
        a=a, nx=nx, ny=ny, nz=nz,
        oversize=oversize,
        overlap_tol=overlap_tol,
        crop_eps=crop_eps,
        gb_gap=gb_gap,
        ratio_tol=1.0,                  # don't re-trip the ratio gate here
        gb_shift_x=gb_shift_x,
        gb_shift_z=gb_shift_z,
        v1_dirs=v1_dirs,
    )


# Sorted positional comparison ---------------------------------------

def positions_match(pos_file: np.ndarray, pos_ref: np.ndarray,
                    tol: float) -> tuple[bool, float, int]:
    """Return (match, max_dev, n_unmatched).

    Assigns each `pos_ref` atom to its nearest neighbour in `pos_file`
    via KDTree; reports the max nearest-neighbour distance and the
    number of reference atoms with no file atom within `tol`.
    """
    if len(pos_file) != len(pos_ref):
        return False, np.inf, abs(len(pos_file) - len(pos_ref))
    tree = cKDTree(pos_file)
    d, _ = tree.query(pos_ref, k=1)
    return bool(d.max() < tol), float(d.max()), int((d > tol).sum())


# Orientation from atomic positions ---------------------------------

def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Best-fit proper rotation R minimising sum_i ||R @ P_i - Q_i||^2.

    P, Q are (N, 3). Returns the 3x3 rotation matrix with det = +1.
    """
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    D = np.diag([1.0, 1.0, d])
    return (Vt.T @ D @ U.T)


def rodrigues(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix for a rotation by `angle_deg` around `axis`.
    `axis` need not be unit-normalised."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0.0:
        return np.eye(3)
    axis = axis / n
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) * c + s * K + (1.0 - c) * np.outer(axis, axis)


def axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (unit axis, angle in degrees) for rotation matrix R."""
    cos_th = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(cos_th))
    if th < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if np.pi - th < 1e-9:
        M = (R + np.eye(3)) / 2.0
        col = int(np.argmax(np.linalg.norm(M, axis=0)))
        ax = M[:, col] / np.linalg.norm(M[:, col])
        return ax, np.degrees(th)
    ax = np.array([R[2, 1] - R[1, 2],
                   R[0, 2] - R[2, 0],
                   R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return ax, np.degrees(th)


def fit_box_frame_rotation(pos: np.ndarray, a: float,
                           v1_basis_mat: np.ndarray) -> np.ndarray:
    """Recover a grain's rotation R_box directly from a single atom and
    its 8 nearest neighbours.

    The trick: pick the most-interior atom of the grain. In a BCC
    crystal, this atom's 8 first-NN form a perfect cube (corners at
    +/-a/2 in each cubic direction relative to the centre). Whatever
    rotation R was applied to V1 to make this grain, the cube is just
    R times its unrotated counterpart.

    So we have 8 known reference points (V1's unrotated cube corners
    in V1's box frame) and 8 observed points (the actual NN positions
    relative to the central atom), and need to recover R = R_box.
    See the inline comments below for the matching algorithm (a
    3-anchor brute force; the obvious 2-anchor variant is degenerate
    on a line of cube symmetries and produces the cubic-inverse R).

    The 8-corner cube has cubic symmetry, so R is defined modulo the
    24-element point group -- exactly the right ambiguity for
    misorientation analysis. The 24 x 24 cubic sweep in `check_file`
    removes this ambiguity downstream.
    """
    # Pick an interior atom that has a complete BCC NN shell.
    # Restrict candidates to atoms strictly inside the grain (at least
    # one NN-distance away from every face of the grain's AABB), then
    # walk outwards from the centroid until we find one whose 8 NN are
    # all at (and only at) the BCC NN distance.
    nn_dist = a * np.sqrt(3) / 2.0
    pad = 1.05 * nn_dist
    lo = pos.min(axis=0) + pad
    hi = pos.max(axis=0) - pad
    interior = np.where(((pos >= lo) & (pos <= hi)).all(axis=1))[0]
    if len(interior) == 0:
        raise RuntimeError("grain too thin: no interior atoms for NN fit")
    centre = pos[interior].mean(axis=0)
    order = interior[np.argsort(np.linalg.norm(pos[interior] - centre, axis=1))]
    tree = cKDTree(pos)
    i_central = -1
    j_nn = None
    # The 9th neighbour (1st 2NN of BCC) sits at distance `a`; require
    # the 9th to be clearly farther than the 8th to be sure we have a
    # full first shell and nothing else.
    for cand in order:
        d, j = tree.query(pos[cand], k=10)      # self + 8 NN + 1 guard
        if d[1:9].max() < 1.05 * nn_dist and d[9] > 1.10 * nn_dist:
            i_central = int(cand)
            j_nn = j
            break
    if i_central < 0:
        raise RuntimeError(
            "no interior atom with complete BCC NN shell found")
    j = j_nn

    # Observed NN cube corners, in V1's box frame, centred on
    # the central atom.
    obs_corners = pos[j[1:9]] - pos[i_central]  # (8, 3)

    # Reference NN cube corners for an UNROTATED V1 atom: the cubic
    # vertices a/2 * (+/-1, +/-1, +/-1), expressed in V1's box frame.
    corners_cubic = (a / 2.0) * np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        float)
    ref_corners = corners_cubic @ v1_basis_mat     # (8, 3) in V1 box frame

    # The 8 obs corners and the 8 ref corners are two BCC NN cubes,
    # equal as sets after some unknown proper rotation R = R_box,
    # with an unknown per-corner labelling (kdtree NN order vs
    # canonical sign-product order).
    #
    # We anchor on TRIPLES: for every choice of distinct ref corners
    # (j0, j1, j2) paired with obs[0], obs[1], obs[2], a 3-point
    # Kabsch fit yields a unique proper rotation (3 non-degenerate
    # points fix R; 2 points leave a flip ambiguity about the line
    # through them, which is the source of the cubic-inverse failure
    # mode). We then verify R on all 8 corners by nearest-neighbour
    # distance to R @ ref. Best worst-corner residual wins.
    #
    # 8 * 7 * 6 = 336 trials, each O(1); cheap.
    obs_tree = cKDTree(obs_corners)
    best_R, best_res = np.eye(3), np.inf
    found_exact = False
    for j0 in range(8):
        if found_exact:
            break
        for j1 in range(8):
            if j1 == j0:
                continue
            if found_exact:
                break
            for j2 in range(8):
                if j2 == j0 or j2 == j1:
                    continue
                R = kabsch(ref_corners[[j0, j1, j2]],
                           obs_corners[[0, 1, 2]])
                pred = ref_corners @ R.T
                d, _ = obs_tree.query(pred, k=1)
                res = float(d.max())
                if res < best_res:
                    best_res, best_R = res, R
                    if res < 1e-6 * a:              # exact: stop early
                        found_exact = True
                        break

    if best_res > 1e-3 * a:
        raise RuntimeError(
            f"orientation fit residual {best_res:.3g} A "
            f"(grain may not be pure BCC?)")
    return best_R


# -------------------- per-file checks --------------------

def check_file(path: Path) -> tuple[bool, list[str]]:
    """Run all checks on one .lmp file. Returns (ok, messages)."""
    info = parse_provenance(path)
    msgs: list[str] = []
    ok = True

    if "lattice (a, A)" not in info:
        return False, ["missing lattice header"]

    cfg = cfg_from_provenance(info)
    a = cfg.a

    # Load with ASE.
    atoms = ase_read(str(path), format="lammps-data", atom_style="atomic")
    pos_file = atoms.get_positions()
    cell = atoms.cell.array
    Lx, Ly, Lz = np.diag(cell)

    # 1) Box dimensions and atom count match the provenance comment.
    if "box (A)" in info:
        bx, by, bz = (float(s) for s in info["box (A)"].replace("x", " ").split())
        for name, want, got in [("Lx", bx, Lx), ("Ly", by, Ly), ("Lz", bz, Lz)]:
            if abs(want - got) > 1e-3:
                ok = False
                msgs.append(f"{name} mismatch: header={want:.4f} file={got:.4f}")

    if "n_atoms" in info:
        want_n = int(info["n_atoms"])
        if want_n != len(atoms):
            ok = False
            msgs.append(f"n_atoms mismatch: header={want_n} file={len(atoms)}")

    is_reference = path.name == "v1_reference.lmp"

    # 2) Two grains visible: a thin slab at y_mid should not be denser
    #    than bulk. Skipped for the V1 reference (single grain).
    #    For non-integer V1 axes the lattice planes don't align cleanly
    #    with the GB, so the thin-slab density is dominated by which
    #    atomic plane happens to land there - the check is reported as
    #    a warning (not a failure) in that case.
    if not is_reference:
        y_mid = Ly / 2.0
        half = 0.5 * a
        n_seam = int((np.abs(pos_file[:, 1] - y_mid) < half).sum())
        rho_bulk = len(atoms) / (Lx * Ly * Lz)
        rho_seam = n_seam / (Lx * Lz * 2 * half)
        if rho_seam > 1.05 * rho_bulk:
            seam_msg = (f"no GB seam visible: rho_seam={rho_seam:.5f} > "
                        f"rho_bulk={rho_bulk:.5f}")
            if _is_integer_miller(cfg.v1_dirs):
                ok = False
                msgs.append(seam_msg)
            else:
                msgs.append("warning: " + seam_msg +
                            " (non-integer V1: thin-slab heuristic "
                            "unreliable, ignored)")

    # 3) Report gap choices that are allowed by the builder but are
    #    physically aggressive for an unrelaxed bicrystal.
    if not is_reference:
        warning = gb_gap_warning(cfg.gb_gap, a)
        if warning:
            if 0.0 < cfg.gb_gap < bcc_nn_distance(a):
                lower = pos_file[pos_file[:, 1] < Ly / 2.0]
                upper = pos_file[pos_file[:, 1] >= Ly / 2.0]
                d_min = float(cKDTree(upper).query(lower, k=1)[0].min())
                msgs.append(
                    f"warning: {warning} Nearest cross-grain separation in "
                    f"this file is {d_min:.3f} A."
                )
            else:
                msgs.append(f"warning: {warning}")

    # 4) Reconstruct the file from its recipe and compare positions.
    V1_BASIS_FILE = v1_basis(cfg.v1_dirs)
    v1 = build_oriented(cfg.v1_dirs, cfg)
    v1.set_pbc(False)
    v1_dims = np.diag(v1.cell.array).copy()

    if is_reference:
        ref_atoms = v1
    else:
        # Recover the misorientation from the comment. Use
        # `ast.literal_eval` (NOT `eval`) so a hostile header cannot
        # execute arbitrary Python.
        try:
            axis = np.array(ast.literal_eval(info["axis (cubic a)"]), float)
            angle = float(info["angle (deg)"])
        except (KeyError, ValueError, SyntaxError) as e:
            return False, [f"bad axis/angle in header: {e}"]
        g2 = rotated_grain(angle, tuple(axis), v1_dims, cfg)
        ref_atoms = stack_along_y(v1, g2, cfg)

    pos_ref = ref_atoms.get_positions()
    matched, max_dev, n_bad = positions_match(pos_file, pos_ref, POSITION_TOL)
    if not matched:
        ok = False
        if len(pos_file) != len(pos_ref):
            msgs.append(f"reconstructed atom count differs: "
                        f"file={len(pos_file)} reconstructed={len(pos_ref)}")
        else:
            msgs.append(f"reconstruction mismatch: max position deviation "
                        f"{max_dev:.4g} A > {POSITION_TOL} A "
                        f"({n_bad} atoms unmatched)")
    else:
        msgs.append(f"reconstruction OK: max dev = {max_dev:.2e} A")

    # 5) INDEPENDENT orientation check from atomic positions only.
    #    Skipped for the V1 reference (no second grain).
    if is_reference:
        return ok, msgs

    try:
        R1_box = fit_box_frame_rotation(pos_file[pos_file[:, 1] < Ly / 2], a, V1_BASIS_FILE)
        R2_box = fit_box_frame_rotation(pos_file[pos_file[:, 1] >= Ly / 2], a, V1_BASIS_FILE)
    except RuntimeError as e:
        ok = False
        msgs.append(f"orientation fit failed: {e}")
        return ok, msgs

    # Misorientation in V1's box frame, then rotated into V1's cubic
    # alpha frame: R_cubic = M @ R_box @ M^T, with M = V1_BASIS
    # (whose columns are the box-basis vectors expressed in the cubic
    # frame).
    R_mis_box = R2_box @ R1_box.T
    R_meas = V1_BASIS_FILE @ R_mis_box @ V1_BASIS_FILE.T

    ax_hdr = np.array(ast.literal_eval(info["axis (cubic a)"]), float)
    ang_hdr = float(info["angle (deg)"])
    R_hdr = rodrigues(ax_hdr, ang_hdr)

    # Two cubic-crystal misorientations R_meas and R_hdr are physically
    # equivalent iff there exist S1, S2 in O such that
    #     S1 @ R_meas @ S2 == R_hdr ,
    # because cubic symmetry on EACH grain leaves the bicrystal
    # invariant. We measure how close they are by minimising the
    # rotation angle of (S1 R_meas S2) @ R_hdr.T over the 24x24 = 576
    # cubic pairs (the residual is 0 iff they are equivalent).
    #
    # Note: comparing only via R_meas @ R_hdr.T (no symmetry sweep)
    # would be wrong -- it tests for equality modulo the *diagonal*
    # action S R S.T, not the two-sided action that crystal symmetry
    # actually provides.
    residual_deg = 360.0
    for S1 in CUBIC_24:
        for S2 in CUBIC_24:
            _, ang = axis_angle(S1 @ R_meas @ S2 @ R_hdr.T)
            if ang < residual_deg:
                residual_deg = ang

    # Also report what the disorientation reduction picks for the
    # measured rotation, so the log is human-readable.
    ax_meas, ang_meas = disorientation(R_meas)

    if residual_deg > MISORIENT_TOL_DEG:
        ok = False
        msgs.append(
            f"orientation-from-atoms: rotation residual {residual_deg:.3f} deg "
            f"> {MISORIENT_TOL_DEG} deg "
            f"(measured {ang_meas:.2f} deg about {ax_meas.round(4).tolist()}, "
            f"header {ang_hdr:.2f} deg about {ax_hdr.round(4).tolist()})"
        )
    else:
        msgs.append(
            f"orientation-from-atoms OK: residual {residual_deg:.2e} deg "
            f"(measured {ang_meas:.2f} deg about {ax_meas.round(3).tolist()})"
        )

    return ok, msgs


# -------------------- driver --------------------

def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "out"
    if not out_dir.is_dir():
        print(f"ERROR: {out_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    files = sorted(out_dir.glob("*.lmp"))
    if not files:
        print(f"ERROR: no .lmp files in {out_dir}", file=sys.stderr)
        sys.exit(2)

    n_ok = 0
    for path in files:
        ok, msgs = check_file(path)
        tag = "OK  " if ok else "FAIL"
        print(f"  {tag} {path.name}")
        for m in msgs:
            print(f"         - {m}")
        if ok:
            n_ok += 1

    print()
    if n_ok == len(files):
        print(f"All {len(files)} files passed.")
        sys.exit(0)
    else:
        print(f"{len(files) - n_ok} / {len(files)} files failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
