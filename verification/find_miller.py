#!/usr/bin/env python3
"""
Detect crystal directions for each grain in BCC Fe bicrystal LAMMPS files.

Recovers orientations purely from atomic positions (BCC 2NN shell detection),
then canonicalizes using the grain1 directions from the file header.
"""

import re
import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# =============================================================================
# FILE I/O
# =============================================================================

def read_lammps(filename):
    """Read a LAMMPS data file (atomic style).
    Returns (positions, box_dims, box_lo)."""
    with open(filename) as f:
        lines = f.readlines()

    natoms = None
    for line in lines:
        if "atoms" in line and "atom" in line.split()[-1]:
            natoms = int(line.split()[0])
            break

    box_lo, box_hi = np.zeros(3), np.zeros(3)
    for line in lines:
        for key, d in [("xlo", 0), ("ylo", 1), ("zlo", 2)]:
            if key in line:
                p = line.split()
                box_lo[d], box_hi[d] = float(p[0]), float(p[1])

    start = next(i for i, l in enumerate(lines) if l.strip().startswith("Atoms")) + 2
    pos = np.zeros((natoms, 3))
    for i in range(natoms):
        p = lines[start + i].split()
        pos[i] = float(p[2]), float(p[3]), float(p[4])

    return pos, box_hi - box_lo, box_lo


def parse_header_grain1(filename):
    """Parse grain1 (cubic) directions from the file header.
    Returns (3,3) unit vectors or None."""
    pattern = re.compile(
        r"#\s+grain1\s+\(cubic\)\s*:\s*"
        r"X=\[([^\]]+)\]\s*Y=\[([^\]]+)\]\s*Z=\[([^\]]+)\]"
    )
    with open(filename) as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = pattern.match(line)
            if m:
                dirs = np.array([
                    [float(x) for x in m.group(i).split(",")]
                    for i in (1, 2, 3)
                ])
                norms = np.linalg.norm(dirs, axis=1, keepdims=True)
                if np.all(norms > 1e-10):
                    return dirs / norms
    return None


# =============================================================================
# CUBIC SYMMETRY (24 proper rotations)
# =============================================================================

def _build_cubic_ops():
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


CUBIC_OPS = _build_cubic_ops()


# =============================================================================
# ORIENTATION DETECTION (BCC 2NN shell)
# =============================================================================

def _detect_axes_single(pos, tree, idx):
    """Detect BCC cubic axes from one atom's 2NN shell. Returns (3,3) or None."""
    ref = pos[idx]
    _, nn = tree.query(ref, k=20)
    deltas = tree.data[nn[1:]] - ref
    d = np.linalg.norm(deltas, axis=1)
    order = np.argsort(d)
    d, deltas = d[order], deltas[order]

    # Need clear gap between 1NN (8 atoms) and 2NN (6 atoms)
    if d[8] - d[7] < 0.1:
        return None
    if np.std(d[8:14]) / np.mean(d[8:14]) > 0.03:
        return None

    nn2 = deltas[8:14]

    # Pair antiparallel vectors to get 3 cubic axes
    axes, used = [], set()
    for i in range(6):
        if i in used:
            continue
        for j in range(i + 1, 6):
            if j in used:
                continue
            if np.linalg.norm(nn2[i] + nn2[j]) < 0.3:
                axes.append(nn2[i])
                used |= {i, j}
                break
    if len(axes) != 3:
        return None

    axes = np.array(axes)
    if not all(
        abs(np.dot(axes[i], axes[j])) / (np.linalg.norm(axes[i]) * np.linalg.norm(axes[j])) < 0.05
        for i in range(3) for j in range(i + 1, 3)
    ):
        return None

    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]
    return axes


def get_orientation(pos, box, indices, n=40):
    """Determine crystal orientation via majority-vote over multiple atoms."""
    imgs = [pos + np.array([dx, dy, dz]) * box
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    tree = cKDTree(np.vstack(imgs))

    all_axes = []
    for idx in indices[:n]:
        ax = _detect_axes_single(pos, tree, idx)
        if ax is not None:
            all_axes.append(ax)

    if not all_axes:
        return None
    if len(all_axes) == 1:
        return all_axes[0]

    # Cluster by disorientation < 5deg, return majority
    clusters = []
    for ax in all_axes:
        R = rot_matrix(ax)
        matched = False
        for i, (rep, cnt) in enumerate(clusters):
            if disorientation_angle(R, rot_matrix(rep)) < 5.0:
                clusters[i] = (rep, cnt + 1)
                matched = True
                break
        if not matched:
            clusters.append((ax, 1))

    clusters.sort(key=lambda x: -x[1])
    return clusters[0][0]


# =============================================================================
# ROTATION AND DISORIENTATION
# =============================================================================

def rot_matrix(axes):
    """Rotation matrix from lattice vectors (columns = unit vectors)."""
    R = np.column_stack([a / np.linalg.norm(a) for a in axes])
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R


def disorientation_angle(R1, R2):
    """Minimum misorientation angle (degrees) over cubic symmetry."""
    dR = R1.T @ R2
    min_angle = 180.0
    for S in CUBIC_OPS:
        tr = np.clip((np.trace(S @ dR) - 1) / 2, -1, 1)
        min_angle = min(min_angle, np.degrees(np.arccos(tr)))
    return min_angle


# =============================================================================
# CRYSTAL DIRECTIONS AND RELABELING
# =============================================================================

def get_raw_directions(axes):
    """Crystal direction [hkl] along each lab axis. Returns (3,3) array."""
    R = rot_matrix(axes)
    return np.linalg.inv(R).T  # row j = crystal direction along lab axis j


def best_relabel(dirs, ref_unit):
    """Find the cubic symmetry op S that best aligns dirs with ref_unit.
    Ties broken by lexicographic order of rotated components."""
    dirs_unit = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    best_score, best_key, best_S = -np.inf, None, None
    for S in CUBIC_OPS:
        rotated = (S @ dirs_unit.T).T
        score = sum(np.dot(rotated[j], ref_unit[j]) for j in range(3))
        key = tuple(rotated.ravel())
        if (score > best_score + 1e-12) or (abs(score - best_score) < 1e-12 and key > best_key):
            best_score, best_key, best_S = score, key, S
    return best_S


# =============================================================================
# FORMATTING
# =============================================================================

def fmt_dir(v, integer_atol=0.05):
    """Format as integers when rational, else 4-decimal reals."""
    v = np.array(v, dtype=float)
    v[np.abs(v) < 0.005] = 0.0
    rounded = np.round(v)
    if np.max(np.abs(v - rounded)) < integer_atol:
        return "[{:+2d} {:+2d} {:+2d}]".format(*rounded.astype(int))
    return "[{:+8.4f} {:+8.4f} {:+8.4f}]".format(*v)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process(filename):
    """Process one bicrystal file.
    Returns (dirs_grain1, dirs_grain2, disorientation_degrees) or None."""
    pos, box, box_lo = read_lammps(filename)

    # Split grains along GB normal (longest box dimension)
    gb_ax = np.argmax(box)
    frac = (pos[:, gb_ax] - box_lo[gb_ax]) / box[gb_ax]
    g1 = np.where((frac > 0.1) & (frac < 0.4))[0]
    g2 = np.where((frac > 0.6) & (frac < 0.9))[0]
    if len(g1) < 5 or len(g2) < 5:
        return None

    # Detect orientations
    np.random.seed(42)
    s1 = np.random.choice(g1, min(40, len(g1)), replace=False)
    s2 = np.random.choice(g2, min(40, len(g2)), replace=False)
    ax1 = get_orientation(pos, box, s1, n=40)
    ax2 = get_orientation(pos, box, s2, n=40)
    if ax1 is None or ax2 is None:
        return None

    # Canonicalize grain 1 against header reference
    ref = parse_header_grain1(filename)
    if ref is None:
        raise ValueError(f"No grain1 header found in {filename}")

    dirs1 = get_raw_directions(ax1)
    S1 = best_relabel(dirs1, ref)
    dirs1_can = (S1 @ dirs1.T).T

    # Canonicalize grain 2 against grain 1
    ref1_unit = dirs1_can / np.linalg.norm(dirs1_can, axis=1, keepdims=True)
    dirs2 = get_raw_directions(ax2)
    S2 = best_relabel(dirs2, ref1_unit)
    dirs2_can = (S2 @ dirs2.T).T

    # Scale grain 2 to grain 1 norms for display
    ref_norms = np.linalg.norm(dirs1_can, axis=1)
    dir2_norms = np.linalg.norm(dirs2_can, axis=1)
    dirs2_scaled = dirs2_can * (ref_norms / dir2_norms)[:, np.newaxis]

    angle = disorientation_angle(rot_matrix(ax1), rot_matrix(ax2))
    return dirs1_can, dirs2_scaled, angle


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Report grain directions and disorientation for bicrystal .lmp files.")
    ap.add_argument("out_dir", nargs="?",
                    default=str(Path(__file__).resolve().parent.parent / "out"),
                    help="directory containing .lmp files (default: ../out)")
    args = ap.parse_args()

    files = sorted(
        glob.glob(str(Path(args.out_dir) / "ks_*.lmp")),
        key=lambda f: int(Path(f).stem.split('v')[-1]),
    )
    if not files:
        print(f"ERROR: no ks_*.lmp files in {args.out_dir}")
        raise SystemExit(2)

    print(f"{'File':<18} {'Grain 1: x / y / z':<96} {'Grain 2: x / y / z':<96} {'Disori.'}")
    print("-" * 225)
    for f in files:
        name = Path(f).stem
        try:
            result = process(f)
        except ValueError as e:
            print(f"{name:<18} ERROR: {e}")
            continue
        if result:
            s1, s2, angle = result
            g1s = " / ".join(fmt_dir(v) for v in s1)
            g2s = " / ".join(fmt_dir(v) for v in s2)
            print(f"{name:<18} {g1s:<96} {g2s:<96} {angle:5.1f}\u00b0")
        else:
            print(f"{name:<18} FAILED")
