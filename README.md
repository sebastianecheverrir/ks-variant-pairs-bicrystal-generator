# BCC-Fe K–S variant-pair bicrystal generator

Builds two-grain (bicrystal) BCC iron simulation boxes for every
Kurdjumov–Sachs (K–S) martensite variant pair $V_1$–$V_n$
($n = 2, \dots, 24$), as a single LAMMPS `.lmp` data file per pair.

Output: 1 single-grain reference (`v1_reference.lmp`) + 23 bicrystals
(`ks_v1v2.lmp` … `ks_v1v24.lmp`). Each file is self-documenting via a
provenance comment header (lattice constant, grid sizes, oversize,
tolerances, gap, V1 orientation, pair name, angle, axis, atom count,
git SHA, timestamp).

**References**
- Kurdjumow, G., & Sachs, G. (1930). Der Mechanismus der Stahlhärtung.
  *Die Naturwissenschaften*, **18**(22), 534–534.
  <https://doi.org/10.1007/BF01513427>
- Morito, S., Tanaka, H., Konishi, R., Furuhara, T., & Maki, T. (2003).
  The morphology and crystallography of lath martensite in Fe–C alloys.
  *Acta Materialia*, **51**(6), 1789–1799.
  <https://doi.org/10.1016/S1359-6454(02)00577-3>
  Definitive listing of the 24 variants and their pairwise
  misorientations (Tables 1 & 2).

---

## Quick start

```bash
# 1. install dependencies (numpy, scipy, ase)
python3 -m venv .venv && source .venv/bin/activate
pip install ase scipy numpy

# 2. generate v1_reference.lmp + 23 bicrystals into ./out
python3 gb_generator.py
```

`gb_generator.py` exposes every knob via `argparse` — see
`python3 gb_generator.py --help`. Most-used flags:

| Flag | Default | Meaning |
|------|--------:|---------|
| `--lattice` | 2.866 Å | BCC Fe lattice constant |
| `--nx --ny --nz` | 15 5 10 | V1 cells per grain along box X / Y / Z |
| `--oversize OX OY OZ` | 5 5 5 | rotate-and-crop oversize factors |
| `--overlap-tol` | 0.5 Å | merge atoms closer than this at the GB |
| `--gb-gap` | 0.0 Å | vacuum slit at the GB (see §4 below) |
| `--gb-shift-x` | 0.0 Å | rigid in-plane displacement of grain 2 along box X |
| `--gb-shift-z` | 0.0 Å | rigid in-plane displacement of grain 2 along box Z |
| `--ratio-tol` | 0.02 | max $\lvert n_\text{kept}/n_\text{expected} - 1 \rvert$ |
| `--boundary-plane H K L` | `1 1 -2` | V1 GB-plane normal; compatible X and Z axes are generated automatically |
| `--v1-x H K L` | `1 1 1` | V1 box X axis (Miller in cubic-$\alpha$; integer or real) |
| `--v1-y H K L` | `1 1 -2` | V1 box Y axis = GB-plane normal |
| `--v1-z H K L` | `-1 1 0` | V1 box Z axis |
| `--out-dir` | `./out` | output directory |
| `-v` | off | DEBUG-level logging |

For very small cells, the effective density-ratio tolerance has a
$1/\sqrt{N}$ floor because the inclusion or exclusion of a few atoms on the
rotated crop faces otherwise dominates the ratio.

---

## 1. The K–S orientation relationship

Between parent austenite ($\gamma$, FCC) and product martensite
($\alpha$, BCC):

$$
\{1\,1\,1\}_\gamma \;\|\; \{0\,1\,1\}_\alpha,
\qquad
\langle 1\,0\,\bar 1\rangle_\gamma \;\|\; \langle 1\,1\,\bar 1\rangle_\alpha.
$$

Counting: 4 close-packed {111}<sub>γ</sub> planes × 3 in-plane ⟨101̄⟩<sub>γ</sub> directions × 2 equivalent ⟨111̄⟩<sub>α</sub> directions per plane = **24 crystallographically distinct variants**, partitioned into 4 packets (one per {111}<sub>γ</sub> plane, 6 variants each).


## 2. V1 reference orientation

Adopting Morito's V1, the simulation box for V1 uses by default the
integer Miller axes

$$
X_\text{box} = [1\,1\,1]_\alpha,\qquad
Y_\text{box} = [1\,1\,\bar 2]_\alpha,\qquad
Z_\text{box} = [\bar 1\,1\,0]_\alpha.
$$

**The interface normal is always along box Y; its crystallographic
direction in V1 is configurable.** The direction $[1\,1\,\bar 2]_\alpha$
and corresponding interface plane $(1\,1\,\bar 2)_\alpha$ are the
defaults only. With `--boundary-plane H K L`, box Y is aligned with
$[H\,K\,L]_\alpha$ in V1, so the interface is V1's $(H\,K\,L)_\alpha$
plane and remains parallel to the box XZ plane.

For example, to select a $(1\,1\,1)_\alpha$ interface in V1:

```bash
python gb_generator.py --boundary-plane 1 1 1 --out-dir out_111
```

Compatible X and Z directions are generated automatically while
preserving primitive integer Miller directions when possible. Changing
the interface plane preserves each V1–Vn pair's K–S misorientation;
the same physical interface generally has different Miller indices in
Vn, determined by that pair's rotation. The `Y` entries in the output
header's `grain1 (cubic)` and `grain2 (cubic)` lines report the interface
normal in each grain's crystal coordinates. This boundary normal and
the pair's misorientation rotation axis are separate quantities.

Alternatively, all three axes can be controlled on the CLI via
`--v1-x`, `--v1-y`,
`--v1-z`. Each takes three numbers (integer Miller indices *or*
real-valued direction cosines) in V1's cubic-$\alpha$ frame. The
triple must be mutually orthogonal (within ~0.06°) and right-handed;
the generator validates this and exits with an error otherwise. For
integer Miller inputs the V1 supercell is tiled by ASE
(`BodyCenteredCubic`) and is exactly periodic. For real-valued
inputs — useful for aligning V1 with an irrational direction such as a
K–S close-packed axis (e.g. $[1\,1\,1]_\alpha/\sqrt3$ entered as
`0.5774 0.5774 0.5774`) — the builder uses a cubic-aligned BCC source
and rotate-and-crop; the result is non-periodic but carries the
correct lattice in the requested orientation. Slightly non-orthogonal
user input (typical residual $\sim10^{-4}$ from 4-decimal rounding)
is cleaned up by QR orthonormalization before the rotation is
applied. The Morito misorientation table in §3 is expressed in the
cubic-$\alpha$ frame and remains valid for any user choice of box
axes; only its projection onto the box (which sets the rotation axis
in box coordinates) changes.

Oriented-cell periodicities for the default V1 axes are:
$p_X = a\sqrt{3} \approx 4.964$ Å,
$p_Y = a\sqrt{6} \approx 7.020$ Å,
$p_Z = a\sqrt{2} \approx 4.053$ Å.

The triad is right-handed (positive determinant).

## 3. V1–Vn misorientations (Morito 2003 Table 2)

Each row gives the misorientation that takes V1 → Vn, as a unit
3-vector axis in the V1 cubic-$\alpha$ frame and a rotation angle.
Reproduced verbatim from Morito 2003 Table 2; this is exactly the
table consumed by `gb_generator.PAIRS`.

> **Convention.** The tabulated rotation axes are unit Cartesian vectors
> in V1's cubic-$\alpha$ crystal frame. The generator projects them into
> the selected box frame before rotating grain 2. They specify the
> misorientation rotation axis; the interface normal is the separately
> selected box Y direction described in §2. Most K–S misorientation axes are *irrational* with
> respect to the cubic lattice and cannot be represented as low-index
> Miller triplets 

| Pair | Angle (deg) | Axis (V1 $\alpha$, unit) | Σ |
|------|-----------:|---------------------------|---|
| V1–V2  | 60.00 | $[\,0.5774\;-0.5774\;\hspace{0.45em}0.5774]$ | $\Sigma 3$ |
| V1–V3  | 60.00 | $[\,0.0000\;-0.7071\;-0.7071]$           | $\Sigma 3$ |
| V1–V4  | 10.53 | $[\,0.0000\;\hspace{0.45em}0.7071\;\hspace{0.45em}0.7071]$ | — |
| V1–V5  | 60.00 | $[\,0.0000\;\hspace{0.45em}0.7071\;\hspace{0.45em}0.7071]$ | $\Sigma 3$ |
| V1–V6  | 49.47 | $[\,0.0000\;-0.7071\;-0.7071]$           | — |
| V1–V7  | 49.47 | $[-0.5774\;\hspace{0.45em}0.5774\;\hspace{0.45em}0.5774]$ | — |
| V1–V8  | 10.53 | $[\,0.5774\;-0.5774\;\hspace{0.45em}0.5774]$ | — |
| V1–V9  | 50.51 | $[-0.1862\;\hspace{0.45em}0.7666\;\hspace{0.45em}0.6145]$ | — |
| V1–V10 | 50.51 | $[-0.4904\;-0.4625\;\hspace{0.45em}0.7387]$  | — |
| V1–V11 | 14.88 | $[\,0.3543\;-0.9329\;-0.0650]$           | — |
| V1–V12 | 57.21 | $[\,0.3568\;-0.7136\;\hspace{0.45em}0.6029]$ | — |
| V1–V13 | 14.88 | $[\,0.9329\;\hspace{0.45em}0.3543\;\hspace{0.45em}0.0650]$ | — |
| V1–V14 | 50.51 | $[-0.7387\;\hspace{0.45em}0.4625\;-0.4904]$  | — |
| V1–V15 | 57.21 | $[-0.2461\;-0.6278\;-0.7384]$            | — |
| V1–V16 | 20.61 | $[\,0.6589\;\hspace{0.45em}0.6589\;\hspace{0.45em}0.3628]$ | — |
| V1–V17 | 51.73 | $[-0.6589\;\hspace{0.45em}0.3628\;-0.6589]$  | — |
| V1–V18 | 47.11 | $[-0.3022\;-0.6255\;-0.7193]$            | — |
| V1–V19 | 50.51 | $[-0.6145\;\hspace{0.45em}0.1862\;-0.7666]$  | — |
| V1–V20 | 57.21 | $[-0.3568\;-0.6029\;-0.7136]$            | — |
| V1–V21 | 20.61 | $[\,0.9551\;\hspace{0.45em}0.0000\;-0.2962]$ | — |
| V1–V22 | 47.11 | $[-0.7193\;\hspace{0.45em}0.3022\;-0.6255]$  | — |
| V1–V23 | 57.21 | $[-0.7384\;-0.2461\;\hspace{0.45em}0.6278]$  | — |
| V1–V24 | 21.06 | $[\,0.9121\;\hspace{0.45em}0.4100\;\hspace{0.45em}0.0000]$ | — |


Only the three $60^\circ/\langle 111 \rangle$ rows (V2, V3, V5) are
exactly rational ($\Sigma 3$ coherent twins); all other rows are
irrational with respect to the cubic lattice.

## 4. Builder pipeline

For every pair, `build_pair()` does:

1. Build the V1 reference crystal once (oriented BCC Fe with the §2
   axes).
2. **Grain-2 construction (rotate-and-crop).** Build V1 oversized by
   `--oversize`, rotate about the §3 axis by the §3 angle around the
   box centre, crop a centred V1-sized block. The number of atoms
   that survive the crop is compared against the analytical
   expectation; if it deviates by more than `--ratio-tol`, the build
   aborts with an error (this catches silent crop bugs and
   under-sized oversize factors). The same rotate-and-crop path is
   used for *all* 23 pairs, including the three $\Sigma 3$ rows
   (V2, V3, V5) — they are not special-cased.
3. **Stack along Y.** Concatenate grain 1 and grain 2 along the
   GB-normal direction, optionally separated by `--gb-gap`. With
   `gap = 0` (the default), atoms within `--overlap-tol` of the
   central GB plane are merged using a tie-break that keeps the atom
   *closer* to the GB.
4. **Write LAMMPS data** with a provenance comment header.

### LAMMPS boundary recommendation

All bicrystals are non-periodic (`pbc=False`); always use
`boundary s s s` in LAMMPS.


### `--gb-gap` safety

Default `--gb-gap = 0`: the two grains are placed in contact. Atom
pairs from opposite grains that end up within `--overlap-tol` of
each other are deduplicated (one atom of each pair is removed; the
tie-break keeps the atom closer to the seam plane so neither grain
is systematically thinned). Without this step, touching grains
would contain a thin layer of near-coincident atoms at the seam
that an MD code would interpret as overlapping particles.

Use `--gb-gap >= 2.5 Å` if you want a real free-surface slit (BCC
Fe NN distance = $a\sqrt{3}/2 \approx 2.482$ Å).

### In-plane grain-2 displacement (`--gb-shift-x`, `--gb-shift-z`)

Both grains live with the GB plane normal to box Y, so the two
in-plane directions of the GB are box X and box Z. `--gb-shift-x` and
`--gb-shift-z` apply a rigid translation (in Å) of *grain 2 only*
along those two directions, on top of the K–S misorientation. Defaults
are zero.

Typical use cases:

- **γ-surface scans.** Sweep `(--gb-shift-x, --gb-shift-z)` over a
  2D grid and relax each configuration to map the
  GB-translation-vs-energy surface.
- **Coincidence shifts.** For the three Σ3 pairs (V2, V3, V5) some
  in-plane shifts reproduce alternative coincidence sites.

Implementation notes:

- The shift is applied during stacking; the box is expanded in X and
  Z to bound both grains exactly, then the atoms are translated so
  the box origin sits at $(0, 0, 0)$. No atoms are wrapped or
  discarded (`pbc=False`, `boundary s s s`).
- The shift is recorded in the provenance header
  (`gb_shift_x (A)`, `gb_shift_z (A)`) and both verifiers
  (`check_outputs.py`, `check_orientations_independent.py`) account
  for it.

---

## 5. Verification

Four independent checks are provided in
[verification/](verification/README.md). They cover both the
*recipe* (the K–S misorientation table itself) and the *atoms on
disk* (do the generated files actually carry the intended
orientations). See
[verification/README.md](verification/README.md) for the full
description and a comparison of the scripts.

Short version:

| Script | What it verifies |
|---|---|
| [ks_misorientations.py](verification/ks_misorientations.py) | The Morito table is reproduced from first principles ($R_\text{KS}$ + cubic right-coset orbit). Also prints two Miller-index tables. |
| [check_outputs.py](verification/check_outputs.py) | Bit-exact recipe reproduction of every `.lmp` file + orientation recovered from atoms equals the header axis/angle. |
| [check_orientations_independent.py](verification/check_orientations_independent.py) | Re-recovers the orientation of each grain with an independent fitter (different math, no shared code with `check_outputs.py`). |
| [find_miller.py](verification/find_miller.py) | Human-readable reporting of grain-2 Miller indices in the lab frame for each file. |

Run all four and exit code 0 from each confirms full consistency.

---

## 6. Project layout

```
python_gb/
├── README.md                     this file
├── gb_generator.py               bicrystal builder (CLI entry point)
├── out/                          generated .lmp files
└── verification/
    ├── README.md                          verification documentation
    ├── ks_misorientations.py             recipe-level check
    ├── check_outputs.py                  bit-exact file + orientation check
    ├── check_orientations_independent.py independent orientation re-fit
    └── find_miller.py                    lab-frame Miller-index reporter
```

## 7. Dependencies

- Python ≥ 3.9
- `numpy`, `scipy`, `ase`

No compiled extensions; everything runs in pure Python via NumPy.

## Notes
For the habit plane $[575]_{\gamma}$ the corresponding orientation in the $\alpha$ reference frame is [-0.134 0.7645 0.6305]. To generate all the variants, having this plane as GB from the crystal one side, the following command can be used

```
python gb_generator.py --v1-x -0.5774 -0.5774  0.5774 --v1-y -0.134 0.7645 0.6305 --v1-z -0.8054  0.2867 -0.5188
```
