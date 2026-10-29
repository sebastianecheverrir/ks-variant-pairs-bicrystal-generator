# Verification of the K–S bicrystal generator

This folder contains four independent scripts that verify the output of
[`gb_generator.py`](../gb_generator.py) at two complementary levels:

- **Recipe level.** Is the table of K–S misorientations built into
  `gb_generator.PAIRS` actually the Kurdjumov–Sachs misorientation
  table?
- **File level.** Does each `.lmp` file on disk really contain the
  bicrystal it claims to contain?

Together the scripts give defence-in-depth: even if one were to have a
bug, the others would catch it.

---

## Quick start

From the project root (one level up from this folder):

```bash
python3 verification/ks_misorientations.py
python3 verification/check_outputs.py                 [out_dir]
python3 verification/check_orientations_independent.py [out_dir]
python3 verification/find_miller.py                   [out_dir]
```

All four accept an optional output directory; the default for the
file-level scripts is `../out` (i.e. the generator's default).

Each verifier prints its result and exits 0 on success.

---

## The four scripts

### `ks_misorientations.py` — recipe-level check

Reproduces every row of `gb_generator.PAIRS` from first principles:

1. Build $R_\text{KS}$ from
   $\{1\,1\,1\}_\gamma \,\|\, \{0\,1\,1\}_\alpha$ and
   $\langle 1\,0\,\bar 1\rangle_\gamma \,\|\, \langle 1\,1\,\bar 1\rangle_\alpha$
   (the $\gamma \to \alpha$ rotation).
2. Generate the 24 K–S variants as the right-coset orbit
   $\{R_\text{KS}\,S : S \in O_\gamma\}$. Right-multiplication is
   essential: left-multiplication by $S \in O_\alpha$ collapses under
   the disorientation reduction and gives a trivial result.
3. For each row of `PAIRS`, find the orbit member whose disorientation
   with V1 reproduces the tabulated (axis, angle) modulo cubic
   symmetry, within $0.05^\circ$ in angle and $0.5^\circ$ in axis
   direction (loose enough to absorb the 4-decimal rounding of
   Morito's printed unit vectors).

In addition, the script prints two Miller-index tables:

- **K–S parallelism** for each variant: which $\{111\}_\gamma$ plane
  and which $\langle 110\rangle_\gamma$ direction lie along which
  $\{011\}_\alpha$ plane and $\langle 111\rangle_\alpha$ direction.
- **Box-frame Miller directions** of grain 2: for each variant, the
  cubic-$\alpha$ Miller indices that lie along the global box X, Y, Z
  axes. This table uses the generator's default V1 orientation; the
  interface normal in generated files is configurable. The default V1
  row reproduces $[1\,1\,1]$, $[1\,1\,\bar 2]$,
  $[\bar 1\,1\,0]$ exactly. The other rows are real-valued (the
  three $\Sigma 3$ pairs V2, V3, V5 are rational; the rest are
  irrational, as the K–S rotation itself is irrational with respect
  to the cubic lattice).

For files generated with a different interface plane, use
`find_miller.py out_dir` to inspect the actual box-axis directions.
Its Y entries give the interface normal in each grain's crystal frame.

Exit code 0 means every row in `PAIRS` is consistent with the K–S
construction.

### `check_outputs.py` — file-level check (recipe + orientation)

Loads every `out/*.lmp` and verifies, per file:

1. Box dimensions and atom count match the provenance comment.
2. A visible GB seam at $y = L_y / 2$ (a thin slab there is not denser
   than bulk; rules out a single accidentally-bonded crystal).
3. The script reports a warning if the gap is in the physically
   aggressive range $0 < \text{gap} < a\sqrt{3}/2$.
4. **Bit-exact recipe reproduction.** The script parses the provenance
   header (`a`, `NX/NY/NZ`, `oversize`, `overlap_tol`, `crop_eps`,
   `gb_gap`, `gb_shift_x`, `gb_shift_z`, `axis`, `angle`), re-runs the
   same `build_oriented` + `rotated_grain` + `stack_along_y` pipeline
   from `gb_generator.py`, and compares every atom in the file to its
   KDTree-nearest reconstructed atom. The maximum per-atom deviation
   must be below $10^{-3}$ Å (in practice $\sim 10^{-14}$ Å —
   machine precision).
5. **Orientation from atoms.** Picks an interior atom with a complete
   BCC first-NN shell, fits the 8-corner NN cube via Kabsch SVD
   against the unrotated reference cube, recovers both grains'
   rotations $R_1, R_2$, and compares $R_2 R_1^{T}$ (in V1's cubic
   frame) to $\text{Rodrigues}(\vec a_\text{hdr}, \theta_\text{hdr})$
   from the header modulo two-sided cubic symmetry (a 24 × 24 sweep,
   576 cases). Tolerance: $0.5^\circ$.

This is the gold-standard file check: if the file came from the
documented recipe with the documented axis/angle, both step 4 and
step 5 pass.

### `check_orientations_independent.py` — independent re-fit

A second opinion on the orientation check (step 5 above) using a
deliberately different algorithm and no shared fitter code:

1. Find an interior atom of each grain with a complete 8-NN BCC
   shell.
2. From the $\binom{8}{2} = 28$ pairwise differences of those NN
   positions, pick three orthonormal directions of length $a$
   (the cubic cube edges). These form a basis $B$ for the grain.
3. $R_\text{box} = B_2 B_1^{T}$;
   $R_\text{cubic} = V \cdot R_\text{box} \cdot V^{T}$, with
   $V = $ V1-basis matrix.
4. Compare to the header axis/angle modulo two-sided cubic symmetry,
   exactly as in `check_outputs.py`, tolerance $0.5^\circ$.

This script imports nothing from `gb_generator` and re-implements the
cubic group, the Rodrigues formula, and the disorientation reduction
locally, so it cannot share a bug with the main verifier. It reads the
V1 directions from the file's provenance header.

### `find_miller.py` — lab-frame Miller-index reporter

Human-readable reporting tool. For every bicrystal `.lmp` file in the
chosen directory, it prints:

- the three integer Miller directions of grain 1 along the lab X, Y, Z
  axes (parsed from the provenance header);
- the three real-valued Miller directions of grain 2 along the same
  lab axes, scaled to the same magnitudes as grain 1;
- the disorientation angle between the two grains.

It uses the BCC **second**-nearest-neighbour shell (6 atoms at distance
$a$) to read the cubic axes directly, rather than the 1NN shell that
the other two file-level scripts use. The script reads the provenance
header to obtain the grain 1 reference directions for canonicalization.
It is not a pass/fail tool — it is for inspection and for
cross-checking the Miller-index tables printed by
`ks_misorientations.py`.

---

## How the four scripts relate

Each script answers a different question:

| # | Script | Question answered | Layer | Reads header? | Imports `gb_generator`? | Pass / fail? |
|---|---|---|---|:---:|:---:|:---:|
| 1 | `ks_misorientations.py` | Is the table in `gb_generator.PAIRS` actually the K–S misorientation table? | recipe | n/a | yes (table only) | yes |
| 2 | `check_outputs.py` | Did the atoms in each `.lmp` come from the documented recipe with the documented axis/angle? | file | yes | yes (full pipeline) | yes |
| 3 | `check_orientations_independent.py` | Does the orientation recovered from atoms alone match the header axis/angle? | file | yes | no | yes |
| 4 | `find_miller.py` | What are the lab-frame Miller indices of each grain in each file? | file | yes (grain 1 dirs) | no | no (report only) |

Scripts 2 and 3 both verify the same fact (the file's grain
misorientation equals the header axis/angle), but with disjoint code
paths: script 2 uses a Kabsch SVD fit on the BCC 1NN cube and also
re-runs the generator pipeline, while script 3 reconstructs the cube
edges directly from pairwise NN differences and shares no helper code
with the generator. If both pass, a bug would have to exist
simultaneously in two independent implementations.

The recommended order:

1. `ks_misorientations.py` — the table itself is correct.
2. `check_outputs.py` — every file matches its recipe and its
   recorded orientation.
3. `check_orientations_independent.py` — same orientation fact,
   re-confirmed by a different method.
4. `find_miller.py` — eyeball the Miller indices for sanity.

---

## Notes on cubic symmetry

For any cubic↔cubic misorientation comparison, the correct equivalence
test is the **two-sided** one:

$$
R_\text{meas} \equiv R_\text{ref}
\quad\iff\quad
\exists \, S_1, S_2 \in O \;:\; S_1 R_\text{meas} S_2 = R_\text{ref}.
$$

This is what `check_outputs.py` and `check_orientations_independent.py`
do (24 × 24 = 576 candidates). A one-sided sweep
($S R_\text{meas} S^{T}$) is **not** equivalent and can disagree by
several degrees. `find_miller.py` uses a single-sided sweep, which is
why it is only used for reporting, not for pass/fail verification.

## Dependencies

Same as the generator: `numpy`, `scipy`, `ase`. No extra packages are
required for verification.
