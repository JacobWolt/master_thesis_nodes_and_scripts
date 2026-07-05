"""
Master clustering script with integrated free-volume analysis.

For every lipid, evaluate three independent criteria and select those that
satisfy at least --min-criteria (default 2):

  1. CYLINDER     — P atom lies inside a cylinder of radius --radius (default
                    17 Å) around the per-frame pore center. The pore center
                    is detected from waters in the membrane slab; the cavity
                    centroid (from .dx or per-frame free-volume) acts as an
                    anchor so the cylinder consistently tracks the same pore.
  2. CAVITY       — P atom lies within --cavity-cutoff Å of the pore-cavity
                    surface (default 6 Å). The cavity is either:
                       * read from a passed .dx file (static, one shape used
                         for all frames), OR
                       * computed per frame internally using Falck-style
                         free-volume analysis (slower but tracks the cavity
                         as it evolves).
  3. TILT         — Lipid head→tail tilt is in the high-tilt cluster of a
                    per-frame 2-component Gaussian Mixture Model.

Two operating modes (auto-detected from positional arguments):

  A) STATIC CAVITY MODE — pass a .dx file as input:
        python master_cluster_lipids.py system.gro pore_cavity.dx
        python master_cluster_lipids.py system.gro traj.xtc pore_cavity.dx

  B) PER-FRAME FREE-VOLUME MODE — no .dx file, the script computes the
     cavity internally for every frame:
        python master_cluster_lipids.py system.gro
        python master_cluster_lipids.py system.gro traj.xtc

Outputs:
  - <prefix>_per_lipid.csv   : one row per (frame, lipid) with all criteria
  - <prefix>_per_frame.csv   : per-frame summary. First two rows record the
                                parameters used for this run; row 3 is the
                                normal data header; row 4+ is the per-frame
                                data. vmd_sel_selected is in column H.
  - <prefix>_resid_groups.csv: trajectory-averaged groups
  - <prefix>_lipid_counts.csv: per-lipid frequency of selection
  - <prefix>_cavity_dx/      : per-frame cavity .dx files (only if
                                --save-cavity-dx is passed and running in
                                per-frame mode)
  - <prefix>_cylinder_dx/    : per-frame cylinder .dx files (only if
                                --write-cylinders is passed)
"""

# =========================================================================
# DEFAULT PARAMETERS  (override on the command line)
# =========================================================================
DEFAULT_RADIUS = 17.0
DEFAULT_CAVITY_CUTOFF = 6.0
DEFAULT_ISOVALUE = 0.5
DEFAULT_HEAD_SEL = "name P"
DEFAULT_TAIL_SEL = "name C218 C316"
DEFAULT_P_SEL = "name P"
DEFAULT_WATER_SEL = "resname SOL TIP3 W HOH and name OW OH2 O W"
DEFAULT_SEED_CUTOFF = 14.0
DEFAULT_MIN_SEED_SIZE = 2
DEFAULT_MIN_CRITERIA = 2
DEFAULT_LIPID_SEL = "resname POPC POPS POPE DOPC DPPC POPG"
DEFAULT_FV_DELTA = 1.0          
DEFAULT_FV_PADDING = 2.0      
DEFAULT_PROBE_RADIUS = 0.0      
DEFAULT_SPAN_MARGIN = 2.0       
DEFAULT_MIN_CAVITY_VOLUME = 100.0
# =========================================================================

import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import MDAnalysis as mda
except ImportError:
    sys.exit("MDAnalysis is required. Install with: pip install MDAnalysis")

try:
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import label as cc_label
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.mixture import GaussianMixture
except ImportError:
    sys.exit("scipy and scikit-learn are required. "
             "Install with: pip install scipy scikit-learn")


# =========================================================================
# vdW radii and element guessing (for free-volume mode)
# =========================================================================

VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52,
    "P": 1.80, "S": 1.80,
}
DEFAULT_VDW = 1.70


def element_from_name(name: str) -> str:
    if not name:
        return ""
    if name[0] in "CHONPSF":
        if name[:2] in ("Cl", "CL"): return "Cl"
        if name[:2] in ("Na", "NA"): return "Na"
        return name[0].upper()
    twol = name[:2].capitalize()
    if twol in VDW_RADII:
        return twol
    return name[0].upper()


# =========================================================================
# DX reader / writer
# =========================================================================

def read_dx(path: Path):
    """Read an OpenDX scalar file. Returns (grid, origin, delta)."""
    with open(path) as f:
        lines = f.readlines()

    nx = ny = nz = None
    origin = None
    deltas = []
    data_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("object 1") and "gridpositions" in s:
            parts = s.split()
            nx, ny, nz = int(parts[-3]), int(parts[-2]), int(parts[-1])
        elif s.startswith("origin"):
            parts = s.split()
            origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        elif s.startswith("delta"):
            parts = s.split()
            deltas.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
        elif "data follows" in s:
            data_start = i + 1
            break

    if nx is None or origin is None or len(deltas) < 3 or data_start is None:
        sys.exit(f"Could not parse DX header in {path}")

    delta = np.array([deltas[0][0], deltas[1][1], deltas[2][2]])

    vals = []
    for line in lines[data_start:]:
        s = line.strip()
        if not s or s.startswith("attribute") or s.startswith("object"):
            break
        vals.extend(float(v) for v in s.split())
    arr = np.array(vals, dtype=float).reshape(nx, ny, nz, order="C")
    return arr, origin, delta


def write_dx(grid: np.ndarray, origin: np.ndarray, delta: np.ndarray,
             path: Path) -> None:
    """Write a 3D scalar grid as an OpenDX file (VMD-readable)."""
    nx, ny, nz = grid.shape
    n_total = nx * ny * nz
    with open(path, "w") as f:
        f.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        f.write(f"origin {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n")
        f.write(f"delta {delta[0]:.6f} 0 0\n")
        f.write(f"delta 0 {delta[1]:.6f} 0\n")
        f.write(f"delta 0 0 {delta[2]:.6f}\n")
        f.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        f.write(f"object 3 class array type double rank 0 items {n_total} data follows\n")
        flat = grid.ravel(order="C").astype(float)
        for i in range(0, n_total, 3):
            chunk = flat[i:i + 3]
            f.write(" ".join(f"{v:.6g}" for v in chunk) + "\n")
        f.write('attribute "dep" string "positions"\n')
        f.write('object "density" class field\n')


def tile_grid_xy(grid: np.ndarray, origin: np.ndarray, delta: np.ndarray,
                 box: np.ndarray, n_tiles: int = 3):
    """Tile a 3D grid (n_tiles × n_tiles) in xy so it covers all periodic
    images around the primary box. The grid is assumed to already span the
    primary box in xy. Returns (tiled_grid, tiled_origin, delta).

    With n_tiles=3 the result is a 3×3 super-grid centered on the primary
    cell — so you can move atoms by up to ±1 box length in either direction
    in VMD and still see a continuous isosurface."""
    if n_tiles < 1:
        return grid, origin, delta

    nx, ny, nz = grid.shape

    # Voxels per box length. We assume the grid spacing divides the box
    # length evenly enough; we use the actual grid extent to tile.
    tiled_nx = nx * n_tiles
    tiled_ny = ny * n_tiles
    tiled = np.tile(grid, (n_tiles, n_tiles, 1))

    # Shift the origin so the central cell sits at the primary box position.
    # With n_tiles=3 the central cell is at offset (1, 1) in tile-index.
    offset_cells = (n_tiles - 1) // 2
    new_origin = origin.copy()
    new_origin[0] -= offset_cells * box[0]
    new_origin[1] -= offset_cells * box[1]
    return tiled, new_origin, delta


# =========================================================================
# Free-volume building blocks (from free_volume_analysis.py)
# =========================================================================

def build_occupancy(positions: np.ndarray, radii: np.ndarray,
                    origin: np.ndarray, delta: np.ndarray,
                    shape: tuple, box=None) -> np.ndarray:
    """Mark voxels inside any atom's vdW sphere. Returns a bool grid.
    If `box` is given, atoms near grid edges also paint their PBC image."""
    nx, ny, nz = shape
    occ = np.zeros(shape, dtype=bool)
    inv_delta = 1.0 / delta

    for (x, y, z), r in zip(positions, radii):
        if box is None:
            shifts = [(0.0, 0.0, 0.0)]
        else:
            sx = [0.0]
            if x - r < origin[0]: sx.append(box[0])
            if x + r > origin[0] + nx * delta[0]: sx.append(-box[0])
            sy = [0.0]
            if y - r < origin[1]: sy.append(box[1])
            if y + r > origin[1] + ny * delta[1]: sy.append(-box[1])
            sz = [0.0]
            if z - r < origin[2]: sz.append(box[2])
            if z + r > origin[2] + nz * delta[2]: sz.append(-box[2])
            shifts = [(dx, dy, dz) for dx in sx for dy in sy for dz in sz]

        for dx_s, dy_s, dz_s in shifts:
            xx = x + dx_s; yy = y + dy_s; zz = z + dz_s
            ix_lo = max(0, int(np.floor((xx - r - origin[0]) * inv_delta[0])))
            ix_hi = min(nx, int(np.ceil((xx + r - origin[0]) * inv_delta[0])) + 1)
            iy_lo = max(0, int(np.floor((yy - r - origin[1]) * inv_delta[1])))
            iy_hi = min(ny, int(np.ceil((yy + r - origin[1]) * inv_delta[1])) + 1)
            iz_lo = max(0, int(np.floor((zz - r - origin[2]) * inv_delta[2])))
            iz_hi = min(nz, int(np.ceil((zz + r - origin[2]) * inv_delta[2])) + 1)
            if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
                continue

            xs = origin[0] + (np.arange(ix_lo, ix_hi) + 0.5) * delta[0]
            ys = origin[1] + (np.arange(iy_lo, iy_hi) + 0.5) * delta[1]
            zs = origin[2] + (np.arange(iz_lo, iz_hi) + 0.5) * delta[2]
            dx2 = (xs - xx) ** 2
            dy2 = (ys - yy) ** 2
            dz2 = (zs - zz) ** 2
            d2 = dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :]
            inside = d2 <= (r * r)
            occ[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi] |= inside

    return occ


def find_pore_cavity(free_grid, origin, delta, z_upper, z_lower,
                     margin=2.0, min_cavity_volume=0.0,
                     combine_all_in_slab=True):
    """Find pore-forming cavities — same logic as in free_volume_analysis.py."""
    voxel_volume = float(np.prod(delta))
    empty_info = {
        "n_components": 0, "n_spanning": 0, "is_spanning": 0,
        "n_cavities_included": 0, "pore_voxels": 0, "pore_volume_A3": 0.0,
        "pore_x": float("nan"), "pore_y": float("nan"),
        "pore_z": float("nan"),
    }
    structure = np.ones((3, 3, 3), dtype=int)
    labels, n_components = cc_label(free_grid, structure=structure)
    if n_components == 0:
        return np.zeros_like(free_grid, dtype=bool), empty_info

    nz = free_grid.shape[2]
    z_coords = origin[2] + (np.arange(nz) + 0.5) * delta[2]
    upper_band = (z_coords >= z_upper - margin) & (z_coords <= z_upper + margin)
    lower_band = (z_coords >= z_lower - margin) & (z_coords <= z_lower + margin)
    in_slab = (z_coords >= z_lower - margin) & (z_coords <= z_upper + margin)

    labels_upper = set(np.unique(labels[:, :, upper_band])) - {0}
    labels_lower = set(np.unique(labels[:, :, lower_band])) - {0}
    spanning = labels_upper & labels_lower

    labels_in_slab = set(np.unique(labels[:, :, in_slab])) - {0}
    sizes = {lab: int((labels == lab).sum()) for lab in labels_in_slab}

    included = set()
    if spanning:
        biggest_spanning = max(spanning, key=lambda lab: sizes.get(lab, 0))
        included.add(biggest_spanning)
    min_voxels = int(np.ceil(min_cavity_volume / voxel_volume))
    if combine_all_in_slab:
        for lab, sz in sizes.items():
            if lab not in included and sz >= max(min_voxels, 1):
                if lab in labels_upper or lab in labels_lower:
                    included.add(lab)
    if not included and sizes:
        included.add(max(sizes, key=sizes.get))

    if not included:
        return (np.zeros_like(free_grid, dtype=bool),
                {**empty_info, "n_components": int(n_components),
                 "n_spanning": len(spanning),
                 "is_spanning": 1 if spanning else 0})

    mask = np.isin(labels, list(included))
    total_voxels = int(mask.sum())
    idx = np.argwhere(mask)
    if len(idx) > 0:
        centroid = origin + (idx.mean(axis=0) + 0.5) * delta
    else:
        centroid = np.array([np.nan, np.nan, np.nan])
    info = {
        "n_components": int(n_components),
        "n_spanning": len(spanning),
        "is_spanning": 1 if spanning else 0,
        "n_cavities_included": len(included),
        "pore_voxels": total_voxels,
        "pore_volume_A3": total_voxels * voxel_volume,
        "pore_x": float(centroid[0]),
        "pore_y": float(centroid[1]),
        "pore_z": float(centroid[2]),
    }
    return mask, info


def compute_cavity_per_frame(lipids_pos, lipid_radii, box, z_up, z_lo, args):
    """Run free-volume analysis on the current frame and return
    (cavity_mask, origin, delta, info). Honors --use-box / --fv-wrap /
    --pbc-paint / --pbc-halo (same semantics as free_volume_analysis.py)."""
    pos = lipids_pos.copy()
    if args.fv_wrap or args.pbc_halo:
        for k in range(3):
            if box[k] > 0:
                pos[:, k] = pos[:, k] % box[k]

    radii = lipid_radii + args.probe_radius

    # Build grid bounds
    if args.pbc_halo:
        halo_offsets = [(ix, iy) for ix in (-1, 0, 1) for iy in (-1, 0, 1)]
        all_pos = np.vstack([
            pos + np.array([ix * box[0], iy * box[1], 0.0])
            for ix, iy in halo_offsets
        ])
        all_radii = np.tile(radii, len(halo_offsets))
        grid_pos = all_pos
        grid_radii = all_radii
        xy_min = np.array([-box[0], -box[1]])
        xy_max = np.array([2 * box[0], 2 * box[1]])
        z_min, z_max = 0.0, box[2]
    else:
        grid_pos = pos
        grid_radii = radii
        if args.use_box:
            xy_min = np.array([0.0, 0.0])
            xy_max = np.array([box[0], box[1]])
            z_min, z_max = 0.0, box[2]
        else:
            xy_min = pos[:, :2].min(axis=0) - args.fv_padding
            xy_max = pos[:, :2].max(axis=0) + args.fv_padding
            z_min = pos[:, 2].min() - args.fv_padding
            z_max = pos[:, 2].max() + args.fv_padding

    origin = np.array([xy_min[0], xy_min[1], z_min])
    extent = np.array([xy_max[0] - xy_min[0],
                       xy_max[1] - xy_min[1],
                       z_max - z_min])
    delta = np.array([args.fv_delta] * 3)
    shape = tuple(int(np.ceil(e / d)) for e, d in zip(extent, delta))

    paint_box = box if (args.pbc_paint and not args.pbc_halo) else None
    occ = build_occupancy(grid_pos, grid_radii, origin, delta, shape,
                          box=paint_box)
    free = ~occ

    pore_mask, info = find_pore_cavity(
        free, origin, delta, z_up, z_lo,
        margin=args.span_margin,
        min_cavity_volume=args.min_cavity_volume,
        combine_all_in_slab=not args.no_combine_cavities,
    )

    # Halo crop back to primary box
    if args.pbc_halo:
        inv_delta = 1.0 / delta
        ix0 = int(round((0.0 - origin[0]) * inv_delta[0]))
        ix1 = int(round((box[0] - origin[0]) * inv_delta[0]))
        iy0 = int(round((0.0 - origin[1]) * inv_delta[1]))
        iy1 = int(round((box[1] - origin[1]) * inv_delta[1]))
        pore_mask = pore_mask[ix0:ix1, iy0:iy1, :]
        origin = np.array([origin[0] + ix0 * delta[0],
                           origin[1] + iy0 * delta[1],
                           origin[2]])

    # Clip to slab in z so the cavity represents the transmembrane part only
    nz = pore_mask.shape[2]
    z_voxel = origin[2] + (np.arange(nz) + 0.5) * delta[2]
    in_slab = (z_voxel >= z_lo - args.span_margin) & \
              (z_voxel <= z_up + args.span_margin)
    pore_mask_clipped = pore_mask.copy()
    pore_mask_clipped[:, :, ~in_slab] = False

    return pore_mask_clipped, origin, delta, info


# =========================================================================
# Pore-detection & PBC helpers
# =========================================================================

def mic_vector(v, box):
    v = np.array(v, dtype=float).copy()
    for k in range(3):
        if box[k] > 0:
            v[k] -= box[k] * np.round(v[k] / box[k])
    return v


def xy_distance_pbc(px, py, cx, cy, box_xy):
    dx = px - cx
    dy = py - cy
    if box_xy is not None:
        dx -= box_xy[0] * np.round(dx / box_xy[0])
        dy -= box_xy[1] * np.round(dy / box_xy[1])
    return np.sqrt(dx * dx + dy * dy)


def _split_leaflets_z(p_z, box_z):
    """Return (z_upper, z_lower, z_shift) in the PBC-unwrapped frame."""
    zs = np.sort(p_z)
    gaps = np.diff(zs)
    wrap_gap = (zs[0] + box_z) - zs[-1]
    all_gaps = np.append(gaps, wrap_gap)
    i = int(np.argmax(all_gaps))
    shift = -zs[i + 1] if i < len(gaps) else -zs[0]
    z_shifted = (p_z + shift) % box_z
    zmean = z_shifted.mean()
    z_up = z_shifted[z_shifted > zmean].mean()
    z_lo = z_shifted[z_shifted <= zmean].mean()
    z_shift = (z_shifted[0] - p_z[0]) % box_z
    return float(z_up), float(z_lo), float(z_shift)


def membrane_planes_original_frame(p_atoms, box_z):
    """Return (z_up, z_lo) in the ORIGINAL atom-coordinate frame (the one
    VMD displays). Handles the wrap-around case where z_lo may end up below
    z_up."""
    z_up_s, z_lo_s, z_shift = _split_leaflets_z(p_atoms.positions[:, 2], box_z)
    z_up = (z_up_s - z_shift) % box_z
    z_lo = (z_lo_s - z_shift) % box_z
    if z_lo > z_up:
        z_lo, z_up = z_up, z_lo
    return z_up, z_lo


def _pbc_pdist_xy(xy, box_xy):
    n = len(xy)
    out = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n - 1):
        dx = xy[i + 1:, 0] - xy[i, 0]
        dy = xy[i + 1:, 1] - xy[i, 1]
        dx -= box_xy[0] * np.round(dx / box_xy[0])
        dy -= box_xy[1] * np.round(dy / box_xy[1])
        out[k:k + len(dx)] = np.sqrt(dx * dx + dy * dy)
        k += len(dx)
    return out


def _circular_mean(coords, box_len):
    angles = 2 * np.pi * coords / box_len
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return (mean_angle / (2 * np.pi)) * box_len % box_len


def detect_pore_centers(p_atoms, water_o, box, seed_cutoff, min_seed_size):
    """Detect pore xy center(s) by clustering slab waters."""
    box_xy = box[:2]
    box_z = box[2]
    p_z = p_atoms.positions[:, 2]
    z_up, z_lo, z_shift = _split_leaflets_z(p_z, box_z)
    o_pos = water_o.positions.copy()
    o_pos[:, 2] = (o_pos[:, 2] + z_shift) % box_z
    in_slab = (o_pos[:, 2] > z_lo) & (o_pos[:, 2] < z_up)
    slab_xy = o_pos[in_slab, :2]
    if len(slab_xy) == 0:
        return []
    if len(slab_xy) == 1:
        return [(float(slab_xy[0, 0]), float(slab_xy[0, 1]), 1)] if min_seed_size <= 1 else []
    d = _pbc_pdist_xy(slab_xy, box_xy)
    Z = linkage(d, method="single")
    labels = fcluster(Z, t=seed_cutoff, criterion="distance")
    centers = []
    for cl in np.unique(labels):
        m = labels == cl
        if m.sum() >= min_seed_size:
            cx = _circular_mean(slab_xy[m, 0], box_xy[0])
            cy = _circular_mean(slab_xy[m, 1], box_xy[1])
            centers.append((float(cx), float(cy), int(m.sum())))
    return centers


def closest_center_to_anchor(centers, anchor_xy, box_xy):
    if not centers:
        return None, None
    ax, ay = anchor_xy
    best = None
    best_d = float("inf")
    for cx, cy, _ in centers:
        d = float(xy_distance_pbc(np.array([cx]), np.array([cy]),
                                  ax, ay, box_xy)[0])
        if d < best_d:
            best_d = d
            best = (cx, cy)
    return best


# =========================================================================
# Cylinder volume builder (for --write-cylinders)
# =========================================================================

def build_cylinder_dx(cx, cy, radius, z_lo, z_up, box,
                      delta_xy=1.0, delta_z=1.0):
    nx = int(np.ceil(box[0] / delta_xy))
    ny = int(np.ceil(box[1] / delta_xy))
    z_min = z_lo - 1.0
    z_max = z_up + 1.0
    nz = int(np.ceil((z_max - z_min) / delta_z))
    origin = np.array([0.0, 0.0, z_min])
    delta = np.array([delta_xy, delta_xy, delta_z])
    xs = origin[0] + (np.arange(nx) + 0.5) * delta_xy
    ys = origin[1] + (np.arange(ny) + 0.5) * delta_xy
    zs = origin[2] + (np.arange(nz) + 0.5) * delta_z
    dx = xs - cx
    dy = ys - cy
    dx -= box[0] * np.round(dx / box[0])
    dy -= box[1] * np.round(dy / box[1])
    d2_xy = dx[:, None] ** 2 + dy[None, :] ** 2
    in_cyl_xy = d2_xy < radius * radius
    in_z = (zs >= z_lo) & (zs <= z_up)
    mask = in_cyl_xy[:, :, None] & in_z[None, None, :]
    return mask.astype(np.uint8), origin, delta


# =========================================================================
# Tilt clustering & per-frame tilt vector
# =========================================================================

def cluster_tilts_2comp(tilts, random_state=0):
    X = tilts.reshape(-1, 1)
    if len(X) < 5 or np.std(tilts) < 1e-6:
        return (np.zeros(len(X), dtype=int), np.zeros(len(X)),
                np.array([float(np.mean(tilts))] * 2),
                np.array([float(np.std(tilts) + 1e-6)] * 2))
    try:
        gmm = GaussianMixture(n_components=2, random_state=random_state).fit(X)
    except Exception:
        return (np.zeros(len(X), dtype=int), np.zeros(len(X)),
                np.array([float(np.mean(tilts))] * 2),
                np.array([float(np.std(tilts) + 1e-6)] * 2))
    means = gmm.means_.flatten()
    order = np.argsort(means)
    relabel = {old: new for new, old in enumerate(order)}
    raw = gmm.predict(X)
    labels = np.array([relabel[l] for l in raw])
    prob_high = gmm.predict_proba(X)[:, order[-1]]
    sigmas = np.array([np.sqrt(gmm.covariances_[i, 0, 0]) for i in order])
    return labels, prob_high, means[order], sigmas


def compute_lipid_tilts(u, head_sel, tail_sel, p_sel):
    p_atoms = u.select_atoms(p_sel)
    if len(p_atoms) == 0:
        sys.exit(f"No atoms matched P selection '{p_sel}'.")
    resids = []; head_groups = []; tail_groups = []; p_groups = []
    for p in p_atoms:
        res = p.residue
        h = res.atoms.select_atoms(head_sel)
        t = res.atoms.select_atoms(tail_sel)
        if len(h) == 0 or len(t) == 0:
            continue
        resids.append(int(res.resid))
        head_groups.append(h)
        tail_groups.append(t)
        p_groups.append(res.atoms.select_atoms(p_sel))
    return resids, head_groups, tail_groups, p_groups


def compute_frame_tilts(head_groups, tail_groups, p_groups, box):
    """Per-frame tilts. Returns (tilt, raw, tilt_x, tilt_y, px, py, pz)."""
    n = len(head_groups)
    tilt = np.full(n, np.nan); raw = np.full(n, np.nan)
    tilt_x = np.full(n, np.nan); tilt_y = np.full(n, np.nan)
    px = np.full(n, np.nan); py = np.full(n, np.nan); pz = np.full(n, np.nan)
    for i, (h, t, p) in enumerate(zip(head_groups, tail_groups, p_groups)):
        hp = h.center_of_mass(); tp = t.center_of_mass(); pp = p.center_of_mass()
        v = mic_vector(tp - hp, box)
        nrm = np.linalg.norm(v)
        if nrm < 1e-9:
            continue
        cos_t = np.clip(v[2] / nrm, -1.0, 1.0)
        ang = float(np.degrees(np.arccos(cos_t)))
        raw[i] = ang
        tilt[i] = ang if ang <= 90.0 else 180.0 - ang
        v_fold = v if v[2] >= 0 else -v
        tilt_x[i] = float(np.degrees(np.arctan2(v_fold[0], v_fold[2])))
        tilt_y[i] = float(np.degrees(np.arctan2(v_fold[1], v_fold[2])))
        px[i], py[i], pz[i] = float(pp[0]), float(pp[1]), float(pp[2])
    return tilt, raw, tilt_x, tilt_y, px, py, pz


# =========================================================================
# Main
# =========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("topology", help=".gro / .pdb")
    ap.add_argument("second", nargs="?", default=None,
                    help="Either: pore_cavity.dx (static cavity mode), or a "
                         ".xtc/.trr trajectory (then arg 3 may be a .dx). If "
                         "omitted, run single-frame in free-volume mode.")
    ap.add_argument("third", nargs="?", default=None,
                    help="pore_cavity.dx (when a trajectory is given as the "
                         "second argument). Omit to compute the cavity "
                         "internally per frame.")
    # --- master clustering parameters ---
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                    help=f"Cylinder radius around pore center (Å). "
                         f"Default: {DEFAULT_RADIUS}")
    ap.add_argument("--cavity-cutoff", type=float, default=DEFAULT_CAVITY_CUTOFF,
                    help=f"Distance to cavity surface for 'close to cavity' "
                         f"(Å). Default: {DEFAULT_CAVITY_CUTOFF}")
    ap.add_argument("--isovalue", type=float, default=DEFAULT_ISOVALUE,
                    help=f"DX isovalue defining the cavity. "
                         f"Default: {DEFAULT_ISOVALUE}")
    ap.add_argument("--min-criteria", type=int, default=DEFAULT_MIN_CRITERIA,
                    help=f"In legacy mode only (see --legacy-combine): how "
                         f"many of the 3 criteria must be satisfied "
                         f"(default {DEFAULT_MIN_CRITERIA}). With "
                         f"--min-criteria 1, being inside the cylinder alone "
                         f"is enough. Ignored in the default mode.")
    ap.add_argument("--legacy-combine", action="store_true",
                    help="Use the old combine rule: selected if at least "
                         "--min-criteria of the three criteria are met. "
                         "Default rule (without this flag) is the stricter "
                         "'cylinder AND (cavity OR tilt)', which filters out "
                         "artifact lipids that happen to be near unrelated "
                         "cavities elsewhere in the membrane.")
    ap.add_argument("--head-sel", default=DEFAULT_HEAD_SEL,
                    help=f"Head reference (default '{DEFAULT_HEAD_SEL}')")
    ap.add_argument("--tail-sel", default=DEFAULT_TAIL_SEL,
                    help=f"Tail end carbons (default '{DEFAULT_TAIL_SEL}')")
    ap.add_argument("--p-sel", default=DEFAULT_P_SEL,
                    help=f"P atom selection (default '{DEFAULT_P_SEL}')")
    ap.add_argument("--water-sel", default=DEFAULT_WATER_SEL,
                    help="Water selection for per-frame pore detection")
    ap.add_argument("--seed-cutoff", type=float, default=DEFAULT_SEED_CUTOFF,
                    help=f"xy clustering cutoff for slab waters "
                         f"(default {DEFAULT_SEED_CUTOFF} Å)")
    ap.add_argument("--min-seed-size", type=int, default=DEFAULT_MIN_SEED_SIZE,
                    help=f"Min slab waters to call a pore "
                         f"(default {DEFAULT_MIN_SEED_SIZE})")
    ap.add_argument("--static-center", action="store_true",
                    help="Disable per-frame pore detection (use cavity "
                         "centroid for every frame).")
    ap.add_argument("--write-cylinders", action="store_true",
                    help="Write a .dx of the cylinder per frame.")
    ap.add_argument("--cylinder-dx-delta", type=float, default=1.0,
                    help="Voxel spacing for cylinder .dx (default 1.0 Å).")
    ap.add_argument("--wrap", action="store_true",
                    help="Wrap P positions into [0, box] before grid lookup "
                         "(use if your trajectory has periodic-image atoms).")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out-prefix", default=None)

    # --- free-volume parameters (used in per-frame mode) ---
    fvg = ap.add_argument_group("Free-volume analysis (used when no .dx is given)")
    fvg.add_argument("--lipid-sel", default=DEFAULT_LIPID_SEL,
                     help=f"Lipid atom selection for the occupancy grid "
                          f"(default common phospholipids).")
    fvg.add_argument("--fv-delta", type=float, default=DEFAULT_FV_DELTA,
                     help=f"Free-volume grid spacing (Å, default "
                          f"{DEFAULT_FV_DELTA}). 0.5 = sharper, much slower.")
    fvg.add_argument("--fv-padding", type=float, default=DEFAULT_FV_PADDING,
                     help=f"Padding around lipid bounding box (Å, default "
                          f"{DEFAULT_FV_PADDING}). Ignored with --use-box.")
    fvg.add_argument("--probe-radius", type=float, default=DEFAULT_PROBE_RADIUS,
                     help=f"Probe sphere radius (Å, default "
                          f"{DEFAULT_PROBE_RADIUS}). 1.4 = water-accessible.")
    fvg.add_argument("--span-margin", type=float, default=DEFAULT_SPAN_MARGIN,
                     help=f"Leaflet-plane tolerance for 'spanning' cavity "
                          f"(Å, default {DEFAULT_SPAN_MARGIN}).")
    fvg.add_argument("--min-cavity-volume", type=float,
                     default=DEFAULT_MIN_CAVITY_VOLUME,
                     help=f"Min volume (Å³) for a non-spanning cavity to be "
                          f"included (default {DEFAULT_MIN_CAVITY_VOLUME}). "
                          f"Catches pores forming from both sides.")
    fvg.add_argument("--no-combine-cavities", action="store_true",
                     help="Only output the largest cavity (don't combine).")
    fvg.add_argument("--use-box", action="store_true",
                     help="Build the free-volume grid on the full simulation "
                          "box (xy) instead of the lipid bounding region.")
    fvg.add_argument("--fv-wrap", action="store_true",
                     help="Wrap atoms into [0, box] before building the free-"
                          "volume grid (recommended with --use-box).")
    fvg.add_argument("--pbc-paint", action="store_true",
                     help="Paint atom periodic images on opposite box edges.")
    fvg.add_argument("--pbc-halo", action="store_true",
                     help="Most thorough PBC: 3×3 super-grid in xy, then "
                          "crop back. 9× memory but correct for edge-crossing "
                          "cavities.")
    fvg.add_argument("--save-cavity-dx", action="store_true",
                     help="In free-volume mode, save the per-frame cavity .dx "
                          "files into <prefix>_cavity_dx/ for animations or "
                          "VMD inspection. Off by default to save disk space.")
    fvg.add_argument("--tile-output", action="store_true",
                     help="Tile the cavity and cylinder .dx files 3×3 in xy "
                          "so they cover all periodic images around the "
                          "primary box. Use this if you want to translate "
                          "atoms in VMD (with `moveby` or similar) and still "
                          "see a continuous isosurface across box boundaries. "
                          "The resulting .dx files are 9× larger. The cavity "
                          "centroid you'd normally look up is still in the "
                          "central cell, so analysis is unchanged.")

    args = ap.parse_args()

    # --- Disambiguate positional args ---
    # Possibilities (top is the topology, always required):
    #   master.py top                 → second=None, third=None: FV mode, single frame
    #   master.py top file1           → second=file1, third=None:
    #                                   if file1 is .dx  → STATIC mode, single frame
    #                                   else            → FV mode with trajectory
    #   master.py top file1 file2     → second=traj, third=.dx: STATIC mode
    top = Path(args.topology)
    if not top.exists():
        sys.exit(f"Topology not found: {top}")

    traj_path = None
    dx_path = None
    if args.second is None and args.third is None:
        pass  # FV mode, single frame
    elif args.third is None:
        # Single non-topology arg — detect from suffix
        p = Path(args.second)
        if not p.exists():
            sys.exit(f"Input not found: {p}")
        if p.suffix.lower() == ".dx":
            dx_path = p
        else:
            traj_path = p
    else:
        # Both extra args present: traj + dx
        traj_path = Path(args.second)
        dx_path = Path(args.third)
        if not traj_path.exists():
            sys.exit(f"Trajectory not found: {traj_path}")
        if not dx_path.exists():
            sys.exit(f"DX file not found: {dx_path}")
        if dx_path.suffix.lower() != ".dx":
            sys.exit(f"Third argument must be a .dx file (got {dx_path})")

    fv_mode = dx_path is None
    if fv_mode:
        print("Mode: PER-FRAME FREE-VOLUME (cavity computed every frame)")
    else:
        print(f"Mode: STATIC CAVITY (cavity loaded from {dx_path.name})")

    prefix = args.out_prefix or top.with_suffix("").as_posix()

    # ----- Load Universe -----
    if traj_path:
        u = mda.Universe(str(top), str(traj_path))
        print(f"Loaded {top} + {traj_path} ({len(u.trajectory)} frames)")
    else:
        u = mda.Universe(str(top))
        print(f"Loaded {top} (single frame)")

    # ----- Static DX cavity (if provided) -----
    static_grid = static_origin = static_delta = None
    static_dist_grid = None
    static_anchor = None
    if not fv_mode:
        static_grid, static_origin, static_delta = read_dx(dx_path)
        cavity_mask = static_grid > args.isovalue
        n_cav_vox = int(cavity_mask.sum())
        if n_cav_vox == 0:
            sys.exit(f"No cavity voxels above isovalue {args.isovalue} in {dx_path}")
        print(f"Loaded cavity from {dx_path.name}: shape {static_grid.shape}, "
              f"{n_cav_vox} voxels above isovalue {args.isovalue}")
        cav_idx = np.argwhere(cavity_mask)
        cav_centroid = static_origin + (cav_idx.mean(axis=0) + 0.5) * static_delta
        static_anchor = (float(cav_centroid[0]), float(cav_centroid[1]))
        print(f"Cavity centroid: ({static_anchor[0]:.2f}, {static_anchor[1]:.2f}, "
              f"{cav_centroid[2]:.2f})")
        print("Computing static distance transform...")
        static_dist_grid = distance_transform_edt(~cavity_mask, sampling=static_delta)

    # ----- Lipid head/tail/P groups -----
    resids, head_groups, tail_groups, p_groups = compute_lipid_tilts(
        u, args.head_sel, args.tail_sel, args.p_sel
    )
    n_lipids = len(resids)
    if n_lipids == 0:
        sys.exit("No lipids matched head/tail/P selections.")
    print(f"Tracking {n_lipids} lipids")
    print(f"Cylinder radius: {args.radius} Å, cavity cutoff: {args.cavity_cutoff} Å")
    if args.legacy_combine:
        print(f"Selection rule (legacy): lipid flagged if >= "
              f"{args.min_criteria} of 3 criteria met.")
    else:
        print("Selection rule (default): cylinder mandatory AND (cavity OR tilt).")

    # P atoms for membrane geometry
    p_all = u.select_atoms(args.p_sel)
    if len(p_all) == 0:
        sys.exit(f"No atoms matched P selection '{args.p_sel}'.")

    # Lipid atoms + pre-computed vdW radii (used only in fv_mode)
    if fv_mode:
        lipid_atoms = u.select_atoms(args.lipid_sel)
        if len(lipid_atoms) == 0:
            sys.exit(f"No atoms matched lipid selection: '{args.lipid_sel}'\n"
                     f"Available resnames: {sorted(set(u.atoms.resnames))}")
        print(f"Free-volume mode: {len(lipid_atoms)} lipid atoms, "
              f"grid spacing {args.fv_delta} Å, probe radius {args.probe_radius} Å")
        elements = [element_from_name(a.name) for a in lipid_atoms]
        lipid_radii = np.array([VDW_RADII.get(e, DEFAULT_VDW) for e in elements])
    else:
        lipid_atoms = None
        lipid_radii = None

    # Water selection (for pore detection)
    if not args.static_center:
        water_o = u.select_atoms(args.water_sel)
        if len(water_o) == 0:
            print(f"Warning: water selection '{args.water_sel}' matched 0 atoms. "
                  f"Falling back to --static-center.")
            args.static_center = True
        else:
            print(f"Per-frame pore detection: {len(water_o)} water atoms, "
                  f"seed cutoff {args.seed_cutoff} Å")
    else:
        print("Static cylinder centering on cavity centroid for every frame.")

    # Output dirs for optional per-frame .dx files
    cyl_dir = Path(f"{prefix}_cylinder_dx") if args.write_cylinders else None
    if cyl_dir is not None:
        cyl_dir.mkdir(exist_ok=True)
        print(f"Per-frame cylinder .dx files → {cyl_dir}/")
    cav_dir = Path(f"{prefix}_cavity_dx") if (args.save_cavity_dx and fv_mode) else None
    if cav_dir is not None:
        cav_dir.mkdir(exist_ok=True)
        print(f"Per-frame cavity .dx files → {cav_dir}/")
    elif args.save_cavity_dx and not fv_mode:
        print("Note: --save-cavity-dx ignored in static cavity mode "
              "(the .dx you passed already exists).")

    frame_indices = list(range(args.start, args.stop or len(u.trajectory), args.step))

    per_lipid_rows = []
    per_frame_rows = []

    for f_idx in frame_indices:
        u.trajectory[f_idx]
        ts = u.trajectory.ts
        box = u.dimensions[:3]
        box_xy = box[:2]

        # Tilts and P positions
        tilt, raw_tilt, tilt_x, tilt_y, p_x, p_y, p_z = compute_frame_tilts(
            head_groups, tail_groups, p_groups, box
        )
        if args.wrap:
            for k, arr in enumerate([p_x, p_y, p_z]):
                arr[~np.isnan(arr)] = arr[~np.isnan(arr)] % box[k]

        # Leaflet planes (in original frame, for cavity slab + cylinder z-range)
        z_up, z_lo = membrane_planes_original_frame(p_all, box[2])

        # ----- Cavity for this frame -----
        if fv_mode:
            cavity_mask, cav_origin, cav_delta, cav_info = compute_cavity_per_frame(
                lipid_atoms.positions, lipid_radii, box, z_up, z_lo, args
            )
            n_cav_vox = int(cavity_mask.sum())
            if n_cav_vox == 0:
                # No cavity detected this frame — set up empty distance grid
                # so nothing gets flagged. Use a tiny dummy grid.
                dist_grid = np.full(cavity_mask.shape, np.inf, dtype=float)
                anchor_xy = (float(np.nan), float(np.nan))
                cav_centroid_z = float("nan")
            else:
                dist_grid = distance_transform_edt(~cavity_mask, sampling=cav_delta)
                idx = np.argwhere(cavity_mask)
                centroid = cav_origin + (idx.mean(axis=0) + 0.5) * cav_delta
                anchor_xy = (float(centroid[0]), float(centroid[1]))
                cav_centroid_z = float(centroid[2])
            grid_for_lookup = cavity_mask
            origin_for_lookup = cav_origin
            delta_for_lookup = cav_delta
            grid_shape_lookup = cavity_mask.shape

            # Save per-frame cavity .dx if requested
            if cav_dir is not None:
                cav_path = cav_dir / f"cavity_frame{ts.frame:06d}.dx"
                if args.tile_output:
                    tg, to, td = tile_grid_xy(cavity_mask.astype(np.uint8),
                                              cav_origin, cav_delta, box,
                                              n_tiles=3)
                    write_dx(tg, to, td, cav_path)
                else:
                    write_dx(cavity_mask.astype(np.uint8), cav_origin,
                             cav_delta, cav_path)
                if f_idx == frame_indices[0]:
                    print(f"  Wrote first cavity .dx: {cav_path}"
                          f"{' (tiled 3x3)' if args.tile_output else ''}")
        else:
            # Static cavity: reuse pre-computed distance grid
            cavity_mask = static_grid > args.isovalue
            dist_grid = static_dist_grid
            anchor_xy = static_anchor
            cav_centroid_z = float("nan")  # cached
            grid_for_lookup = cavity_mask
            origin_for_lookup = static_origin
            delta_for_lookup = static_delta
            grid_shape_lookup = static_grid.shape
            n_cav_vox = int(cavity_mask.sum())
            cav_info = None

        inv_lookup = 1.0 / delta_for_lookup

        # ----- Pore center (for cylinder criterion) -----
        n_pore_centers = 0; n_pore_waters = 0
        if args.static_center or np.isnan(anchor_xy[0]):
            frame_cx, frame_cy = anchor_xy
        else:
            centers = detect_pore_centers(p_all, water_o, box,
                                          args.seed_cutoff, args.min_seed_size)
            n_pore_centers = len(centers)
            n_pore_waters = sum(c[2] for c in centers)
            chosen = closest_center_to_anchor(centers, anchor_xy, box_xy)
            if chosen[0] is None:
                frame_cx, frame_cy = anchor_xy
            else:
                frame_cx, frame_cy = chosen

        # ----- Criterion 1: in cylinder -----
        valid_p = ~np.isnan(p_x)
        cyl_dist = np.full(n_lipids, np.nan)
        if not np.isnan(frame_cx):
            cyl_dist[valid_p] = xy_distance_pbc(
                p_x[valid_p], p_y[valid_p], frame_cx, frame_cy, box_xy
            )
        in_cylinder = (cyl_dist < args.radius).astype(int)
        in_cylinder[~valid_p] = 0

        # Optional cylinder .dx
        if cyl_dir is not None and not np.isnan(frame_cx):
            cyl_grid, cyl_origin, cyl_delta = build_cylinder_dx(
                frame_cx, frame_cy, args.radius, z_lo, z_up, box,
                delta_xy=args.cylinder_dx_delta,
                delta_z=args.cylinder_dx_delta,
            )
            cyl_path = cyl_dir / f"cylinder_frame{ts.frame:06d}.dx"
            if args.tile_output:
                tg, to, td = tile_grid_xy(cyl_grid, cyl_origin, cyl_delta,
                                          box, n_tiles=3)
                write_dx(tg, to, td, cyl_path)
            else:
                write_dx(cyl_grid, cyl_origin, cyl_delta, cyl_path)
            if f_idx == frame_indices[0]:
                print(f"  Wrote first cylinder .dx: {cyl_path} "
                      f"(z {z_lo:.1f}-{z_up:.1f}, center "
                      f"({frame_cx:.1f}, {frame_cy:.1f}))"
                      f"{' (tiled 3x3)' if args.tile_output else ''}")

        # ----- Criterion 2: close to cavity surface -----
        cavity_dist = np.full(n_lipids, np.nan)
        nx, ny, nz = grid_shape_lookup
        for i in range(n_lipids):
            if not valid_p[i]:
                continue
            ix = int(np.floor((p_x[i] - origin_for_lookup[0]) * inv_lookup[0]))
            iy = int(np.floor((p_y[i] - origin_for_lookup[1]) * inv_lookup[1]))
            iz = int(np.floor((p_z[i] - origin_for_lookup[2]) * inv_lookup[2]))
            if (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                cavity_dist[i] = float(dist_grid[ix, iy, iz])
        close_to_cavity = (cavity_dist <= args.cavity_cutoff).astype(int)
        close_to_cavity[np.isnan(cavity_dist)] = 0

        # ----- Criterion 3: high tilt (per-frame GMM) -----
        valid_t = ~np.isnan(tilt)
        high_tilt = np.zeros(n_lipids, dtype=int)
        prob_high = np.full(n_lipids, np.nan)
        if valid_t.sum() >= 5:
            sub_labels, sub_prob, means, sigmas = cluster_tilts_2comp(tilt[valid_t])
            high_tilt[valid_t] = sub_labels
            prob_high[valid_t] = sub_prob
        else:
            means = np.array([np.nan, np.nan])
            sigmas = np.array([np.nan, np.nan])

        # ----- Combine -----
        # n_criteria_met is always recorded (useful diagnostic) but the
        # selection rule depends on --legacy-combine.
        criteria_sum = in_cylinder + close_to_cavity + high_tilt
        if args.legacy_combine:
            # Old rule: at least --min-criteria of the three flags are set
            selected = (criteria_sum >= args.min_criteria).astype(int)
        else:
            # New default: cylinder is mandatory, plus at least one of the
            # other two. Equivalent: cylinder AND (cavity OR tilt).
            selected = (in_cylinder & (close_to_cavity | high_tilt)).astype(int)

        for i, rid in enumerate(resids):
            per_lipid_rows.append({
                "frame": ts.frame, "time_ps": ts.time, "resid": rid,
                "tilt_deg": (float(tilt[i]) if not np.isnan(tilt[i]) else ""),
                "raw_tilt_deg": (float(raw_tilt[i]) if not np.isnan(raw_tilt[i]) else ""),
                "tilt_x_deg": (float(tilt_x[i]) if not np.isnan(tilt_x[i]) else ""),
                "tilt_y_deg": (float(tilt_y[i]) if not np.isnan(tilt_y[i]) else ""),
                "P_x": (float(p_x[i]) if not np.isnan(p_x[i]) else ""),
                "P_y": (float(p_y[i]) if not np.isnan(p_y[i]) else ""),
                "P_z": (float(p_z[i]) if not np.isnan(p_z[i]) else ""),
                "dist_to_cavity_A": (float(cavity_dist[i])
                                     if not np.isnan(cavity_dist[i]) else ""),
                "xy_dist_to_pore_A": (float(cyl_dist[i])
                                      if not np.isnan(cyl_dist[i]) else ""),
                "prob_high_tilt": (float(prob_high[i])
                                   if not np.isnan(prob_high[i]) else ""),
                "in_cylinder": int(in_cylinder[i]),
                "close_to_cavity": int(close_to_cavity[i]),
                "high_tilt": int(high_tilt[i]),
                "n_criteria_met": int(criteria_sum[i]),
                "selected": int(selected[i]),
            })

        def resid_list(flag_array):
            return sorted({resids[i] for i in range(n_lipids) if flag_array[i] == 1})

        sel_resids = resid_list(selected)
        cyl_resids = resid_list(in_cylinder)
        cav_resids = resid_list(close_to_cavity)
        tilt_resids = resid_list(high_tilt)
        vmd_sel = (f"{args.p_sel} and resid {' '.join(str(r) for r in sel_resids)}"
                   if sel_resids else "")

        per_frame_rows.append({
            # Columns A–H — vmd_sel_selected in column H for easy access
            "frame": ts.frame,
            "time_ps": ts.time,
            "n_lipids": n_lipids,
            "n_in_cylinder": int(in_cylinder.sum()),
            "n_close_to_cavity": int(close_to_cavity.sum()),
            "n_high_tilt": int(high_tilt.sum()),
            "n_selected": len(sel_resids),
            "vmd_sel_selected": vmd_sel,
            # Diagnostic info
            "tilt_bulk_mean": float(means[0]) if not np.isnan(means[0]) else "",
            "tilt_high_mean": float(means[1]) if not np.isnan(means[1]) else "",
            "pore_center_x": float(frame_cx) if not np.isnan(frame_cx) else "",
            "pore_center_y": float(frame_cy) if not np.isnan(frame_cy) else "",
            "anchor_x": float(anchor_xy[0]) if not np.isnan(anchor_xy[0]) else "",
            "anchor_y": float(anchor_xy[1]) if not np.isnan(anchor_xy[1]) else "",
            "n_pore_centers_detected": int(n_pore_centers),
            "n_pore_waters": int(n_pore_waters),
            "cavity_voxels": int(n_cav_vox),
            "cavity_volume_A3": (float(cav_info["pore_volume_A3"])
                                 if cav_info else ""),
            "cavity_is_spanning": (int(cav_info["is_spanning"])
                                   if cav_info else ""),
            "n_cavities_included": (int(cav_info["n_cavities_included"])
                                    if cav_info else ""),
            "resids_selected": ", ".join(str(r) for r in sel_resids),
            "resids_in_cylinder": ", ".join(str(r) for r in cyl_resids),
            "resids_close_to_cavity": ", ".join(str(r) for r in cav_resids),
            "resids_high_tilt": ", ".join(str(r) for r in tilt_resids),
        })

        if ts.frame % 25 == 0 or f_idx == frame_indices[-1]:
            print(f"  frame {ts.frame}: cylinder={int(in_cylinder.sum())}, "
                  f"cavity={int(close_to_cavity.sum())}, "
                  f"tilt={int(high_tilt.sum())}, "
                  f"selected={len(sel_resids)}")

    # ----- Build DataFrames -----
    per_lipid = pd.DataFrame(per_lipid_rows)
    per_frame = pd.DataFrame(per_frame_rows)

    per_lipid = per_lipid.sort_values(
        by=["frame", "selected", "n_criteria_met", "resid"],
        ascending=[True, False, False, True], kind="mergesort"
    ).reset_index(drop=True)

    counts = (per_lipid.groupby("resid")["selected"]
              .agg(["sum", "count"]).reset_index()
              .rename(columns={"sum": "n_frames_selected",
                               "count": "n_frames_total"}))
    counts["fraction_selected"] = counts["n_frames_selected"] / counts["n_frames_total"]

    persistent = counts[counts["fraction_selected"] >= 0.5]["resid"].tolist()
    ever = counts[counts["n_frames_selected"] >= 1]["resid"].tolist()
    summary = pd.DataFrame([
        {"group": "persistently_selected_>=50pct",
         "n_lipids": len(persistent),
         "residue_numbers": ", ".join(str(r) for r in sorted(persistent))},
        {"group": "ever_selected",
         "n_lipids": len(ever),
         "residue_numbers": ", ".join(str(r) for r in sorted(ever))},
    ])

    # ----- Write outputs -----
    per_lipid.to_csv(f"{prefix}_per_lipid.csv", index=False)

    # Per-frame CSV: two parameter rows on top, then header, then data.
    param_names = [
        "mode", "combine_rule",
        "cylinder_radius_A", "cavity_cutoff_A", "min_criteria", "isovalue",
        "head_sel", "tail_sel", "p_sel", "water_sel",
        "seed_cutoff_A", "min_seed_size", "static_center",
        "wrap", "write_cylinders", "cylinder_dx_delta",
        "lipid_sel", "fv_delta_A", "fv_padding_A", "probe_radius_A",
        "span_margin_A", "min_cavity_volume_A3", "no_combine_cavities",
        "use_box", "fv_wrap", "pbc_paint", "pbc_halo", "save_cavity_dx",
        "tile_output",
        "start", "stop", "step",
        "topology", "trajectory", "dx_file",
    ]
    mode_str = "free_volume_per_frame" if fv_mode else "static_dx"
    combine_str = ("legacy: criteria_sum >= min_criteria" if args.legacy_combine
                   else "default: cylinder AND (cavity OR tilt)")
    param_values = [
        mode_str, combine_str,
        args.radius, args.cavity_cutoff, args.min_criteria, args.isovalue,
        args.head_sel, args.tail_sel, args.p_sel, args.water_sel,
        args.seed_cutoff, args.min_seed_size, args.static_center,
        args.wrap, args.write_cylinders, args.cylinder_dx_delta,
        args.lipid_sel, args.fv_delta, args.fv_padding, args.probe_radius,
        args.span_margin, args.min_cavity_volume, args.no_combine_cavities,
        args.use_box, args.fv_wrap, args.pbc_paint, args.pbc_halo,
        args.save_cavity_dx, args.tile_output,
        args.start, args.stop, args.step,
        str(top), str(traj_path) if traj_path else "",
        str(dx_path) if dx_path else "",
    ]
    pf_path = f"{prefix}_per_frame.csv"
    with open(pf_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(param_names)
        w.writerow(param_values)
    per_frame.to_csv(pf_path, mode="a", index=False)

    summary.to_csv(f"{prefix}_resid_groups.csv", index=False)
    counts.sort_values("fraction_selected", ascending=False).to_csv(
        f"{prefix}_lipid_counts.csv", index=False
    )

    print(f"\nWrote:\n  {prefix}_per_lipid.csv\n  {prefix}_per_frame.csv\n"
          f"  {prefix}_resid_groups.csv\n  {prefix}_lipid_counts.csv")
    if cyl_dir is not None:
        print(f"  {cyl_dir}/  (per-frame cylinder .dx)")
    if cav_dir is not None:
        print(f"  {cav_dir}/  (per-frame cavity .dx)")

    if len(per_frame_rows) == 1:
        row = per_frame_rows[0]
        print(f"\nSelected lipids ({row['n_selected']}):")
        print(f"  Resids: {row['resids_selected']}")
        if row['vmd_sel_selected']:
            print(f"  VMD:    {row['vmd_sel_selected']}")


if __name__ == "__main__":
    main()