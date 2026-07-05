"""
Pore water analysis for a lipid bilayer with pore(s).

Strategy:
  1. Compute membrane half-width automatically from mean P-P distance.
  2. Find waters in the membrane.
  3. Locate the pore center(s) in xy from those waters' positions.
  4. Count ALL waters inside a cylinder around each pore center,
     between the leaflets. This catches pore waters even
     if the column is broken into disconnected fragments.

Outputs:
  - pore_metrics.csv : time series of pore metrics (one row per frame)
  - pore_waters.csv  : list of (frame, time, resid, pore_id) for every
                       water flagged as pore water

Usage:
    python pore_water_analysis.py system.gro traj.xtc   # trajectory
    python pore_water_analysis.py system.gro            # single frame
"""

import sys
import numpy as np
import pandas as pd
import MDAnalysis as mda
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


# ---------- user-tunable parameters ----------
CYLINDER_RADIUS  = 17.0    # Å, radius of pore-counting cylinder
SEED_CUTOFF      = 14.0    # Å, xy-distance to merge slab waters into pore seeds
MIN_SEED_SIZE    = 2      # waters needed to define a pore center
LIPID_P_SEL      = "name P"
WATER_O_SEL      = "resname SOL TIP3 W and name OW OH2 O"
# ---------------------------------------------


def split_leaflets(p_atoms, box_z=None):
    """
    Split phosphate atoms into upper/lower leaflet by z.
 
    If box_z is given, first unwrap z so the membrane is contiguous even
    when it straddles the periodic boundary. We do this by shifting z so
    the largest gap in the sorted z-distribution lies at the box edge.
    """
    z = p_atoms.positions[:, 2].copy()
 
    if box_z is not None:
        # find largest gap in sorted z (with wrap-around). shift so that
        # gap sits at z=0/box_z, leaving the membrane intact in between.
        zs = np.sort(z)
        gaps = np.diff(zs)
        wrap_gap = (zs[0] + box_z) - zs[-1]
        all_gaps = np.append(gaps, wrap_gap)
        i = np.argmax(all_gaps)
        if i < len(gaps):
            shift = -zs[i + 1]    # put the gap's upper edge at z=0
        else:
            shift = -zs[0]        # wrap_gap is biggest -> already aligned
        z = (z + shift) % box_z
 
    zmean = z.mean()
    upper_mask = z > zmean
    upper = p_atoms[upper_mask]
    lower = p_atoms[~upper_mask]
    return upper, lower, z  # also return shifted z for downstream use
 
 
def membrane_geometry(p_atoms, box_z):
    """
    Return (midplane, slab_half_width, z_shift).
 
    z_shift must be added to any other z-coordinate (e.g., water z) and
    then mod-box_z to put it in the same unwrapped frame as the membrane.
    """
    upper, lower, z_shifted = split_leaflets(p_atoms, box_z=box_z)
    # which atoms got which (uses indices from p_atoms ordering)
    z_up = z_shifted[np.isin(p_atoms.indices, upper.indices)].mean()
    z_lo = z_shifted[np.isin(p_atoms.indices, lower.indices)].mean()
    mid = 0.5 * (z_up + z_lo)
    thickness = z_up - z_lo
    slab_half = thickness / 2.2
 
    # work out the shift that was applied inside split_leaflets so callers
    # can reproduce it. we re-derive it from one P atom.
    z_orig = p_atoms.positions[0, 2]
    z_new = z_shifted[0]
    z_shift = (z_new - z_orig) % box_z
    return mid, slab_half, z_shift
 
 
def pbc_pdist_xy(xy, box_xy):
    """Pairwise xy distances under minimum-image convention."""
    n = len(xy)
    out = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n - 1):
        dx = xy[i + 1:, 0] - xy[i, 0]
        dy = xy[i + 1:, 1] - xy[i, 1]
        dx -= box_xy[0] * np.round(dx / box_xy[0])
        dy -= box_xy[1] * np.round(dy / box_xy[1])
        d = np.sqrt(dx * dx + dy * dy)
        out[k:k + len(d)] = d
        k += len(d)
    return out
 
 
def circular_mean(coords, box_len):
    """Mean of 1D periodic coordinates using the circular-mean trick."""
    angles = 2 * np.pi * coords / box_len
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return (mean_angle / (2 * np.pi)) * box_len % box_len
 
 
def find_pore_centers(slab_xy, cutoff, min_size, box_xy):
    """
    Cluster slab waters in xy (with PBC) to find pore locations.
    Returns list of (cx, cy, n_seed_waters) tuples.
    """
    if len(slab_xy) == 0:
        return []
    if len(slab_xy) == 1:
        return [(slab_xy[0, 0], slab_xy[0, 1], 1)] if min_size <= 1 else []
 
    d = pbc_pdist_xy(slab_xy, box_xy)
    Z = linkage(d, method="single")
    labels = fcluster(Z, t=cutoff, criterion="distance")
 
    centers = []
    for cl in np.unique(labels):
        m = labels == cl
        if m.sum() >= min_size:
            cx = circular_mean(slab_xy[m, 0], box_xy[0])
            cy = circular_mean(slab_xy[m, 1], box_xy[1])
            centers.append((cx, cy, int(m.sum())))
    return centers
 
 
def waters_in_cylinder(water_o, o_pos_shifted, cx, cy, radius,
                       z_lo, z_hi, box_xy):
    """
    Return resids of waters within radius (xy, with PBC) of (cx, cy)
    and within the slab z_lo < z < z_hi. o_pos_shifted has already had
    the membrane z-shift applied.
    """
    dx = o_pos_shifted[:, 0] - cx
    dy = o_pos_shifted[:, 1] - cy
    dx -= box_xy[0] * np.round(dx / box_xy[0])
    dy -= box_xy[1] * np.round(dy / box_xy[1])
    in_xy = (dx * dx + dy * dy) < radius * radius
    in_z = (o_pos_shifted[:, 2] > z_lo) & (o_pos_shifted[:, 2] < z_hi)
    mask = in_xy & in_z
    return water_o[mask].resids
 
 
def analyze_frame(u, p_atoms, water_o):
    """Return metrics dict and list of (resid, pore_id) for this frame."""
    box = u.dimensions
    box_xy = box[:2]
    box_z = box[2]
 
    mid, slab_half, z_shift = membrane_geometry(p_atoms, box_z)
    z_top = mid + slab_half
    z_bot = mid - slab_half
 
    # apply same z-shift to waters so they live in the same frame as the
    # (un-wrapped) membrane
    o_pos = water_o.positions.copy()
    o_pos[:, 2] = (o_pos[:, 2] + z_shift) % box_z
 
    in_slab = (o_pos[:, 2] > z_bot) & (o_pos[:, 2] < z_top)
    slab_xy = o_pos[in_slab, :2]
 
    # find pore centers in xy (PBC-aware)
    centers = find_pore_centers(slab_xy, SEED_CUTOFF, MIN_SEED_SIZE, box_xy)
 
    metrics = {
        "n_slab_waters_total": int(in_slab.sum()),
        "slab_half_width": float(slab_half),
        "midplane_z": float(mid),
        "n_pores": len(centers),
        "n_pore_waters": 0,
        "cylinder_radius": CYLINDER_RADIUS,
    }
    pore_water_records = []
 
    for pore_id, (cx, cy, _) in enumerate(centers, start=1):
        cyl_resids = waters_in_cylinder(
            water_o, o_pos, cx, cy, CYLINDER_RADIUS, z_bot, z_top, box_xy
        )
        metrics[f"pore{pore_id}_x"] = float(cx)
        metrics[f"pore{pore_id}_y"] = float(cy)
        metrics[f"pore{pore_id}_nwaters"] = len(cyl_resids)
        metrics["n_pore_waters"] += len(cyl_resids)
        for rid in cyl_resids:
            pore_water_records.append((int(rid), pore_id))
 
    return metrics, pore_water_records
 
 
def main(topology, trajectory=None):
    if trajectory is None:
        u = mda.Universe(topology)
        print(f"Loaded {topology} (single frame).")
    else:
        u = mda.Universe(topology, trajectory)
        print(f"Loaded {topology} + {trajectory} ({len(u.trajectory)} frames).")
 
    p_atoms = u.select_atoms(LIPID_P_SEL)
    water_o = u.select_atoms(WATER_O_SEL)
 
    if len(p_atoms) == 0:
        raise SystemExit(f"No P atoms found with selection '{LIPID_P_SEL}'")
    if len(water_o) == 0:
        raise SystemExit(f"No water O atoms found with selection "
                         f"'{WATER_O_SEL}'.")
 
    # report geometry once at the start
    mid, slab_half, _ = membrane_geometry(p_atoms, u.dimensions[2])
    print(f"Membrane: thickness {2*slab_half:.1f} Å, "
          f"midplane z={mid:.1f}, slab ±{slab_half:.1f} Å")
    print(f"Pore cylinder radius: {CYLINDER_RADIUS} Å")
 
    metric_rows = []
    pore_water_rows = []
 
    for ts in u.trajectory:
        metrics, pore_records = analyze_frame(u, p_atoms, water_o)
        row = {"frame": ts.frame, "time_ps": ts.time, **metrics}
        metric_rows.append(row)
 
        # group resids per pore for this frame
        by_pore = {}
        for rid, pid in pore_records:
            by_pore.setdefault(pid, []).append(rid)
        all_resids = sorted({rid for rid, _ in pore_records})
 
        frame_row = {
            "frame": ts.frame,
            "time_ps": ts.time,
            "n_pore_waters": len(all_resids),
            "resids": " ".join(map(str, all_resids)),
        }
        # one column per pore so you can separate them if needed
        for pid, resids in sorted(by_pore.items()):
            frame_row[f"pore{pid}_resids"] = " ".join(
                map(str, sorted(set(resids)))
            )
        pore_water_rows.append(frame_row)
 
        if ts.frame % 50 == 0:
            print(f"  frame {ts.frame}: {metrics['n_pores']} pore(s), "
                  f"{metrics['n_pore_waters']} pore waters")
 
    pd.DataFrame(metric_rows).to_csv("pore_metrics.csv", index=False)
    pd.DataFrame(pore_water_rows).to_csv("pore_waters.csv", index=False)
    print("\nWrote pore_metrics.csv and pore_waters.csv")
 
    # for single-frame, print resids and a VMD selection string
    if trajectory is None and pore_water_rows:
        row = pore_water_rows[0]
        for key, val in row.items():
            if key.startswith("pore") and key.endswith("_resids") and val:
                pid = key.replace("pore", "").replace("_resids", "")
                resids = val.split()
                print(f"\nPore {pid}: {len(resids)} waters")
                print(f"  resids: {val}")
                print(f'  VMD: resname SOL and resid {val}')
 
 
if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python pore_water_analysis.py system.gro [traj.xtc]")
        sys.exit(1)
 