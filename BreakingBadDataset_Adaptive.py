"""
BreakingBadDataset — PyTorch Geometric-compatible wrapper
=========================================================
CSV columns used:
  path               → OBJ file path (relative to dataset_root)
  class              → object category string  → data.y (label idx)
  caption            → natural language description → data.caption
  parse_error        → skip rows where TRUE
  fragment_location, fracture_surface, missing_piece_size,
  break_type, fragment_guess, confidence
                     → packed into data.meta (JSON-serialisable dict)

data layout (torch_geometric.data.Data):
  data.pos       [N, 3]   FPS-sampled surface points (xyz)
  data.norm      [N, 3]   unit normals at FPS points  (nx,ny,nz)
  data.curv      [N, 6]   curvature features per point
                          (k1, k2, mean, gaussian, curvedness, shape_index)
  data.data      [N, C, D, D]  tangent-disk images  (C channels, D×D grid)
  data.pos_enc   [N, 6]   positional encoding = [xyz_norm | nx,ny,nz]
  data.x         str      object class label
  data.y         int      class index
  data.caption   str
  data.meta      dict     all other CSV fields

Dependencies
------------
  pip install torch torch-geometric open3d numpy scipy pandas trimesh
"""

from __future__ import annotations

import json
import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RBFInterpolator
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    # Sampling
    n_fps: int = 1024           # number of FPS points
    fps_mode: str = "euclidean" # "euclidean" | "geodesic" (geodesic is slow)

    # Tangent disk
    disk_radius_frac: float = 0.05  # radius as fraction of bounding-sphere diameter
    disk_grid: int = 16             # D: output grid is D×D
    disk_channels: int = 6          # curvature channels written into disk image
    disk_fill: str = "rbf"          # "rbf" | "nearest" | "zero"

    # Curvature
    curv_knn: int = 10            # was 30; now controls adaptive radius k, not fixed kNN
    curv_radius_scale: float = 3.0   # NEW
    curv_min_neighbors: int = 10     # NEW
    curv_max_condition: float = 1e6  # NEW

    # Positional encoding
    fourier_bands: int = 0      # 0 = raw xyz; >0 = Fourier features (2*bands*3 dims)

    # Misc
    cache_processed: bool = True  # save .pt next to OBJ for fast reload
    skip_parse_errors: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_mesh(obj_path: Path):
    """Load OBJ with open3d; return (vertices [V,3], triangles [F,3], normals [V,3])."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(obj_path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # ── manifold + normal consistency repair ──────────────────────────────────
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()

    verts  = np.asarray(mesh.vertices,       dtype=np.float32)   # [V,3]
    tris   = np.asarray(mesh.triangles,      dtype=np.int64)     # [F,3]
    norms  = np.asarray(mesh.vertex_normals, dtype=np.float32)   # [V,3]

    # Re-normalise (repair can produce slightly off normals)
    n_len = np.linalg.norm(norms, axis=1, keepdims=True).clip(1e-8)
    norms = norms / n_len

    return verts, tris, norms


def normalize_mesh(verts: np.ndarray):
    """Translate to centroid, scale to unit bounding sphere."""
    centroid = verts.mean(axis=0)
    verts    = verts - centroid
    radius   = np.linalg.norm(verts, axis=1).max()
    if radius > 1e-8:
        verts = verts / radius
    return verts, centroid, radius


def fps_euclidean(verts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """
    Farthest Point Sampling (Euclidean).
    Returns indices of shape [n] into verts.
    """
    rng   = np.random.default_rng(seed)
    V     = len(verts)
    n     = min(n, V)
    sel   = np.zeros(n, dtype=np.int64)
    dist  = np.full(V, np.inf, dtype=np.float64)
    sel[0] = rng.integers(V)
    for i in range(1, n):
        d = np.sum((verts - verts[sel[i - 1]]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        sel[i] = dist.argmax()
    return sel


# ──────────────────────────────────────────────────────────────────────────────
# Curvature — adaptive quadric fitting
# Replaces: estimate_curvature, get_boundary_vertices
# New exports: compute_adaptive_radii, compute_principal_curvatures_adaptive,
#              compute_all_curvature_fields
# ──────────────────────────────────────────────────────────────────────────────

from scipy.spatial import cKDTree   # add to top-of-file imports


def compute_adaptive_radii(
    points: np.ndarray,
    k: int = 10,
    scale: float = 3.0,
) -> np.ndarray:
    """
    Per-point neighbourhood radius = mean distance to k nearest neighbours × scale.

    Parameters
    ----------
    points : (V, 3) float32  — mesh vertices (already normalised)
    k      : int             — number of neighbours used to measure local spacing
    scale  : float           — multiplier; 3.0 gives a disk ~3× the local spacing

    Returns
    -------
    radii : (V,) float32
    """
    tree = cKDTree(points)
    # query k+1 so we can drop self (distance 0)
    dists, _ = tree.query(points, k=k + 1)
    # dists[:, 0] == 0 (self), so take [:, 1:]
    mean_nn_dist = dists[:, 1:].mean(axis=1)          # (V,)
    return (mean_nn_dist * scale).astype(np.float32)


def compute_principal_curvatures_adaptive(
    points: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray,
    min_neighbors: int = 10,
    max_condition: float = 1e6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-vertex principal curvatures via adaptive-radius quadric fitting.

    For each vertex i:
      1. Collect neighbours within radii[i].  If fewer than min_neighbors,
         fall back to the min_neighbors nearest neighbours (same as old kNN
         fallback but condition-gated instead of rank-gated).
      2. Build local tangent frame from SVD of the neighbourhood cloud
         (more robust than the _perp() heuristic for near-degenerate normals).
      3. Fit quadric h = a·u² + b·v² + c·u·v to heights above tangent plane.
      4. Guard: if cond(A) > max_condition skip → leave zeros.
         This replaces the old boundary_mask + post-hoc median smoothing.
      5. Derive kmin, kmax from the 2×2 Hessian trace/determinant.

    Convention kept identical to estimate_curvature:
      returned kmax has |kmax| >= |kmin|   (column 1 = "k1", column 0 = "k2")

    Parameters
    ----------
    points       : (V, 3) float32
    normals      : (V, 3) float32  — unit normals (from open3d, already normalised)
    radii        : (V,)  float32   — per-vertex adaptive radius
    min_neighbors: int             — minimum neighbours; triggers kNN fallback
    max_condition: float           — condition number ceiling for design matrix A

    Returns
    -------
    kmin : (V,) float64   — smaller principal curvature  (may be negative)
    kmax : (V,) float64   — larger  |curvature|          (|kmax| >= |kmin|)
    """
    V    = len(points)
    kmin = np.zeros(V, dtype=np.float64)
    kmax = np.zeros(V, dtype=np.float64)
    tree = cKDTree(points)

    for i in range(V):
        # ── 1. Neighbourhood ─────────────────────────────────────────────────
        idx = tree.query_ball_point(points[i], r=radii[i])
        idx = [j for j in idx if j != i]          # drop self

        if len(idx) < min_neighbors:
            # kNN fallback — guarantees enough points even on sparse regions
            _, idx = tree.query(points[i], k=min_neighbors + 1)
            idx = [j for j in idx.tolist() if j != i]

        nbr_pts = points[np.array(idx)] - points[i]   # (M, 3) centred

        # ── 2. Local tangent frame from SVD ──────────────────────────────────
        # SVD gives the best-fit plane; Vt[-1] is the fitted normal.
        # We then check sign consistency with the open3d normal.
        _, _, Vt = np.linalg.svd(nbr_pts, full_matrices=True)
        fitted_normal = Vt[-1]
        if np.dot(fitted_normal, normals[i]) < 0:
            fitted_normal = -fitted_normal
        t1, t2 = Vt[0], Vt[1]          # tangent axes (orthonormal by SVD)

        # ── 3. Project onto tangent plane & get heights ───────────────────────
        u       = nbr_pts @ t1          # (M,)
        v       = nbr_pts @ t2          # (M,)
        heights = nbr_pts @ fitted_normal  # (M,)

        # Design matrix for quadric:  h ≈ a·u² + b·v² + c·u·v
        # (no linear or constant terms — centred, so they vanish for a smooth surface)
        A = np.column_stack([u**2, v**2, u * v])   # (M, 3)

        # ── 4. Condition guard ────────────────────────────────────────────────
        if np.linalg.cond(A) > max_condition:
            continue                    # leave zeros — same as old rank check
                                        # but catches near-singular, not just rank-0

        coeffs = np.linalg.lstsq(A, heights, rcond=None)[0]
        a, b, c = coeffs               # h ≈ a·u² + b·v² + c·u·v

        # ── 5. Principal curvatures from Hessian eigenvalues ─────────────────
        # Hessian of quadric surface: H = [[2a, c], [c, 2b]]
        trace = 2.0 * (a + b)
        det   = 4.0 * a * b - c ** 2
        disc  = trace ** 2 - 4.0 * det

        if disc < 0:
            disc = 0.0                  # numerical noise — treat as umbilic point

        sq           = np.sqrt(disc)
        ki           = (trace - sq) / 2.0   # smaller eigenvalue
        ka           = (trace + sq) / 2.0   # larger  eigenvalue

        # Keep |kmax| >= |kmin| — same convention as estimate_curvature
        if abs(ka) >= abs(ki):
            kmin[i], kmax[i] = ki, ka
        else:
            kmin[i], kmax[i] = ka, ki

    return kmin, kmax


def compute_all_curvature_fields(
    points: np.ndarray,
    normals: np.ndarray,
    k_neighbors: int = 10,
    radius_scale: float = 3.0,
    min_neighbors: int = 10,
    max_condition: float = 1e6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full curvature pipeline: adaptive radii → principal curvatures → 6 features.

    Drop-in replacement for estimate_curvature().  The only difference in the
    call site is that this returns TWO arrays instead of one.

    Parameters
    ----------
    points       : (V, 3) float32
    normals      : (V, 3) float32
    k_neighbors  : int    — k for adaptive radius estimation (≈ cfg.curv_knn)
    radius_scale : float  — radius = mean_k_dist × scale
    min_neighbors: int    — kNN fallback threshold
    max_condition: float  — condition ceiling

    Returns
    -------
    feats : (V, 6) float32
            columns: [kmin, kmax, mean_curv, gauss_curv, curvedness, shape_index]
            identical column order to estimate_curvature — nothing downstream changes
    radii : (V,)  float32
            per-vertex adaptive radius — pass radii[fps_idx] to build_tangent_disk
    """
    radii = compute_adaptive_radii(points, k=k_neighbors, scale=radius_scale)

    kmin, kmax = compute_principal_curvatures_adaptive(
        points, normals, radii,
        min_neighbors=min_neighbors,
        max_condition=max_condition,
    )

    # ── Derived features (vectorised — no loop) ───────────────────────────────
    eps   = 1e-8
    mean  = (kmin + kmax) / 2.0
    gauss = kmin * kmax
    curv  = np.sqrt((kmin**2 + kmax**2) / 2.0)

    # Shape index: (2/π)·arctan((kmax+kmin)/(kmax−kmin))
    # denom collapses to eps at umbilic points (kmax==kmin) → shape_index = 0
    denom = np.where(np.abs(kmax - kmin) < eps, eps, kmax - kmin)
    si    = (2.0 / np.pi) * np.arctan((kmax + kmin) / denom)

    feats = np.stack([kmin, kmax, mean, gauss, curv, si], axis=1).astype(np.float32)

    # Final safety clip — catches any residual NaN/Inf from degenerate geometry
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return feats, radii

def _perp(n: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to n."""
    t = np.array([1.0, 0.0, 0.0])
    if abs(n @ t) > 0.9:
        t = np.array([0.0, 1.0, 0.0])
    return np.cross(n, t) / np.linalg.norm(np.cross(n, t))



# ──────────────────────────────────────────────────────────────────────────────
# Tangent disk
# ──────────────────────────────────────────────────────────────────────────────

def build_tangent_disk(
    center_vert:  np.ndarray,   # [3]   FPS point
    center_norm:  np.ndarray,   # [3]   normal at FPS point
    center_curv:  np.ndarray,   # [6]   curvature features at center
    curv_k1_dir:  np.ndarray,   # [3]   principal direction for k1 (canonical U)
    nbr_verts:    np.ndarray,   # [K,3] neighbouring vertices (original mesh)
    nbr_curv:     np.ndarray,   # [K,6] curvature at neighbours
    radius:       float,
    grid:         int,
    fill:         str = "rbf",
) -> np.ndarray:
    """
    Project neighbourhood onto the local tangent plane defined by
    (U=k1_dir, V=normal×k1_dir), scatter curvature values onto a D×D grid.

    Returns [6, D, D] float32 image (6 curvature channels).
    """
    # Canonical tangent frame anchored to principal curvature direction
    U = curv_k1_dir / (np.linalg.norm(curv_k1_dir) + 1e-8)
    V = np.cross(center_norm, U)
    V = V / (np.linalg.norm(V) + 1e-8)

    # Project neighbours into (u,v) plane
    delta = nbr_verts - center_vert              # [K,3]
    u_coords = delta @ U                          # [K]
    v_coords = delta @ V                          # [K]

    # Keep only within-radius points
    in_disk = (u_coords**2 + v_coords**2) <= radius**2
    u_c = u_coords[in_disk]
    v_c = v_coords[in_disk]
    feat = nbr_curv[in_disk]                     # [M,6]

    # Add center point itself
    u_c = np.concatenate([[0.0], u_c])
    v_c = np.concatenate([[0.0], v_c])
    feat = np.vstack([center_curv[np.newaxis], feat])

    # Grid coordinates in [-1, 1]
    lin = np.linspace(-1, 1, grid)
    gu, gv = np.meshgrid(lin, lin)               # [D,D]
    gpts = np.column_stack([gu.ravel(), gv.ravel()])

    # Normalise source coords to [-1,1]
    r_inv = 1.0 / (radius + 1e-8)
    src   = np.column_stack([u_c * r_inv, v_c * r_inv])

    disk = np.zeros((6, grid, grid), dtype=np.float32)

    if fill == "rbf" and len(src) >= 4:
        try:
            interp = RBFInterpolator(src, feat, kernel="linear", smoothing=0.1)
            vals   = interp(gpts).reshape(grid, grid, 6)
            disk   = vals.transpose(2, 0, 1).astype(np.float32)
        except Exception:
            fill = "nearest"  # fallback

    if fill == "nearest" or (fill == "rbf" and len(src) < 4):
        if len(src) > 0:
            from scipy.spatial import KDTree
            tree  = KDTree(src)
            _, ii = tree.query(gpts)              # [D*D]
            vals  = feat[ii].reshape(grid, grid, 6)
            disk  = vals.transpose(2, 0, 1).astype(np.float32)

    # "zero" fill → already zeros

    return disk


def principal_k1_direction(verts: np.ndarray, norms: np.ndarray,
                            idx: int, nbr_idx: np.ndarray) -> np.ndarray:
    """
    Estimate the k1 (max principal curvature) direction at vertex `idx`
    using the neighbourhood `nbr_idx`.
    Returns a unit 3-vector in the tangent plane.
    """
    ni  = norms[idx]
    u   = _perp(ni)
    v   = np.cross(ni, u)

    pts = verts[nbr_idx] - verts[idx]
    pu  = pts @ u
    pv  = pts @ v
    pn  = pts @ ni

    A = np.column_stack([pu**2, pu * pv, pv**2])
    if np.linalg.matrix_rank(A) < 3:
        return u  # fallback

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, pn, rcond=None)
    except np.linalg.LinAlgError:
        return u

    a, b, c = coeffs
    H = np.array([[2 * a, b], [b, 2 * c]])
    _, evecs = np.linalg.eigh(H)
    k1_2d = evecs[:, -1]           # eigenvector for largest eigenvalue
    return k1_2d[0] * u + k1_2d[1] * v


# ──────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ──────────────────────────────────────────────────────────────────────────────

def positional_encoding(pos: np.ndarray, norms: np.ndarray,
                        fourier_bands: int = 0) -> np.ndarray:
    """
    pos   [N,3] normalised xyz
    norms [N,3] unit normals

    Returns [N, 6] if fourier_bands==0 else [N, 6 + 2*bands*3].
    """
    if fourier_bands == 0:
        return np.concatenate([pos, norms], axis=1).astype(np.float32)

    freqs = 2.0 ** np.arange(fourier_bands) * np.pi   # [B]
    # pos: [N,3] → [N, 2*B*3]
    angles = pos[:, :, None] * freqs[None, None, :]    # [N,3,B]
    fourier = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
    fourier = fourier.reshape(len(pos), -1)
    return np.concatenate([pos, norms, fourier], axis=1).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Per-mesh processing
# ──────────────────────────────────────────────────────────────────────────────

def process_mesh(obj_path: Path, cfg: DatasetConfig) -> dict:
    """
    Full pipeline for one OBJ file.
    Returns a dict ready to be packed into a Data object.
    """
    cache_path = obj_path.with_suffix(".pt")
    print(cache_path)
    if cfg.cache_processed and cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    # 1. Load & repair
    verts, tris, norms = load_mesh(obj_path)

    # 2. Normalise to unit bounding sphere
    verts, _centroid, _radius_orig = normalize_mesh(verts)

    # 3. REMOVED: bnd_mask = get_boundary_vertices(verts, tris)
    #    compute_all_curvature_fields handles ill-conditioning internally

    # 4. Per-vertex curvature — UPGRADED
    curv_full, radii = compute_all_curvature_fields(
        verts, norms,
        k_neighbors  = cfg.curv_knn,     # same config knob, same meaning
        radius_scale = 3.0,              # new: controls adaptive radius spread
        min_neighbors = 10,
        max_condition = 1e6,
    )
    # curv_full: [V, 6]  same layout as before — k1,k2,mean,gauss,curvedness,shape_index
    # radii:     [V]     per-vertex adaptive neighbourhood radius (NEW — used below)


    # 5. FPS
    fps_idx = fps_euclidean(verts, cfg.n_fps)
    fps_verts = verts[fps_idx]     # [N,3]
    fps_norms = norms[fps_idx]     # [N,3]
    fps_curv  = curv_full[fps_idx] # [N,6]
    fps_radii = radii[fps_idx]   
    

    # 6. Disk radius — UPGRADED: per-point instead of one global value
    from scipy.spatial import KDTree
    mesh_tree = KDTree(verts)
    # Keep the old global as a floor so very dense regions still get coverage
    global_disk_r = cfg.disk_radius_frac * 2.0   # bsphere radius = 1.0

    # 7. Build tangent disks [N, 6, D, D] — loop body changes, signature unchanged
    D     = cfg.disk_grid
    disks = np.zeros((len(fps_idx), cfg.disk_channels, D, D), dtype=np.float32)

    for i in range(len(fps_idx)):
        # UPGRADED: use per-point adaptive radius, floored at global_disk_r
        disk_r  = float(max(fps_radii[i], global_disk_r))
        nbr_radius = disk_r * 1.5

        nbr_i = np.array(mesh_tree.query_ball_point(fps_verts[i], r=nbr_radius),
                          dtype=np.int64)
        if len(nbr_i) < 3:
            _, nbr_i = mesh_tree.query(fps_verts[i], k=min(30, len(verts)))
            nbr_i = np.array(nbr_i)

        k1_dir = principal_k1_direction(verts, norms, fps_idx[i], nbr_i)

        disks[i] = build_tangent_disk(
            center_vert = fps_verts[i],
            center_norm = fps_norms[i],
            center_curv = fps_curv[i],
            curv_k1_dir = k1_dir,
            nbr_verts   = verts[nbr_i],
            nbr_curv    = curv_full[nbr_i],
            radius      = disk_r,          # NOW per-point, was global_disk_r
            grid        = D,
            fill        = cfg.disk_fill,
        )

    # 8. Positional encoding  [N, 6 or more]
    pos_enc = positional_encoding(fps_verts, fps_norms, cfg.fourier_bands)
    fps_curv_norm = robust_norm(fps_curv)
    
    # Also normalise disk channels across spatial dims
    for c in range(cfg.disk_channels):
        ch = disks[:, c, :, :]
        med = np.median(ch)
        mad = max(np.median(np.abs(ch - med)), 1e-8) * 1.4826
        disks[:, c, :, :] = (ch - med) / mad

    result = dict(
        pos     = torch.from_numpy(fps_verts).float(),          # [N,3]
        norm    = torch.from_numpy(fps_norms).float(),          # [N,3]
        curv    = torch.from_numpy(fps_curv_norm).float(),      # [N,6]
        data    = torch.from_numpy(disks).float(),              # [N,6,D,D]
        pos_enc = torch.from_numpy(pos_enc).float(),            # [N,6+]
    )

    if cfg.cache_processed:
        torch.save(result, cache_path)

    return result
    
def robust_norm(x):  # move outside process_mesh so it's reusable
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0).clip(1e-8)
    return (x - med) / (mad * 1.4826)

# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

class MeshData:
    """
    Lightweight data container (no torch_geometric dependency required).

    Attributes
    ----------
    pos      [N,3]      FPS xyz (normalised)
    norm     [N,3]      normals
    curv     [N,6]      curvature features (robust-normalised)
    data     [N,C,D,D]  tangent disk images
    pos_enc  [N,P]      positional encoding
    x        str        class label string
    y        int        class label index
    caption  str
    meta     dict       all other CSV metadata
    """
    __slots__ = ("pos", "norm", "curv", "data", "pos_enc",
                 "x", "y", "caption", "meta")

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        shapes = {k: tuple(getattr(self, k).shape)
                  for k in ("pos", "norm", "curv", "data", "pos_enc")}
        return (f"MeshData(y={self.y}, x={self.x!r}, "
                f"caption={self.caption[:40]!r}…\n  tensors={shapes})")

    def to(self, device):
        """Move all tensors to device."""
        for k in ("pos", "norm", "curv", "data", "pos_enc"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def token_sequence(self) -> torch.Tensor:
        """
        Flatten tangent disk images and concatenate positional encoding.
        Returns [N, C*D*D + P] ready for a linear projection into a transformer.
        """
        N = self.data.shape[0]
        flat_disk = self.data.view(N, -1)          # [N, C*D*D]
        return torch.cat([flat_disk, self.pos_enc], dim=1)  # [N, C*D*D + P]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

META_COLS = [
    "fragment_location", "fracture_surface", "missing_piece_size",
    "break_type", "fragment_guess", "confidence",
]

class BreakingBadDataset(Dataset):
    """
    PyTorch Dataset wrapping the Breaking Bad fracture CSV.

    Parameters
    ----------
    csv_path     : path to the annotations CSV
    dataset_root : root directory; OBJ paths in the CSV are joined here
    cfg          : DatasetConfig
    transform    : optional callable applied to MeshData before returning
    classes      : optional list of class names to keep (None = all)
    """

    def __init__(
        self,
        csv_path: str | Path,
        dataset_root: str | Path,
        cfg: Optional[DatasetConfig] = None,
        transform=None,
        classes: Optional[list[str]] = None,
    ):
        self.root      = Path(dataset_root)
        self.cfg       = cfg or DatasetConfig()
        self.transform = transform

        df = pd.read_csv(csv_path, sep=",", dtype=str)
        df.columns = df.columns.str.strip()
        # print(df.columns)

        # ── Parse error filter ────────────────────────────────────────────────
        n_before = len(df)
        if self.cfg.skip_parse_errors:
            df = df[df["parse_error"].str.upper().str.strip() != "TRUE"]
            n_skip = n_before - len(df)
            if n_skip:
                log.warning(f"Skipped {n_skip} rows with parse_error=TRUE")
        df = df[:-1]
        # ── Class filter ──────────────────────────────────────────────────────
        if classes is not None:
            df = df[df["class"].isin(classes)]

        # ── Build class index ─────────────────────────────────────────────────
        print(df["class"].unique().tolist())
        all_classes = sorted(df["class"].unique().tolist())
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(all_classes)}
        self.idx_to_class: list[str] = all_classes

        self.records = df.reset_index(drop=True)
        log.info(f"Dataset: {len(self.records)} samples, "
                 f"{len(all_classes)} classes: {all_classes[:8]}{'…' if len(all_classes)>8 else ''}")

    # ── Accessors ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> MeshData:
        row = self.records.iloc[idx]

        obj_path = self.root / row["path"].strip()
        if not obj_path.exists():
            raise FileNotFoundError(f"OBJ not found: {obj_path}")

        # Process mesh → tensors
        tensors = process_mesh(obj_path, self.cfg)

        # Metadata
        meta = {}
        for col in META_COLS:
            if col in row:
                meta[col] = row[col]

        # images_used: stored as JSON list string
        try:
            images_used = ast.literal_eval(row.get("images_used", "[]"))
        except Exception:
            images_used = []
        meta["images_used"] = images_used

        # Caption (may be JSON-escaped)
        caption = _clean_caption(row.get("caption", ""))

        sample = MeshData(
            **tensors,
            x       = row["class"],
            y       = self.class_to_idx[row["class"]],
            caption = caption,
            meta    = meta,
        )

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ── Utilities ─────────────────────────────────────────────────────────────

    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def token_dim(self) -> int:
        """Feature dimension per token = C*D*D + P (for linear projection)."""
        C = self.cfg.disk_channels
        D = self.cfg.disk_grid
        P = 6 + (2 * self.cfg.fourier_bands * 3 if self.cfg.fourier_bands else 0)
        return C * D * D + P

    def collate_fn(self, batch: list[MeshData]) -> dict:
        """
        Stack a list of MeshData into a batch dict with tensors [B, N, ...].
        All meshes must have the same N (guaranteed by fixed cfg.n_fps).
        """
        return {
            "pos"     : torch.stack([s.pos     for s in batch]),   # [B,N,3]
            "norm"    : torch.stack([s.norm    for s in batch]),   # [B,N,3]
            "curv"    : torch.stack([s.curv    for s in batch]),   # [B,N,6]
            "data"    : torch.stack([s.data    for s in batch]),   # [B,N,C,D,D]
            "pos_enc" : torch.stack([s.pos_enc for s in batch]),   # [B,N,P]
            "tokens"  : torch.stack([s.token_sequence() for s in batch]), # [B,N,T]
            "y"       : torch.tensor([s.y for s in batch], dtype=torch.long),
            "x"       : [s.x       for s in batch],
            "caption" : [s.caption for s in batch],
            "meta"    : [s.meta    for s in batch],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_caption(raw: str) -> str:
    """Strip markdown fences or outer JSON if the caption is double-encoded."""
    raw = raw.strip()
    # Strip ```json ... ``` fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    # Try to parse as JSON dict and extract caption key
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "caption" in obj:
                return obj["caption"]
        except json.JSONDecodeError:
            pass
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    csv_path     = sys.argv[1] if len(sys.argv) > 1 else "annotations.csv"
    dataset_root = sys.argv[2] if len(sys.argv) > 2 else "."

    cfg = DatasetConfig(
        n_fps           = 512,
        disk_grid       = 16,
        disk_channels   = 6,
        disk_fill       = "rbf",
        curv_knn        = 20,
        cache_processed = True,
        fourier_bands   = 4,
    )

    ds = BreakingBadDataset(csv_path, dataset_root, cfg=cfg)
    print(f"\nDataset size : {len(ds)}")
    print(f"Classes      : {ds.idx_to_class}")
    print(f"Token dim    : {ds.token_dim()}")

    sample = ds[0]
    print(f"\nSample 0:\n{sample}")
    print(f"  token_sequence : {sample.token_sequence().shape}")
    print(f"  meta keys      : {list(sample.meta.keys())}")

    # DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn, shuffle=True)
    batch  = next(iter(loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:10s} {tuple(v.shape)}")
        else:
            print(f"  {k:10s} {v}")