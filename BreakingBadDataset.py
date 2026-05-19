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
    curv_knn: int = 30          # neighbours for curvature estimation
    boundary_sigma: float = 2.0 # stddev multiplier for boundary masking

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


def estimate_curvature(verts: np.ndarray, norms: np.ndarray, knn: int = 30,
                       boundary_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimate per-vertex principal curvatures and derived features.

    Returns [V, 6]: (k1, k2, mean_curv, gauss_curv, curvedness, shape_index)

    Strategy: quadric fitting in local tangent frame for each vertex's kNN.
    Boundary vertices are smoothed post-hoc to suppress blow-up.
    """
    from scipy.spatial import KDTree

    V    = len(verts)
    curv = np.zeros((V, 6), dtype=np.float32)
    tree = KDTree(verts)

    _, nbrs = tree.query(verts, k=knn + 1)   # includes self
    nbrs = nbrs[:, 1:]                         # drop self

    for i in range(V):
        ni   = norms[i]
        pts  = verts[nbrs[i]] - verts[i]       # [k,3] centred

        # Build tangent frame (u,v) orthonormal to normal
        u = _perp(ni)
        v = np.cross(ni, u)

        # Project neighbours onto tangent plane
        pu = pts @ u   # [k]
        pv = pts @ v   # [k]
        pn = pts @ ni  # [k]  height above tangent plane

        # Fit quadric: h = a*pu^2 + b*pu*pv + c*pv^2
        A  = np.column_stack([pu**2, pu * pv, pv**2])
        if np.linalg.matrix_rank(A) < 3:
            continue  # degenerate neighbourhood → leave zeros
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, pn, rcond=None)
        except np.linalg.LinAlgError:
            continue

        a, b, c = coeffs
        # Shape operator eigenvalues → principal curvatures
        # Hessian of quadric = [[2a, b], [b, 2c]]
        H  = np.array([[2 * a, b], [b, 2 * c]])
        k1, k2 = np.linalg.eigvalsh(H)   # sorted ascending
        # Convention: |k1| >= |k2|
        if abs(k1) < abs(k2):
            k1, k2 = k2, k1

        mean_k   = (k1 + k2) / 2.0
        gauss_k  = k1 * k2
        curved   = np.sqrt((k1**2 + k2**2) / 2.0)
        denom    = k1 - k2
        if abs(denom) > 1e-8:
            si = (2.0 / np.pi) * np.arctan((k1 + k2) / denom)
        else:
            si = 0.0

        curv[i] = [k1, k2, mean_k, gauss_k, curved, si]

    # ── Boundary smoothing: replace blow-up values with neighbour median ──────
    if boundary_mask is not None:
        bad = boundary_mask
        for feat in range(6):
            col = curv[:, feat]
            # also flag statistical outliers
            q1, q3 = np.percentile(col, [5, 95])
            outlier = (col < q1 - 3 * (q3 - q1)) | (col > q3 + 3 * (q3 - q1))
            mask = bad | outlier
            for i in np.where(mask)[0]:
                col[i] = np.median(col[nbrs[i]])
        # curv[:, :] = col  # already modified in place via view

    # Clip to finite
    curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
    return curv


def _perp(n: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to n."""
    t = np.array([1.0, 0.0, 0.0])
    if abs(n @ t) > 0.9:
        t = np.array([0.0, 1.0, 0.0])
    return np.cross(n, t) / np.linalg.norm(np.cross(n, t))


def get_boundary_vertices(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Return boolean mask of boundary (open-edge) vertices."""
    from collections import defaultdict
    edge_count = defaultdict(int)
    for tri in tris:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge_count[tuple(sorted((a, b)))] += 1
    boundary_edges = {e for e, c in edge_count.items() if c == 1}
    mask = np.zeros(len(verts), dtype=bool)
    for a, b in boundary_edges:
        mask[a] = mask[b] = True
    return mask


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


def _cache_path(obj_path: Path, cfg: DatasetConfig) -> Path:
    """
    Build a per-config cache filename next to the OBJ:
 
        piece_0__n1024_g16_c6_r050_rbf_k30_f0.pt
 
    Format: <stem>__n<n_fps>_g<disk_grid>_c<disk_channels>
                   _r<radius_frac*1000:03d>_<disk_fill>
                   _k<curv_knn>_f<fourier_bands>.pt
 
    Each config produces a distinct file, so multiple caches coexist
    in the same directory without collision or staleness issues.
    """
    r   = int(round(cfg.disk_radius_frac * 1000))   # e.g. 0.05 -> 050
    tag = (
        f"n{cfg.n_fps}"
        f"_g{cfg.disk_grid}"
        f"_c{6}"
        f"_r{r:03d}"
        f"_{cfg.disk_fill}"
        f"_k{cfg.curv_knn}"
        f"_f{cfg.fourier_bands}"
    )
    return obj_path.with_name(f"{obj_path.stem}__{tag}.pt")

# ──────────────────────────────────────────────────────────────────────────────
# Per-mesh processing
# ──────────────────────────────────────────────────────────────────────────────

def process_mesh(obj_path: Path, cfg: DatasetConfig) -> dict:
    """
    Full pipeline for one OBJ file.
    Returns a dict ready to be packed into a Data object.
    """
    cache_path = _cache_path(obj_path, cfg)
    if cfg.cache_processed and cache_path.exists():
        # print(f"using : {cache_path}")
        return torch.load(cache_path, weights_only=False)   
    # 1. Load & repair
    verts, tris, norms = load_mesh(obj_path)

    # 2. Normalise to unit bounding sphere
    verts, _centroid, _radius_orig = normalize_mesh(verts)

    # 3. Boundary mask (fracture surface)
    bnd_mask = get_boundary_vertices(verts, tris)

    # 4. Per-vertex curvature (full mesh)
    curv_full = estimate_curvature(verts, norms, knn=cfg.curv_knn,
                                   boundary_mask=bnd_mask)

    # 5. FPS
    fps_idx = fps_euclidean(verts, cfg.n_fps)
    fps_verts = verts[fps_idx]     # [N,3]
    fps_norms = norms[fps_idx]     # [N,3]
    fps_curv  = curv_full[fps_idx] # [N,6]

    # 6. Disk radius tied to scale
    from scipy.spatial import KDTree
    mesh_tree  = KDTree(verts)

    bsphere_r  = 1.0              # already normalised
    disk_r     = cfg.disk_radius_frac * 2 * bsphere_r

    # 7. Build tangent disks  [N, 6, D, D]
    D     = cfg.disk_grid
    disks = np.zeros((len(fps_idx), cfg.disk_channels, D, D), dtype=np.float32)

    # Query full-mesh neighbours for each FPS point (use all mesh vertices)
    # We use a larger radius to guarantee enough coverage
    nbr_radius = disk_r * 1.5
    nbr_lists  = mesh_tree.query_ball_point(fps_verts, r=nbr_radius)

    for i in range(len(fps_idx)):
        nbr_i = np.array(nbr_lists[i], dtype=np.int64)
        if len(nbr_i) < 3:
            # Fallback: kNN
            _, nbr_i = mesh_tree.query(fps_verts[i], k=min(30, len(verts)))
            nbr_i = np.array(nbr_i)

        k1_dir = principal_k1_direction(verts, norms, fps_idx[i], nbr_i)

        disks[i] = build_tangent_disk(
            center_vert  = fps_verts[i],
            center_norm  = fps_norms[i],
            center_curv  = fps_curv[i],
            curv_k1_dir  = k1_dir,
            nbr_verts    = verts[nbr_i],
            nbr_curv     = curv_full[nbr_i],
            radius       = disk_r,
            grid         = D,
            fill         = cfg.disk_fill,
        )

    # 8. Positional encoding  [N, 6 or more]
    pos_enc = positional_encoding(fps_verts, fps_norms, cfg.fourier_bands)

    # 9. Normalise curvature channels per-feature (robust z-score)
    def robust_norm(x):
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med), axis=0).clip(1e-8)
        return (x - med) / (mad * 1.4826)

    fps_curv_norm = robust_norm(fps_curv)
    # Also normalise disk channels across spatial dims
    for c in range(6):
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
        log.info(f"Cached → {cache_path} ")
    return result


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
        example = Path("example.obj")
        log.info(f"Example cache: {_cache_path(example, self.cfg)}")

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