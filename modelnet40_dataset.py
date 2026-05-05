"""
modelnet40_dataset.py
=====================
PyTorch Dataset for ModelNet40 classification loaded from a CSV index file.

CSV format (tab-separated, header required):
    object_id        class      split   object_path
    airplane_0627    airplane   test    airplane/test/airplane_0627.off
    …

`object_path` is relative to `root`.  Comma-separated CSVs are also accepted
(delimiter is auto-detected from the header line).

Returned dict per __getitem__:
    "points"  FloatTensor  (n_points, 3)
    "normals" FloatTensor  (n_points, 3)
    "feats"   FloatTensor  (n_points, 6)   — curvature features (compute_feats=True)
    "radius"  FloatTensor  (n_points,)     — adaptive radius per point
    "label"   LongTensor   scalar          — class index
    "name"    str                          — object_id from CSV
    "index"   int                          — dataset index

Extra methods (call after preprocess_all):
    statistic_feats()
        → computes global [max, mean, min] per feature channel; stored in self.stats

    _export_to_image(channel, n_pix, out_dir)
        → projects each point's neighborhood onto its local tangent plane and
          saves a disk-shaped PNG (greyscale for 1 channel, RGB for 3 channels)
"""

from __future__ import annotations

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal, Optional, Tuple, List
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from tqdm import tqdm


# ── geometry helpers ──────────────────────────────────────────────────────────

def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals."""
    normals = np.zeros_like(vertices, dtype=np.float32)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return (normals / norm).astype(np.float32)


def sample_points_uniform(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Area-weighted random surface sampling. Returns (points, normals)."""
    rng = np.random.default_rng(seed)
    vnormals = compute_vertex_normals(vertices, faces)

    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    n0, n1, n2 = vnormals[faces[:, 0]], vnormals[faces[:, 1]], vnormals[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = areas.sum()

    if total == 0:
        idx = rng.integers(0, len(vertices), size=n_points)
        return vertices[idx].astype(np.float32), vnormals[idx].astype(np.float32)

    probs = areas / total

    cf = rng.choice(len(faces), size=n_points, p=probs)

    r1, r2 = rng.random(n_points), rng.random(n_points)
    sqrt_r1 = np.sqrt(r1)
    u, v, w = 1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2

    points = (
        u[:, None] * vertices[faces[cf, 0]]
        + v[:, None] * vertices[faces[cf, 1]]
        + w[:, None] * vertices[faces[cf, 2]]
    ).astype(np.float32)

    normals = u[:, None] * n0[cf] + v[:, None] * n1[cf] + w[:, None] * n2[cf]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return points, normals.astype(np.float32)


def compute_adaptive_radii(
    points: np.ndarray, k: int = 10, scale: float = 3.0
) -> np.ndarray:
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    return (scale * dists[:, 1:].mean(axis=1)).astype(np.float32)


def compute_principal_curvatures_adaptive(
    points: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray,
    min_neighbors: int = 10,
    max_condition: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(points)
    kmin = np.zeros(N, dtype=np.float64)
    kmax = np.zeros(N, dtype=np.float64)
    tree = cKDTree(points)

    for i in range(N):
        idx = tree.query_ball_point(points[i], r=radii[i])
        if len(idx) < min_neighbors:
            _, idx = tree.query(points[i], k=min_neighbors + 1)
            idx = idx.tolist()

        Q_c = points[np.array(idx)] - points[i]
        _, _, Vt = np.linalg.svd(Q_c, full_matrices=True)
        normal = Vt[-1]
        if np.dot(normal, normals[i]) < 0:
            normal = -normal
        t1, t2 = Vt[0], Vt[1]

        heights = Q_c @ normal
        proj = Q_c - np.outer(heights, normal)
        x, y = proj @ t1, proj @ t2

        A = np.column_stack([x**2, y**2, x * y, np.ones(len(x))])
        if np.linalg.cond(A) > max_condition:
            continue

        coeffs = np.linalg.lstsq(A, heights, rcond=None)[0]
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        trace = 2 * (a + b)
        det   = 4 * a * b - c**2
        disc  = trace**2 - 4 * det
        if disc >= 0:
            sq = np.sqrt(disc)
            kmin[i] = (trace - sq) / 2
            kmax[i] = (trace + sq) / 2

    return kmin, kmax


def compute_all_curvature_fields(
    points: np.ndarray,
    normals: np.ndarray,
    k_neighbors: int = 10,
    radius_scale: float = 3.0,
    min_neighbors: int = 10,
    max_condition: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        feats  (N, 6) float32 — [k1, k2, mean_curv, gauss_curv, curvedness, shape_index]
        radii  (N,)   float32 — adaptive per-point neighbourhood radius
    """
    radii = compute_adaptive_radii(points, k=k_neighbors, scale=radius_scale)
    kmin, kmax = compute_principal_curvatures_adaptive(
        points, normals, radii,
        min_neighbors=min_neighbors, max_condition=max_condition,
    )
    eps   = 1e-8
    denom = np.where(np.abs(kmax - kmin) < eps, eps, kmax - kmin)
    feats = np.stack([
        kmin,
        kmax,
        (kmin + kmax) / 2,
        kmin * kmax,
        np.sqrt((kmin**2 + kmax**2) / 2),
        (2 / np.pi) * np.arctan((kmax + kmin) / denom),
    ], axis=1).astype(np.float32)
    return feats, radii


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _parse_off(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a (possibly header-fused) OFF file → (vertices, faces)."""
    with open(path, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    header = lines[0]
    if header == "OFF":
        start = 1
    elif header.upper().startswith("OFF"):
        lines[0] = header[3:].strip()
        start = 0
    else:
        raise ValueError(f"Not a valid OFF file: {path}")

    counts  = lines[start].split()
    n_verts = int(counts[0])
    n_faces = int(counts[1])
    start  += 1

    verts = np.array(
        [list(map(float, lines[start + i].split()[:3])) for i in range(n_verts)],
        dtype=np.float64,
    )
    faces = np.array(
        [list(map(int, lines[start + n_verts + i].split()[1:4])) for i in range(n_faces)],
        dtype=np.int32,
    )
    return verts, faces


def _parse_txt(path: str, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse pre-sampled ModelNet40 .txt files (x,y,z,nx,ny,nz per row)."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    pts, nrm = data[:, :3], data[:, 3:6]
    if len(pts) == n_points:
        return pts, nrm
    rng = np.random.default_rng(None)
    idx = rng.choice(len(pts), size=n_points, replace=(len(pts) < n_points))
    return pts[idx], nrm[idx]


# ── CSV index loader ──────────────────────────────────────────────────────────

def _load_csv_index(
    csv_path: str,
    root: str,
    split: Optional[str],
) -> Tuple[List[Tuple[str, int, str]], List[str]]:
    """
    Parse the CSV index.

    Returns:
        samples     — list of (abs_file_path, class_idx, object_id)
        class_names — sorted list of unique class strings (stable ordering)

    Auto-detects tab vs comma delimiter from the header line.
    Rows whose ``split`` column does not match `split` are skipped
    (pass split=None to load all rows).
    """
    with open(csv_path, newline="") as fh:
        header_line = fh.readline()
    dialect = "excel-tab" if "\t" in header_line else "excel"

    rows: List[dict] = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh, dialect=dialect):
            rows.append(row)

    # Build sorted class list from all rows (not just this split) for index stability
    seen: dict[str, None] = {}
    for row in rows:
        seen.setdefault(row["class"].strip(), None)
    class_names   = sorted(seen.keys())
    class_to_idx  = {c: i for i, c in enumerate(class_names)}

    samples: List[Tuple[str, int, str]] = []
    for row in rows:
        if split is not None and row["split"].strip() != split:
            continue
        cls      = row["class"].strip()
        obj_id   = row["object_id"].strip()
        rel_path = row["object_path"].strip()
        abs_path = os.path.normpath(os.path.join(root, rel_path))
        samples.append((abs_path, class_to_idx[cls], obj_id))

    return samples, class_names


# ── Dataset ───────────────────────────────────────────────────────────────────

class ModelNet40Dataset(Dataset):
    """
    ModelNet40 classification dataset driven by a CSV index file.

    Args:
        root:          Root directory; ``object_path`` values in the CSV are
                       joined to this path.
        csv_path:      Tab- or comma-separated index file with columns:
                       object_id, class, split, object_path
        split:         "train", "test", or None (load all rows).
        n_points:      Number of surface points per shape.
        normalize:     Centre + scale point cloud to unit sphere.
        augment:       Random Y-rotation + Gaussian jitter (training only).
        compute_feats: Compute 6-channel curvature features (slow).
        seed:          Global RNG seed; per-sample seed = seed + idx.
        indices:       Optional list of integer indices to subset the data.
        preprocessed:  Set to True after preprocess_all() so __getitem__
                       returns directly from self.data.
    """

    def __init__(
        self,
        root: str | Path,
        csv_path: str | Path,
        split: Optional[Literal["train", "test"]] = "train",
        n_points: int = 1024,
        normalize: bool = True,
        augment: bool = False,
        compute_feats: bool = False,
        seed: Optional[int] = 42,
        indices: Optional[List[int]] = None,
        preprocessed: bool = False,
    ):
        self.root          = str(root)
        self.csv_path      = str(csv_path)
        self.split         = split
        self.n_points      = n_points
        self.normalize     = normalize
        self.augment       = augment
        self.compute_feats = compute_feats
        self.seed          = seed
        self.preprocessed  = preprocessed
        self.data:  List[dict]        = []   # filled by preprocess_all()
        self.stats: List[List[float]] = []   # filled by statistic_feats()

        all_samples, self.class_names = _load_csv_index(
            self.csv_path, self.root, split
        )
        if not all_samples:
            raise FileNotFoundError(
                f"No samples found for split='{split}' in {csv_path}"
            )

        self.num_classes = len(self.class_names)
        self.samples = (
            [all_samples[i] for i in indices] if indices is not None else all_samples
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_raw(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch to correct parser by file extension."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            return _parse_txt(path, self.n_points)
        if ext == ".off":
            verts, faces = _parse_off(path)
            return sample_points_uniform(verts, faces, self.n_points)
        raise ValueError(f"Unsupported file extension '{ext}': {path}")

    @staticmethod
    def _normalize_points(points: np.ndarray) -> np.ndarray:
        pts = points - points.mean(axis=0)
        scale = np.linalg.norm(pts, axis=1).max()
        if scale > 0:
            pts /= scale
        return pts.astype(np.float32)

    @staticmethod
    def _augment_points(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        theta = rng.uniform(0, 2 * np.pi)
        c, s  = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        points = points @ R.T
        points += rng.normal(0, 0.02, points.shape).astype(np.float32)
        return points

    def _compute_features(
        self, points: np.ndarray, normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return compute_all_curvature_fields(
            points, normals,
            k_neighbors=10, radius_scale=3.0,
            min_neighbors=10, max_condition=1e6,
        )

    def _build_sample_dict(
        self,
        idx: int,
        points: np.ndarray,
        normals: np.ndarray,
        label: int,
        name: str,
        feats:  Optional[np.ndarray] = None,
        radius: Optional[np.ndarray] = None,
    ) -> dict:
        return {
            "points":  torch.from_numpy(points),
            "normals": torch.from_numpy(normals),
            "feats":   torch.from_numpy(feats)   if feats  is not None
                       else torch.zeros(self.n_points, 6,  dtype=torch.float32),
            "radius":  torch.from_numpy(radius)  if radius is not None
                       else torch.zeros(self.n_points,     dtype=torch.float32),
            "label":   torch.tensor(label, dtype=torch.long),
            "name":    name,
            "index":   idx,
        }

    # ── preprocess + cache ────────────────────────────────────────────────────

    def preprocess_all(
        self, cache_dir: str = "cache_modelnet40", overwrite: bool = False
    ):
        """
        Load every sample, optionally compute curvature features, and cache
        each as a ``.pt`` file.  After this call, ``__getitem__`` returns
        from ``self.data`` without any further disk I/O.

        Cache filename pattern: ``{split}_{object_id}.pt``
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.data = []
        split_tag = self.split or "all"

        print(
            f"[ModelNet40Dataset] Preprocessing {len(self.samples)} "
            f"'{split_tag}' samples → {cache_dir}"
        )

        for idx, (path, cidx, name) in tqdm(
            enumerate(self.samples), total=len(self.samples),
            desc=f"preprocess({split_tag})",
        ):
            cache_path = os.path.join(cache_dir, f"{split_tag}_{name}.pt")

            if os.path.exists(cache_path) and not overwrite:
                self.data.append(torch.load(cache_path, weights_only=False))
                continue

            all_cache_path = os.path.join(cache_dir, f"all_{name}.pt")

            if os.path.exists(all_cache_path) and not overwrite:
                self.data.append(torch.load(all_cache_path, weights_only=False))
                continue

            points, normals = self._load_raw(path)
            sample_seed = None if self.seed is None else self.seed + idx

            if self.normalize:
                points = self._normalize_points(points)

            if self.augment:
                points = self._augment_points(points, np.random.default_rng(sample_seed))

            feats = radius = None
            if self.compute_feats:
                feats, radius = self._compute_features(points, normals)

            sample = self._build_sample_dict(idx, points, normals, cidx, name, feats, radius)
            torch.save(sample, cache_path)
            self.data.append(sample)

        self.preprocessed = True
        print(f"[ModelNet40Dataset] Done. {len(self.data)} samples ready.")

    # ── statistics ────────────────────────────────────────────────────────────

    def statistic_feats(self) -> List[List[float]]:
        """
        Compute global per-channel statistics across all preprocessed samples.

        Populates and returns ``self.stats``:
            [ [max, mean, min],   # ch 0  k1 (kmin)
              [max, mean, min],   # ch 1  k2 (kmax)
              [max, mean, min],   # ch 2  mean curvature
              [max, mean, min],   # ch 3  Gaussian curvature
              [max, mean, min],   # ch 4  curvedness
              [max, mean, min] ]  # ch 5  shape index

        Requires preprocess_all() with compute_feats=True.
        """
        if not self.preprocessed or not self.data:
            raise RuntimeError("Call preprocess_all() before statistic_feats().")

        all_feats = []
        for sample in self.data:
            f = sample["feats"]
            all_feats.append(f.numpy() if isinstance(f, torch.Tensor) else np.asarray(f))

        all_feats = np.concatenate(all_feats, axis=0)   # (Total_N, 6)

        self.stats = []
        for i in range(all_feats.shape[1]):
            col = all_feats[:, i]
            self.stats.append([float(np.max(col)), float(np.mean(col)), float(np.min(col))])

        ch_names = ["k1(kmin)", "k2(kmax)", "mean_curv", "gauss_curv", "curvedness", "shape_idx"]
        print("[ModelNet40Dataset] Feature statistics (max / mean / min):")
        for ch, stat in zip(ch_names, self.stats):
            print(f"  {ch:>12s} : max={stat[0]:+.4f}  mean={stat[1]:+.4f}  min={stat[2]:+.4f}")

        return self.stats

    # ── disk-image export ─────────────────────────────────────────────────────

    def _export_to_image(
        self,
        channel: List[int],
        n_pix:   int = 128,
        out_dir: str = "./cache_images",
        local_lim=False
    ):
        """
        Project each point's neighbourhood onto its local tangent plane and
        save a disk-shaped PNG image encoding curvature values as pixel colours.

        Args:
            channel:  1 or 3 feature-channel indices (into the 6-column feats).
                      1 entry → greyscale, 3 entries → RGB.
            n_pix:    Output image side length in pixels.
            out_dir:  Destination directory.  Files: ``{object_id}_{pid}.png``.

        Requires:
            preprocess_all(compute_feats=True) and statistic_feats() called first.
        """
        if not self.preprocessed or not self.data:
            raise RuntimeError("Call preprocess_all() before _export_to_image().")
        if not self.stats:
            raise RuntimeError("Call statistic_feats() before _export_to_image().")

        n = len(channel)
        if n not in (1, 3):
            raise ValueError("channel must have 1 (greyscale) or 3 (RGB) entries.")

        os.makedirs(out_dir, exist_ok=True)

        # Global (vmin, vmax) per requested channel — derived from statistic_feats
        # self.stats[ch] = [max, mean, min]  →  vmin = stats[2], vmax = stats[0]
        channel_stats = [(self.stats[ch][2], self.stats[ch][0]) for ch in channel]

        total_pts = sum(
            s["points"].shape[0] if isinstance(s["points"], torch.Tensor)
            else len(s["points"])
            for s in self.data
        )
        saved = 0

        for sample in self.data:
            name     = sample["name"]
            xyz      = sample["points"]
            norms    = sample["normals"]
            feats    = sample["feats"]
            radius   = sample["radius"]
            name_dir = os.path.join(out_dir, f"{name}")
            if os.path.isdir(name_dir):
                saved += len(xyz)
                continue
            
            os.makedirs(name_dir)
            # Ensure numpy
            xyz      = xyz.numpy()    if isinstance(xyz,    torch.Tensor) else np.asarray(xyz)
            normals  = norms.numpy()  if isinstance(norms,  torch.Tensor) else np.asarray(norms)
            feats_np = feats.numpy()  if isinstance(feats,  torch.Tensor) else np.asarray(feats)
            radius_np= radius.numpy() if isinstance(radius, torch.Tensor) else np.asarray(radius)

            if local_lim:
                channel_stats = []
                for chid in channel:
                    vals = feats_np[:, chid].astype(np.float64)
                    channel_stats.append((float(np.max(vals)), float(np.min(vals))))
            tree = cKDTree(xyz)

            for pid in range(len(xyz)):
                r = float(radius_np[pid])

                neighbor_ids = tree.query_ball_point(xyz[pid], r=r)
                neighbor_ids = np.array(
                    [nid for nid in neighbor_ids if nid != pid], dtype=int
                )
                # print(len(neighbor_ids),flush=True)
                if len(neighbor_ids) < 3:
                    saved += 1
                    continue

                # ── tangent frame at pid ──────────────────────────────────
                n_pid = normals[pid].astype(np.float64)
                n_pid /= np.linalg.norm(n_pid) + 1e-12

                tangent_x = np.cross(n_pid, np.array([0., 0., 1.]))
                if np.linalg.norm(tangent_x) < 1e-5:
                    tangent_x = np.cross(n_pid, [1., 0., 0.])
                tangent_x /= np.linalg.norm(tangent_x)
                tangent_y  = np.cross(n_pid, tangent_x)
                tangent_y /= np.linalg.norm(tangent_y)

                # ── project neighbors onto tangent plane ──────────────────
                rel = xyz[neighbor_ids] - xyz[pid]
                pxy = np.stack([rel @ tangent_x, rel @ tangent_y], axis=1)

                # ── pixel grid ────────────────────────────────────────────
                xs = np.linspace(-r, r, n_pix)
                ys = np.linspace(-r, r, n_pix)
                xx, yy    = np.meshgrid(xs, ys, indexing="xy")
                in_disk   = (xx**2 + yy**2) <= r**2
                grid_pts  = np.stack([xx[in_disk], yy[in_disk]], axis=1)

                # ── rasterise channels ────────────────────────────────────
                if n == 1:
                    vmin, vmax = channel_stats[0]
                    vals = feats_np[neighbor_ids, channel[0]].astype(np.float64)

                    img    = np.zeros((n_pix, n_pix), dtype=np.float32)
                    interp_linear = griddata(pxy, vals, grid_pts, method="linear")
                    # interp_nearest = griddata(pxy, vals, grid_pts, method="nearest")

                    # interp = np.where(np.isnan(interp_linear), interp_nearest, interp_linear)
                    img[in_disk] = interp_linear

                    normed = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
                    normed[~in_disk | np.isnan(img)] = 1.0   # white background
                    out_img = (normed * 255).astype(np.uint8)

                else:  # RGB
                    out_img = np.ones((n_pix, n_pix, 3), dtype=np.uint8) * 255
                    for ci, ch_idx in enumerate(channel):
                        vmin, vmax = channel_stats[ci]
                        vals = feats_np[neighbor_ids, ch_idx].astype(np.float64)

                        img    = np.zeros((n_pix, n_pix), dtype=np.float32)
                        try:
                            interp = griddata(pxy, vals, grid_pts, method="linear", fill_value=np.nan)
                        except Exception:
                            pxy = np.nan_to_num(pxy, nan=0.0, posinf=0.0, neginf=0.0)
                            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                            interp = griddata(pxy, vals, grid_pts, method="nearest", fill_value=np.nan)
                        img[in_disk] = interp

                        normed = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
                        normed[~in_disk | np.isnan(img)] = 0.0
                        out_img[..., ci] = (normed * 255).astype(np.uint8)

                img_path = os.path.join(name_dir, f"{pid}.png")
                plt.imsave(img_path, out_img, cmap="gray" if n == 1 else None)

                saved += 1
                print(
                    f"\rExporting images: {saved}/{total_pts}  [{name}  pid={pid}]",
                    end="", flush=True,
                )

        print(f"\n[ModelNet40Dataset] Saved {saved} images → {out_dir}")

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.preprocessed:
            return self.data[idx]

        path, cidx, name = self.samples[idx]
        sample_seed = None if self.seed is None else self.seed + idx

        points, normals = self._load_raw(path)

        if self.normalize:
            points = self._normalize_points(points)
        if self.augment:
            points = self._augment_points(points, np.random.default_rng(sample_seed))

        feats = radius = None
        if self.compute_feats:
            feats, radius = self._compute_features(points, normals)

        return self._build_sample_dict(idx, points, normals, cidx, name, feats, radius)

    # ── convenience ───────────────────────────────────────────────────────────

    def class_name(self, label: int) -> str:
        return self.class_names[int(label)]

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted cross-entropy."""
        from collections import Counter
        counts: Counter = Counter(cidx for _, cidx, _ in self.samples)
        w = np.array(
            [1.0 / counts.get(c, 1) for c in range(self.num_classes)],
            dtype=np.float32,
        )
        w /= w.sum()
        return torch.from_numpy(w)

    def split_indices(
        self,
        val_ratio: float = 0.1,
        shuffle:   bool = True,
        seed:      Optional[int] = None,
    ) -> Tuple["ModelNet40Dataset", "ModelNet40Dataset"]:
        """
        Carve out a validation subset.  Returns (train_subset, val_subset).
        Useful because ModelNet40 has no official validation split.
        """
        n   = len(self)
        idx = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        n_val     = int(n * val_ratio)
        val_idx   = idx[:n_val].tolist()
        train_idx = idx[n_val:].tolist()

        def _make(idxs):
            return ModelNet40Dataset(
                root=self.root, csv_path=self.csv_path, split=self.split,
                n_points=self.n_points, normalize=self.normalize,
                augment=self.augment, compute_feats=self.compute_feats,
                seed=self.seed, indices=idxs, preprocessed=False,
            )

        return _make(train_idx), _make(val_idx)

    def __repr__(self) -> str:
        return (
            f"ModelNet40Dataset("
            f"split='{self.split}', n_samples={len(self)}, "
            f"n_points={self.n_points}, classes={self.num_classes}, "
            f"normalize={self.normalize}, augment={self.augment}, "
            f"compute_feats={self.compute_feats})"
        )


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time
    from torch.utils.data import DataLoader

    if len(sys.argv) < 3:
        print("Usage: python modelnet40_dataset.py <root> <csv_path> [split]")
        sys.exit(1)

    root     = sys.argv[1]
    csv_path = sys.argv[2]
    split    = sys.argv[3] if len(sys.argv) > 3 else "train"

    ds = ModelNet40Dataset(
        root=root, csv_path=csv_path, split=split,
        n_points=1024, normalize=True,
        augment=(split == "train"), compute_feats=True,
    )
    print(ds)
    print(f"  Classes ({ds.num_classes}): {ds.class_names[:5]} …")

    # 1. preprocess & cache
    t0 = time.perf_counter()
    ds.preprocess_all(cache_dir="cache_modelnet40", overwrite=False)
    print(f"  preprocess_all: {time.perf_counter() - t0:.1f}s")

    # 2. statistics
    ds.statistic_feats()

    # 3. export images (first sample only for speed)
    _backup, ds.data = ds.data, ds.data[:1]
    ds._export_to_image(channel=[0, 1, 2], n_pix=64, out_dir="./smoke_images")
        

    # 4. DataLoader check
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    batch  = next(iter(loader))
    print(f"\n  points  : {batch['points'].shape}")
    print(f"  normals : {batch['normals'].shape}")
    print(f"  feats   : {batch['feats'].shape}")
    print(f"  radius  : {batch['radius'].shape}")
    print(f"  labels  : {batch['label'].shape}  unique={batch['label'].unique().tolist()}")
    print("\nAll checks passed.")