"""
render_multiview.py
-------------------
For every row in a CSV with columns 'path' and 'base_path':

  'path'      → fractured piece .obj
  'base_path' → complete (unfractured) base .obj

A .txt file named <stem>_labels.txt lives in the same folder as 'path'
and contains one integer per line (one per face):
  0 → normal face   (light grey)
  1 → fracture face (vivid orange-red highlight)

VIEW SCHEME
-----------
  view_overview  : one wide shot of the whole object from a neutral 3/4 angle
  view_frac00–NN : (n_views-1) cameras distributed over the hemisphere that
                   faces the fracture surface, all aimed at the fracture
                   centroid so the fracture face is always visible and
                   covered from many different angles.

The same camera poses are reused for the base mesh, so every fractured
image has a directly comparable base image.

Outputs (next to the fractured .obj):
  <stem>_frac_view_overview.png
  <stem>_frac_view_frac00.png … <stem>_frac_view_fracNN.png
  <stem>_base_view_overview.png
  <stem>_base_view_frac00.png … <stem>_base_view_fracNN.png

No apt-get required — tries EGL → osmesa → default pyrender.

Usage:
    python render_multiview.py meshes.csv
    python render_multiview.py meshes.csv --views 6 --size 512

Requirements:
    pip install trimesh pyrender numpy pillow pandas scipy
    export PYOPENGL_PLATFORM=egl    # recommended on headless servers
"""

import argparse
import math
import os
import sys
import traceback

import numpy as np
import pandas as pd

# ── Colour palette (RGBA uint8) ───────────────────────────────────────────────
NORMAL_COLOUR   = np.array([210, 210, 210, 255], dtype=np.uint8)  # light grey
FRACTURE_COLOUR = np.array([255,  75,  35, 255], dtype=np.uint8)  # vivid orange-red
BASE_COLOUR     = np.array([160, 200, 235, 255], dtype=np.uint8)  # steel blue


# ══════════════════════════════════════════════════════════════════════════════
# Mesh loading & normalisation
# ══════════════════════════════════════════════════════════════════════════════

def load_mesh(obj_path: str):
    """Load .obj → single trimesh.Trimesh (flattens Scenes)."""
    import trimesh
    raw = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(raw, trimesh.Scene):
        geoms = list(raw.geometry.values())
        if not geoms:
            raise ValueError(f"No geometry in {obj_path}")
        return trimesh.util.concatenate(geoms)
    return raw


def normalise_to_ref(mesh, ref_mesh):
    """
    Centre + scale `mesh` using `ref_mesh`'s bounding-box centroid and radius.
    Applying the same transform to frac and base keeps them in one shared frame.
    Returns (normalised_mesh, centroid, scale) so callers can transform points.
    """
    mesh     = mesh.copy()
    centroid = ref_mesh.bounding_box.centroid
    scale    = np.linalg.norm(ref_mesh.vertices - centroid, axis=1).max()
    scale    = scale if scale > 1e-8 else 1.0
    mesh.vertices = (mesh.vertices - centroid) / scale
    return mesh, centroid, scale


# ══════════════════════════════════════════════════════════════════════════════
# Face labels
# ══════════════════════════════════════════════════════════════════════════════

def load_face_labels(obj_path: str, n_faces: int) -> np.ndarray:
    """
    Load per-face labels from <stem>.txt (one int per line).
    Returns int32 array length n_faces; all-zero if file missing.
    """
    txt    = os.path.splitext(obj_path)[0] + "_labels.txt"
    labels = np.zeros(n_faces, dtype=np.int32)
    if not os.path.isfile(txt):
        print(f"  [warn] label file not found: {txt} — treating all faces as normal")
        return labels
    with open(txt) as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    parsed = np.array([int(x) for x in raw], dtype=np.int32)
    n = min(len(parsed), n_faces)
    labels[:n] = parsed[:n]
    if len(parsed) != n_faces:
        print(f"  [warn] label count {len(parsed)} ≠ face count {n_faces}; "
              f"{'padding with 0' if len(parsed) < n_faces else 'truncating'}")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Mesh colouring
# ══════════════════════════════════════════════════════════════════════════════

def colour_by_labels(mesh, labels: np.ndarray):
    """
    Per-face colouring without colour bleed: vertices are duplicated so
    every face owns its colour independently.
    """
    import trimesh
    new_verts = mesh.vertices[mesh.faces.reshape(-1)]
    new_faces = np.arange(len(new_verts)).reshape(-1, 3)
    face_cols = np.where(labels[:, None] == 1, FRACTURE_COLOUR, NORMAL_COLOUR)
    vert_cols = np.repeat(face_cols, 3, axis=0)
    m = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=vert_cols)
    return m


def colour_uniform(mesh, colour: np.ndarray):
    """Return a copy of mesh painted with one uniform RGBA colour."""
    import trimesh
    m    = mesh.copy()
    cols = np.tile(colour, (len(m.vertices), 1))
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=cols)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Fracture geometry analysis
# ══════════════════════════════════════════════════════════════════════════════

def fracture_geometry(mesh, labels: np.ndarray):
    """
    Returns:
      normal   : unit normal of fracture surface (area-weighted mean)
      centroid : area-weighted centroid of fracture face centres
    Falls back to k-means heuristic if no label-1 faces exist.
    """
    mask = labels == 1
    if mask.any():
        normals  = mesh.face_normals[mask]
        areas    = mesh.area_faces[mask]
        centers  = mesh.triangles_center[mask]
        w_normal = (normals * areas[:, None]).sum(0)
        norm_len = np.linalg.norm(w_normal)
        normal   = w_normal / norm_len if norm_len > 1e-8 else np.array([0., 1., 0.])
        centroid = (centers * areas[:, None]).sum(0) / areas.sum()
        return normal, centroid

    # Fallback
    normal = _fracture_normal_kmeans(mesh)
    return normal, mesh.bounding_box.centroid


def _fracture_normal_kmeans(mesh) -> np.ndarray:
    from scipy.cluster.vq import kmeans2
    normals, areas = mesh.face_normals, mesh.area_faces
    k = min(12, max(2, len(mesh.faces) // 50))
    try:
        centroids, lbls = kmeans2(normals, k, minit="points", nit=20)
    except Exception:
        w = (normals * areas[:, None]).sum(0)
        return w / (np.linalg.norm(w) + 1e-12)
    best_n, best_a, best_flat = None, -1., False
    for ci in range(k):
        mask = lbls == ci
        if not mask.any():
            continue
        mn   = centroids[ci] / (np.linalg.norm(centroids[ci]) + 1e-12)
        tot  = areas[mask].sum()
        flat = mesh.triangles_center[mask].dot(mn).std() < 0.05
        if flat and tot > best_a:
            best_n, best_a, best_flat = mn, tot, True
        elif not best_flat and tot > best_a:
            best_n, best_a = mn, tot
    return best_n if best_n is not None else np.array([0., 1., 0.])


# ══════════════════════════════════════════════════════════════════════════════
# View direction generation
# ══════════════════════════════════════════════════════════════════════════════

def hemisphere_directions(n: int, normal: np.ndarray) -> list:
    """
    Generate `n` unit directions that lie in the open hemisphere facing
    `normal`, spread as evenly as possible using a sunflower / golden-angle
    spiral on the hemisphere.

    The first direction is exactly `normal` (straight-on view of the fracture).
    The remaining n-1 are spread angularly around it, up to ~75° from the
    normal so the fracture face stays clearly visible from every viewpoint.
    """
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    # Build an orthonormal basis {u, v, normal}
    up = np.array([0., 1., 0.])
    if abs(normal.dot(up)) > 0.95:
        up = np.array([1., 0., 0.])
    u = np.cross(normal, up);  u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    dirs = [normal.copy()]     # first: dead-on fracture view

    if n == 1:
        return dirs

    # Golden-angle spiral to spread remaining n-1 points on hemisphere
    # Polar angle from 0 → max_polar, azimuth by golden angle steps
    max_polar   = math.radians(75)   # don't go more than 75° off the fracture
    golden      = math.pi * (3. - math.sqrt(5.))   # golden angle ≈ 137.5°

    for i in range(1, n):
        # Linear ramp from small angle to max_polar
        polar = max_polar * i / (n - 1) if n > 2 else max_polar * 0.5
        azim  = golden * i

        # Direction in hemisphere basis
        sin_p = math.sin(polar)
        d = (math.cos(polar) * normal
             + sin_p * math.cos(azim) * u
             + sin_p * math.sin(azim) * v)
        dirs.append(d / np.linalg.norm(d))

    return dirs


def overview_direction() -> np.ndarray:
    """
    A classic 3/4 view: elevated 30°, rotated 45° in azimuth.
    Always uses world coords — gives context regardless of fracture orientation.
    """
    elev = math.radians(30)
    azim = math.radians(45)
    return np.array([
        math.cos(elev) * math.cos(azim),
        math.sin(elev),
        math.cos(elev) * math.sin(azim),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Camera maths
# ══════════════════════════════════════════════════════════════════════════════

def look_at(eye: np.ndarray, target=None) -> np.ndarray:
    """4×4 OpenGL-convention camera pose (camera looks down -Z)."""
    if target is None:
        target = np.zeros(3)
    fwd = target - eye
    d   = np.linalg.norm(fwd)
    if d < 1e-8:
        return np.eye(4)
    fwd /= d
    up = np.array([0., 1., 0.])
    if abs(fwd.dot(up)) > 0.95:
        up = np.array([0., 0., 1.])
    right = np.cross(fwd, up);  right /= np.linalg.norm(right)
    up    = np.cross(right, fwd)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, -fwd, eye
    return M


def fit_distance_to_target(mesh, direction: np.ndarray, target: np.ndarray,
                            fov_y: float, img_w: int, img_h: int,
                            margin: float = 1.02) -> float:
    """
    Compute the minimum camera distance D from  along 
    so that every vertex projects inside the FOV.

    For a camera at distance D from target, a vertex at lateral offset L
    and depth offset dz along the view axis satisfies:

        L / (D - dz) <= tan(half_fov)
        =>  D >= L / tan(half_fov) + dz

    Taking the max over all vertices gives the exact tight-fit distance.
    proj_d = vertex . camera_direction (positive = vertex is behind target,
    i.e. further from the camera, so camera must be further back).
    """
    d  = direction / (np.linalg.norm(direction) + 1e-12)
    up = np.array([0., 1., 0.])
    if abs(d.dot(up)) > 0.95:
        up = np.array([0., 0., 1.])
    right  = np.cross(-d, up);  right /= np.linalg.norm(right)
    up_cam = np.cross(right, -d)

    v_rel  = mesh.vertices - target
    proj_r = v_rel @ right           # horizontal lateral offset from target
    proj_u = v_rel @ up_cam          # vertical lateral offset from target
    proj_d = v_rel @ d               # depth: positive = behind target (away from cam)

    half_fov_y = fov_y / 2
    half_fov_x = math.atan(math.tan(half_fov_y) * img_w / img_h)

    # Per-vertex required D; take element-wise max across both axes
    D_required = np.maximum(
        np.abs(proj_r) / math.tan(half_fov_x) + proj_d,
        np.abs(proj_u) / math.tan(half_fov_y) + proj_d,
    )

    return float(max(D_required.max() * margin, 0.3))


# ══════════════════════════════════════════════════════════════════════════════
# Renderer bootstrap (EGL → osmesa → default)
# ══════════════════════════════════════════════════════════════════════════════

def get_renderer(w: int, h: int):
    import pyrender
    last_exc = None
    for platform in ["egl", "osmesa", None]:
        if platform:
            os.environ["PYOPENGL_PLATFORM"] = platform
        elif "PYOPENGL_PLATFORM" in os.environ:
            del os.environ["PYOPENGL_PLATFORM"]
        try:
            r = pyrender.OffscreenRenderer(w, h)
            if platform:
                print(f"  [renderer] PYOPENGL_PLATFORM={platform}")
            return r
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        f"pyrender OffscreenRenderer failed on all backends. "
        f"Last error: {last_exc}\n"
        "Tip: pip install open3d  for a fully self-contained renderer."
    )


def make_scene(pr_mesh, cam_pose, fov_y: float, light_cfgs, ambient):
    import pyrender
    scene  = pyrender.Scene(ambient_light=ambient)
    camera = pyrender.PerspectiveCamera(yfov=fov_y)
    scene.add(pr_mesh)
    scene.add(camera, pose=cam_pose)
    for ldir, lint in light_cfgs:
        lp = look_at(ldir / np.linalg.norm(ldir) * 5.)
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=lint),
            pose=lp,
        )
    return scene


# ══════════════════════════════════════════════════════════════════════════════
# Core render routine
# ══════════════════════════════════════════════════════════════════════════════

def render_pair(frac_path: str, base_path: str,
                n_views: int, img_size: int) -> list:
    import pyrender
    from PIL import Image

    frac_path = os.path.abspath(frac_path)
    base_path = os.path.abspath(base_path)
    out_dir   = os.path.dirname(frac_path)
    stem      = os.path.splitext(os.path.basename(frac_path))[0]

    # ── Load & normalise ──────────────────────────────────────────────────
    frac_raw = load_mesh(frac_path)
    base_raw = load_mesh(base_path)

    # Both normalised to the BASE frame so camera poses are directly comparable
    base_norm, ref_centroid, ref_scale = normalise_to_ref(base_raw, base_raw)
    frac_norm, _,            _         = normalise_to_ref(frac_raw, base_raw)

    # ── Labels & fracture geometry ────────────────────────────────────────
    labels          = load_face_labels(frac_path, len(frac_norm.faces))
    n_frac_faces    = (labels == 1).sum()
    frac_normal, frac_centroid = fracture_geometry(frac_norm, labels)

    print(f"  {n_frac_faces}/{len(labels)} fracture faces")
    print(f"  fracture normal   ≈ [{frac_normal[0]:.2f}, "
          f"{frac_normal[1]:.2f}, {frac_normal[2]:.2f}]")
    print(f"  fracture centroid ≈ [{frac_centroid[0]:.2f}, "
          f"{frac_centroid[1]:.2f}, {frac_centroid[2]:.2f}]")

    # ── Colour meshes ─────────────────────────────────────────────────────
    frac_coloured = colour_by_labels(frac_norm, labels)
    base_coloured = colour_uniform(base_norm, BASE_COLOUR)

    frac_pr = pyrender.Mesh.from_trimesh(frac_coloured, smooth=False)
    base_pr = pyrender.Mesh.from_trimesh(base_coloured, smooth=True)

    # ── Build camera list ─────────────────────────────────────────────────
    #
    # Each entry is (name, eye_direction, look_target):
    #   • overview  → direction from overview_direction(), target = mesh centroid (0,0,0)
    #   • frac views → directions on the fracture hemisphere, target = frac_centroid
    #
    fov_y       = math.radians(45.)
    mesh_center = np.zeros(3)          # already centred by normalise_to_ref

    n_frac_views = n_views - 1         # all slots minus the overview
    hemi_dirs    = hemisphere_directions(n_frac_views, frac_normal)

    cameras = []

    # Slot 0: overview of whole object — fit to base so the full shape is visible
    ov_dir = overview_direction()
    ov_dist = fit_distance_to_target(
        base_norm, ov_dir, mesh_center, fov_y, img_size, img_size
    )
    cameras.append(("overview", ov_dir * ov_dist, mesh_center))

    # Slots 1…N: fracture-hemisphere views aimed at the fracture centroid
    # Fit distance to the FRACTURED piece (not base) so it fills the frame
    for i, hd in enumerate(hemi_dirs):
        dist = fit_distance_to_target(
            frac_norm, hd, frac_centroid, fov_y, img_size, img_size
        )
        cameras.append((f"frac{i:02d}", frac_centroid + hd * dist, frac_centroid))

    # ── Lighting ──────────────────────────────────────────────────────────
    light_cfgs = [
        (np.array([ 1.,  2.,  1.5]), 3.5),   # key
        (np.array([-1.5,  0.5, -1.]), 1.5),  # fill
        (np.array([ 0., -1.,  0.5]), 1.0),   # rim
    ]
    ambient = np.array([0.25, 0.25, 0.25])

    renderer = get_renderer(img_size, img_size)
    saved    = []

    for view_name, eye, target in cameras:
        cam_pose = look_at(eye, target)

        for tag, pr_mesh in [("frac", frac_pr), ("base", base_pr)]:
            scene    = make_scene(pr_mesh, cam_pose, fov_y, light_cfgs, ambient)
            color, _ = renderer.render(scene)
            out_name = f"{stem}_{tag}_view_{view_name}.png"
            out_path = os.path.join(out_dir, out_name)
            Image.fromarray(color).save(out_path)
            saved.append(out_path)
            print(f"  ✔ {out_name}")

    renderer.delete()
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render multi-view PNGs for fractured .obj pieces + matching base mesh.\n\n"
            "VIEW SCHEME\n"
            "  1 overview  : whole-object 3/4 shot\n"
            "  N-1 frac*   : cameras spread over the fracture-face hemisphere,\n"
            "                all aimed at the fracture centroid\n\n"
            "CSV columns required: 'path' (fractured piece), 'base_path' (whole mesh).\n"
            "A <stem>_labels.txt with per-face labels (0=normal, 1=fracture) must live\n"
            "next to each 'path' .obj."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv",            help="Input CSV file")
    parser.add_argument("--views", "-v",  type=int, default=6,
                        help="Total views per mesh pair (default: 6; min: 2)")
    parser.add_argument("--size",  "-s",  type=int, default=512,
                        help="Output image size in pixels (default: 512)")
    args = parser.parse_args()

    if args.views < 2:
        sys.exit("[error] --views must be >= 2 (1 overview + at least 1 fracture view)")

    df = pd.read_csv(args.csv)
    missing = [c for c in ("path", "base_path") if c not in df.columns]
    if missing:
        sys.exit(f"[error] CSV missing column(s): {missing}. Found: {list(df.columns)}")

    rows = df[["path", "base_path"]].dropna()
    n_img = args.views * 2
    print(
        f"Found {len(rows)} pair(s). "
        f"Rendering {args.views} views × 2 meshes = {n_img} images per pair "
        f"at {args.size}×{args.size}px …\n"
        f"  (1 overview + {args.views - 1} fracture-hemisphere views)\n"
    )

    ok = fail = 0
    for _, row in rows.iterrows():
        fp, bp = str(row["path"]), str(row["base_path"])
        print(f"→ {fp}")
        if not os.path.isfile(fp):
            print("  [skip] fractured mesh not found");  fail += 1;  continue
        if not os.path.isfile(bp):
            print("  [skip] base mesh not found");       fail += 1;  continue
        try:
            render_pair(fp, bp, args.views, args.size)
            ok += 1
        except Exception as exc:
            print(f"  [error] {exc}")
            traceback.print_exc()
            fail += 1

    print(f"\nDone — {ok} pair(s) rendered, {fail} skipped/failed.")


if __name__ == "__main__":
    main()