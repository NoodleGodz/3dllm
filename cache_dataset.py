"""
cache_dataset.py
----------------
Pre-processes every mesh in the dataset and saves a .pt cache file next to
each OBJ.  Run this once before training so the DataLoader never has to
process a mesh on-the-fly, which would block GPU training.

Usage
-----
    python cache_dataset.py --csv   captions_dataset_shorttest.csv  --root  . --n_fps 1024 --disk_grid 32 --curv_channels 3 --force --workers 4

    # Force reprocess everything (delete existing caches first):
    python cache_dataset.py --csv captions.csv --root . --force

    # Parallel workers (faster on multi-core, safe because each OBJ writes
    # to its own .pt file):
    python cache_dataset.py --csv captions.csv --root . --workers 8
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Pre-cache Breaking Bad dataset .pt files")
    p.add_argument("--csv",          required=True)
    p.add_argument("--root",         required=True)
    p.add_argument("--n_fps",        type=int,   default=512)
    p.add_argument("--disk_grid",    type=int,   default=16)
    p.add_argument("--curv_channels",type=int,   default=6,
                   choices=[2, 3, 4, 6],
                   help="Must match the value used during training")
    p.add_argument("--curv_knn",     type=int,   default=30)
    p.add_argument("--disk_fill",    default="rbf", choices=["rbf","nearest","zero"])
    p.add_argument("--disk_radius_frac", type=float, default=0.05)
    p.add_argument("--fourier_bands",    type=int,   default=0)
    p.add_argument("--workers",      type=int,   default=1,
                   help="Parallel worker processes. 1=serial (safer). "
                        "Use 4-8 on a multi-core machine.")
    p.add_argument("--force",        action="store_true",
                   help="Delete existing .pt files and reprocess everything")
    p.add_argument("--skip_errors",  action="store_true", default=True,
                   help="Skip meshes that fail processing instead of crashing")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Worker function (must be top-level for multiprocessing pickling)
# ──────────────────────────────────────────────────────────────────────────────

def _cache_one(args_tuple):
    """
    Process a single OBJ and save its .pt cache.
    Returns (obj_path, status) where status is 'cached', 'skipped', or 'error:...'
    """
    obj_path_str, cfg_dict, force = args_tuple

    obj_path  = Path(obj_path_str)
    cache_path = obj_path.with_suffix(".pt")

    # Skip if already cached and not forcing
    if cache_path.exists() and not force:
        return obj_path_str, "skipped"

    # Delete stale cache if forcing
    if cache_path.exists() and force:
        cache_path.unlink()

    # Import here so it works in subprocess context
    sys.path.insert(0, str(Path(__file__).parent))
    from BreakingBadDataset import DatasetConfig, process_mesh

    cfg = DatasetConfig(**cfg_dict)

    try:
        process_mesh(obj_path, cfg)
        return obj_path_str, "cached"
    except Exception as e:
        return obj_path_str, f"error: {e}"

def cache_path(obj_path: Path, n_fps, disk_grid, disk_channels,
               disk_radius_frac, disk_fill, curv_knn, fourier_bands) -> Path:
    """
    Mirrors _cache_path() in BreakingBadDataset.py exactly.
    piece_0__n512_g16_c6_r050_rbf_k10_f0.pt
    """
    r   = int(round(disk_radius_frac * 1000))
    tag = (
        f"n{n_fps}"
        f"_g{disk_grid}"
        f"_c{disk_channels}"
        f"_r{r:03d}"
        f"_{disk_fill}"
        f"_k{curv_knn}"
        f"_f{fourier_bands}"
    )
    return obj_path.with_name(f"{obj_path.stem}__{tag}.pt")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv, sep=",", dtype=str)
    df.columns = df.columns.str.strip()

    # Filter parse errors
    if "parse_error" in df.columns:
        before = len(df)
        df = df[df["parse_error"].str.upper().str.strip() != "TRUE"]
        skipped_rows = before - len(df)
        if skipped_rows:
            log.info(f"Skipped {skipped_rows} parse_error rows")

    # Drop last row if it's a summary/blank (matches BreakingBadDataset behaviour)
    df = df[:-1]

    root = Path(args.root)

    # Build list of unique OBJ paths (deduplicate — same mesh may appear
    # multiple times in the CSV with different captions)
    seen    = set()
    jobs    = []
    missing = []

    for _, row in df.iterrows():
        rel  = row["path"].strip()
        obj  = root / rel
        key  = str(obj.resolve())
        if key in seen:
            continue
        seen.add(key)
        if not obj.exists():
            missing.append(str(obj))
            continue
        jobs.append(str(obj))

    log.info(f"CSV rows       : {len(df)}")
    log.info(f"Unique meshes  : {len(seen)}")
    log.info(f"Missing OBJ    : {len(missing)}")
    log.info(f"To process     : {len(jobs)}")
    if missing:
        log.warning(f"First 5 missing: {missing[:5]}")

    if not jobs:
        log.info("Nothing to do.")
        return

    # ── Config dict (picklable for multiprocessing) ───────────────────────────
    cfg_dict = dict(
        n_fps           = args.n_fps,
        disk_grid       = args.disk_grid,
        disk_channels   = 6,        # always compute all 6, select at training time
        disk_fill       = args.disk_fill,
        curv_knn        = args.curv_knn,
        disk_radius_frac= args.disk_radius_frac,
        cache_processed = True,
        skip_parse_errors = True,
        fourier_bands   = args.fourier_bands,
    )

    # ── Check how many are already cached ────────────────────────────────────
    already = sum(1 for p in jobs if cache_path(
        Path(p),
        args.n_fps, args.disk_grid, args.disk_channels,
        args.disk_radius_frac, args.disk_fill,
        args.curv_knn, args.fourier_bands,
    ).exists())
    if already and not args.force:
        log.info(f"Already cached : {already}  (use --force to reprocess)")
    to_run = len(jobs) if args.force else (len(jobs) - already)
    log.info(f"Will process   : {to_run}")

    # ── Process ───────────────────────────────────────────────────────────────
    t0       = time.time()
    n_cached = n_skipped = n_errors = 0
    error_list = []

    job_args = [(p, cfg_dict, args.force) for p in jobs]

    if args.workers <= 1:
        # Serial — easier to debug, no multiprocessing overhead
        for i, ja in enumerate(job_args):
            path, status = _cache_one(ja)
            if status == "cached":
                n_cached += 1
            elif status == "skipped":
                n_skipped += 1
            else:
                n_errors += 1
                print(status)
                error_list.append((path, status))
                if not args.skip_errors:
                    log.error(f"Failed: {path}  →  {status}")
                    sys.exit(1)

            # Progress every 50 meshes
            done = i + 1
            if done % 50 == 0 or done == len(job_args):
                elapsed = time.time() - t0
                rate    = n_cached / max(elapsed, 1)
                eta_s   = (to_run - n_cached) / max(rate, 1e-6)
                log.info(
                    f"[{done}/{len(job_args)}]  "
                    f"cached={n_cached}  skipped={n_skipped}  errors={n_errors}  "
                    f"rate={rate:.1f}/s  ETA={eta_s/60:.1f}min"
                )
    else:
        # Parallel
        log.info(f"Running with {args.workers} parallel workers")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_cache_one, ja): ja[0] for ja in job_args}
            done_count = 0
            for fut in as_completed(futures):
                path, status = fut.result()
                done_count += 1
                if status == "cached":
                    n_cached += 1
                elif status == "skipped":
                    n_skipped += 1
                else:
                    n_errors += 1
                    error_list.append((path, status))

                if done_count % 50 == 0 or done_count == len(job_args):
                    elapsed = time.time() - t0
                    rate    = n_cached / max(elapsed, 1)
                    eta_s   = (to_run - n_cached) / max(rate, 1e-6)
                    log.info(
                        f"[{done_count}/{len(job_args)}]  "
                        f"cached={n_cached}  skipped={n_skipped}  "
                        f"errors={n_errors}  "
                        f"rate={rate:.1f}/s  ETA={eta_s/60:.1f}min"
                    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info(f"\n{'─'*50}")
    log.info(f"Done in {elapsed/60:.1f} min")
    log.info(f"  Cached  : {n_cached}")
    log.info(f"  Skipped : {n_skipped}  (already existed)")
    log.info(f"  Errors  : {n_errors}")

    if error_list:
        log.warning(f"\nFailed meshes ({len(error_list)}):")
        for path, err in error_list[:20]:
            log.warning(f"  {Path(path).name}  →  {err}")
        if len(error_list) > 20:
            log.warning(f"  ... and {len(error_list)-20} more")

        # Save error list to file for inspection
        err_path = Path(args.root) / "cache_errors.txt"
        with open(err_path, "w") as f:
            for path, err in error_list:
                f.write(f"{path}\t{err}\n")
        log.info(f"Error list saved → {err_path}")

    # Verify cache completeness
    total_pts = sum(1 for p in jobs if Path(p).with_suffix(".pt").exists())
    log.info(f"\nCache completeness: {total_pts}/{len(jobs)} meshes have .pt files")
    if total_pts < len(jobs):
        log.warning(f"  {len(jobs)-total_pts} meshes still missing cache — "
                    f"check cache_errors.txt and re-run")


if __name__ == "__main__":
    main()
