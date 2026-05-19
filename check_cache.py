"""
check_cache.py — validate per-config .pt cache files against current config.

Cache filename format (from _cache_path in BreakingBadDataset.py):
    <stem>__n<n_fps>_g<disk_grid>_c<disk_channels>_r<radius*1000>_<fill>_k<knn>_f<fourier>.pt
    e.g.  piece_0__n512_g16_c6_r050_rbf_k10_f0.pt

Checks per file:
  1. File exists (with the correct config-encoded name)
  2. torch.load succeeds (not corrupt)
  3. Tensor shapes match config

Usage
-----
    python check_cache.py --csv captions.csv --root .

    # With explicit config (must match cache_dataset.py / training args):
    python check_cache.py --csv captions.csv --root . \\
        --n_fps 512 --disk_grid 16 --disk_channels 6 \\
        --disk_radius_frac 0.05 --disk_fill rbf \\
        --curv_knn 10 --fourier_bands 0

    # Auto-delete bad files so cache_dataset.py will rebuild them:
    python check_cache.py --csv captions.csv --root . --delete_bad
"""

import argparse
import sys
from pathlib import Path
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import pandas as pd
import torch


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate per-config .pt cache files"
    )
    p.add_argument("--csv",              required=True)
    p.add_argument("--root",             default=".")

    # Config — must match values used in cache_dataset.py and training
    p.add_argument("--n_fps",            type=int,   default=512)
    p.add_argument("--disk_grid",        type=int,   default=16)
    p.add_argument("--disk_channels",    type=int,   default=6)
    p.add_argument("--disk_radius_frac", type=float, default=0.05)
    p.add_argument("--disk_fill",        default="rbf",
                   choices=["rbf", "nearest", "zero"])
    p.add_argument("--curv_knn",         type=int,   default=10)
    p.add_argument("--fourier_bands",    type=int,   default=0)

    p.add_argument("--delete_bad",  action="store_true",
                   help="Delete corrupt / wrong-shape files so they get rebuilt")
    p.add_argument("--quiet",       action="store_true",
                   help="Only print summary")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Replicate _cache_path from BreakingBadDataset.py
# ──────────────────────────────────────────────────────────────────────────────

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
# Expected tensor shapes
# ──────────────────────────────────────────────────────────────────────────────

def expected_shapes(n_fps, disk_grid, disk_channels, fourier_bands) -> dict:
    pos_enc_dim = 6 + (2 * fourier_bands * 3 if fourier_bands > 0 else 0)
    return {
        "pos"    : (n_fps, 3),
        "norm"   : (n_fps, 3),
        "curv"   : (n_fps, 6),                              # always all 6 curvature features
        "data"   : (n_fps, disk_channels, disk_grid, disk_grid),
        "pos_enc": (n_fps, pos_enc_dim),
    }


def check_shapes(cache: dict, expected: dict) -> list[str]:
    out = []
    for key, exp in expected.items():
        if key not in cache:
            out.append(f"missing key '{key}'")
            continue
        got = tuple(cache[key].shape)
        if got != exp:
            out.append(f"{key}={got} expected {exp}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv, dtype=str)
    df.columns = df.columns.str.strip()
    if "parse_error" in df.columns:
        df = df[df["parse_error"].str.upper().str.strip() != "TRUE"]
    df = df[:-1]

    root  = Path(args.root)
    paths = sorted({(root / r["path"].strip()) for _, r in df.iterrows()})
    total = len(paths)

    # ── Build expected cache name for one example ─────────────────────────────
    example_pt = cache_path(
        Path("piece_0.obj"),
        args.n_fps, args.disk_grid, args.disk_channels,
        args.disk_radius_frac, args.disk_fill,
        args.curv_knn, args.fourier_bands,
    )
    exp_shapes = expected_shapes(
        args.n_fps, args.disk_grid, args.disk_channels, args.fourier_bands
    )

    # ── Print config header ───────────────────────────────────────────────────
    print(f"\nConfig")
    print(f"{'─'*56}")
    print(f"  n_fps          : {args.n_fps}")
    print(f"  disk_grid      : {args.disk_grid}")
    print(f"  disk_channels  : {args.disk_channels}")
    print(f"  disk_radius_frac: {args.disk_radius_frac}")
    print(f"  disk_fill      : {args.disk_fill}")
    print(f"  curv_knn       : {args.curv_knn}")
    print(f"  fourier_bands  : {args.fourier_bands}")
    print(f"\nCache filename pattern:")
    print(f"  {example_pt.name}")
    print(f"\nExpected tensor shapes:")
    for k, s in exp_shapes.items():
        print(f"  {k:<10}: {s}")
    print(f"{'─'*56}\n")

    # ── Validate ──────────────────────────────────────────────────────────────
    ok = missing = corrupt = wrong_shape = 0
    bad = []   # (obj_str, reason, pt_path)

    for i, obj in enumerate(paths, 1):
        pt = cache_path(
            obj,
            args.n_fps, args.disk_grid, args.disk_channels,
            args.disk_radius_frac, args.disk_fill,
            args.curv_knn, args.fourier_bands,
        )

        print(
            f"\r[{i}/{total}]  "
            f"ok={ok}  missing={missing}  "
            f"corrupt={corrupt}  wrong_shape={wrong_shape}   ",
            end="", flush=True,
        )

        # 1. Existence
        if not pt.exists():
            missing += 1
            bad.append((str(obj), f"missing  (expected: {pt.name})", pt))
            continue

        # 2. Loadability
        try:
            cache = torch.load(pt, weights_only=False, map_location="cpu")
        except Exception as e:
            corrupt += 1
            bad.append((str(obj), f"corrupt: {str(e)[:80]}", pt))
            if args.delete_bad:
                pt.unlink(missing_ok=True)
            continue

        if not isinstance(cache, dict):
            corrupt += 1
            bad.append((str(obj), f"corrupt: not a dict ({type(cache).__name__})", pt))
            if args.delete_bad:
                pt.unlink(missing_ok=True)
            continue

        # 3. Shape check
        mismatches = check_shapes(cache, exp_shapes)
        if mismatches:
            wrong_shape += 1
            bad.append((str(obj), "wrong shape: " + "; ".join(mismatches), pt))
            if args.delete_bad:
                pt.unlink(missing_ok=True)
            continue

        ok += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\r{'─'*56}")
    print(f"Total        : {total}")
    print(f"OK           : {ok}  ({ok/total*100:.1f}%)")
    print(f"Missing      : {missing}")
    print(f"Corrupt      : {corrupt}")
    print(f"Wrong shape  : {wrong_shape}"
          + ("  ← shapes inside file don't match config" if wrong_shape else ""))
    print(f"{'─'*56}")

    if bad and not args.quiet:
        print(f"\nBad files ({len(bad)}):")
        for obj_path, reason, _ in bad[:30]:
            print(f"  {Path(obj_path).name:<42}  {reason}")
        if len(bad) > 30:
            print(f"  ... and {len(bad)-30} more")

    if bad:
        out = Path("bad_cache.txt")
        out.write_text("\n".join(f"{p}\t{r}" for p, r, _ in bad))
        print(f"\nFull list → {out}")

        if args.delete_bad:
            print("Bad .pt files deleted — rebuild with:")
        else:
            print("\nTo delete bad files and rebuild:")
            print("  python check_cache.py [same args] --delete_bad")
            print("\nRebuild command:")

        print(
            f"  python cache_dataset.py --csv {args.csv} --root {args.root} --force \\\n"
            f"      --n_fps {args.n_fps} --disk_grid {args.disk_grid} "
            f"--curv_channels {args.disk_channels} \\\n"
            f"      --disk_radius_frac {args.disk_radius_frac} "
            f"--disk_fill {args.disk_fill} \\\n"
            f"      --curv_knn {args.curv_knn} --fourier_bands {args.fourier_bands}"
        )
        sys.exit(1)
    else:
        print("\nAll caches valid ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()