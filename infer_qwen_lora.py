"""
infer_qwen_lora.py
==================
Inference script for the LoRA fine-tuned Qwen2.5-VL fracture captioner.

Given one or more OBJ files, runs the full pipeline:
  OBJ → process_mesh → patches + pos_enc
      → GeometryProjector → geometry prefix tokens
      → [geo prefix | prompt] → Qwen2.5-VL (merged LoRA)
      → structured JSON caption (all 8 fields)

Usage
-----
  # Single file
  python infer_qwen_lora.py \
      --ckpt  ./checkpoints/ckpt_best \
      --obj   path/to/fragment.obj \
      --class teapot

  # Batch from a CSV (same format as training CSV)
  python infer_qwen_lora.py \
      --ckpt  ./checkpoints/ckpt_best \
      --csv   test_set.csv \
      --root  /path/to/breaking_bad \
      --out   predictions.csv

  # Must match the hyperparameters used during training:
  #   --n_fps, --disk_grid, --curv_channels, --proj_hidden

Notes
-----
  * The checkpoint directory must contain:
      ckpt_best/
        lora_adapter/          ← HF adapter weights + tokenizer
        geometry_projector.pt  ← plain state_dict

  * use_cache is re-enabled at inference (was off during training for
    gradient checkpointing) — this enables KV-cache for fast generation.

  * Structured JSON is parsed and pretty-printed field by field.
    If the model output is not valid JSON, the raw text is returned.
"""

from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference — Qwen2.5-VL LoRA fracture captioner"
    )

    # Checkpoint
    p.add_argument("--ckpt",     required=True,
                   help="Checkpoint dir (e.g. checkpoints/ckpt_best)")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="Base model ID (must match training)")

    # Input — single file OR batch CSV
    p.add_argument("--obj",      default=None,
                   help="Single OBJ file path")
    p.add_argument("--cls",      default="unknown",
                   help="Object class label for single-file mode")
    p.add_argument("--csv",      default=None,
                   help="CSV with columns: path, class  (batch mode)")
    p.add_argument("--root",     default=".",
                   help="Dataset root prepended to CSV paths")

    # Output
    p.add_argument("--out",      default=None,
                   help="Output CSV path (batch mode). Omit to print to stdout.")

    # Must match training hyperparameters
    p.add_argument("--n_fps",         type=int, default=512)
    p.add_argument("--disk_grid",     type=int, default=16)
    p.add_argument("--curv_channels", type=int, default=3, choices=[2, 3, 4, 6])
    p.add_argument("--proj_hidden",   type=int, default=512)

    # Generation
    p.add_argument("--max_new_tokens", type=int,   default=512)
    p.add_argument("--temperature",    type=float, default=0.1,
                   help="Low temperature for deterministic structured output")
    p.add_argument("--do_sample",      action="store_true", default=False)
    p.add_argument("--batch_size",     type=int,   default=1,
                   help="Inference batch size (1 is safest for variable N)")
    p.add_argument("--device",         default="cuda")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Constants — must match train_qwen_lora.py exactly
# ──────────────────────────────────────────────────────────────────────────────

CURV_CHANNEL_MAP = {
    2: [0, 1],
    3: [1, 4, 5],
    4: [0, 1, 2, 3],
    6: [0, 1, 2, 3, 4, 5],
}

CURV_CHANNEL_NAMES = {
    0: "kmin", 1: "kmax", 2: "mean_curv",
    3: "gauss_curv", 4: "curvedness", 5: "shape_index",
}

TARGET_FIELDS = [
    "object_description",
    "fragment_location",
    "fracture_surface",
    "missing_piece_size",
    "break_type",
    "fragment_guess",
    "confidence",
    "caption",
]

SYSTEM_MSG = (
    "You are a precise 3D fracture analyst. "
    "Given geometry features of a fractured object, respond with a single "
    "valid JSON object describing the fracture. "
    "No extra text, no markdown, no code blocks."
)

USER_MSG = (
    "Analyse the geometry of this fractured 3D object (class: {cls}) "
    "and return the structured JSON caption."
)

USER_MSG_NO_CLASS = (
    "Analyse the geometry of this fractured 3D object "
    "and return the structured JSON caption."
)
# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(
    ckpt_dir: str | Path,
    model_id: str,
    curv_channels: int,
    disk_grid: int,
    proj_hidden: int,
    device: str,
):
    """
    Load the fine-tuned LLM (base + LoRA adapter) and GeometryProjector.

    Checkpoint layout expected:
        ckpt_dir/
            lora_adapter/          HF PeftModel weights + tokenizer
            geometry_projector.pt  plain state_dict
    """
    from transformers import AutoTokenizer
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    # Import GeometryProjector from the training script
    sys.path.insert(0, str(Path(__file__).parent))
    from train_qwen_lora import GeometryProjector

    ckpt_dir = Path(ckpt_dir)
    adapter_dir = ckpt_dir / "lora_adapter"
    proj_path   = ckpt_dir / "geometry_projector.pt"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"No lora_adapter dir found at {adapter_dir}")
    if not proj_path.exists():
        raise FileNotFoundError(f"No geometry_projector.pt found at {proj_path}")

    log.info(f"Loading tokenizer from {adapter_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, trust_remote_code=True
    )

    log.info(f"Loading base model {model_id} ...")
    # Must use the VL-specific class — AutoModelForCausalLM does not
    # recognise Qwen2_5_VLConfig and will raise ValueError.
    base_llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype   = torch.bfloat16,
        device_map    = device,
        trust_remote_code = True,
    )

    log.info("Merging LoRA adapter ...")
    llm = PeftModel.from_pretrained(base_llm, adapter_dir)

    # Option A: keep adapters separate (faster load, same output)
    # Option B: merge for slightly faster inference — uncomment if needed:
    # llm = llm.merge_and_unload()

    # Re-enable KV cache for fast autoregressive generation
    llm.config.use_cache = True
    llm.eval()

    log.info("Loading GeometryProjector ...")
    d_model = llm.config.text_config.hidden_size   # 3584 for Qwen2.5-VL-7B

    projector = GeometryProjector(
        c_in        = curv_channels,
        d_grid      = disk_grid,
        d_model     = d_model,
        pos_enc_dim = 6,                      # fourier_bands=0 during training
        hidden      = proj_hidden,
    ).to(device)

    projector.load_state_dict(
        torch.load(proj_path, map_location=device, weights_only=True)
    )
    projector.eval()

    log.info("Models ready.")
    return llm, projector, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Mesh processing
# ──────────────────────────────────────────────────────────────────────────────

def process_single_obj(
    obj_path: str | Path,
    n_fps: int,
    disk_grid: int,
    curv_ch_idx: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the full geometry pipeline on a single OBJ file.

    Returns
    -------
    patches : (1, N, C_in, D, D)  — curvature patch tensor ready for projector
    pos_enc : (1, N, 6)           — positional encoding
    """
    from BreakingBadDataset import DatasetConfig, process_mesh

    cfg = DatasetConfig(
        n_fps          = n_fps,
        disk_grid      = disk_grid,
        disk_channels  = 6,        # always compute all 6, select below
        disk_fill      = "rbf",
        curv_knn       = 10,
        cache_processed= True,     # reuse .pt cache if already computed
        fourier_bands  = 0,
    )

    tensors = process_mesh(Path(obj_path), cfg)

    # Select the curvature channels used during training
    # tensors["data"]: [N, 6, D, D]
    patches = tensors["data"][:, curv_ch_idx, :, :]   # [N, C_in, D, D]
    pos_enc = tensors["pos_enc"]                        # [N, 6]

    # Add batch dim
    patches = patches.unsqueeze(0)    # [1, N, C_in, D, D]
    pos_enc = pos_enc.unsqueeze(0)    # [1, N, 6]

    # patches = torch.randn_like(patches)
    # pos_enc = torch.randn_like(pos_enc)
    print("AHHHHHHH")
    return patches, pos_enc


# ──────────────────────────────────────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────────────────────────────────────
# USER_MSG.format(cls=obj_class)
def build_prompt_ids(obj_class: str, tokenizer) -> torch.Tensor:
    """Build tokenised prompt (system + user turn) without geometry prefix."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": USER_MSG_NO_CLASS},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # appends the assistant turn opening
    )
    ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids   # [1, T_prompt]
    return ids


# ──────────────────────────────────────────────────────────────────────────────
# Generation with geometry prefix
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()

# ──────────────────────────────────────────────────────────────────────────────
# Embed-token resolver — version/PEFT agnostic
# ──────────────────────────────────────────────────────────────────────────────

def _get_embed_tokens(llm):
    """
    Find the token embedding layer regardless of PEFT wrapping depth or
    transformers version.  Qwen2_5_VL embeds can live at:
      bare model : llm.model.embed_tokens
      PEFT wrap  : llm.base_model.model.model.embed_tokens  (or deeper)
    Walking named_modules() is always correct.
    """
    for name, mod in llm.named_modules():
        if name.endswith("embed_tokens"):
            return mod
    raise AttributeError(
        "embed_tokens not found. "
        f"First 40 module names: {[n for n,_ in llm.named_modules()][:40]}"
    )

def generate_caption(
    llm,
    projector,
    tokenizer,
    patches: torch.Tensor,       # (B, N, C_in, D, D)
    pos_enc: torch.Tensor,       # (B, N, 6)
    obj_class: str,
    device: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """
    Run geometry prefix injection + autoregressive generation.

    Returns the raw model output string (JSON or fallback text).
    """
    B = patches.shape[0]
    dev = torch.device(device)

    patches = patches.to(dev)
    pos_enc = pos_enc.to(dev)

    # 1. Geometry prefix tokens: (B, N, d_model)  — projector runs in float32
    geo_embeds = projector(patches, pos_enc)

    # 2. Prompt token embeddings: (B, T_prompt, d_model) — LLM dtype (bfloat16)
    prompt_ids  = build_prompt_ids(obj_class, tokenizer).to(dev)
    prompt_ids  = prompt_ids.expand(B, -1)
    text_embeds = _get_embed_tokens(llm)(prompt_ids)

    # Cast geo_embeds to match LLM dtype before concatenation.
    # Projector is float32; LLM embedding table is bfloat16 — mismatched dtypes
    # cause RuntimeError in lm_head matmul downstream.
    geo_embeds  = geo_embeds.to(text_embeds.dtype)

    # 3. Concatenate [geo prefix | prompt]
    full_embeds = torch.cat([geo_embeds, text_embeds], dim=1)   # (B, N+T, d_model)

    # 4. Attention mask — all ones (no padding in inference, single sample)
    full_mask = torch.ones(
        B, full_embeds.shape[1],
        dtype=torch.long, device=dev,
    )

    # 5. Generate — pass inputs_embeds so the model continues from our prefix
    output_ids = llm.generate(
        inputs_embeds      = full_embeds,
        attention_mask     = full_mask,
        max_new_tokens     = max_new_tokens,
        do_sample          = do_sample,
        temperature        = temperature if do_sample else None,
        pad_token_id       = tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id       = tokenizer.eos_token_id,
    )
    # output_ids contains the newly generated tokens only
    # (because we passed inputs_embeds, not input_ids, the output
    #  starts from position 0 of the generated sequence)
    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# JSON parsing + pretty print
# ──────────────────────────────────────────────────────────────────────────────

def parse_output(raw: str) -> dict:
    """
    Extract JSON from model output.  Handles:
      - clean JSON string
      - JSON wrapped in ```json ... ``` fences
      - partial / malformed JSON (returns raw in '_raw' field)
    """
    text = raw.strip()

    # Strip markdown fences if present
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback — return raw so the caller can still log it
    return {"_parse_error": True, "_raw": raw}


def pretty_print(result: dict, obj_path: str):
    """Print parsed result field by field."""
    print(f"\n{'─'*60}")
    print(f"  OBJ: {obj_path}")
    print(f"{'─'*60}")

    if result.get("_parse_error"):
        print(f"  [PARSE ERROR] Raw output:\n{result.get('_raw', '')}")
        return

    for field in TARGET_FIELDS:
        val = result.get(field, "—")
        print(f"  {field:<24} {val}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    sys.path.insert(0, str(Path(__file__).parent))

    curv_ch_idx = CURV_CHANNEL_MAP[args.curv_channels]
    log.info(
        f"Curvature channels: "
        f"{[CURV_CHANNEL_NAMES[i] for i in curv_ch_idx]}"
    )

    # ── Load model ────────────────────────────────────────────────────────────
    llm, projector, tokenizer = load_model(
        ckpt_dir      = args.ckpt,
        model_id      = args.model_id,
        curv_channels = args.curv_channels,
        disk_grid     = args.disk_grid,
        proj_hidden   = args.proj_hidden,
        device        = args.device,
    )

    # ── Build job list ────────────────────────────────────────────────────────
    # Each job = (obj_path, obj_class)
    jobs: list[tuple[str, str]] = []

    if args.obj:
        # Single file mode
        jobs.append((args.obj, args.cls))

    elif args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv, sep=",", dtype=str)
        df.columns = df.columns.str.strip()

        if "path" not in df.columns or "class" not in df.columns:
            sys.exit("[error] CSV must have 'path' and 'class' columns")

        root = Path(args.root)
        for _, row in df.iterrows():
            obj_path = str(root / row["path"].strip())
            jobs.append((obj_path, row["class"].strip()))

        log.info(f"Batch mode: {len(jobs)} files from {args.csv}")

    else:
        sys.exit("[error] Provide either --obj or --csv")

    # ── Run inference ─────────────────────────────────────────────────────────
    results = []

    for i, (obj_path, obj_class) in enumerate(jobs):
        log.info(f"[{i+1}/{len(jobs)}] {obj_path}  class={obj_class}")

        # ── Geometry processing ───────────────────────────────────────────────
        try:
            patches, pos_enc = process_single_obj(
                obj_path    = obj_path,
                n_fps       = args.n_fps,
                disk_grid   = args.disk_grid,
                curv_ch_idx = curv_ch_idx,
            )
        except FileNotFoundError:
            log.error(f"  OBJ not found: {obj_path} — skipping")
            results.append({
                "path": obj_path, "class": obj_class,
                "_parse_error": True, "_raw": "OBJ not found",
            })
            continue
        except Exception as e:
            log.error(f"  Mesh processing failed: {e} — skipping")
            results.append({
                "path": obj_path, "class": obj_class,
                "_parse_error": True, "_raw": str(e),
            })
            continue

        # ── Generate ──────────────────────────────────────────────────────────
        try:
            raw = generate_caption(
                llm            = llm,
                projector      = projector,
                tokenizer      = tokenizer,
                patches        = patches,
                pos_enc        = pos_enc,
                obj_class      = obj_class,
                device         = args.device,
                max_new_tokens = args.max_new_tokens,
                temperature    = args.temperature,
                do_sample      = args.do_sample,
            )
        except Exception as e:
            log.error(f"  Generation failed: {e}")
            results.append({
                "path": obj_path, "class": obj_class,
                "_parse_error": True, "_raw": str(e),
            })
            continue

        # ── Parse ─────────────────────────────────────────────────────────────
        parsed = parse_output(raw)
        parsed["path"]  = obj_path
        parsed["class"] = obj_class
        results.append(parsed)

        pretty_print(parsed, obj_path)

    # ── Save results ──────────────────────────────────────────────────────────
    if args.out and results:
        import pandas as pd

        # Normalise: ensure all TARGET_FIELDS are columns even if missing
        rows = []
        for r in results:
            row = {"path": r.get("path", ""), "class": r.get("class", "")}
            for field in TARGET_FIELDS:
                row[field] = r.get(field, "")
            row["parse_error"] = r.get("_parse_error", False)
            row["raw_output"]  = r.get("_raw", "")
            rows.append(row)

        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.out, index=False)
        log.info(f"\nSaved {len(out_df)} predictions → {args.out}")

    elif not args.out:
        # Already pretty-printed above — nothing else to do
        pass

    log.info("Done.")


if __name__ == "__main__":
    main()