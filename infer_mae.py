"""
infer_mae.py
============
Inference script for the MAE-based LoRA fine-tuned Qwen2.5-VL fracture captioner.

Matches train_mae.py exactly:
  - GeometryProjector: MAE encoder (chunked, Flash Attention SDPA)
  - Decoder not used at inference (mask_ratio=0)
  - _get_embed_tokens: PEFT-agnostic embed resolver
  - max_geo_tokens: adaptive avg pool before LLM
  - CURV_CHANNEL_MAP: same channel selection
  - Checkpoint layout: ckpt_dir/lora_adapter/ + geometry_projector.pt

Usage
-----
  # Single OBJ file
  python infer_mae.py \
      --ckpt  ./checkpoints/ckpt_best \
      --obj   path/to/fragment.obj \
      --cls   teapot

  # Batch from CSV (same format as training CSV)
  python infer_mae.py \
      --ckpt  ./checkpoints/ckpt_best \
      --csv   test_set.csv \
      --root  /path/to/breaking_bad \
      --out   predictions.csv

  # Must match training hyperparameters:
  python infer_mae.py \
      --ckpt ./checkpoints/ckpt_best \
      --csv  test_set.csv --root . \
      --n_fps 1024 --disk_grid 16 --curv_channels 3 \
      --proj_hidden 256 --mae_enc_layers 4 --mae_n_heads 4 \
      --max_geo_tokens 256

Notes
-----
  * Decoder weights (dec_embed, decoder, dec_norm, dec_pred) are loaded but
    never called — they stay in RAM. Use --merge to merge LoRA and drop decoder
    weights for a leaner inference binary.
  * use_cache is re-enabled at inference for fast KV-cache autoregressive decoding.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference — MAE GeometryProjector + Qwen2.5-VL LoRA fracture captioner"
    )

    # Checkpoint
    p.add_argument("--ckpt",     required=True,
                   help="Checkpoint dir (e.g. checkpoints/ckpt_best)")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="Base model ID (must match training)")

    # Input — single file OR batch CSV
    p.add_argument("--obj",   default=None, help="Single OBJ file path")
    p.add_argument("--cls",   default="unknown", help="Object class for single-file mode")
    p.add_argument("--csv",   default=None, help="CSV with columns: path, class (batch mode)")
    p.add_argument("--root",  default=".", help="Dataset root prepended to CSV paths")

    # Output
    p.add_argument("--out",   default=None,
                   help="Output CSV path (batch mode). Omit to print to stdout.")

    # Geometry — must match training exactly
    p.add_argument("--n_fps",         type=int,   default=512,
                   help="Must match training --n_fps")
    p.add_argument("--disk_grid",     type=int,   default=16,
                   help="Must match training --disk_grid")
    p.add_argument("--curv_channels", type=int,   default=3, choices=[2, 3, 4, 6],
                   help="Must match training --curv_channels")
    p.add_argument("--proj_hidden",   type=int,   default=256,
                   help="Must match training --proj_hidden")
    p.add_argument("--mae_enc_layers",type=int,   default=4,
                   help="Must match training --mae_enc_layers")
    p.add_argument("--mae_dec_layers",type=int,   default=2,
                   help="Must match training --mae_dec_layers (loaded but unused at inference)")
    p.add_argument("--mae_n_heads",   type=int,   default=4,
                   help="Must match training --mae_n_heads")
    p.add_argument("--mae_chunk_size",type=int,   default=128,
                   help="Chunked encoder batch size — can differ from training, lower = less VRAM")
    p.add_argument("--max_geo_tokens",type=int,   default=512,
                   help="Must match training --max_geo_tokens (caps prefix length to LLM)")

    # Generation
    p.add_argument("--max_new_tokens",type=int,   default=512)
    p.add_argument("--temperature",   type=float, default=0.1,
                   help="Low temperature for deterministic structured output")
    p.add_argument("--do_sample",     action="store_true", default=False)
    p.add_argument("--flash_attn",    action="store_true", default=False,
                   help="Enable Flash Attention 2 for LLM. "
                        "Requires: pip install flash-attn --no-build-isolation")
    p.add_argument("--device",        default="cuda")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Constants — must match train_mae.py exactly
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
    "object_description", "fragment_location", "fracture_surface",
    "missing_piece_size", "break_type", "fragment_guess", "confidence", "caption",
]

SYSTEM_MSG = (
    "You are a precise 3D fracture analyst. "
    "You will receive geometry embeddings of a fractured 3D object as a prefix. "
    "Based ONLY on the geometry, respond with a single valid JSON object "
    "describing the fracture. No extra text, no markdown, no code blocks."
)

USER_MSG = (
    "Analyse the geometry prefix tokens of this fractured object "
    "and return the structured JSON caption describing the fracture."
)


# ──────────────────────────────────────────────────────────────────────────────
# GeometryProjector — mirrors train_mae.py exactly
# ──────────────────────────────────────────────────────────────────────────────

class _TransformerBlock(nn.Module):
    """
    Single pre-norm transformer block — Flash Attention via SDPA (PyTorch 2.0+).
    Mirrors train_mae.py _TransformerBlock exactly.
    """
    def __init__(self, hidden: int, n_heads: int, mlp_ratio: float = 2.0,
                 dropout: float = 0.0):
        super().__init__()
        assert hidden % n_heads == 0, \
            f"hidden={hidden} must be divisible by n_heads={n_heads}"
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.dropout  = dropout

        self.norm1 = nn.LayerNorm(hidden)
        self.qkv   = nn.Linear(hidden, hidden * 3, bias=False)
        self.out   = nn.Linear(hidden, hidden,     bias=False)
        self.norm2 = nn.LayerNorm(hidden)
        dim_ff = int(hidden * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, dim_ff, bias=False),
            nn.GELU(),
            nn.Linear(dim_ff, hidden, bias=False),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        nh, hd  = self.n_heads, self.head_dim
        h   = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        dp = self.dropout if self.training else 0.0
        h  = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dp, is_causal=False,
        )
        h = h.permute(0, 2, 1, 3).reshape(B, S, H)
        h = self.out(h)
        x = x + self.drop(h)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class GeometryProjector(nn.Module):
    """
    MAE-style geometry projector — inference-mode.

    At inference mask_ratio=0: encoder sees all spatial tokens, decoder is unused.
    Encoder runs in chunks of chunk_size to keep QKV bounded.
    """
    def __init__(
        self,
        c_in: int,
        d_grid: int,
        d_model: int,
        pos_enc_dim: int = 6,
        hidden: int = 256,
        n_heads: int = 4,
        enc_layers: int = 4,
        dec_layers: int = 2,
        mask_ratio: float = 0.0,   # always 0 at inference
        dropout: float = 0.1,
        chunk_size: int = 128,
    ):
        super().__init__()
        self.c_in        = c_in
        self.d_grid      = d_grid
        self.pos_enc_dim = pos_enc_dim
        self.hidden      = hidden
        self.mask_ratio  = mask_ratio
        self.chunk_size  = chunk_size
        self.n_spatial   = d_grid * d_grid

        self.patch_embed = nn.Linear(c_in + pos_enc_dim, hidden, bias=False)
        self.spatial_pos = nn.Parameter(torch.zeros(1, self.n_spatial, hidden))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

        # mask_token loaded from checkpoint but never used at inference
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.encoder = nn.ModuleList([
            _TransformerBlock(hidden, n_heads, dropout=dropout)
            for _ in range(enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(hidden)

        # Decoder weights — loaded from checkpoint, unused at inference
        self.dec_embed = nn.Linear(hidden, hidden, bias=False)
        self.decoder   = nn.ModuleList([
            _TransformerBlock(hidden, n_heads, dropout=dropout)
            for _ in range(dec_layers)
        ])
        self.dec_norm  = nn.LayerNorm(hidden)
        self.dec_pred  = nn.Linear(hidden, c_in + pos_enc_dim, bias=True)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

    def _spatial_tokens(self, patches: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        BN, C, S = patches.shape
        x  = patches.permute(0, 2, 1)
        pe = pos_enc.unsqueeze(1).expand(-1, S, -1)
        x  = torch.cat([x, pe], dim=-1)
        x  = self.patch_embed(x)
        x  = x + self.spatial_pos.reshape(1, S, self.hidden)
        return x

    @torch.no_grad()
    def forward(
        self,
        patches: torch.Tensor,   # (B, N, C_in, D, D)
        pos_enc: torch.Tensor,   # (B, N, pos_enc_dim)
    ) -> torch.Tensor:           # (B, N, d_model)
        """
        Inference-mode forward — no masking, decoder not called.
        Chunked encoder keeps memory bounded.
        """
        B, N, C, D, _ = patches.shape
        BN = B * N
        S  = D * D

        x  = patches.reshape(BN, C, S)
        pe = pos_enc.reshape(BN, self.pos_enc_dim)
        tok = self._spatial_tokens(x, pe)   # (BN, S, hidden) — full, no masking

        chunk = self.chunk_size
        out_chunks = []
        for start in range(0, BN, chunk):
            end   = min(start + chunk, BN)
            t_ch  = tok[start:end]
            for blk in self.encoder:
                t_ch = blk(t_ch)
            t_ch = self.enc_norm(t_ch)
            pooled_ch = t_ch.mean(dim=1)         # (C, hidden)
            out_ch    = self.out_proj(pooled_ch)  # (C, d_model)
            out_chunks.append(out_ch)

        out = torch.cat(out_chunks, dim=0)        # (BN, d_model)
        return out.reshape(B, N, -1)              # (B, N, d_model)


# ──────────────────────────────────────────────────────────────────────────────
# Embed-token resolver — mirrors train_mae.py exactly
# ──────────────────────────────────────────────────────────────────────────────

def _get_embed_tokens(llm):
    for name, mod in llm.named_modules():
        if name.endswith("embed_tokens"):
            return mod
    raise AttributeError(
        "embed_tokens not found. "
        f"First 40 modules: {[n for n, _ in llm.named_modules()][:40]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(args) -> tuple:
    """
    Load the fine-tuned LLM (base + LoRA adapter) and GeometryProjector.

    Checkpoint layout:
        ckpt_dir/
            lora_adapter/          HF PeftModel weights + tokenizer
            geometry_projector.pt  plain state_dict
    """
    from transformers import AutoTokenizer
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    ckpt_dir    = Path(args.ckpt)
    adapter_dir = ckpt_dir / "lora_adapter"
    proj_path   = ckpt_dir / "geometry_projector.pt"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"No lora_adapter dir at {adapter_dir}")
    if not proj_path.exists():
        raise FileNotFoundError(f"No geometry_projector.pt at {proj_path}")

    log.info(f"Loading tokenizer from {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    # Flash Attention for LLM — mirrors train_mae.py logic
    if args.flash_attn:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            log.info("Flash Attention 2 enabled for LLM")
        except ImportError:
            attn_impl = "sdpa"
            log.warning("flash-attn not installed — falling back to SDPA")
    else:
        attn_impl = "sdpa"
        log.info("LLM attention: SDPA (PyTorch built-in Flash Attention)")

    log.info(f"Loading base model {args.model_id}")
    base_llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype          = torch.bfloat16,
        device_map           = args.device,
        trust_remote_code    = True,
        attn_implementation  = attn_impl,
    )

    log.info("Merging LoRA adapter")
    llm = PeftModel.from_pretrained(base_llm, adapter_dir)

    # Re-enable KV cache — was disabled during training for gradient checkpointing
    llm.config.use_cache = True
    llm.eval()

    log.info("Loading GeometryProjector")
    d_model = llm.config.text_config.hidden_size   # 3584 for Qwen2.5-VL-7B

    projector = GeometryProjector(
        c_in        = args.curv_channels,
        d_grid      = args.disk_grid,
        d_model     = d_model,
        pos_enc_dim = 6,                      # fourier_bands=0 during training
        hidden      = args.proj_hidden,
        n_heads     = args.mae_n_heads,
        enc_layers  = args.mae_enc_layers,
        dec_layers  = args.mae_dec_layers,
        mask_ratio  = 0.0,                    # always 0 at inference
        dropout     = 0.0,                    # no dropout at inference
        chunk_size  = args.mae_chunk_size,
    ).to(args.device)

    projector.load_state_dict(
        torch.load(proj_path, map_location=args.device, weights_only=True)
    )
    projector.eval()

    log.info(f"Projector params: {sum(p.numel() for p in projector.parameters()):,}")
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
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the full geometry pipeline on one OBJ file.
    Returns patches (1, N, C_in, D, D) and pos_enc (1, N, 6) on device.
    Reuses .pt cache if present — must match config used during training.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from BreakingBadDataset import DatasetConfig, process_mesh

    cfg = DatasetConfig(
        n_fps          = n_fps,
        disk_grid      = disk_grid,
        disk_channels  = 6,        # always compute all 6, select below
        disk_fill      = "rbf",
        curv_knn       = 30,
        cache_processed= True,
        fourier_bands  = 0,
    )

    tensors = process_mesh(Path(obj_path), cfg)

    # Select curvature channels used during training
    patches = tensors["data"][:, curv_ch_idx, :, :]   # [N, C_in, D, D]
    pos_enc = tensors["pos_enc"]                        # [N, 6]

    return (
        patches.unsqueeze(0).to(device),    # [1, N, C_in, D, D]
        pos_enc.unsqueeze(0).to(device),    # [1, N, 6]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt_ids(tokenizer, device: str) -> torch.Tensor:
    """Tokenise the system+user prompt. No class name — mirrors train_mae.py."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": USER_MSG},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer(
        text, add_special_tokens=False, return_tensors="pt",
    ).input_ids
    return ids.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Generation with geometry prefix
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_caption(
    llm,
    projector: GeometryProjector,
    tokenizer,
    patches: torch.Tensor,     # (1, N, C_in, D, D)
    pos_enc: torch.Tensor,     # (1, N, 6)
    args,
) -> str:
    """
    Geometry prefix injection + autoregressive generation.
    Mirrors the inference path of forward_with_geo_prefix in train_mae.py.
    """
    device = args.device

    # 1. Geometry prefix: (1, N, d_model) — chunked encoder, no masking
    geo_embeds = projector(patches, pos_enc)           # float32

    # 2. Text embeddings
    prompt_ids  = build_prompt_ids(tokenizer, device)  # (1, T_prompt)
    text_embeds = _get_embed_tokens(llm)(prompt_ids)   # (1, T_prompt, d_model)

    # 3. Cast geo to LLM dtype (bfloat16)
    geo_embeds_f32 = geo_embeds
    geo_embeds     = geo_embeds.to(text_embeds.dtype)
    del geo_embeds_f32

    # 4. Pool geo tokens if > max_geo_tokens — mirrors train_mae.py step 3
    max_geo = args.max_geo_tokens
    if geo_embeds.shape[1] > max_geo:
        ge_t       = geo_embeds.permute(0, 2, 1)
        ge_t       = torch.nn.functional.adaptive_avg_pool1d(ge_t, max_geo)
        geo_embeds = ge_t.permute(0, 2, 1)

    # 5. Concatenate [geo prefix | prompt]
    full_embeds = torch.cat([geo_embeds, text_embeds], dim=1)   # (1, N+T, d_model)

    # 6. Attention mask — all ones (no padding)
    full_mask = torch.ones(
        1, full_embeds.shape[1], dtype=torch.long, device=device,
    )

    # 7. Generate — inputs_embeds so model continues from our prefix
    output_ids = llm.generate(
        inputs_embeds  = full_embeds,
        attention_mask = full_mask,
        max_new_tokens = args.max_new_tokens,
        do_sample      = args.do_sample,
        temperature    = args.temperature if args.do_sample else None,
        pad_token_id   = tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id   = tokenizer.eos_token_id,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────────────
# JSON parsing + pretty print
# ──────────────────────────────────────────────────────────────────────────────

def parse_output(raw: str) -> dict:
    text = raw.strip()
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
    return {"_parse_error": True, "_raw": raw}


def pretty_print(result: dict, obj_path: str):
    print(f"\n{'─'*60}")
    print(f"  OBJ: {obj_path}")
    print(f"{'─'*60}")
    if result.get("_parse_error"):
        print(f"  [PARSE ERROR] Raw:\n{result.get('_raw', '')}")
        return
    for field in TARGET_FIELDS:
        print(f"  {field:<24} {result.get(field, '—')}")
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
        f"Config  n_fps={args.n_fps}  disk_grid={args.disk_grid}  "
        f"curv_channels={args.curv_channels} {[CURV_CHANNEL_NAMES[i] for i in curv_ch_idx]}  "
        f"max_geo_tokens={args.max_geo_tokens}"
    )

    # ── Load model ────────────────────────────────────────────────────────────
    llm, projector, tokenizer = load_model(args)

    # ── Build job list ────────────────────────────────────────────────────────
    jobs: list[tuple[str, str]] = []

    if args.obj:
        jobs.append((args.obj, args.cls))
    elif args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv, sep=",", dtype=str)
        df.columns = df.columns.str.strip()
        if "path" not in df.columns or "class" not in df.columns:
            sys.exit("[error] CSV must have 'path' and 'class' columns")
        root = Path(args.root)
        for _, row in df.iterrows():
            jobs.append((str(root / row["path"].strip()), row["class"].strip()))
        log.info(f"Batch mode: {len(jobs)} files")
    else:
        sys.exit("[error] Provide either --obj or --csv")

    # ── Run inference ─────────────────────────────────────────────────────────
    results = []

    for i, (obj_path, obj_class) in enumerate(jobs):
        log.info(f"[{i+1}/{len(jobs)}] {obj_path}  class={obj_class}")

        # Geometry
        try:
            patches, pos_enc = process_single_obj(
                obj_path, args.n_fps, args.disk_grid, curv_ch_idx, args.device,
            )
        except Exception as e:
            log.error(f"  Mesh processing failed: {e}")
            results.append({"path": obj_path, "class": obj_class,
                             "_parse_error": True, "_raw": str(e)})
            continue

        # Generate
        try:
            raw    = generate_caption(llm, projector, tokenizer, patches, pos_enc, args)
            parsed = parse_output(raw)
        except Exception as e:
            log.error(f"  Generation failed: {e}")
            results.append({"path": obj_path, "class": obj_class,
                             "_parse_error": True, "_raw": str(e)})
            continue

        parsed["path"]  = obj_path
        parsed["class"] = obj_class
        results.append(parsed)
        pretty_print(parsed, obj_path)

    # ── Save results ──────────────────────────────────────────────────────────
    if args.out and results:
        import pandas as pd
        rows = []
        for r in results:
            row = {"path": r.get("path", ""), "class": r.get("class", "")}
            for field in TARGET_FIELDS:
                row[field] = r.get(field, "")
            row["parse_error"] = r.get("_parse_error", False)
            row["raw_output"]  = r.get("_raw", "")
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        log.info(f"Saved {len(rows)} predictions → {args.out}")

    log.info("Done.")


if __name__ == "__main__":
    main()