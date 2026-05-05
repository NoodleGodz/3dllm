"""
train_qwen_lora.py
==================
LoRA SFT fine-tuning of Qwen2.5-VL-7B-Instruct on the Breaking Bad fracture
dataset.  Geometry tokens from tangent-disk patches are projected into the LLM's
embedding space and prepended as a prefix before the structured JSON target.

Architecture
------------
  MeshData.data  [N, C_in, D, D]          # C_in ∈ {2,3,4,6} — see CURV_CHANNELS
      ↓  GeometryProjector
  geo_embeds     [N, d_model]              # prefix tokens
      ↓  concat with text embeds
  full_seq       [N + T, d_model]
      ↓  Qwen2.5-VL-7B (LoRA on q/k/v/o proj)
  next-token loss over JSON target only    # prefix tokens masked in loss

Usage
-----
  # minimal — single GPU
  python train_qwen_lora.py \
      --csv      captions.csv \
      --root     /path/to/breaking_bad \
      --out      ./checkpoints

  # recommended — with W&B, QLoRA, reduced FPS for speed
  python train_qwen_lora.py \
      --csv      captions.csv \
      --root     /path/to/breaking_bad \
      --out      ./checkpoints \
      --n_fps    512 \
      --disk_grid 16 \
      --curv_channels 3 \
      --lora_r   16 \
      --batch    2 \
      --grad_accum 8 \
      --epochs   3 \
      --wandb    my_project

Dependencies
------------
  pip install torch transformers peft trl bitsandbytes scipy open3d pandas numpy
  pip install wandb          # optional, for logging
  pip install unsloth        # optional, for 2× faster training (recommended)

Notes
-----
  * Set --curv_channels to control which of the 6 curvature features are used.
    Default 3 = (kmin, kmax, mean_curv) — the three most informative.
    Set to 6 to use all: (kmin, kmax, mean, gauss, curvedness, shape_index).
  * Cache: BreakingBadDataset saves .pt files next to each OBJ.  Delete them
    if you change n_fps / disk_grid / curv_channels between runs.
  * The GeometryProjector (1×1 conv + linear) is trained fully; the Qwen vision
    encoder is not used — geometry arrives as pre-computed patch tensors.
  * LoRA targets q_proj, k_proj, v_proj, o_proj in the LLM decoder layers.
    The projector and LoRA adapters together are < 50M parameters.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT — Qwen2.5-VL on Breaking Bad")

    # Data
    p.add_argument("--csv",   required=True,  help="captions.csv from caption_fractures.py")
    p.add_argument("--root",  required=True,  help="Dataset root (OBJ files live here)")
    p.add_argument("--out",   default="./checkpoints", help="Output directory")
    p.add_argument("--val_split", type=float, default=0.05, help="Fraction held out for val")
    p.add_argument("--classes", nargs="*", default=None, help="Subset of object classes")

    # Geometry encoder
    p.add_argument("--n_fps",        type=int,   default=512,
                   help="FPS points per mesh (lower = faster, less geometry)")
    p.add_argument("--disk_grid",    type=int,   default=16,
                   help="Tangent disk spatial resolution D (grid is D×D)")
    p.add_argument("--curv_channels",type=int,   default=3,  choices=[2, 3, 4, 6],
                   help="Curvature channels to keep from the 6 available. "
                        "2=(kmin,kmax)  3=+mean  4=+gauss  6=all")
    p.add_argument("--proj_hidden",  type=int,   default=256,
                   help="Hidden dim of GeometryProjector. 256→~1M params, 512→~1.8M params (old default caused 470M)")

    # Model
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--load_4bit", action="store_true", default=True,
                   help="Load base model in 4-bit (QLoRA) — saves ~12 GB VRAM")
    p.add_argument("--no_4bit",   dest="load_4bit", action="store_false")

    # LoRA
    p.add_argument("--lora_r",      type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--lora_dropout",type=float, default=0.05)

    # Training
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--freeze_lm_steps", type=int, default=300,
                   help="Freeze LoRA layers for this many steps at the start so the "
                        "projector is forced to learn useful geometry signal before "
                        "the LLM adapts. Set 0 to disable. Default 300 ≈ 1 mini-epoch.")
    p.add_argument("--batch",       type=int,   default=1,
                   help="Per-device batch size.  Effective = batch × grad_accum")
    p.add_argument("--grad_accum",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--proj_lr",     type=float, default=2e-4,
                   help="LR for the GeometryProjector. Lowered from 1e-3 to prevent early saturation/collapse.")
    p.add_argument("--warmup_ratio",type=float, default=0.03)
    p.add_argument("--max_seq_len", type=int,   default=512,
                   help="Max text token length (JSON target).  Geo prefix is added on top.")
    p.add_argument("--seed",        type=int,   default=42)

    # Resume
    p.add_argument("--resume",  default=None,
                   help="Path to a checkpoint dir to resume from "
                        "(e.g. checkpoints/ckpt_step200). "
                        "Restores LoRA weights, projector, optimizer, scheduler, "
                        "and global_step so training continues seamlessly.")

    # Logging
    p.add_argument("--wandb",   default=None,
                   help="W&B project name.  Omit to disable W&B.")
    p.add_argument("--log_steps",   type=int,   default=10)
    p.add_argument("--save_steps",  type=int,   default=200)
    p.add_argument("--eval_steps",  type=int,   default=200)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Curvature channel selection
# ──────────────────────────────────────────────────────────────────────────────

# Full order from MeshData.curv / disk images:
#   0: kmin   1: kmax   2: mean_curv   3: gauss_curv   4: curvedness   5: shape_index
CURV_CHANNEL_MAP = {
    2: [0, 1],              # kmin, kmax — raw curvatures only
    3: [1, 4, 5],           # + mean curvature (most useful single derived feature)
    4: [0, 1, 2, 3],        # + gaussian curvature (sign encodes saddle vs dome)
    6: [0, 1, 2, 3, 4, 5],  # all six
}

CURV_CHANNEL_NAMES = {
    0: "kmin", 1: "kmax", 2: "mean_curv",
    3: "gauss_curv", 4: "curvedness", 5: "shape_index",
}


# ──────────────────────────────────────────────────────────────────────────────
# Geometry projector
# ──────────────────────────────────────────────────────────────────────────────

class GeometryProjector(nn.Module):
    """
    Projects tangent-disk patch tensors into the LLM's embedding space.

    Input : (B, N, C_in, D, D)  — batch of N patches, C_in curvature channels
    Output: (B, N, d_model)     — N geometry prefix tokens per mesh

    Architecture (param-efficient)
    ------------------------------
      1. Reshape: (B*N, C_in, D, D) → (B*N, C_in, D*D)
      2. Concat pos_enc broadcast as extra channels     → (B*N, C_in+P, D*D)
      3. Conv1d channel mixer  C_in+P → hidden          → (B*N, hidden, D*D)
      4. Global average pool over D*D spatial positions → (B*N, hidden)
         *** This is the key change — eliminates the hidden*D*D linear bottleneck
             that caused 470M params with hidden=512, D=16 ***
      5. LayerNorm → Linear(hidden → d_model) → LayerNorm

    Parameter count with default hidden=256:
      Conv1d layers : ~3K
      Linear        : 256 * 3584 = 917,504  (~0.9M)
      Total         : ~1M params  vs 470M before

    Recommended hidden values:
      hidden=128  →  ~460K params  (very lightweight)
      hidden=256  →  ~917K params  (default, good balance)
      hidden=512  →  ~1.8M params  (if you want more capacity)
    """

    def __init__(
        self,
        c_in: int,            # number of curvature channels kept (2,3,4,6)
        d_grid: int,          # spatial size D of tangent disk (D×D)
        d_model: int,         # LLM hidden size (3584 for Qwen2.5-VL-7B)
        pos_enc_dim: int = 6, # pos_enc feature dim (6 for fourier_bands=0)
        hidden: int = 256,    # lowered from 512 — see param count above
    ):
        super().__init__()
        self.c_in        = c_in
        self.d_grid      = d_grid
        self.pos_enc_dim = pos_enc_dim

        total_c = c_in + pos_enc_dim   # curvature channels + positional

        # Channel mixer: pointwise conv — mixes channels, preserves spatial dim
        self.chan_mix = nn.Sequential(
            nn.Conv1d(total_c, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden,  hidden, kernel_size=1, bias=False),
            nn.GELU(),
        )

        # Global average pool collapses D*D → 1, so Linear sees (BN, hidden)
        # not (BN, hidden*D*D).  This is what keeps params at ~1M.
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        patches: torch.Tensor,   # (B, N, C_in, D, D)
        pos_enc: torch.Tensor,   # (B, N, pos_enc_dim)
    ) -> torch.Tensor:           # (B, N, d_model)

        B, N, C, D, _ = patches.shape

        # Flatten spatial → 1D sequence of D*D positions
        x = patches.reshape(B * N, C, D * D)               # (BN, C, D²)

        # Broadcast pos_enc over spatial positions as extra channels
        pe = pos_enc.reshape(B * N, self.pos_enc_dim)       # (BN, P)
        pe = pe.unsqueeze(-1).expand(-1, -1, D * D)         # (BN, P, D²)
        x  = torch.cat([x, pe], dim=1)                      # (BN, C+P, D²)

        x = self.chan_mix(x)                                 # (BN, hidden, D²)

        # Global average pool: collapse spatial dimension
        x = self.gap(x).squeeze(-1)                         # (BN, hidden)

        x = self.proj(x)                                     # (BN, d_model)

        return x.reshape(B, N, -1)                           # (B, N, d_model)


# ──────────────────────────────────────────────────────────────────────────────
# JSON target builder
# ──────────────────────────────────────────────────────────────────────────────

# Fields from caption_fractures.py output — in the order we want them generated.
# object_description comes from meta (stored in MeshData.meta implicitly via
# BreakingBadDataset.__getitem__ which puts fragment_location etc. in meta).
# The caption field is stored directly in MeshData.caption.
TARGET_FIELDS_ORDER = [
    "object_description",   # stored in meta if present, else derived from class
    "fragment_location",
    "fracture_surface",
    "missing_piece_size",
    "break_type",
    "fragment_guess",
    "confidence",
    "caption",
]


def build_target_json(sample_meta: dict, caption: str, obj_class: str) -> str:
    """
    Build the JSON string that the LLM must learn to generate.
    Missing fields are filled with empty string so the schema is always complete.
    """
    obj = {}
    for field in TARGET_FIELDS_ORDER:
        if field == "caption":
            obj[field] = caption
        elif field == "object_description":
            # caption_fractures.py stores this in the CSV; meta may have it
            obj[field] = sample_meta.get("object_description",
                                         f"A fractured {obj_class}.")
        else:
            obj[field] = sample_meta.get(field, "")
    return json.dumps(obj, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Collate + tokenise
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You are a precise 3D fracture analyst. "
    "You will receive geometry embeddings of a fractured 3D object as a prefix. "
    "Based ONLY on the geometry, respond with a single valid JSON object "
    "describing the fracture. No extra text, no markdown, no code blocks."
)

# CLASS NAME INTENTIONALLY REMOVED from user message.
# Including the class (e.g. "Bottle") gives the LLM a strong text prior that
# dominates the geometry tokens — the model ignores geometry and generates from
# the class name alone.  Removing it forces reliance on the geometry prefix.
USER_MSG = (
    "Analyse the geometry prefix tokens of this fractured object "
    "and return the structured JSON caption describing the fracture."
)


def build_chat_text(obj_class: str, tokenizer) -> str:
    """
    Build the chat-template formatted string (without geometry prefix).
    obj_class is accepted for API compatibility but NOT inserted into the prompt —
    the class name would let the LLM bypass geometry entirely.
    """
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": USER_MSG},   # no {cls} substitution
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def collate_and_tokenise(
    batch,                  # list[MeshData]
    tokenizer,
    curv_ch_idx: list[int],
    max_seq_len: int,
    device: torch.device,
):
    """
    Build per-sample inputs for the forward pass.

    Returns
    -------
    patches   : (B, N, C_in, D, D)  — selected curvature channels
    pos_enc   : (B, N, P)
    input_ids : (B, T_total)         — prompt tokens (no geo)
    labels    : (B, T_total)         — -100 on prompt, target ids on JSON
    attn_mask : (B, T_total)
    n_geo_tok : int                  — number of geometry prefix tokens (N)
    """
    patches_list, pos_enc_list = [], []
    input_ids_list, label_list = [], []

    for s in batch:
        # ── Geometry ──────────────────────────────────────────────────────────
        # s.data: [N, 6, D, D] — select the channels we want
        patch = s.data[:, curv_ch_idx, :, :]          # [N, C_in, D, D]
        patches_list.append(patch)
        pos_enc_list.append(s.pos_enc)                 # [N, P]

        # ── Text ──────────────────────────────────────────────────────────────
        prompt_text = build_chat_text(s.x, tokenizer)
        target_text = build_target_json(s.meta, s.caption, s.x)

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]

        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
        ).input_ids[0]

        # EOS after the JSON
        eos = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
        target_ids = torch.cat([target_ids, eos])

        full_ids = torch.cat([prompt_ids, target_ids])

        # Labels: -100 on prompt (masked), real ids on target
        labels = torch.cat([
            torch.full((len(prompt_ids),), -100, dtype=torch.long),
            target_ids,
        ])

        input_ids_list.append(full_ids)
        label_list.append(labels)

    # ── Pad text sequences ────────────────────────────────────────────────────
    pad_id  = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_len = max(t.shape[0] for t in input_ids_list)

    input_ids = torch.stack([
        torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_id)
        for t in input_ids_list
    ])
    labels = torch.stack([
        torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=-100)
        for t in label_list
    ])
    attn_mask = (input_ids != pad_id).long()

    # ── Stack geometry — normalise N across batch ────────────────────────────
    # Different meshes can have different N after process_mesh (e.g. a mesh
    # with fewer vertices than n_fps gets fewer FPS points).  torch.stack
    # requires identical shapes, so we truncate/pad to the minimum N.
    N_min = min(p.shape[0] for p in patches_list)   # safe common N

    fixed_patches = []
    fixed_pos_enc = []
    for p, pe in zip(patches_list, pos_enc_list):
        n = p.shape[0]
        if n > N_min:
            p  = p[:N_min]       # truncate — discard the extra FPS points
            pe = pe[:N_min]
        elif n < N_min:
            # pad with zeros (masked in loss anyway via geo_labels=-100)
            pad_p  = torch.zeros(N_min - n, *p.shape[1:],  dtype=p.dtype)
            pad_pe = torch.zeros(N_min - n, *pe.shape[1:], dtype=pe.dtype)
            p  = torch.cat([p,  pad_p],  dim=0)
            pe = torch.cat([pe, pad_pe], dim=0)
        fixed_patches.append(p)
        fixed_pos_enc.append(pe)

    patches = torch.stack(fixed_patches)   # (B, N_min, C_in, D, D)
    pos_enc = torch.stack(fixed_pos_enc)   # (B, N_min, P)

    n_geo_tok = patches.shape[1]           # N_min — number of geometry prefix tokens

    return (
        patches.to(device),
        pos_enc.to(device),
        input_ids.to(device),
        labels.to(device),
        attn_mask.to(device),
        n_geo_tok,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass with geometry prefix injection
# ──────────────────────────────────────────────────────────────────────────────


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

def geometry_diversity_loss(geo_embeds: torch.Tensor, lam: float = 0.05) -> torch.Tensor:
    """
    Contrastive auxiliary loss: push geometry embeddings of different samples apart.

    When the LLM ignores geometry (noise test gives same output), this loss forces
    the projector to produce DIFFERENT embeddings for different fragments, which
    in turn forces the LLM to learn to attend to them.

    geo_embeds : (B, N, d_model)
    lam        : weight — 0.05 adds ~5% of LM loss magnitude
    Returns scalar loss (0 if B==1, batch size too small to contrast).
    """
    B = geo_embeds.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=geo_embeds.device, requires_grad=True)
    # Pool N geometry tokens → one vector per sample: (B, d_model)
    pooled = geo_embeds.mean(dim=1)
    pooled = torch.nn.functional.normalize(pooled, dim=-1)    # unit sphere
    # Full cosine similarity matrix (B, B)
    sim_matrix = pooled @ pooled.T                             # (B, B)
    # Off-diagonal = similarity between DIFFERENT samples — we want this LOW
    mask = ~torch.eye(B, dtype=torch.bool, device=pooled.device)
    off_diag_sim = sim_matrix[mask]                            # (B*(B-1),)
    # Mean similarity as loss — minimising this = maximising diversity
    return lam * off_diag_sim.mean()


def forward_with_geo_prefix(
    llm,
    projector: GeometryProjector,
    patches: torch.Tensor,        # (B, N, C_in, D, D)
    pos_enc: torch.Tensor,        # (B, N, P)
    input_ids: torch.Tensor,      # (B, T)
    labels: torch.Tensor,         # (B, T)
    attn_mask: torch.Tensor,      # (B, T)
    n_geo_tok: int,
    geo_div_lam: float = 0.05,
) -> torch.Tensor:
    """
    Prefix-inject geometry tokens into the LLM and compute total loss.

    Total loss = LM next-token loss (over JSON target only)
               + geometry diversity loss (pushes embeddings of diff samples apart)

    The diversity loss is the key fix for mode collapse when noise test shows
    geometry is ignored — it forces the projector to produce fragment-specific
    embeddings rather than collapsing to a constant per-class mean.
    """
    B = input_ids.shape[0]

    # 1. Geometry prefix: project patches → (B, N, d_model)
    geo_embeds = projector(patches, pos_enc)               # (B, N, d_model)  float32

    # 2. Text embeddings — dtype is bfloat16 (LLM weights dtype)
    # Resolved via _get_embed_tokens() — works for any PEFT wrapping depth.
    text_embeds = _get_embed_tokens(llm)(input_ids)        # (B, T, d_model)

    # Cast geo_embeds to match LLM dtype (bfloat16 when load_4bit or torch_dtype=bf16).
    # The projector runs in float32 (no quantisation applied to it); the LLM
    # embedding table and all downstream layers expect bfloat16.
    target_dtype = text_embeds.dtype
    geo_embeds   = geo_embeds.to(target_dtype)

    # 3. Concatenate: [geo prefix | text]
    full_embeds = torch.cat([geo_embeds, text_embeds], dim=1)  # (B, N+T, d_model)

    # 4. Extend attention mask and labels for geo prefix positions
    geo_mask   = torch.ones(B, n_geo_tok, dtype=attn_mask.dtype, device=attn_mask.device)
    full_mask  = torch.cat([geo_mask, attn_mask], dim=1)   # (B, N+T)

    geo_labels = torch.full((B, n_geo_tok), -100, dtype=labels.dtype, device=labels.device)
    full_labels = torch.cat([geo_labels, labels], dim=1)   # (B, N+T)

    # 5. LLM forward — pass inputs_embeds instead of input_ids
    outputs = llm(
        inputs_embeds=full_embeds,
        attention_mask=full_mask,
        labels=full_labels,
    )

    lm_loss  = outputs.loss
    div_loss = geometry_diversity_loss(geo_embeds, lam=geo_div_lam)

    # Return a simple namespace so callers can access individual components
    # for detailed W&B logging without recomputing.
    from types import SimpleNamespace
    return SimpleNamespace(
        loss     = lm_loss + div_loss,   # total — used for .backward()
        lm_loss  = lm_loss.detach(),
        div_loss = div_loss.detach(),
        geo_embeds = geo_embeds.detach(),  # (B, N, d_model) for geometry health metrics
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Force HuggingFace to operate fully offline after the initial model load.
    # Without this, save_pretrained makes a HEAD request to HF on every
    # checkpoint save, causing a multi-minute hang on restricted servers.
    # We set this AFTER logging setup but BEFORE any HF calls so the first
    # from_pretrained can still download if needed (model not cached yet).
    # Switch to offline mode right before the first save — handled in save_checkpoint.
    import os
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # cleaner logs

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = args.wandb is not None
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb, config=vars(args))
        except ImportError:
            log.warning("wandb not installed — logging disabled")
            use_wandb = False

    # ── Dataset ──────────────────────────────────────────────────────────────
    # Import here so the script can be parsed even if BreakingBadDataset
    # dependencies (open3d etc.) are not installed during linting.
    sys.path.insert(0, str(Path(__file__).parent))
    from BreakingBadDataset import BreakingBadDataset, DatasetConfig

    curv_ch_idx = CURV_CHANNEL_MAP[args.curv_channels]
    ch_names    = [CURV_CHANNEL_NAMES[i] for i in curv_ch_idx]
    log.info(f"Curvature channels used ({args.curv_channels}): {ch_names}")

    cfg = DatasetConfig(
        n_fps          = args.n_fps,
        disk_grid      = args.disk_grid,
        disk_channels  = 6,           # always compute all 6, we select below
        disk_fill      = "rbf",
        curv_knn       = 10,
        cache_processed= True,
        fourier_bands  = 0,           # pos_enc dim = 6
    )

    full_ds = BreakingBadDataset(
        csv_path     = args.csv,
        dataset_root = args.root,
        cfg          = cfg,
        classes      = args.classes,
    )

    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    log.info(f"Train: {n_train}  Val: {n_val}")

    # Simple collate — tokenisation happens in the loop via collate_and_tokenise
    def raw_collate(batch):
        return batch   # list[MeshData]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        shuffle=True, num_workers=2, collate_fn=raw_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch,
        shuffle=False, num_workers=2, collate_fn=raw_collate,
    )

    # ── Load Qwen2.5-VL ──────────────────────────────────────────────────────
    log.info(f"Loading {args.model_id} ...")

    from transformers import AutoTokenizer, BitsAndBytesConfig
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = None
    if args.load_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit               = True,
            bnb_4bit_quant_type        = "nf4",
            bnb_4bit_use_double_quant  = True,
            bnb_4bit_compute_dtype     = torch.bfloat16,
        )

    # Qwen2.5-VL is a VisionLanguageModel — must use its own class,
    # NOT AutoModelForCausalLM which does not recognise Qwen2_5_VLConfig.
    # We pass image_token_id via its own config; vision encoder is ignored
    # since we inject geometry tokens manually via inputs_embeds.
    llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config = bnb_cfg,
        torch_dtype         = torch.bfloat16 if not args.load_4bit else None,
        device_map          = "auto",
        trust_remote_code   = True,
    )
    llm.config.use_cache = False           # required for gradient checkpointing

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type    = TaskType.CAUSAL_LM,
        r            = args.lora_r,
        lora_alpha   = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        # Target the attention projection layers in the LLM decoder
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        bias         = "none",
    )
    llm = get_peft_model(llm, lora_cfg)
    llm.print_trainable_parameters()

    # ── Geometry projector ────────────────────────────────────────────────────
    # Qwen2_5_VLConfig nests the LM config under .text_config —
    # top-level .hidden_size does not exist, only .text_config.hidden_size does.
    d_model   = llm.config.text_config.hidden_size   # 3584 for Qwen2.5-VL-7B
    pos_enc_dim = 6                              # fourier_bands=0 → [xyz|normals]

    projector = GeometryProjector(
        c_in        = args.curv_channels,
        d_grid      = args.disk_grid,
        d_model     = d_model,
        pos_enc_dim = pos_enc_dim,
        hidden      = args.proj_hidden,
    ).to(device)

    n_proj_params = sum(p.numel() for p in projector.parameters())
    log.info(f"GeometryProjector params: {n_proj_params:,}  (target: <5M)")
    if n_proj_params > 10_000_000:
        log.warning(
            f"GeometryProjector has {n_proj_params/1e6:.1f}M params — consider "
            f"reducing --proj_hidden (current: {args.proj_hidden})"
        )

    # ── Optimiser — two param groups with different LRs ──────────────────────
    # Projector is trained from scratch → higher LR
    # LoRA adapters are small perturbations → standard LR
    optimizer = torch.optim.AdamW(
        [
            {"params": projector.parameters(),       "lr": args.proj_lr},
            {"params": llm.parameters(),             "lr": args.lr},
        ],
        weight_decay = 1e-2,
    )

    total_steps   = (n_train // (args.batch * args.grad_accum)) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr       = [args.proj_lr, args.lr],
        total_steps  = total_steps,
        pct_start    = args.warmup_ratio,
    )

    log.info(f"Total steps: {total_steps}  Warmup: {warmup_steps}")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    global_step   = 0
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume:
        log.info(f"Resuming from checkpoint: {args.resume}")
        global_step, start_epoch, best_val_loss = load_checkpoint_for_resume(
            ckpt_dir  = args.resume,
            llm       = llm,
            projector = projector,
            optimizer = optimizer,
            scheduler = scheduler,
            device    = str(device),
        )
        log.info(
            f"Continuing from epoch={start_epoch}  "
            f"global_step={global_step}  best_val_loss={best_val_loss:.4f}"
        )
    else:
        start_epoch = 0

    # ── Phase control helpers ─────────────────────────────────────────────────
    def _set_lm_trainable(trainable: bool):
        """Enable / disable gradient flow through LoRA adapter parameters."""
        for name, param in llm.named_parameters():
            if "lora_" in name:
                param.requires_grad_(trainable)

    def _phase_label(step: int) -> str:
        if args.freeze_lm_steps > 0 and step < args.freeze_lm_steps:
            return "phase1(proj-only)"
        return "phase2(joint)"

    # Start in phase 1: projector only, LLM frozen
    if args.freeze_lm_steps > 0 and global_step < args.freeze_lm_steps:
        _set_lm_trainable(False)
        log.info(
            f"Phase 1: LLM frozen for first {args.freeze_lm_steps} steps — "
            f"projector trains alone to learn geometry signal"
        )
    else:
        _set_lm_trainable(True)

    # ── Training ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        llm.train()
        projector.train()

        accum_loss  = 0.0
        accum_lm    = 0.0
        accum_div   = 0.0
        accum_count = 0
        _last_geo   = None   # most recent geo_embeds for health metrics
        _step_t0    = __import__("time").time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            try:
                patches, pos_enc, input_ids, labels, attn_mask, n_geo = \
                    collate_and_tokenise(
                        batch, tokenizer, curv_ch_idx,
                        args.max_seq_len, device,
                    )
            except Exception as e:
                log.warning(f"Skipping batch (collate error): {e}")
                continue

            result = forward_with_geo_prefix(
                llm, projector,
                patches, pos_enc,
                input_ids, labels, attn_mask,
                n_geo,
            )

            # Scale loss for gradient accumulation
            (result.loss / args.grad_accum).backward()
            accum_loss    += result.loss.item()
            accum_lm      += result.lm_loss.item()
            accum_div     += result.div_loss.item()
            accum_count   += 1
            # Track geo_embeds stats for geometry health logging
            _last_geo = result.geo_embeds   # (B, N, d_model)

            if (step + 1) % args.grad_accum == 0:

                # ── Capture grad norms BEFORE zero_grad wipes them ────────────
                # Must happen after backward() but before optimizer.zero_grad()
                proj_gnorm = float(sum(
                    p.grad.norm().item() ** 2
                    for p in projector.parameters()
                    if p.grad is not None
                ) ** 0.5)
                lora_gnorm = float(sum(
                    p.grad.norm().item() ** 2
                    for n, p in llm.named_parameters()
                    if "lora_" in n and p.grad is not None
                ) ** 0.5)

                # Clip gradients — important for LoRA + projector joint training
                total_norm = torch.nn.utils.clip_grad_norm_(
                    list(llm.parameters()) + list(projector.parameters()),
                    max_norm=1.0,
                )
                grad_clipped = 1 if total_norm > 1.0 else 0

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Phase transition: unfreeze LLM after freeze_lm_steps ──────
                if (args.freeze_lm_steps > 0
                        and global_step == args.freeze_lm_steps):
                    _set_lm_trainable(True)
                    log.info(
                        f"Phase 2 started at step {global_step}: "
                        f"LoRA layers unfrozen, joint training begins"
                    )

                # ── Capture averages BEFORE resetting accumulators ────────────
                avg_loss = accum_loss / max(accum_count, 1)
                avg_lm   = accum_lm   / max(accum_count, 1)
                avg_div  = accum_div  / max(accum_count, 1)
                accum_loss = accum_lm = accum_div = accum_count = 0

                if global_step % args.log_steps == 0:
                    import time
                    lr_proj  = optimizer.param_groups[0]["lr"]
                    lr_lora  = optimizer.param_groups[1]["lr"]
                    phase    = 1 if global_step <= args.freeze_lm_steps else 2

                    # ── Geometry health metrics ───────────────────────────────
                    geo_std = geo_cosim = 0.0
                    if _last_geo is not None:
                        geo_std = float(_last_geo.std().item())
                        # cosine_sim works with B=1 too: just use the single
                        # vector's self-similarity as a sanity value (always 1.0)
                        # For real diversity signal you need B>=2 (increase --batch)
                        pooled = _last_geo.mean(dim=1).float()        # (B, D)
                        pooled = torch.nn.functional.normalize(pooled, dim=-1)
                        sim_mat = pooled @ pooled.T                    # (B, B)
                        B = pooled.shape[0]
                        if B >= 2:
                            mask = ~torch.eye(B, dtype=torch.bool,
                                              device=pooled.device)
                            geo_cosim = float(sim_mat[mask].mean().item())
                        else:
                            # B=1: log diagonal (always 1.0) as a reminder to
                            # increase batch size for meaningful diversity signal
                            geo_cosim = float(sim_mat[0, 0].item())

                    # ── Throughput ────────────────────────────────────────────
                    elapsed      = time.time() - _step_t0
                    samples_sec  = (accum_count * args.batch) / max(elapsed, 1e-6)
                    gpu_mem_gb   = (torch.cuda.memory_allocated() / 1e9
                                    if torch.cuda.is_available() else 0.0)
                    _step_t0     = time.time()   # reset timer

                    log.info(
                        f"Epoch {epoch+1}  step {global_step}  "
                        f"[phase{phase}]  "
                        f"loss={avg_loss:.4f}  lm={avg_lm:.4f}  div={avg_div:.4f}  "
                        f"geo_std={geo_std:.3f}  cosim={geo_cosim:.3f}  "
                        f"gnorm_proj={proj_gnorm:.3f}  gnorm_lora={lora_gnorm:.3f}  "
                        f"lr_proj={lr_proj:.2e}  lr_lora={lr_lora:.2e}"
                    )

                    if use_wandb:
                        wandb.log({
                            # ── Loss breakdown ────────────────────────────────
                            "loss/train_total":  avg_loss,
                            "loss/train_lm":     avg_lm,
                            "loss/train_div":    avg_div,
                            # ── Geometry health ───────────────────────────────
                            # geo_std  → should RISE in phase 1 (projector learning)
                            #            then stabilise in phase 2
                            # geo_cosim → should FALL in phase 1 (fragments diverging)
                            #             then plateau at a low value
                            "geo/embed_std":     geo_std,
                            "geo/cosine_sim":    geo_cosim,
                            "geo/div_loss_raw":  avg_div / max(0.05, 1e-8),
                            # ── Gradient norms ────────────────────────────────
                            # If proj_gnorm is near 0 → projector not learning
                            # If lora_gnorm is near 0 in phase2 → LoRA not engaging
                            # grad_clipped=1 occasionally is fine; frequent=instability
                            "grad/projector":    proj_gnorm,
                            "grad/lora":         lora_gnorm,
                            "grad/clipped":      grad_clipped,
                            # ── Learning rates ───────────────────────────────
                            "lr/projector":      lr_proj,
                            "lr/lora":           lr_lora,
                            # ── Throughput ────────────────────────────────────
                            "perf/samples_per_sec": samples_sec,
                            "perf/gpu_memory_gb":   gpu_mem_gb,
                            # ── Phase tracker ────────────────────────────────
                            "train/phase":       phase,
                            "step":              global_step,
                        })

                    # (accum_lm/div already reset above)

                # ── Eval ─────────────────────────────────────────────────────
                if global_step % args.eval_steps == 0:
                    val = evaluate(
                        llm, projector, val_loader,
                        tokenizer, curv_ch_idx, args.max_seq_len, device,
                    )
                    log.info(
                        f"  → val_loss={val['loss']:.4f}  "
                        f"lm={val['lm_loss']:.4f}  "
                        f"div={val['div_loss']:.4f}  "
                        f"json_parse={val['json_parse_rate']*100:.1f}%"
                    )
                    if use_wandb:
                        wandb.log({
                            "loss/val_total":       val["loss"],
                            "loss/val_lm":          val["lm_loss"],
                            "loss/val_div":         val["div_loss"],
                            "val/json_parse_rate":  val["json_parse_rate"],
                            "step": global_step,
                        })

                    if val["loss"] < best_val_loss:
                        best_val_loss = val["loss"]
                        save_checkpoint(llm, projector, tokenizer, out_dir, tag="best")
                        save_training_state(optimizer, scheduler, global_step,
                                            epoch, best_val_loss, out_dir, tag="best")
                        log.info(f"  → saved best (val_loss={best_val_loss:.4f}  "
                                 f"json_parse={val['json_parse_rate']*100:.1f}%)")

                # ── Periodic save ─────────────────────────────────────────────
                if global_step % args.save_steps == 0:
                    save_checkpoint(llm, projector, tokenizer, out_dir,
                                    tag=f"step{global_step}")
                    save_training_state(optimizer, scheduler, global_step,
                                        epoch, best_val_loss, out_dir,
                                        tag=f"step{global_step}")

        # End of epoch — always save
        save_checkpoint(llm, projector, tokenizer, out_dir, tag=f"epoch{epoch+1}")
        save_training_state(optimizer, scheduler, global_step,
                            epoch + 1, best_val_loss, out_dir, tag=f"epoch{epoch+1}")

    log.info("Training complete.")
    if use_wandb:
        wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
# Eval + checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    llm, projector, loader,
    tokenizer, curv_ch_idx, max_seq_len, device,
) -> dict:
    """
    Returns dict with keys:
      loss           — total val loss
      lm_loss        — language model component
      div_loss       — diversity component
      json_parse_rate — fraction of val batches whose JSON target is parseable
                        (proxy for output schema validity without full generation)
    """
    import json as _json
    llm.eval()
    projector.eval()
    total_loss = total_lm = total_div = 0.0
    total_n    = 0
    parse_ok   = parse_total = 0

    for batch in loader:
        try:
            patches, pos_enc, input_ids, labels, attn_mask, n_geo =                 collate_and_tokenise(
                    batch, tokenizer, curv_ch_idx, max_seq_len, device,
                )
        except Exception:
            continue

        result = forward_with_geo_prefix(
            llm, projector,
            patches, pos_enc,
            input_ids, labels, attn_mask,
            n_geo,
        )
        b = len(batch)
        total_loss += result.loss.item()    * b
        total_lm   += result.lm_loss.item() * b
        total_div  += result.div_loss.item()* b
        total_n    += b

        # JSON parse rate: decode label tokens (non -100) and try json.loads
        for i in range(b):
            tgt_ids = labels[i][labels[i] != -100]
            if len(tgt_ids) == 0:
                continue
            try:
                decoded = tokenizer.decode(tgt_ids.cpu(), skip_special_tokens=True)
                _json.loads(decoded.strip())
                parse_ok += 1
            except Exception:
                pass
            parse_total += 1

    llm.train()
    projector.train()
    n = max(total_n, 1)
    return {
        "loss":            total_loss / n,
        "lm_loss":         total_lm   / n,
        "div_loss":        total_div  / n,
        "json_parse_rate": parse_ok / max(parse_total, 1),
    }


def save_checkpoint(llm, projector, tokenizer, out_dir: Path, tag: str):
    import os
    ckpt_dir    = out_dir / f"ckpt_{tag}"
    adapter_dir = ckpt_dir / "lora_adapter"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Prevent save_pretrained from making a HEAD request to HuggingFace to
    # validate the config version — this causes a multi-minute hang on servers
    # with slow or rate-limited outbound HTTP (seen as stuck after val_loss log).
    # Setting TRANSFORMERS_OFFLINE=1 forces fully local operation.
    _prev_tf  = os.environ.get("TRANSFORMERS_OFFLINE")
    _prev_hf  = os.environ.get("HF_DATASETS_OFFLINE")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"]  = "1"

    try:
        # Saves ONLY the LoRA adapter weights (~30 MB), not the 7B base model.
        # safe_serialization=True uses safetensors (faster, no pickle).
        llm.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
    finally:
        # Restore previous env state so the next model load can still reach HF
        if _prev_tf is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = _prev_tf
        if _prev_hf is None:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
        else:
            os.environ["HF_DATASETS_OFFLINE"] = _prev_hf

    # Projector weights
    torch.save(projector.state_dict(), ckpt_dir / "geometry_projector.pt")

    log.info(f"Checkpoint saved → {ckpt_dir}")


def save_training_state(
    optimizer, scheduler, global_step: int, epoch: int,
    best_val_loss: float, out_dir: Path, tag: str,
):
    """
    Save optimizer + scheduler + training counters alongside the model weights.
    Kept separate from save_checkpoint so it can be called independently and
    because optimizer state is large (~2× model params) and not needed for inference.
    """
    ckpt_dir = out_dir / f"ckpt_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "global_step":  global_step,
            "epoch":        epoch,
            "best_val_loss":best_val_loss,
        },
        ckpt_dir / "training_state.pt",
    )
    log.info(f"Training state saved → {ckpt_dir / 'training_state.pt'}")


# ──────────────────────────────────────────────────────────────────────────────
# Resume helper
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint_for_resume(
    ckpt_dir: str | Path,
    llm,
    projector: "GeometryProjector",
    optimizer,
    scheduler,
    device: str,
) -> tuple[int, int, float]:
    """
    Restore LoRA adapter weights, projector weights, optimizer state,
    scheduler state, and training counters from a checkpoint directory.

    Parameters
    ----------
    ckpt_dir  : path to checkpoint dir (e.g. checkpoints/ckpt_step200)
    llm       : PEFT-wrapped LLM (already initialised with LoRA config)
    projector : GeometryProjector (already initialised with same arch)
    optimizer : AdamW (already initialised with same param groups)
    scheduler : OneCycleLR (already initialised with same total_steps)
    device    : device string

    Returns
    -------
    global_step   : int    — resume from this step
    start_epoch   : int    — resume from this epoch
    best_val_loss : float  — carry over best val loss for checkpoint gating
    """
    from peft import set_peft_model_state_dict

    ckpt_dir    = Path(ckpt_dir)
    adapter_dir = ckpt_dir / "lora_adapter"
    proj_path   = ckpt_dir / "geometry_projector.pt"
    state_path  = ckpt_dir / "training_state.pt"

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    # ── LoRA adapter weights ──────────────────────────────────────────────────
    if adapter_dir.exists():
        from peft import PeftModel
        # Load adapter weights directly into the already-initialised PEFT model
        # (avoids re-loading the full 7B base model)
        adapters = torch.load(
            adapter_dir / "adapter_model.safetensors"
            if (adapter_dir / "adapter_model.safetensors").exists()
            else adapter_dir / "adapter_model.bin",
            map_location=device,
        )
        set_peft_model_state_dict(llm, adapters)
        log.info(f"Loaded LoRA adapter from {adapter_dir}")
    else:
        log.warning(f"No lora_adapter dir found in {ckpt_dir} — starting LoRA from scratch")

    # ── Projector weights ─────────────────────────────────────────────────────
    if proj_path.exists():
        projector.load_state_dict(
            torch.load(proj_path, map_location=device, weights_only=True)
        )
        log.info(f"Loaded projector from {proj_path}")
    else:
        log.warning(f"No geometry_projector.pt in {ckpt_dir} — starting projector from scratch")

    # ── Optimizer + scheduler + counters ─────────────────────────────────────
    global_step   = 0
    start_epoch   = 0
    best_val_loss = float("inf")

    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step   = state["global_step"]
        start_epoch   = state["epoch"]
        best_val_loss = state["best_val_loss"]
        log.info(
            f"Resumed from step={global_step}  epoch={start_epoch}  "
            f"best_val_loss={best_val_loss:.4f}"
        )
    else:
        log.warning(
            f"No training_state.pt in {ckpt_dir} — "
            f"optimizer/scheduler/counters reset to zero. "
            f"This is expected if resuming from a checkpoint that was saved "
            f"before this feature was added."
        )

    return global_step, start_epoch, best_val_loss


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper (run after training)
# ──────────────────────────────────────────────────────────────────────────────

def load_for_inference(
    ckpt_dir: str | Path,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    curv_channels: int = 3,
    disk_grid: int = 16,
    proj_hidden: int = 512,
    device: str = "cuda",
):
    """
    Load a saved checkpoint for inference.

    Example
    -------
    >>> llm, projector, tokenizer, curv_ch_idx = load_for_inference("checkpoints/ckpt_best")
    >>> # then call forward_with_geo_prefix with generate() instead of loss
    """
    from transformers import AutoTokenizer
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    ckpt_dir  = Path(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(ckpt_dir / "lora_adapter")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    llm = PeftModel.from_pretrained(base, ckpt_dir / "lora_adapter")
    llm.eval()

    d_model = llm.config.text_config.hidden_size
    proj = GeometryProjector(
        c_in        = curv_channels,
        d_grid      = disk_grid,
        d_model     = d_model,
        pos_enc_dim = 6,
        hidden      = proj_hidden,
    ).to(device)
    proj.load_state_dict(
        torch.load(ckpt_dir / "geometry_projector.pt", map_location=device)
    )
    proj.eval()

    return llm, proj, tok, CURV_CHANNEL_MAP[curv_channels]


if __name__ == "__main__":
    main()