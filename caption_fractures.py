"""
caption_fractures.py
--------------------
Auto-captions fractured 3D object pair images using Qwen2.5-VL.

For each row in the input CSV (columns: path, class, pair_images_path):
  - Loads up to MAX_IMAGES_PER_RUN pair images
  - Runs the model N_CAPTIONS times to generate caption variants
  - Saves all captions to an output CSV

Usage:
    python caption_fractures.py --csv data.csv --output captions.csv
    python caption_fractures.py --csv data.csv --output captions.csv \
        --n_captions 5 --max_images 4 --batch_size 4

Requirements:
    pip install transformers torch pandas pillow qwen-vl-utils
"""

import argparse
import ast
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
import random
import pandas as pd
import torch
from PIL import Image
import gc
import torch


def unload_model(model=None, processor=None):

    try:
        del model
    except:
        pass

    try:
        del processor
    except:
        pass

    gc.collect()

    torch.cuda.empty_cache()

    torch.cuda.ipc_collect()

    print("Model unloaded.")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise 3D object analyst specializing in fracture and damage assessment. "
    "You will be given one or more rendered images of a 3D object pair and must respond "
    "with a single valid JSON object — no extra text, no markdown, no code blocks."
)

USER_PROMPT_TEMPLATE = """\
Each image contains a side-by-side pair of 3D rendered views of the same object:
- LEFT: the complete, intact object
- RIGHT: the main body of the object after a small fragment has broken off \
(this is the LARGER remaining piece, not the fragment itself), with the \
fracture surface highlighted in orange showing exactly where the piece broke away

The object class is: {class_name}

If multiple images are provided, they show the same pair from different viewpoints. \
Use all views together before answering.

Respond with exactly this JSON structure:

{{
  "object_description": "Describe the intact object (shown on the left): include its overall shape, structure, proportions, and any notable features.",

  "fragment_location": "Specify where the piece broke off from on the object. Use precise positional language (e.g. 'bottom-left corner of the base', 'upper section of the handle','the bottom base',...).". Indicate clearly if the fragment is Interior break or Boundary fracture

  "fracture_surface": "Describe the highlighted (orange) fracture region: include its shape (e.g., 'thin curved strip', 'irregular polygon',..), its relative size compared to the whole object (e.g., 'very small', 'about 10% of the surface'), and its orientation (e.g., 'curving along the rim', 'cutting diagonally across the body').",

  "missing_piece_size": "Estimate the size of the missing fragment relative to the whole object (e.g., 'tiny chip', 'small shard ~5% of the object', 'large fragment ~25% of the object').",

  "break_type": "Describe how the object likely broke, including direction and nature (e.g., 'a small chip from the rim due to impact', 'a clean horizontal break across the body', 'a diagonal fracture removing a corner').",

  "fragment_guess": "Describe what the missing fragment likely looked like, based on the fracture surface and object geometry (e.g., 'a curved rim segment', 'a flat rectangular shard', 'a wedge-shaped corner piece','an arm of a toy robot').",

  "confidence": "low | medium | high — indicate how confident you are in the above descriptions based on the available views and image quality.",

  "caption": "Provide a single concise sentence describing the object and its damage. Only include features you are confident about (e.g., 'A teapot with a chipped spout tip', 'A ceramic mug with a fracture near the handle base','A TeaCup missing a piece of its plate'). Indicate clearly if the fragment is Interior break or Boundary fracture"
}}
For every field:
- Use exactly ONE or TWO sentence
- Maximum 25–30 words
- No repetition or alternative explanations
- No uncertainty phrases unless confidence is low
- Try not using rim.
"""


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_id: str):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    log.info(f"Loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    log.info("Model ready.")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────

def load_and_resize(img_path, image_size):
    img = Image.open(img_path).convert("RGB")
    img.thumbnail(image_size)
    return img


def build_messages(image_paths: list[str], class_name: str, max_size=(256,512)) -> list[dict]:
    """Build the Qwen VL message list with interleaved images."""
    from qwen_vl_utils import process_vision_info  # noqa: F401 — imported for side effects

    content = []
    for p in image_paths:
        img = load_and_resize(p, max_size)

        content.append({
            "type": "image",
            "image": img,   # 🔥 pass PIL image instead of file://
        })

    content.append({
        "type": "text",
        "text": USER_PROMPT_TEMPLATE.format(class_name=class_name),
    })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]


def run_inference(model, processor, messages: list[dict],
                  max_new_tokens: int = 512) -> str:
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,          # sample for caption diversity across runs
            temperature=0.6,
            top_p=0.9,
        )

    # Strip the input tokens from the output
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def parse_json_output(raw: str) -> dict:
    """Try to extract a JSON object from model output, return raw on failure."""
    # Strip markdown fences if present
    cleaned = raw
    for fence in ["```json", "```"]:
        if fence in cleaned:
            cleaned = cleaned.split(fence, 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Return a structured error dict so the row is still logged
        return {
            "object_description": None,
            "fragment_location": None,
            "fracture_surface": None,
            "missing_piece_size": None,
            "break_type": None,
            "fragment_guess": None,
            "confidence": None,
            "caption": None,
            "_parse_error": True,
            "_raw_output": raw,
        }


# ── CSV helpers ───────────────────────────────────────────────────────────────

def parse_image_list(value: str) -> list[str]:
    """Parse the pair_images_path column — stored as a Python list literal."""
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(p) for p in parsed]
    except Exception:
        pass
    # Fallback: comma-separated
    return [p.strip().strip("'\"[]") for p in value.split(",") if p.strip()]


def load_existing_output(output_path: str) -> pd.DataFrame:
    """Load existing output CSV so we can resume interrupted runs."""
    if os.path.isfile(output_path):
        df = pd.read_csv(output_path)
        log.info(f"Resuming — found {len(df)} existing rows in {output_path}")
        return df
    return pd.DataFrame()


def already_done(existing_df: pd.DataFrame, obj_path: str, caption_idx: int) -> bool:
    if existing_df.empty:
        return False
    mask = (existing_df["path"] == obj_path) & (existing_df["caption_idx"] == caption_idx)
    return bool(mask.any())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-caption fractured 3D object pair images with Qwen2.5-VL."
    )
    parser.add_argument("--csv",        required=True,
                        help="Input CSV with columns: path, class, pair_images_path")
    parser.add_argument("--output",     default="captions.csv",
                        help="Output CSV path (default: captions.csv)")
    parser.add_argument("--model",      default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--n_captions", type=int, default=5,
                        help="Number of caption runs per object (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random (default: 42)")
    parser.add_argument("--max_images", type=int, default=4,
                        help="Max pair images to feed per inference call (default: 4). "
                             "Higher = more context but more VRAM. "
                             "Overview image is always included if present.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens per generation (default: 512)")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip rows already present in the output CSV")
    args = parser.parse_args()
        
    # ── Load input CSV ────────────────────────────────────────────────────
    df_in = pd.read_csv(args.csv, sep=",")
    # Normalise column names (strip whitespace)
    df_in.columns = [c.strip() for c in df_in.columns]

    required = {"path", "class", "pair_images_path"}
    missing  = required - set(df_in.columns)
    if missing:
        sys.exit(f"[error] Input CSV missing columns: {missing}. Found: {list(df_in.columns)}")

    log.info(f"Input CSV: {len(df_in)} rows")

    # ── Resume support ────────────────────────────────────────────────────
    existing_df = load_existing_output(args.output) if args.resume else pd.DataFrame()

    # ── Load model ────────────────────────────────────────────────────────
    model, processor = load_model(args.model)

    # ── Output buffer ──────────────────────────────────────────────────────
    output_rows = []

    def flush_to_csv():
        """Append buffered rows to output CSV and clear buffer."""
        if not output_rows:
            return
        out_df  = pd.DataFrame(output_rows)
        write_h = not os.path.isfile(args.output)
        out_df.to_csv(args.output, mode="a", header=write_h, index=False)
        output_rows.clear()
        log.info(f"  → flushed to {args.output}")

    # ── Main loop ─────────────────────────────────────────────────────────
    total   = len(df_in)
    ok      = 0
    skipped = 0
    failed  = 0

    for row_idx, row in df_in.iterrows():
        obj_path   = str(row["path"]).strip()
        class_name = str(row["class"]).strip()
        images_raw = row["pair_images_path"]

        log.info(f"[{row_idx + 1}/{total}] {obj_path}")

        # ── Select images ──────────────────────────────────────────────
        all_images = parse_image_list(images_raw)

        # Always include the overview image (contains "overview" in name)
        overview   = [p for p in all_images if "overview" in Path(p).stem.lower()]
        frac_views = [p for p in all_images if "overview" not in Path(p).stem.lower()]

        # Fill remaining slots with frac views (up to max_images - 1 for overview)
        n_slots       = max(1, args.max_images - len(overview))
        if len(frac_views) <= n_slots:
            selected_frac = frac_views
        else:
            selected_frac = random.sample(frac_views, n_slots)

        selected      = overview + selected_frac

        # Validate files exist
        missing_files = [p for p in selected if not os.path.isfile(p)]
        if missing_files:
            log.warning(f"  Missing image files: {missing_files}")
            selected = [p for p in selected if os.path.isfile(p)]

        if not selected:
            log.error(f"  No valid images found — skipping")
            failed += 1
            continue

        log.info(f"  Using {len(selected)}/{len(all_images)} images "
                 f"({len(overview)} overview + {len(selected_frac)} frac views)")

        messages = build_messages(selected, class_name)

        # ── N caption runs ─────────────────────────────────────────────
        obj_ok = True
        for cap_idx in range(args.n_captions):
            # if len(frac_views) <= n_slots:
            #     selected_frac = frac_views
            # else:
            #     selected_frac = random.sample(frac_views, n_slots)
    
            # selected      = overview + selected_frac
    
            # # Validate files exist
            # missing_files = [p for p in selected if not os.path.isfile(p)]
            # if missing_files:
            #     log.warning(f"  Missing image files: {missing_files}")
            #     selected = [p for p in selected if os.path.isfile(p)]
    
            # if not selected:
            #     log.error(f"  No valid images found — skipping")
            #     failed += 1
            #     continue
    
            # log.info(f"  Using {len(selected)}/{len(all_images)} images "
            #          f"({len(overview)} overview + {len(selected_frac)} frac views)")
    
            # messages = build_messages(selected, class_name)
            if args.resume and already_done(existing_df, obj_path, cap_idx):
                log.info(f"  Caption {cap_idx + 1}/{args.n_captions} — already done, skipping")
                skipped += 1
                continue

            log.info(f"  Caption {cap_idx + 1}/{args.n_captions} ...")
            t0 = time.time()
            error_threshold = 11
            try:
                raw    = run_inference(model, processor, messages, args.max_new_tokens)
                parsed = parse_json_output(raw)
                elapsed = time.time() - t0

                output_rows.append({
                    "path":               obj_path,
                    "class":              class_name,
                    "caption_idx":        cap_idx,
                    "images_used":        json.dumps(selected),
                    "object_description": parsed.get("object_description"),
                    "fragment_location":  parsed.get("fragment_location"),
                    "fracture_surface":   parsed.get("fracture_surface"),
                    "missing_piece_size": parsed.get("missing_piece_size"),
                    "break_type":         parsed.get("break_type"),
                    "fragment_guess":     parsed.get("fragment_guess"),
                    "confidence":         parsed.get("confidence"),
                    "caption":            parsed.get("caption"),
                    "parse_error":        parsed.get("_parse_error", False),
                    "raw_output":         raw,
                    "elapsed_s":          round(elapsed, 1),
                })
                if parsed.get("_parse_error"):
                    status = "⚠ parse error" 
                else:
                    status = "✔"
                    error_threshold = 11
                log.info(f"    {status} ({elapsed:.1f}s) confidence={parsed.get('confidence')}")
                
            except Exception as exc:
                error_threshold = error_threshold - 1
                log.error(f"    [error] caption {cap_idx}: {exc}")
                traceback.print_exc()
                output_rows.append({
                    "path":        obj_path,
                    "class":       class_name,
                    "caption_idx": cap_idx,
                    "images_used": json.dumps(selected),
                    "parse_error": True,
                    "raw_output":  str(exc),
                    "elapsed_s":   round(time.time() - t0, 1),
                })
                obj_ok = False
                if error_threshold == 0:
                    log.error(f"Too much crash. Program terminated at line {(row_idx + 1)*args.n_captions+cap_idx}")
                    #exit program
                    import sys
                    sys.exit(1)
                elif error_threshold % 3:
                    #try to fix
                    # unload old broken CUDA state
                    unload_model(model, processor)
            
                    # reload clean model
                    model, processor = load_model(args.model)
                    log.warning(f"Recovered model. Remaining retries: {error_threshold}")

        # Flush after every object so progress is saved incrementally
        flush_to_csv()

        if obj_ok:
            ok += 1
        else:
            failed += 1

    log.info(f"\nDone — {ok} objects captioned, {skipped} skipped, {failed} failed.")
    log.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()