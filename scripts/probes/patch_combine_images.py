#!/usr/bin/env python3
"""
Patch + combine image soft tokens from two images, then probe with
combined_ideology dimension, visualize token heatmaps, and generate descriptions.

Tiers:  baseline_A (vanilla img A), baseline_B (vanilla img B),
        patched (each combination: half_left_right, half_top_bottom, add, avg,
                 weighted, interleave).

For patched inputs: embeds are extracted from model.visual for both images,
combined, and injected into inputs_embeds at image-token positions.
The model runs without pixel_values (no deepstack), using simple position IDs.
Baselines run the full normal pipeline with pixel_values + deepstack.

Example:
    python scripts/probes/patch_combine_images.py \
        --model-path /home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct \
        --image-a data/congress_images/G000587.jpg \
        --image-b data/congress_images/N000190.jpg \
        --methods half_left_right half_top_bottom add avg weighted interleave \
        --alpha 0.5 --top-k 16 \
        --output-dir results/patch_combine/
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from probes.headwise_linear_probe import HeadwiseLinearProbe


# ===========================================================================
# Qwen3-VL grid helpers
# ===========================================================================

def _round_by_factor(number: float, factor: int) -> int:
    return int(round(number / factor) * factor)


def smart_resize_qwen(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int,
) -> Tuple[int, int]:
    """Reimplementation of HF smart_resize to avoid import issues."""
    if max(height, width) / max(min(height, width), 1) > 200:
        raise ValueError(f"Aspect ratio too large: {max(height, width) / min(height, width)}")
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = (h_bar * w_bar / max_pixels) ** 0.5
        h_bar = max(factor, _round_by_factor(h_bar / beta, factor))
        w_bar = max(factor, _round_by_factor(w_bar / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (h_bar * w_bar)) ** 0.5
        h_bar = _round_by_factor(h_bar * beta, factor)
        w_bar = _round_by_factor(w_bar * beta, factor)
    return int(h_bar), int(w_bar)


def _processor_factor(processor) -> int:
    ip = processor.image_processor
    return int(ip.patch_size * ip.merge_size)


def qwen3_grid_hw(img: Image.Image, processor) -> Tuple[int, int]:
    """Compute (grid_h, grid_w) *after merge* for a PIL image."""
    ip = processor.image_processor
    factor = _processor_factor(processor)
    min_px = ip.size["shortest_edge"]
    max_px = ip.size["longest_edge"]
    rh, rw = smart_resize_qwen(img.height, img.width, factor, min_px, max_px)
    return int(rh // factor), int(rw // factor)


# ===========================================================================
# Vision embedding extraction
# ===========================================================================

def _vision_preprocess(
    processor, img: Image.Image, device: torch.device, dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (pixel_values, image_grid_thw) on device."""
    ip = processor.image_processor
    out = ip.preprocess([img], return_tensors="pt")
    pv = out["pixel_values"].to(device=device, dtype=dtype)
    gt = out["image_grid_thw"].to(device=device)
    return pv, gt


def extract_vision_embeddings(
    model, img: Image.Image, processor, device: torch.device,
) -> Tuple[torch.Tensor, int, int, torch.Tensor, List[torch.Tensor]]:
    """
    Run a single image through model.visual.
    Returns:
        pooler_embeds: [N, D] float32 cpu  (merged features injected into LLM)
        grid_h, grid_w
        image_grid_thw on device
        deepstack: list of [N_k, D] per deepstack layer (empty if none)
    """
    grid_h, grid_w = qwen3_grid_hw(img, processor)
    factor = _processor_factor(processor)
    target_h, target_w = grid_h * factor, grid_w * factor
    if (img.height, img.width) != (target_h, target_w):
        img = img.resize((target_w, target_h), Image.LANCZOS)

    pv, gt = _vision_preprocess(processor, img, device, model.dtype)
    with torch.no_grad():
        vis_out = model.model.visual(pv, grid_thw=gt)

    pooler = vis_out.pooler_output
    deepstack = vis_out.deepstack_features if hasattr(vis_out, "deepstack_features") else []

    if pooler.dim() == 3:
        pooler = pooler.squeeze(0)
    pooler = pooler.to(dtype=torch.float32).cpu()

    expected = grid_h * grid_w
    actual = pooler.shape[0]
    if actual != expected:
        print(f"  [WARN] pooler tokens {actual} != grid {grid_h}×{grid_w}={expected}")
        # pad or truncate
        if actual < expected:
            pad = torch.zeros(expected - actual, pooler.shape[1],
                              dtype=pooler.dtype, device=pooler.device)
            pooler = torch.cat([pooler, pad], dim=0)
        else:
            pooler = pooler[:expected]

    ds_combined = []
    for ds in deepstack:
        if ds.dim() == 3:
            ds = ds.squeeze(0)
        ds = ds.to(dtype=torch.float32).cpu()
        if ds.shape[0] < expected:
            pad = torch.zeros(expected - ds.shape[0], ds.shape[1],
                              dtype=ds.dtype, device=ds.device)
            ds = torch.cat([ds, pad], dim=0)
        elif ds.shape[0] > expected:
            ds = ds[:expected]
        ds_combined.append(ds)

    return pooler, grid_h, grid_w, gt, ds_combined


# ===========================================================================
# Image token identification
# ===========================================================================

def image_token_ids_set(tokenizer) -> Set[int]:
    ids: Set[int] = set()
    for attr in ("image_token_id", "image_pad_token_id"):
        val = getattr(tokenizer, attr, None)
        if isinstance(val, int) and val >= 0:
            ids.add(int(val))
    val = getattr(tokenizer, "image_token_ids", None)
    if isinstance(val, (list, tuple)):
        ids.update(int(v) for v in val if isinstance(v, int) and v >= 0)
    for tok in ("<|image_pad|>", "<image_soft_token>", "<image>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id and int(tid) >= 0:
            ids.add(int(tid))
    return ids


# ===========================================================================
# Combination:  pooler  [N, D]  (and deepstack layers)
# ===========================================================================

def _reshape_flat(emb: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    return emb.reshape(gh, gw, emb.shape[1])


def _flatten_grid(g: torch.Tensor) -> torch.Tensor:
    return g.reshape(-1, g.shape[2])


def _combine_spatial_fmap(
    a: torch.Tensor, b: torch.Tensor, gh: int, gw: int,  # [N, D] tensors
    mode_left_right: bool,  # True => left/right split; False => top/bottom
) -> torch.Tensor:
    ag = _reshape_flat(a, gh, gw)
    bg = _reshape_flat(b, gh, gw)
    mid = gw // 2 if mode_left_right else gh // 2
    if mode_left_right:
        cg = torch.cat([ag[:, :mid, :], bg[:, mid:, :]], dim=1)
    else:
        cg = torch.cat([ag[:mid, :, :], bg[mid:, :, :]], dim=0)
    return _flatten_grid(cg)


# --- pooler-level combiners (return [N, D]) ---

def combine_half_lr(a: torch.Tensor, b: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    return _combine_spatial_fmap(a, b, gh, gw, mode_left_right=True)


def combine_half_tb(a: torch.Tensor, b: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    return _combine_spatial_fmap(a, b, gh, gw, mode_left_right=False)


def combine_add(a: torch.Tensor, b: torch.Tensor, *_args) -> torch.Tensor:
    return a + b


def combine_avg(a: torch.Tensor, b: torch.Tensor, *_args) -> torch.Tensor:
    return (a + b) / 2.0


def combine_weighted(a: torch.Tensor, b: torch.Tensor, alpha: float, *_args) -> torch.Tensor:
    return alpha * a + (1.0 - alpha) * b


def combine_interleave(a: torch.Tensor, b: torch.Tensor, *_args) -> torch.Tensor:
    n = a.shape[0]
    out = torch.empty_like(a)
    out[0::2] = a[0::2]
    out[1::2] = b[1::2]
    return out


# Map method_name -> (combinator, needs_gh_gw)
COMBINERS: Dict[str, Tuple[callable, bool]] = {
    "half_left_right":  (combine_half_lr,  True),
    "half_top_bottom":  (combine_half_tb,  True),
    "add":              (combine_add,      False),
    "avg":              (combine_avg,      False),
    "weighted":         (combine_weighted, False),
    "interleave":       (combine_interleave, False),
}


def apply_combination(
    method: str, alpha: float,
    pooler_a: torch.Tensor, pooler_b: torch.Tensor,
    deep_a: List[torch.Tensor], deep_b: List[torch.Tensor],
    gh: int, gw: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Combine pooler and deepstack features using the given method."""
    comb, needs_gh = COMBINERS[method]
    args = [gh, gw] if needs_gh else []
    if method == "weighted":
        args = [alpha]

    pooler = comb(pooler_a, pooler_b, *args)
    deep = []
    for da, db in zip(deep_a, deep_b):
        deep.append(comb(da, db, *args))
    return pooler, deep


# ===========================================================================
# Chat template & text decoding
# ===========================================================================

ASSISTANT_PREFIX = "<|im_start|>assistant\n"


def build_chat_encoded(
    processor, image_path: str, text: str, device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Encode [image, text] via chat template, return dict with tensors on device."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": text},
        ],
    }]
    out = processor.apply_chat_template(
        [messages], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    return {k: v.to(device=device) for k, v in out.items()}


def _image_token_positions(input_ids: torch.Tensor, tokenizer) -> np.ndarray:
    """Boolean array [seq_len] where image tokens are."""
    ids_np = input_ids.cpu().numpy().flatten()
    img_ids = image_token_ids_set(tokenizer)
    img_arr = np.array(sorted(img_ids), dtype=np.int64)
    return np.isin(ids_np, img_arr)


def decode_generated(gen_ids: torch.Tensor, prompt_len: int, processor) -> str:
    """Decode generated tokens only (skip prompt)."""
    new_ids = gen_ids[0, prompt_len:]
    text = processor.decode(new_ids, skip_special_tokens=True)
    # Strip trailing <|im_end|> if present
    text = text.replace("<|im_end|>", "").strip()
    # Remove the assistant_prefix if it appears at start
    if text.startswith(ASSISTANT_PREFIX.strip()):
        text = text[len(ASSISTANT_PREFIX.strip()):].strip()
    return text


# ===========================================================================
# Monkey-patch vision model for patched tiers
# ===========================================================================

def patch_vision_model(model, combined_pooler: torch.Tensor, combined_deepstack: List[torch.Tensor]):
    """Temporarily replace model.model.visual.forward to return combined features."""
    orig_forward = model.model.visual.forward

    def patched_forward(hidden_states, grid_thw, **kwargs):
        # Call original to get proper output type/structure, then replace values
        with torch.no_grad():
            out = orig_forward(hidden_states, grid_thw, **kwargs)
        out.pooler_output = combined_pooler.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if hasattr(out, "deepstack_features") and out.deepstack_features:
            out.deepstack_features = [d.to(device=hidden_states.device, dtype=hidden_states.dtype)
                                      for d in combined_deepstack]
        return out

    model.model.visual.forward = patched_forward
    return orig_forward


def unpatch_vision_model(model, orig_forward):
    """Restore original vision model forward."""
    model.model.visual.forward = orig_forward


# ===========================================================================
# Probing infrastructure
# ===========================================================================

def patch_vision_model(model, combined_pooler: torch.Tensor, combined_deepstack: List[torch.Tensor]):
    """Temporarily replace model.model.visual.forward to return combined features."""
    orig_forward = model.model.visual.forward

    def patched_forward(hidden_states, grid_thw, **kwargs):
        # Call original to get proper output type/structure, then replace values
        with torch.no_grad():
            out = orig_forward(hidden_states, grid_thw, **kwargs)
        out.pooler_output = combined_pooler.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if hasattr(out, "deepstack_features") and out.deepstack_features:
            out.deepstack_features = [d.to(device=hidden_states.device, dtype=hidden_states.dtype)
                                      for d in combined_deepstack]
        return out

    model.model.visual.forward = patched_forward
    return orig_forward


def unpatch_vision_model(model, orig_forward):
    """Restore original vision model forward."""
    model.model.visual.forward = orig_forward


# ===========================================================================
# Probing infrastructure
# ===========================================================================

def _get_head_names(model) -> List[str]:
    named = dict(model.named_modules())
    names = []
    for i in range(200):
        path = f"model.language_model.layers.{i}.self_attn.head_out"
        if path in named:
            names.append(path)
        elif names:
            break
    return names


def load_headwise_probe(model_path: str, prefix: str, data_dir: str):
    probe = HeadwiseLinearProbe(
        model_path=model_path, prefix=prefix, mode="vision",
        model_family="qwen3-vl", data_dir=data_dir,
    )
    probe.load()
    return probe


def _build_runtime(model, probe, top_k: int) -> Dict[str, Any]:
    head_names = _get_head_names(model)
    top = probe.topk_indices(k=top_k)
    groups: Dict[str, List[Tuple[int, torch.Tensor]]] = {}
    for li, hi in top:
        li, hi = int(li), int(hi)
        if li >= len(head_names):
            continue
        name = head_names[li]
        coef = np.asarray(probe.weights_[li][hi].coef_, dtype=np.float32)
        groups.setdefault(name, []).append((hi, torch.from_numpy(coef)))

    return {
        "probe_type": "headwise_linear",
        "head_names": head_names,
        "groups": groups,
        "normalizer": float(max(len(top), 1)),
        "module_names": set(groups.keys()),
    }


def _forward_with_hooks(model, encoded: Dict, module_names: Iterable[str]) -> Dict[str, torch.Tensor]:
    named = dict(model.named_modules())
    missing = [n for n in module_names if n not in named]
    if missing:
        raise RuntimeError(f"Missing modules: {missing[:3]}")

    captured: Dict[str, torch.Tensor] = {}
    hooks = []

    def _hook_for(name: str):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach().to(dtype=torch.float32).cpu()
        return fn

    for n in module_names:
        hooks.append(named[n].register_forward_hook(_hook_for(n)))

    try:
        with torch.no_grad():
            _ = model(**encoded)
    finally:
        for h in hooks:
            h.remove()
    return captured


def _score_from_captured(captured: Dict, runtime: Dict) -> torch.Tensor:
    total = None
    for mod_name, entries in runtime["groups"].items():
        t = captured[mod_name]  # [B, S, H, D]
        layer_total: Optional[torch.Tensor] = None
        for hi, coef in entries:
            c = coef.to(t.device, t.dtype)
            hs = torch.einsum("b s d, d -> b s", t[:, :, hi, :], c)
            layer_total = hs if layer_total is None else layer_total + hs
        total = layer_total if total is None else total + layer_total
    return (total / float(runtime["normalizer"])).to(torch.float32)


# ===========================================================================
# Heatmap rendering
# ===========================================================================

def _overlay_heatmap(
    img_pil: Image.Image, heatmap: np.ndarray,
    alpha=0.45, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
) -> np.ndarray:
    arr = np.asarray(img_pil.convert("RGB")).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    hm = heatmap.astype(np.float32)
    span = max(vmax - vmin, 1e-8)
    hm01 = np.clip((hm - vmin) / span, 0.0, 1.0)
    hm_up = np.array(Image.fromarray((hm01 * 255).astype(np.uint8)).resize(
        (W, H), resample=Image.NEAREST)) / 255.0
    cm = plt.get_cmap(cmap)
    rgb = cm(hm_up)[..., :3]
    return np.clip((1 - alpha) * arr + alpha * rgb, 0.0, 1.0)


def render_heatmap_single(
    scores: np.ndarray, img: Image.Image, title: str,
    path: str, vmin=-1.0, vmax=1.0,
):
    if scores.size == 0:
        return
    ov = _overlay_heatmap(img, scores, vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(ov)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_heatmap_side_by_side(
    scores: np.ndarray,
    img_a: Image.Image, img_b: Image.Image,
    method_name: str, path: str,
    gh: int, gw: int, vmin=-1.0, vmax=1.0,
):
    if scores.size == 0:
        return
    oa = _overlay_heatmap(img_a, scores, vmin=vmin, vmax=vmax)
    ob = _overlay_heatmap(img_b, scores, vmin=vmin, vmax=vmax)
    ha, wa = oa.shape[:2]
    hb, wb = ob.shape[:2]
    mh = max(ha, hb)
    canvas = np.ones((mh, wa + wb, 3), dtype=np.float32)
    canvas[:ha, :wa] = oa
    canvas[:hb, wa:wa + wb] = ob
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(canvas)
    ax.set_title(f"{method_name}  (grid {gh}×{gw})", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_score_stats(name: str, scores: np.ndarray):
    if len(scores) == 0:
        print(f"  {name} mean: n/a (no image tokens)")
        return float("nan")
    m = float(np.mean(scores))
    print(f"  {name} mean={m:.4f}  median={float(np.median(scores)):.4f}  "
          f"min={float(np.min(scores)):.4f}  max={float(np.max(scores)):.4f}")
    return m


# ===========================================================================
# Tier runners
# ===========================================================================

DESC_PROMPTS = [
    "Describe this politician's appearance and what their likely political views are.",
    "What would this politician say in a policy interview?",
]


def run_baseline(
    model, processor, image_path: str, probe_text: str,
    runtime: Dict, device: torch.device, out_dir: str, label: str,
) -> Dict[str, Any]:
    """Probe + heatmap + descriptions for a vanilla image."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n  BASELINE: {label}\n{'='*60}")

    img = Image.open(image_path).convert("RGB")
    gh, gw = qwen3_grid_hw(img, processor)
    print(f"  Image: {image_path}")
    print(f"  Grid:  {gh}×{gw}")

    # Probe
    enc = build_chat_encoded(processor, image_path, probe_text, device)
    captured = _forward_with_hooks(model, enc, runtime["head_names"])
    all_scores = _score_from_captured(captured, runtime).squeeze(0)

    img_mask = _image_token_positions(enc["input_ids"], processor.tokenizer)
    img_scores = all_scores.cpu().numpy()[img_mask]
    mean_score = _print_score_stats("image-token", img_scores)

    expected = gh * gw
    scores_2d = np.zeros((gh, gw), dtype=np.float32)
    n_copy = min(len(img_scores), expected)
    if n_copy > 0:
        scores_2d.flat[:n_copy] = img_scores[:n_copy]
    np.save(os.path.join(out_dir, "scores.npy"), scores_2d)

    render_heatmap_single(
        scores_2d, img, f"{label}\n(mean={mean_score:.3f})",
        os.path.join(out_dir, "heatmap.png"))

    # Descriptions
    descriptions = []
    for pi, dp in enumerate(DESC_PROMPTS):
        e = build_chat_encoded(processor, image_path, dp, device)
        p_len = e["input_ids"].shape[1]
        with torch.no_grad():
            gids = model.generate(**e, max_new_tokens=200, do_sample=False)
        txt = decode_generated(gids, p_len, processor)
        desc_path = os.path.join(out_dir, f"description_{pi}.txt")
        with open(desc_path, "w") as f:
            f.write(txt)
        descriptions.append(txt)
        print(f"  Description [{pi}]: {txt[:120]}...")

    return {"mean_score": mean_score, "descriptions": descriptions}


def run_patched(
    model, processor,
    combined_pooler: torch.Tensor,
    combined_deepstack: List[torch.Tensor],
    img_a: Image.Image, img_b: Image.Image,
    probe_text: str, method_name: str,
    runtime: Dict, device: torch.device,
    out_dir: str, gh: int, gw: int,
) -> Dict[str, Any]:
    """Probe + heatmap + descriptions for a patched combination via monkey-patching."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n  PATCHED: {method_name}\n{'='*60}")

    ref_path = "/tmp/_patch_ref_a.jpg"
    img_a.save(ref_path)

    # --- Monkey-patch vision model ---
    orig = patch_vision_model(model, combined_pooler, combined_deepstack)

    # --- Probe scoring ---
    enc = build_chat_encoded(processor, ref_path, probe_text, device)
    captured = _forward_with_hooks(model, enc, runtime["head_names"])
    all_scores = _score_from_captured(captured, runtime).squeeze(0)

    img_mask = _image_token_positions(enc["input_ids"], processor.tokenizer)
    img_scores = all_scores.cpu().numpy()[img_mask]
    mean_score = _print_score_stats("image-token", img_scores)

    expected = gh * gw
    scores_2d = np.zeros((gh, gw), dtype=np.float32)
    n_copy = min(len(img_scores), expected)
    if n_copy > 0:
        scores_2d.flat[:n_copy] = img_scores[:n_copy]
    np.save(os.path.join(out_dir, "scores.npy"), scores_2d)

    render_heatmap_side_by_side(
        scores_2d, img_a, img_b, method_name,
        os.path.join(out_dir, "heatmap.png"), gh, gw)

    # --- Description generation ---
    descriptions = []
    for pi, dp in enumerate(DESC_PROMPTS):
        e = build_chat_encoded(processor, ref_path, dp, device)
        p_len = e["input_ids"].shape[1]
        with torch.no_grad():
            gids = model.generate(**e, max_new_tokens=200, do_sample=False)
        txt = decode_generated(gids, p_len, processor)
        desc_path = os.path.join(out_dir, f"description_{pi}.txt")
        with open(desc_path, "w") as f:
            f.write(txt)
        descriptions.append(txt)
        print(f"  Description [{pi}]: {txt[:120]}...")

    # --- Unpatch ---
    unpatch_vision_model(model, orig)

    return {"mean_score": mean_score, "descriptions": descriptions}


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Patch + combine image soft tokens, probe ideology, heatmap, generate descriptions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--image-a", required=True, help="Image A (template for token layout)")
    p.add_argument("--image-b", required=True, help="Image B")
    p.add_argument("--methods", nargs="+", required=True,
                   choices=list(COMBINERS))
    p.add_argument("--alpha", type=float, default=0.5, help="Weight for weighted blend.")
    p.add_argument("--probe-prefix", default="combined_ideology")
    p.add_argument("--probe-text", default="What would this politician say in a policy interview?")
    p.add_argument("--data-dir", default="results/probes")
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--output-dir", default="results/patch_combine")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()

    from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration

    print("Loading model...")
    torch_dtype = getattr(torch, args.dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch_dtype, device_map=args.device)
    model.eval()
    processor = Qwen3VLProcessor.from_pretrained(args.model_path)
    device = torch.device(args.device) if isinstance(args.device, str) else args.device
    print(f"Model dtype: {model.dtype}  Device: {device}")

    # Load probe
    print("Loading probe...")
    probe = load_headwise_probe(args.model_path, args.probe_prefix, args.data_dir)
    runtime = _build_runtime(model, probe, args.top_k)
    n_heads_total = sum(len(v) for v in runtime["groups"].values())
    print(f"  Loaded {len(runtime['groups'])} head groups ({n_heads_total} heads), "
          f"normalizer={runtime['normalizer']}")

    # Load & resize images to common grid
    img_a = Image.open(args.image_a).convert("RGB")
    img_b = Image.open(args.image_b).convert("RGB")
    gha, gwa = qwen3_grid_hw(img_a, processor)
    ghb, gwb = qwen3_grid_hw(img_b, processor)
    tgh = max(gha, ghb)
    tgw = max(gwa, gwb)
    fac = _processor_factor(processor)
    target_px = (tgw * fac, tgh * fac)
    print(f"\nOriginal grids: A={gha}×{gwa}  B={ghb}×{gwb}")
    print(f"Common target:  {tgh}×{tgw}  ({target_px[0]}×{target_px[1]} px)")

    img_a_r = img_a.resize(target_px, Image.LANCZOS)
    img_b_r = img_b.resize(target_px, Image.LANCZOS)

    # Extract full vision features
    print("\nExtracting vision embeddings...")
    pooler_a, _, _, _, dsa = extract_vision_embeddings(model, img_a_r, processor, device)
    pooler_b, _, _, _, dsb = extract_vision_embeddings(model, img_b_r, processor, device)
    print(f"  A pooler: {list(pooler_a.shape)}  deepstack: {[d.shape for d in dsa]}")
    print(f"  B pooler: {list(pooler_b.shape)}  deepstack: {[d.shape for d in dsb]}")

    base_name = (f"{os.path.splitext(os.path.basename(args.image_a))[0]}_x_"
                 f"{os.path.splitext(os.path.basename(args.image_b))[0]}")
    out_root = os.path.join(args.output_dir, base_name)
    os.makedirs(out_root, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    # Baselines
    da = run_baseline(model, processor, args.image_a, args.probe_text,
                      runtime, device, os.path.join(out_root, "baseline_A"),
                      f"Baseline A ({os.path.basename(args.image_a)})")
    summary_rows.append({
        "tier": "baseline_A", "method": "none",
        "image": os.path.basename(args.image_a),
        "grid": f"{gha}x{gwa}",
        "mean_score": da["mean_score"],
        "desc0": da["descriptions"][0][:300] if da["descriptions"] else "",
        "desc1": da["descriptions"][1][:300] if len(da["descriptions"]) > 1 else "",
    })

    db = run_baseline(model, processor, args.image_b, args.probe_text,
                      runtime, device, os.path.join(out_root, "baseline_B"),
                      f"Baseline B ({os.path.basename(args.image_b)})")
    summary_rows.append({
        "tier": "baseline_B", "method": "none",
        "image": os.path.basename(args.image_b),
        "grid": f"{ghb}x{gwb}",
        "mean_score": db["mean_score"],
        "desc0": db["descriptions"][0][:300] if db["descriptions"] else "",
        "desc1": db["descriptions"][1][:300] if len(db["descriptions"]) > 1 else "",
    })

    # Patched
    for method in args.methods:
        if method not in COMBINERS:
            print(f"\n  [SKIP] unknown method: {method}")
            continue

        cpooler, cdeep = apply_combination(
            method, args.alpha, pooler_a, pooler_b, dsa, dsb, tgh, tgw)
        print(f"\n  {method}: combined pooler shape={list(cpooler.shape)}  "
              f"norm={cpooler.norm():.2f}  mean={cpooler.mean():.4f}")

        dp = run_patched(
            model, processor, cpooler, cdeep,
            img_a_r, img_b_r,
            args.probe_text, method,
            runtime, device, os.path.join(out_root, method),
            tgh, tgw)
        summary_rows.append({
            "tier": "patched", "method": method,
            "image": base_name,
            "grid": f"{tgh}x{tgw}",
            "mean_score": dp["mean_score"],
            "desc0": dp["descriptions"][0][:300] if dp["descriptions"] else "",
            "desc1": dp["descriptions"][1][:300] if len(dp["descriptions"]) > 1 else "",
        })

    # Summary CSV
    csv_path = os.path.join(out_root, "summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "tier", "method", "image", "grid", "mean_score", "desc0", "desc1"])
        w.writeheader()
        w.writerows(summary_rows)

    # Markdown summary
    md_lines = []
    md_lines.append("# Image Soft Token Patching Results")
    md_lines.append("")
    md_lines.append(f"**Image A:** `{os.path.basename(args.image_a)}`  ")
    md_lines.append(f"**Image B:** `{os.path.basename(args.image_b)}`  ")
    md_lines.append(f"**Probe:** {args.probe_prefix} (headwise_linear, top-k={args.top_k})  ")
    md_lines.append(f"**Grid:** {tgh}×{tgw} ({tgh*tgw} image tokens)")
    md_lines.append("")
    md_lines.append("## Probe Scores Summary")
    md_lines.append("")
    md_lines.append("| Tier | Method | Mean Score |")
    md_lines.append("|------|--------|-----------:|")
    for r in summary_rows:
        md_lines.append(f"| {r['tier']} | {r['method']} | {r['mean_score']:.4f} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    for r in summary_rows:
        md_lines.append(f"## {r['tier']} — {r['method']}  (score: {r['mean_score']:.4f})")
        md_lines.append("")
        md_lines.append('**Prompt 0:** "Describe this politician\'s appearance and what their likely political views are."')
        md_lines.append("")
        md_lines.append(f"> {r['desc0']}")
        md_lines.append("")
        md_lines.append('**Prompt 1:** "What would this politician say in a policy interview?"')
        md_lines.append("")
        md_lines.append(f"> {r['desc1']}")
        md_lines.append("")

    md_path = os.path.join(out_root, "summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'='*60}\nDone! Results: {out_root}\nSummary:   {csv_path}\nMarkdown:  {md_path}\n{'='*60}")


if __name__ == "__main__":
    main()
