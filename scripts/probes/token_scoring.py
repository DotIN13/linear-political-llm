"""
Score prompt token internals using trained probes from probes/cli.py artifacts.

This script:
1) Loads one or more trained probes (headwise_linear / layerwise_linear / layerwise_rfm)
2) Runs one forward pass per prompt
3) Reuses captured activations to score all requested probes
4) Saves image-token scores and all-token scores for each probe
5) Saves per-prompt summary CSV for each probe

Example (single probe):
python scripts/token_scoring.py \
  --model-path /home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --mode vision \
  --probe headwise_linear \
  --prefix combined_ideology \
  --data data/probes/some_prompts.jsonl \
  --has-images \
  --top-k 16

Example (multi probe, one pass):
python scripts/token_scoring.py \
  --model-path /home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct \
  --model-family qwen3-vl \
  --mode vision \
  --probe-specs combined_ideology:headwise_linear combined_ideology:layerwise_linear combined_ideology:layerwise_rfm \
  --data data/probes/some_prompts.jsonl \
  --has-images \
  --top-k 16
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
    MllamaForConditionalGeneration,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from probes import HeadwiseLinearProbe, LayerwiseLinearProbe, LayerwiseRFM
from probes.base import get_head_module_names, model_base_name, resolve_module_paths


PROBE_CLASSES = {
    "headwise_linear": HeadwiseLinearProbe,
    "layerwise_linear": LayerwiseLinearProbe,
    "layerwise_rfm": LayerwiseRFM,
}

DEFAULT_MODEL_PATH = "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
DEFAULT_DATA = "data/probes/combined_ideology.jsonl"
DEFAULT_PREFIX = "combined_ideology"
DEFAULT_PROMPT_FALLBACK = "Describe this image."
MAX_IMAGE_WIDTH = 800
MAX_IMAGE_HEIGHT = 800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score input tokens using trained probe artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-family", required=True, choices=["gemma4", "mllama", "qwen3-vl"])
    parser.add_argument("--mode", required=True, choices=["text", "vision"])

    # Backward-compatible single-probe args
    parser.add_argument("--probe", default="headwise_linear", choices=list(PROBE_CLASSES))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)

    # Multi-probe args: PREFIX:PROBE
    parser.add_argument(
        "--probe-specs",
        nargs="+",
        default=None,
        help="Optional list of PREFIX:PROBE (e.g. combined_ideology:headwise_linear).",
    )

    parser.add_argument("--data", default=DEFAULT_DATA, help="JSONL/JSON/CSV prompt manifest.")
    parser.add_argument("--has-images", action="store_true", help="Use image field when messages are absent.")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--image-field", default="image_path")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--default-text", default=DEFAULT_PROMPT_FALLBACK)

    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--data-dir", default="results/probes", help="Probe artifact root used by probes/cli.py")
    parser.add_argument(
        "--module-paths-json",
        default=None,
        metavar="JSON",
        help="JSON string or JSON file for module path overrides (same format as probes/cli.py).",
    )

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--activations-name", default=None)
    parser.add_argument("--all-tokens-name", default=None)
    parser.add_argument("--stats-name", default=None)

    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts to score per forward pass.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar output.")

    return parser.parse_args()


class NoOpProgress:
    def update(self, _n: int = 1) -> None:
        return

    def close(self) -> None:
        return


def make_progress(total: int, enabled: bool):
    if not enabled or tqdm is None:
        return NoOpProgress()
    return tqdm(total=total, desc="Scoring prompts", unit="record")


def batched(records: Sequence[Dict[str, Any]], batch_size: int):
    if batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    for start in range(0, len(records), batch_size):
        yield start, records[start : start + batch_size]


def load_records(path: str) -> List[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()

    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if suffix == ".json":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}.")
        return payload

    if suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    raise ValueError(f"Unsupported data format '{suffix}'. Use .jsonl/.json/.csv")


def ensure_writable(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def parse_module_paths(raw: Optional[str], model_family: str):
    if raw is None:
        return None
    if os.path.isfile(raw):
        with open(raw, encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(raw)
    return resolve_module_paths(model_family, payload)


def parse_probe_specs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.probe_specs:
        specs: List[Tuple[str, str]] = []
        for raw in args.probe_specs:
            if ":" not in raw:
                raise ValueError(f"Invalid --probe-spec '{raw}'. Expected PREFIX:PROBE")
            prefix, probe_name = raw.split(":", 1)
            prefix = prefix.strip()
            probe_name = probe_name.strip()
            if not prefix:
                raise ValueError(f"Invalid --probe-spec '{raw}': empty prefix")
            if probe_name not in PROBE_CLASSES:
                raise ValueError(f"Invalid probe '{probe_name}' in --probe-spec '{raw}'")
            specs.append((prefix, probe_name))
        return specs

    return [(args.prefix, args.probe)]


def resolve_output_paths(
    output_dir: str,
    prefix: str,
    probe_name: str,
    activations_name: Optional[str],
    all_tokens_name: Optional[str],
    stats_name: Optional[str],
) -> Tuple[str, str, str]:
    ridge_tag = f"{prefix}_{probe_name}"
    image_name = activations_name or f"prompt_image_token_scores_{ridge_tag}.pt"
    all_name = all_tokens_name or f"prompt_all_token_scores_{ridge_tag}.pt"
    csv_name = stats_name or f"prompt_token_stats_{ridge_tag}.csv"
    return (
        os.path.join(output_dir, image_name),
        os.path.join(output_dir, all_name),
        os.path.join(output_dir, csv_name),
    )


def resolve_image(raw_path: Any, image_root: Optional[str]) -> str:
    path = os.fspath(raw_path)
    if image_root is not None and not os.path.isabs(path):
        path = os.path.join(image_root, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def prepare_image_for_scoring(
    image_path: str,
    cache_dir: str,
    max_width: int = MAX_IMAGE_WIDTH,
    max_height: int = MAX_IMAGE_HEIGHT,
) -> Optional[str]:
    abs_path = os.path.abspath(image_path)
    try:
        src_mtime = os.path.getmtime(abs_path)
    except OSError:
        return None

    _, ext = os.path.splitext(abs_path)
    ext = ext.lower() if ext else ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        ext = ".jpg"

    cache_key = hashlib.sha1(f"{abs_path}:{src_mtime}:{max_width}x{max_height}".encode("utf-8")).hexdigest()
    resized_path = os.path.join(cache_dir, f"{cache_key}{ext}")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(resized_path):
        return resized_path

    try:
        with Image.open(abs_path) as img:
            img = ImageOps.exif_transpose(img)
            resized = img.copy()
            resized.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            save_img = resized
            if ext in {".jpg", ".jpeg"} and save_img.mode in {"RGBA", "LA", "P"}:
                save_img = save_img.convert("RGB")
            save_img.save(resized_path)
    except (OSError, ValueError):
        return None

    return resized_path


def resolve_messages_images(
    messages: Sequence[Dict[str, Any]],
    image_root: Optional[str],
    cache_dir: str,
) -> Optional[List[Dict[str, Any]]]:
    resolved: List[Dict[str, Any]] = []
    for message in messages:
        new_message = dict(message)
        content = message.get("content")

        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                    new_item = dict(item)
                    source_path = resolve_image(item["image"], image_root)
                    resized_path = prepare_image_for_scoring(source_path, cache_dir=cache_dir)
                    if resized_path is None:
                        return None
                    new_item["image"] = resized_path
                    new_content.append(new_item)
                else:
                    new_content.append(item)
            new_message["content"] = new_content

        resolved.append(new_message)

    return resolved


def build_messages(record: Dict[str, Any], args: argparse.Namespace, cache_dir: str) -> Optional[List[Dict[str, Any]]]:
    raw_messages = record.get(args.messages_field)
    if raw_messages is not None:
        if not isinstance(raw_messages, list):
            raise ValueError(f"Expected '{args.messages_field}' to be a list.")
        return resolve_messages_images(raw_messages, image_root=args.image_root, cache_dir=cache_dir)

    text = str(record.get(args.text_field, "")).strip()
    if not text:
        text = args.default_text

    if args.has_images:
        raw_path = record.get(args.image_field)
        if raw_path is None:
            raise KeyError(f"Missing '{args.image_field}' in record with text '{text[:40]}...'")
        image_path = resolve_image(raw_path, args.image_root)
        image_path = prepare_image_for_scoring(image_path, cache_dir=cache_dir)
        if image_path is None:
            return None
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image", "image": image_path},
            ],
        }]

    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def encode_prompts(processor, messages_batch: Sequence[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
    encoded = processor.apply_chat_template(
        list(messages_batch),
        tokenize=True,
        add_generation_prompt=True,
        processor_kwargs={"padding": True},
        return_dict=True,
        return_tensors="pt",
    )

    return {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, dtype_name)


def select_model_loader(model_family: str, model_path: str):
    path_lower = model_path.lower()

    if model_family == "qwen3-vl" or "qwen3-vl" in path_lower:
        return AutoModelForImageTextToText
    if model_family == "gemma4" or "gemma-4" in path_lower or "gemma4" in path_lower:
        return AutoModelForMultimodalLM
    if model_family == "mllama" or ("llama-3.2" in path_lower and "vision" in path_lower):
        return MllamaForConditionalGeneration
    return AutoModelForImageTextToText


def move_to_device(encoded: Dict[str, Any], model) -> Dict[str, Any]:
    try:
        model_device = model.device
    except AttributeError:
        model_device = next(model.parameters()).device

    return {
        key: value.to(model_device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


def flatten_last_dim(layer_out: torch.Tensor) -> torch.Tensor:
    if layer_out.dim() == 4:
        return layer_out.reshape(layer_out.size(0), layer_out.size(1), -1)
    if layer_out.dim() == 3:
        return layer_out
    raise ValueError(f"Unsupported head_out tensor shape: {tuple(layer_out.shape)}")


def gather_candidate_image_token_ids(tokenizer) -> Set[int]:
    token_ids: Set[int] = set()

    for attr in ("image_token_id", "image_pad_token_id"):
        if hasattr(tokenizer, attr):
            value = getattr(tokenizer, attr)
            if isinstance(value, int) and value >= 0:
                token_ids.add(int(value))

    if hasattr(tokenizer, "image_token_ids"):
        value = getattr(tokenizer, "image_token_ids")
        if isinstance(value, (list, tuple)):
            token_ids.update(int(v) for v in value if isinstance(v, int) and v >= 0)

    token_candidates = ["<|image_pad|>", "<image_soft_token>", "<image>"]
    for token in token_candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id and int(token_id) >= 0:
            token_ids.add(int(token_id))

    return token_ids


def extract_first_image_path(messages: Sequence[Dict[str, Any]]) -> str:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image" and "image" in item and isinstance(item["image"], str):
                return item["image"]
    return ""


def load_image_size_from_path(image_path: str) -> Tuple[int, int]:
    if not image_path:
        return -1, -1
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        return int(img.height), int(img.width)


def qwen3_grid_hw_from_path(image_path: str, processor) -> Tuple[int, int]:
    image_h, image_w = load_image_size_from_path(image_path)
    if image_h <= 0 or image_w <= 0:
        return -1, -1

    image_processor = processor.image_processor
    min_pixels = image_processor.size["shortest_edge"]
    max_pixels = image_processor.size["longest_edge"]
    patch = image_processor.patch_size
    merge = image_processor.merge_size

    factor = patch * merge
    resized_h, resized_w = smart_resize(image_h, image_w, factor, min_pixels=min_pixels, max_pixels=max_pixels)
    grid_h = (resized_h // patch) // merge
    grid_w = (resized_w // patch) // merge
    return int(grid_h), int(grid_w)


def gemma4_grid_hw_from_encoded(encoded: Dict[str, Any], processor, batch_index: int = 0) -> Tuple[int, int]:
    image_pos = encoded.get("image_position_ids")
    if image_pos is None:
        return -1, -1

    pos0 = image_pos[batch_index]
    valid = (pos0[:, 0] >= 0) & (pos0[:, 1] >= 0)
    if not valid.any():
        return 0, 0

    patch_grid_w = int(pos0[valid, 0].max().item() + 1)
    patch_grid_h = int(pos0[valid, 1].max().item() + 1)
    pool_k = getattr(processor.image_processor, "pooling_kernel_size", 1)
    grid_w = patch_grid_w // int(pool_k)
    grid_h = patch_grid_h // int(pool_k)
    return int(grid_h), int(grid_w)


def reconstruct_grid_hw(
    model_family: str,
    image_path: str,
    encoded: Dict[str, Any],
    processor,
    batch_index: int = 0,
) -> Tuple[int, int]:
    if model_family == "qwen3-vl" and image_path:
        return qwen3_grid_hw_from_path(image_path, processor)
    if model_family == "gemma4":
        return gemma4_grid_hw_from_encoded(encoded, processor, batch_index=batch_index)
    return -1, -1


def build_probe_runtime(
    model,
    probe,
    top_k: int,
    mode: str,
    model_family: str,
) -> Dict[str, Any]:
    if probe.weights_ is None or probe.scores_ is None:
        raise ValueError("Probe must be loaded before runtime build.")

    head_names = get_head_module_names(
        model,
        mode=mode,
        model_family=model_family,
        module_paths=probe.module_paths,
    )

    probe_type = probe.probe_type
    runtime: Dict[str, Any] = {
        "probe_type": probe_type,
        "head_names": head_names,
        "groups": {},
        "normalizer": 1.0,
        "module_names": set(),
    }

    if probe_type == "headwise_linear":
        top_units = probe.topk_indices(k=top_k)
        groups: Dict[str, List[Tuple[int, torch.Tensor]]] = {}

        for li, hi in top_units:
            li = int(li)
            hi = int(hi)
            if li >= len(head_names):
                raise IndexError(
                    f"Probe selected layer index {li}, but model exposes only {len(head_names)} probeable layers."
                )
            module_name = head_names[li]
            coef = np.asarray(probe.weights_[li][hi].coef_, dtype=np.float32)
            groups.setdefault(module_name, []).append((hi, torch.from_numpy(coef)))

        runtime["groups"] = groups
        runtime["normalizer"] = float(max(len(top_units), 1))
        runtime["module_names"] = set(groups.keys())
        return runtime

    if probe_type in {"layerwise_linear", "rfm_layerwise"}:
        top_layers = [int(x) for x in probe.topk_layers(k=top_k)]
        groups_layer: Dict[str, torch.Tensor] = {}

        for li in top_layers:
            if li >= len(head_names):
                raise IndexError(
                    f"Probe selected layer index {li}, but model exposes only {len(head_names)} probeable layers."
                )
            module_name = head_names[li]
            if probe_type == "layerwise_linear":
                vec = np.asarray(probe.weights_[li].coef_, dtype=np.float32)
            else:
                vec = np.asarray(probe.weights_[li], dtype=np.float32)
            groups_layer[module_name] = torch.from_numpy(vec)

        runtime["groups"] = groups_layer
        runtime["normalizer"] = float(max(len(top_layers), 1))
        runtime["module_names"] = set(groups_layer.keys())
        return runtime

    raise ValueError(f"Unsupported probe type: {probe_type}")


def capture_module_outputs(
    model,
    encoded: Dict[str, Any],
    module_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    named_modules = dict(model.named_modules())
    missing = [name for name in module_names if name not in named_modules]
    if missing:
        raise ValueError(f"Missing expected modules for capture: {missing[:3]}")

    outputs: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(_module, _inp, out):
            tensor_out = out[0] if isinstance(out, tuple) else out
            outputs[name] = tensor_out.detach().to(dtype=torch.float32).cpu()

        return hook_fn

    for name in module_names:
        hooks.append(named_modules[name].register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            _ = model(**move_to_device(encoded, model))
    finally:
        for hook in hooks:
            hook.remove()

    return outputs


def score_from_captured(
    captured: Dict[str, torch.Tensor],
    runtime: Dict[str, Any],
) -> torch.Tensor:
    probe_type = runtime["probe_type"]
    normalizer = runtime["normalizer"]

    token_scores = None

    if probe_type == "headwise_linear":
        for module_name, head_entries in runtime["groups"].items():
            if module_name not in captured:
                raise RuntimeError(f"Captured outputs missing module {module_name}")
            tensor_out = captured[module_name]
            if tensor_out.dim() != 4:
                raise ValueError(
                    f"Headwise scoring expects [B,S,H,D], got {tuple(tensor_out.shape)} for {module_name}"
                )

            module_score = None
            for head_idx, coef in head_entries:
                coef_t = coef.to(device=tensor_out.device, dtype=tensor_out.dtype)
                head_scores = torch.einsum("b s d, d -> b s", tensor_out[:, :, head_idx, :], coef_t)
                module_score = head_scores if module_score is None else module_score + head_scores

            token_scores = module_score if token_scores is None else token_scores + module_score

    elif probe_type in {"layerwise_linear", "rfm_layerwise"}:
        for module_name, vec in runtime["groups"].items():
            if module_name not in captured:
                raise RuntimeError(f"Captured outputs missing module {module_name}")
            tensor_out = captured[module_name]
            flat = flatten_last_dim(tensor_out)
            vec_t = vec.to(device=flat.device, dtype=flat.dtype)
            layer_scores = torch.einsum("b s d, d -> b s", flat, vec_t)
            token_scores = layer_scores if token_scores is None else token_scores + layer_scores

    else:
        raise ValueError(f"Unsupported probe type: {probe_type}")

    if token_scores is None:
        raise RuntimeError("No token scores computed from captured activations.")

    return (token_scores / float(normalizer)).to(dtype=torch.float32)


def summarize(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def save_outputs(
    image_scores_path: str,
    all_scores_path: str,
    stats_path: str,
    metadata: Dict[str, Any],
    records_meta: List[Dict[str, Any]],
    image_scores_list: List[np.ndarray],
    all_scores_list: List[np.ndarray],
    token_ids_list: List[np.ndarray],
) -> None:
    names = [x["name"] for x in records_meta]
    ids = [x["record_id"] for x in records_meta]
    image_paths = [x["image_path"] for x in records_meta]
    grid_hw = [(int(x["grid_h"]), int(x["grid_w"])) for x in records_meta]

    offsets = [0]
    flat_scores: List[torch.Tensor] = []
    for arr in image_scores_list:
        t = torch.from_numpy(arr.astype(np.float32, copy=False))
        flat_scores.append(t)
        offsets.append(offsets[-1] + int(t.numel()))

    image_payload = {
        "metadata": metadata,
        "record_ids": ids,
        "record_names": names,
        "image_paths": image_paths,
        "grid_hw": torch.tensor(grid_hw, dtype=torch.int32),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "scores": torch.cat(flat_scores, dim=0) if flat_scores else torch.empty(0, dtype=torch.float32),
    }
    torch.save(image_payload, image_scores_path)

    all_payload = {
        "metadata": metadata,
        "record_ids": ids,
        "record_names": names,
        "image_paths": image_paths,
        "seq_lengths": torch.tensor([len(x) for x in all_scores_list], dtype=torch.int64),
        "all_token_scores": [torch.from_numpy(x.astype(np.float32, copy=False)) for x in all_scores_list],
        "token_ids": [torch.from_numpy(x.astype(np.int64, copy=False)) for x in token_ids_list],
    }
    torch.save(all_payload, all_scores_path)

    fieldnames = [
        "record_id",
        "record_name",
        "image_path",
        "image_h",
        "image_w",
        "grid_h",
        "grid_w",
        "num_image_tokens",
        "expected_image_tokens",
        "image_token_mismatch",
        "num_all_tokens",
        "image_mean",
        "image_median",
        "image_min",
        "image_max",
        "image_std",
        "all_mean",
        "all_median",
        "all_min",
        "all_max",
        "all_std",
    ]

    with open(stats_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for meta, image_vals, all_vals in zip(records_meta, image_scores_list, all_scores_list):
            img_stats = summarize(image_vals)
            all_stats = summarize(all_vals)
            writer.writerow(
                {
                    "record_id": meta["record_id"],
                    "record_name": meta["name"],
                    "image_path": meta["image_path"],
                    "image_h": meta["image_h"],
                    "image_w": meta["image_w"],
                    "grid_h": meta["grid_h"],
                    "grid_w": meta["grid_w"],
                    "num_image_tokens": int(image_vals.size),
                    "expected_image_tokens": int(meta["expected_image_tokens"]),
                    "image_token_mismatch": int(meta["image_token_mismatch"]),
                    "num_all_tokens": int(all_vals.size),
                    "image_mean": img_stats["mean"],
                    "image_median": img_stats["median"],
                    "image_min": img_stats["min"],
                    "image_max": img_stats["max"],
                    "image_std": img_stats["std"],
                    "all_mean": all_stats["mean"],
                    "all_median": all_stats["median"],
                    "all_min": all_stats["min"],
                    "all_max": all_stats["max"],
                    "all_std": all_stats["std"],
                }
            )


def main() -> None:
    args = parse_args()

    probe_specs = parse_probe_specs(args)

    # Custom output names are unambiguous only for single spec.
    if len(probe_specs) > 1 and (args.activations_name or args.all_tokens_name or args.stats_name):
        raise ValueError(
            "Custom --activations-name/--all-tokens-name/--stats-name are only supported with a single probe spec."
        )

    output_dir = args.output_dir or os.path.join("results", model_base_name(args.model_path))
    os.makedirs(output_dir, exist_ok=True)

    records = load_records(args.data)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No records loaded from {args.data}")

    module_paths_override = parse_module_paths(args.module_paths_json, args.model_family)

    print(f"Loading model: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model_cls = select_model_loader(args.model_family, args.model_path)

    if args.dtype == "auto":
        default_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        default_dtype = getattr(torch, args.dtype)

    if model_cls is AutoModelForImageTextToText and args.model_family == "qwen3-vl":
        load_kwargs = {
            "dtype": default_dtype,
            "low_cpu_mem_usage": True,
            "device_map": args.device_map,
        }
    elif model_cls is AutoModelForMultimodalLM:
        gemma_dtype = default_dtype
        if args.dtype == "auto" and torch.cuda.is_available():
            gemma_dtype = "auto"
        load_kwargs = {
            "torch_dtype": gemma_dtype,
            "device_map": args.device_map,
        }
    else:
        load_kwargs = {
            "torch_dtype": default_dtype,
            "device_map": args.device_map,
        }

    model = model_cls.from_pretrained(args.model_path, **load_kwargs)
    model.eval()

    # Load probes and build runtimes once.
    runtime_entries: List[Dict[str, Any]] = []
    all_module_names: Set[str] = set()

    for prefix, probe_name in probe_specs:
        probe_cls = PROBE_CLASSES[probe_name]
        probe = probe_cls(
            model_path=args.model_path,
            prefix=prefix,
            mode=args.mode,
            model_family=args.model_family,
            data_dir=args.data_dir,
            module_paths=module_paths_override,
        ).load()

        # Reuse module paths from saved metadata when available.
        if probe.metadata_ is not None:
            meta_paths = probe.metadata_.extra.get("module_paths")
            if isinstance(meta_paths, dict):
                probe.module_paths = resolve_module_paths(args.model_family, meta_paths)

        runtime = build_probe_runtime(
            model=model,
            probe=probe,
            top_k=args.top_k,
            mode=args.mode,
            model_family=args.model_family,
        )
        all_module_names |= runtime["module_names"]

        image_scores_path, all_scores_path, stats_path = resolve_output_paths(
            output_dir=output_dir,
            prefix=prefix,
            probe_name=probe_name,
            activations_name=args.activations_name,
            all_tokens_name=args.all_tokens_name,
            stats_name=args.stats_name,
        )
        ensure_writable(image_scores_path, args.overwrite)
        ensure_writable(all_scores_path, args.overwrite)
        ensure_writable(stats_path, args.overwrite)

        runtime_entries.append(
            {
                "prefix": prefix,
                "probe_name": probe_name,
                "probe": probe,
                "runtime": runtime,
                "image_scores_path": image_scores_path,
                "all_scores_path": all_scores_path,
                "stats_path": stats_path,
                "image_scores_list": [],
                "all_scores_list": [],
            }
        )

    tokenizer = processor.tokenizer
    image_token_ids = gather_candidate_image_token_ids(tokenizer)
    image_token_array = np.asarray(sorted(image_token_ids), dtype=np.int64) if image_token_ids else np.empty(0, dtype=np.int64)

    records_meta: List[Dict[str, Any]] = []
    token_ids_list: List[np.ndarray] = []
    skipped_unreadable = 0

    resized_cache_dir = os.path.join(output_dir, "_resized_images_800")

    progress = make_progress(total=len(records), enabled=not args.no_progress)

    try:
        for batch_start, batch_records in batched(records, args.batch_size):
            valid_batch_indices: List[int] = []
            valid_batch_records: List[Dict[str, Any]] = []
            batch_messages: List[List[Dict[str, Any]]] = []
            for offset, record in enumerate(batch_records):
                messages = build_messages(record, args, cache_dir=resized_cache_dir)
                if messages is None:
                    skipped_unreadable += 1
                    progress.update(1)
                    continue
                valid_batch_indices.append(batch_start + offset)
                valid_batch_records.append(record)
                batch_messages.append(messages)

            if not batch_messages:
                continue

            encoded = encode_prompts(processor, batch_messages)

            captured = capture_module_outputs(
                model=model,
                encoded=encoded,
                module_names=sorted(all_module_names),
            )

            batch_scores: Dict[str, np.ndarray] = {}
            for entry in runtime_entries:
                score_key = f"{entry['prefix']}:{entry['probe_name']}"
                batch_scores[score_key] = score_from_captured(captured, entry["runtime"]).cpu().numpy()

            input_ids_batch = encoded["input_ids"].cpu().numpy().astype(np.int64, copy=False)
            attention_mask = encoded.get("attention_mask")
            if isinstance(attention_mask, torch.Tensor):
                seq_lens = attention_mask.sum(dim=1).cpu().numpy().astype(np.int64, copy=False)
            else:
                seq_lens = np.full(input_ids_batch.shape[0], input_ids_batch.shape[1], dtype=np.int64)

            for batch_idx, record in enumerate(valid_batch_records):
                idx = valid_batch_indices[batch_idx]
                seq_len = int(seq_lens[batch_idx])
                if seq_len <= 0:
                    raise ValueError(f"Empty sequence at record index {idx}")

                input_ids = input_ids_batch[batch_idx, :seq_len]
                token_ids_list.append(input_ids)

                if image_token_array.size > 0:
                    image_mask = np.isin(input_ids, image_token_array)
                else:
                    image_mask = np.zeros_like(input_ids, dtype=bool)

                image_path = extract_first_image_path(batch_messages[batch_idx])
                image_h, image_w = load_image_size_from_path(image_path) if image_path else (-1, -1)
                grid_h, grid_w = reconstruct_grid_hw(
                    model_family=args.model_family,
                    image_path=image_path,
                    encoded=encoded,
                    processor=processor,
                    batch_index=batch_idx,
                )

                expected_image_tokens = -1
                if grid_h >= 0 and grid_w >= 0:
                    expected_image_tokens = int(grid_h * grid_w)

                record_id = record.get(args.id_field, idx)
                records_meta.append(
                    {
                        "record_id": str(record_id),
                        "name": str(record.get("name", f"record_{idx}")),
                        "image_path": image_path,
                        "image_h": int(image_h),
                        "image_w": int(image_w),
                        "grid_h": int(grid_h),
                        "grid_w": int(grid_w),
                        "expected_image_tokens": int(expected_image_tokens),
                        "image_token_mismatch": 0,
                    }
                )

                for entry in runtime_entries:
                    score_key = f"{entry['prefix']}:{entry['probe_name']}"
                    token_scores = batch_scores[score_key][batch_idx, :seq_len].astype(np.float32, copy=False)
                    if token_scores.shape[0] != input_ids.shape[0]:
                        raise ValueError(
                            f"Score/token length mismatch on record {idx} for {entry['prefix']}:{entry['probe_name']}: "
                            f"scores={token_scores.shape[0]} tokens={input_ids.shape[0]}"
                        )

                    image_scores = token_scores[image_mask].astype(np.float32, copy=False)
                    entry["image_scores_list"].append(image_scores)
                    entry["all_scores_list"].append(token_scores)

                progress.update(1)
    finally:
        progress.close()

    if skipped_unreadable > 0:
        print(f"Skipped {skipped_unreadable} records with unreadable images.")

    # Compute mismatch flags using first probe's image-token extraction behavior.
    # Mismatch is image-token count against reconstructed grid when available.
    if runtime_entries:
        first_image_lists = runtime_entries[0]["image_scores_list"]
        for i, meta in enumerate(records_meta):
            expected = meta["expected_image_tokens"]
            if expected >= 0:
                meta["image_token_mismatch"] = int(len(first_image_lists[i]) != expected)

    for entry in runtime_entries:
        probe = entry["probe"]
        prefix = entry["prefix"]
        probe_name = entry["probe_name"]

        metadata = {
            "model_path": args.model_path,
            "model_family": args.model_family,
            "mode": args.mode,
            "probe": probe_name,
            "prefix": prefix,
            "top_k": int(args.top_k),
            "data": args.data,
            "num_records": int(len(records_meta)),
            "num_records_with_images": int(sum(1 for x in records_meta if x["image_path"])),
            "image_token_ids": sorted(int(x) for x in image_token_ids),
            "probe_weights_path": probe.weights_path,
            "probe_scores_path": probe.scores_path,
            "probe_metadata_path": probe.metadata_path,
        }

        save_outputs(
            image_scores_path=entry["image_scores_path"],
            all_scores_path=entry["all_scores_path"],
            stats_path=entry["stats_path"],
            metadata=metadata,
            records_meta=records_meta,
            image_scores_list=entry["image_scores_list"],
            all_scores_list=entry["all_scores_list"],
            token_ids_list=token_ids_list,
        )

        print(f"Saved image token scores: {entry['image_scores_path']}")
        print(f"Saved all token scores: {entry['all_scores_path']}")
        print(f"Saved stats CSV: {entry['stats_path']}")


if __name__ == "__main__":
    main()
