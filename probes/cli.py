#!/usr/bin/env python3
"""
Single-combination probe training CLI.

Trains one probe class for one model and one direction, then saves the result.

Example:
  python -m probes.cli \
    --model-path /path/to/Qwen3-VL-8B-Instruct \
    --model-family qwen3-vl \
    --mode text \
    --probe layerwise_rfm \
    --prefix textual_ideology \
    --data data/probes/textual_ideology.jsonl
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
    MllamaForConditionalGeneration,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from probes import HeadwiseLinearProbe, LayerwiseLinearProbe, LayerwiseRFM
from probes.base import ModulePaths, resolve_module_paths

PROBE_CLASSES = {
    "headwise_linear": HeadwiseLinearProbe,
    "layerwise_linear": LayerwiseLinearProbe,
    "layerwise_rfm": LayerwiseRFM,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a single probe for one model and direction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--model-path", required=True)
    p.add_argument("--mode", required=True, choices=["text", "vision"])
    p.add_argument("--model-family", required=True, choices=["gemma4", "mllama", "qwen3-vl"])
    p.add_argument("--probe", required=True, choices=list(PROBE_CLASSES))
    p.add_argument("--prefix", required=True, help="Direction name, e.g. textual_ideology.")
    p.add_argument("--data", required=True, help="JSONL/JSON/CSV manifest.")
    # Data
    p.add_argument("--has-images", action="store_true", help="Include images using the image_path field.")
    p.add_argument("--text-field", default="text")
    p.add_argument("--label-field", default="label")
    p.add_argument("--image-field", default="image_path")
    p.add_argument("--messages-field", default="messages")
    p.add_argument("--image-root", default=None)
    p.add_argument("--limit", type=int, default=None)
    # Module paths override
    p.add_argument(
        "--module-paths-json",
        default=None,
        metavar="JSON",
        help="JSON string or file path of module_paths overrides (e.g. '{\"layer_prefix\": \"model.layers\"}').",
    )
    # Output
    p.add_argument("--data-dir", default="results/probes")
    # Hyperparams
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha (headwise/layerwise linear).")
    p.add_argument("--n-splits", type=int, default=2, help="KFold splits (headwise/layerwise linear).")
    p.add_argument("--train-fraction", type=float, default=0.8, help="Train split fraction (rfm).")
    p.add_argument("--bandwidths", nargs="+", type=int, default=[1, 10, 100], help="RFM bandwidths.")
    # Model loading
    p.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    p.add_argument("--device-map", default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(path: str) -> List[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}.")
        return payload
    if suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported manifest format '{suffix}'. Use .jsonl, .json, or .csv.")


def _resolve_image(raw_path: Any, image_root: Optional[str]) -> str:
    path = os.fspath(raw_path)
    if image_root is not None and not os.path.isabs(path):
        path = os.path.join(image_root, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def _resolve_messages_images(messages: Sequence[Dict[str, Any]], image_root: Optional[str]) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    for message in messages:
        new_message: Dict[str, Any] = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            new_content: List[Dict[str, Any]] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                    new_item = dict(item)
                    new_item["image"] = _resolve_image(item["image"], image_root)
                    new_content.append(new_item)
                else:
                    new_content.append(item)
            new_message["content"] = new_content
        resolved.append(new_message)
    return resolved


def build_prompts(
    records: Sequence[Dict[str, Any]],
    processor,
    text_field: str,
    label_field: str,
    image_field: str,
    messages_field: str,
    has_images: bool,
    image_root: Optional[str],
    source_path: str,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    prompts: List[Dict[str, Any]] = []
    labels: List[float] = []

    for idx, record in enumerate(records):
        text = str(record.get(text_field, "")).strip()
        if not text:
            raise ValueError(f"Empty '{text_field}' in record {idx} of {source_path}.")

        raw_label = record.get(label_field)
        if raw_label is None:
            raise KeyError(f"Missing '{label_field}' in record {idx} of {source_path}.")

        raw_messages = record.get(messages_field)
        if raw_messages is not None:
            if not isinstance(raw_messages, list):
                raise ValueError(
                    f"Expected list in '{messages_field}' for record {idx} of {source_path}, "
                    f"got {type(raw_messages).__name__}."
                )
            messages = _resolve_messages_images(raw_messages, image_root=image_root)
        else:
            if has_images:
                raw_path = record.get(image_field)
                if raw_path is None:
                    raise KeyError(f"Missing '{image_field}' in record {idx} of {source_path}.")
                image_path = _resolve_image(raw_path, image_root)
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image", "image": image_path},
                ]}]
            else:
                messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompts.append({
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in encoded.items()
        })
        labels.append(float(raw_label))

    return prompts, np.asarray(labels, dtype=np.float32)


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------

def parse_module_paths(raw: Optional[str], model_family: str) -> Optional[ModulePaths]:
    if raw is None:
        return None
    if os.path.isfile(raw):
        with open(raw, encoding="utf-8") as f:
            overrides = json.load(f)
    else:
        overrides = json.loads(raw)
    return resolve_module_paths(model_family, overrides)


def build_probe(args: argparse.Namespace, module_paths: Optional[ModulePaths]):
    common = dict(
        model_path=args.model_path,
        prefix=args.prefix,
        mode=args.mode,
        model_family=args.model_family,
        data_dir=args.data_dir,
        seed=args.seed,
        module_paths=module_paths,
    )
    if args.probe in {"headwise_linear", "layerwise_linear"}:
        return PROBE_CLASSES[args.probe](alpha=args.alpha, n_splits=args.n_splits, **common)
    return PROBE_CLASSES[args.probe](
        train_fraction=args.train_fraction,
        bandwidths=args.bandwidths,
        **common,
    )


def select_model_loader(model_family: str, model_path: str):
    path_lower = model_path.lower()

    if model_family == "qwen3-vl" or "qwen3-vl" in path_lower:
        return AutoModelForImageTextToText
    if model_family == "gemma4" or "gemma-4" in path_lower or "gemma4" in path_lower:
        return AutoModelForMultimodalLM
    if model_family == "mllama" or ("llama-3.2" in path_lower and "vision" in path_lower):
        return MllamaForConditionalGeneration
    return AutoModelForImageTextToText


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    module_paths = parse_module_paths(args.module_paths_json, args.model_family)
    probe = build_probe(args, module_paths=module_paths)

    # Check if results already exist before loading models
    if (os.path.exists(probe.weights_path) and
        os.path.exists(probe.scores_path) and
        os.path.exists(probe.metadata_path)):
        print(f"Results already exist for {args.prefix} ({args.probe}) on {args.model_path}.")
        print(f"Weights : {probe.weights_path}")
        print(f"Scores  : {probe.scores_path}")
        print(f"Metadata: {probe.metadata_path}")
        return

    records = load_records(args.data)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No records loaded from {args.data}.")

    print(f"Loading model: {args.model_path}")
    torch.backends.cuda.matmul.allow_tf32 = True
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

    print(f"Model loader: {model_cls.__name__}")
    model = model_cls.from_pretrained(args.model_path, **load_kwargs)
    model.eval()

    prompts, labels = build_prompts(
        records=records,
        processor=processor,
        text_field=args.text_field,
        label_field=args.label_field,
        image_field=args.image_field,
        messages_field=args.messages_field,
        has_images=args.has_images,
        image_root=args.image_root,
        source_path=args.data,
    )

    probe = build_probe(args, module_paths=module_paths)
    print(f"Fitting {args.probe} on {args.prefix} ({len(labels)} samples)")
    probe.fit(model=model, prompts=prompts, labels=labels)
    probe.save()
    print(f"Saved weights : {probe.weights_path}")
    print(f"Saved scores  : {probe.scores_path}")
    print(f"Saved metadata: {probe.metadata_path}")


if __name__ == "__main__":
    main()
