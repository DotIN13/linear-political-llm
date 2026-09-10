"""
python run_qwen3_vl_news_images.py --recursive --batch-size 4

python scripts/run_qwen3_vl_news_images.py \
  --model-path /project/jevans/tzhang3/models/Qwen3-VL-8B-Instruct \
  --image-dir results/gemini_tie_pairs/red_blue \
  --output-dir results/qwen3-vl-8b-instruct/red_blue \
  --batch-size 4 \
  --top-k 16 \
  --dtype auto \
  --device-map auto \
  --overwrite
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from PIL import UnidentifiedImageError
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lpl.utils import get_top_indices, load_ridge_models, model_base_name


DEFAULT_MODEL_PATH = "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
DEFAULT_IMAGE_DIR = "data/news_images"
DEFAULT_MAX_WIDTH = 800
DEFAULT_PROMPT = (
    "write a possible news article that would best accompany this image, from the best "
    "possible media outlet. Use your knowledge about American political culture and the "
    "media landscape in doing so. If the image seems slanted, write a slanted article. "
    "Do not worry about neutrality: write the article in the style of the media outlet "
    "most likely to have used this image."
)
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score news image patches with Qwen3-VL and save PT/CSV outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--activations-name", default="news_images_patch_scores.pt")
    parser.add_argument("--all-tokens-name", default="news_images_all_tokens.pt")
    parser.add_argument("--stats-name", default="news_images_patch_stats.csv")
    parser.add_argument("--ridge-prefix", default="politician")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join("results", model_base_name(args.model_path))
    os.makedirs(output_dir, exist_ok=True)
    return (
        os.path.join(output_dir, args.activations_name),
        os.path.join(output_dir, args.all_tokens_name),
        os.path.join(output_dir, args.stats_name),
    )


def ensure_writable(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def collect_image_paths(image_dir: str, recursive: bool, limit: int = None) -> List[str]:
    image_paths: List[str] = []
    for pattern in IMAGE_EXTENSIONS:
        query = os.path.join(image_dir, "**", pattern) if recursive else os.path.join(image_dir, pattern)
        image_paths.extend(glob.glob(query, recursive=recursive))

    image_paths = sorted(set(image_paths))
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return image_paths


def chunked(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def build_messages(image_input, prompt: str) -> List[Dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image_input},
        ],
    }]


def vision_grid_hw(orig_h: int, orig_w: int, processor) -> Tuple[int, int]:
    image_processor = processor.image_processor
    min_pixels = image_processor.size["shortest_edge"]
    max_pixels = image_processor.size["longest_edge"]
    patch = image_processor.patch_size
    merge = image_processor.merge_size

    factor = patch * merge
    resized_h, resized_w = smart_resize(
        orig_h,
        orig_w,
        factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    grid_h = (resized_h // patch) // merge
    grid_w = (resized_w // patch) // merge
    return int(grid_h), int(grid_w)


def try_get_image_pad_id(tokenizer) -> int:
    token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise RuntimeError("Could not resolve the <|image_pad|> token id from the tokenizer.")
    return int(token_id)


def encode_sample(processor, image_path: str, prompt: str, max_width: int) -> Dict:
    with Image.open(image_path) as img:
        width, height = img.size
        if max_width and max_width > 0 and width > max_width:
            resized_height = max(1, int(round(height * (float(max_width) / float(width)))))
            img = img.resize((int(max_width), resized_height), resample=Image.Resampling.BICUBIC)

        resized_width, resized_height = img.size
        grid_h, grid_w = vision_grid_hw(resized_height, resized_width, processor)
        encoded = processor.apply_chat_template(
            build_messages(img, prompt),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    sample = {
        "image_path": image_path,
        "image_name": os.path.basename(image_path),
        "grid_h": grid_h,
        "grid_w": grid_w,
        "expected_patches": grid_h * grid_w,
    }
    for key, value in encoded.items():
        if isinstance(value, torch.Tensor):
            sample[key] = value.cpu()
    return sample


def collate_samples(samples: Sequence[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    input_ids = [sample["input_ids"].squeeze(0) for sample in samples]
    attention_masks = [sample["attention_mask"].squeeze(0) for sample in samples]
    max_seq_len = max(tensor.size(0) for tensor in input_ids)

    padded_ids = torch.full((len(samples), max_seq_len), pad_token_id, dtype=input_ids[0].dtype)
    padded_mask = torch.zeros((len(samples), max_seq_len), dtype=attention_masks[0].dtype)

    for row_idx, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
        seq_len = ids.size(0)
        padded_ids[row_idx, :seq_len] = ids
        padded_mask[row_idx, :seq_len] = mask

    batch = {
        "input_ids": padded_ids,
        "attention_mask": padded_mask,
    }

    def pad_and_cat(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            raise ValueError("Cannot collate an empty tensor list")

        rank = tensors[0].dim()
        if any(t.dim() != rank for t in tensors):
            raise RuntimeError("Cannot collate tensors with different ranks")

        if rank <= 1:
            return torch.cat(tensors, dim=0)

        target_shape = [max(t.size(dim) for t in tensors) for dim in range(1, rank)]
        padded_tensors: List[torch.Tensor] = []
        for tensor in tensors:
            if [tensor.size(dim) for dim in range(1, rank)] == target_shape:
                padded_tensors.append(tensor)
                continue

            padded = torch.zeros(
                (tensor.size(0), *target_shape),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            slices = (slice(None),) + tuple(slice(0, tensor.size(dim)) for dim in range(1, rank))
            padded[slices] = tensor
            padded_tensors.append(padded)

        return torch.cat(padded_tensors, dim=0)

    tensor_keys = [key for key in samples[0].keys() if key not in batch and isinstance(samples[0].get(key), torch.Tensor)]
    for key in tensor_keys:
        tensors = [sample[key] for sample in samples]
        batch[key] = pad_and_cat(tensors)

    return batch


def move_batch_to_model_device(batch: Dict[str, torch.Tensor], model) -> Dict[str, torch.Tensor]:
    try:
        model_device = model.device
    except AttributeError:
        model_device = next(model.parameters()).device
    return {key: value.to(model_device) for key, value in batch.items()}


def build_layer_head_map(top_indices: np.ndarray) -> Dict[int, List[int]]:
    layer_to_heads: Dict[int, List[int]] = defaultdict(list)
    for layer_idx, head_idx in top_indices:
        layer_to_heads[int(layer_idx)].append(int(head_idx))
    return dict(layer_to_heads)


def build_coef_map(ridge_models: Dict[int, Dict[int, object]], top_indices: np.ndarray) -> Dict[Tuple[int, int], torch.Tensor]:
    coef_map: Dict[Tuple[int, int], torch.Tensor] = {}
    for layer_idx, head_idx in top_indices:
        coef = ridge_models[int(layer_idx)][int(head_idx)].coef_
        coef_map[(int(layer_idx), int(head_idx))] = torch.tensor(coef, dtype=torch.float32)
    return coef_map


def score_batch_tokens(
    model,
    batch: Dict[str, torch.Tensor],
    layer_to_heads: Dict[int, List[int]],
    coef_map: Dict[Tuple[int, int], torch.Tensor],
    num_top_heads: int,
) -> torch.Tensor:
    named_modules = dict(model.named_modules())
    layer_scores: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(_module, _inp, out):
            score_sum = None
            for head_idx in layer_to_heads[layer_idx]:
                coef = coef_map[(layer_idx, head_idx)].to(device=out.device, dtype=out.dtype)
                head_scores = torch.einsum("bsd,d->bs", out[:, :, head_idx, :], coef)
                score_sum = head_scores if score_sum is None else score_sum + head_scores
            layer_scores[layer_idx] = score_sum.detach().to(dtype=torch.float32).cpu()
        return hook_fn

    for layer_idx in layer_to_heads:
        module_name = f"model.language_model.layers.{layer_idx}.self_attn.head_out"
        hooks.append(named_modules[module_name].register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            _ = model(**batch)
    finally:
        for hook in hooks:
            hook.remove()

    batch_scores = None
    for layer_idx in sorted(layer_scores):
        batch_scores = layer_scores[layer_idx] if batch_scores is None else batch_scores + layer_scores[layer_idx]

    if batch_scores is None:
        raise RuntimeError("No layer scores were captured during the forward pass.")

    return batch_scores / float(num_top_heads)


def summarize_scores(image_scores: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(image_scores)),
        "median": float(np.median(image_scores)),
        "min": float(np.min(image_scores)),
        "max": float(np.max(image_scores)),
        "std": float(np.std(image_scores)),
    }


def save_outputs(
    activations_path: str,
    all_tokens_path: str,
    stats_path: str,
    image_paths: List[str],
    image_names: List[str],
    grid_hw: List[Tuple[int, int]],
    offsets: List[int],
    flat_scores: List[torch.Tensor],
    all_token_scores: List[torch.Tensor],
    all_seq_lengths: List[int],
    stats_rows: List[Dict[str, object]],
    metadata: Dict[str, object],
) -> None:
    # Save patch-level scores (existing behavior)
    score_tensor = torch.cat(flat_scores, dim=0) if flat_scores else torch.empty(0, dtype=torch.float32)
    payload = {
        "metadata": metadata,
        "image_paths": image_paths,
        "image_names": image_names,
        "grid_hw": torch.tensor(grid_hw, dtype=torch.int32),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "scores": score_tensor,
    }
    torch.save(payload, activations_path)

    # Save all-token scores
    all_tokens_payload = {
        "metadata": metadata,
        "image_names": image_names,
        "image_paths": image_paths,
        "seq_lengths": torch.tensor(all_seq_lengths, dtype=torch.int64),
        "all_token_scores": all_token_scores,  # List of tensors, one per image
    }
    torch.save(all_tokens_payload, all_tokens_path)

    # Save statistics CSV
    fieldnames = [
        "image_name",
        "image_path",
        "grid_h",
        "grid_w",
        "num_patches",
        "mean",
        "median",
        "min",
        "max",
        "std",
    ]
    with open(stats_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_width is not None and args.max_width < 1:
        raise ValueError("--max-width must be at least 1")

    activations_path, all_tokens_path, stats_path = resolve_output_paths(args)
    ensure_writable(activations_path, args.overwrite)
    ensure_writable(all_tokens_path, args.overwrite)
    ensure_writable(stats_path, args.overwrite)

    image_paths = collect_image_paths(args.image_dir, recursive=args.recursive, limit=args.limit)

    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True

    processor = AutoProcessor.from_pretrained(args.model_path)
    torch_dtype = resolve_torch_dtype(args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch_dtype,
        device_map=args.device_map,
    )
    model.eval()

    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    image_pad_id = try_get_image_pad_id(tokenizer)

    ridge_models, performance = load_ridge_models(args.model_path, args.ridge_prefix)
    top_indices = get_top_indices(performance, k=args.top_k)
    layer_to_heads = build_layer_head_map(top_indices)
    coef_map = build_coef_map(ridge_models, top_indices)

    all_image_paths: List[str] = []
    all_image_names: List[str] = []
    all_grid_hw: List[Tuple[int, int]] = []
    offsets: List[int] = [0]
    flat_scores: List[torch.Tensor] = []
    all_token_scores: List[torch.Tensor] = []
    all_seq_lengths: List[int] = []
    stats_rows: List[Dict[str, object]] = []
    skipped_images = 0

    progress = tqdm(chunked(image_paths, args.batch_size), total=(len(image_paths) + args.batch_size - 1) // args.batch_size)
    for image_batch in progress:
        samples = []
        for image_path in image_batch:
            try:
                samples.append(encode_sample(processor, image_path, args.prompt, args.max_width))
            except (UnidentifiedImageError, OSError, ValueError) as error:
                skipped_images += 1
                tqdm.write(f"Skipping unreadable image {image_path}: {error}")

        if not samples:
            progress.set_postfix(processed=len(all_image_paths), skipped=skipped_images)
            continue

        batch = collate_samples(samples, pad_token_id=pad_token_id)
        score_batch = score_batch_tokens(
            model=model,
            batch=move_batch_to_model_device(batch, model),
            layer_to_heads=layer_to_heads,
            coef_map=coef_map,
            num_top_heads=len(top_indices),
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].bool()

        for row_idx, sample in enumerate(samples):
            valid_mask = attention_mask[row_idx]
            image_mask = (input_ids[row_idx] == image_pad_id) & valid_mask
            image_scores = score_batch[row_idx][image_mask].numpy().astype(np.float32, copy=False)

            # Collect all-token scores (all valid tokens in the sequence)
            all_valid_scores = score_batch[row_idx][valid_mask].numpy().astype(np.float32, copy=False)

            if image_scores.size == 0:
                raise RuntimeError(f"No image patch scores found for {sample['image_path']}")
            if image_scores.size != sample["expected_patches"]:
                raise ValueError(
                    f"Patch count mismatch for {sample['image_path']}: "
                    f"expected {sample['expected_patches']}, got {image_scores.size}"
                )

            all_image_paths.append(sample["image_path"])
            all_image_names.append(sample["image_name"])
            all_grid_hw.append((sample["grid_h"], sample["grid_w"]))
            flat_scores.append(torch.from_numpy(image_scores.copy()))
            all_token_scores.append(torch.from_numpy(all_valid_scores.copy()))
            all_seq_lengths.append(int(all_valid_scores.size))
            offsets.append(offsets[-1] + image_scores.size)

            row = {
                "image_name": sample["image_name"],
                "image_path": sample["image_path"],
                "grid_h": sample["grid_h"],
                "grid_w": sample["grid_w"],
                "num_patches": int(image_scores.size),
            }
            row.update(summarize_scores(image_scores))
            stats_rows.append(row)

        progress.set_postfix(processed=len(all_image_paths), skipped=skipped_images)

    metadata = {
        "model_path": args.model_path,
        "image_dir": args.image_dir,
        "ridge_prefix": args.ridge_prefix,
        "top_k": int(args.top_k),
        "prompt": args.prompt,
        "max_width": int(args.max_width),
        "image_pad_id": int(image_pad_id),
        "skipped_images": int(skipped_images),
    }
    save_outputs(
        activations_path=activations_path,
        all_tokens_path=all_tokens_path,
        stats_path=stats_path,
        image_paths=all_image_paths,
        image_names=all_image_names,
        grid_hw=all_grid_hw,
        offsets=offsets,
        flat_scores=flat_scores,
        all_token_scores=all_token_scores,
        all_seq_lengths=all_seq_lengths,
        stats_rows=stats_rows,
        metadata=metadata,
    )

    print(f"Saved activations to {activations_path}")
    print(f"Saved all-token scores to {all_tokens_path}")
    print(f"Saved stats to {stats_path}")
    print(f"Processed {len(all_image_paths)} images")
    print(f"Skipped {skipped_images} unreadable images")


if __name__ == "__main__":
    main()