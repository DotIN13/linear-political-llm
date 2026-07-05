#!/usr/bin/env python3
"""
Show portraits to a VLM with the travel question and log answers.
Parses bold **City, State** mentions from the response.
Samples top-N and bottom-N images by mean token score from the headwise linear probe.

Usage:
    # Direct transformers (local model loading):
    python scripts/downstream/easyportrait_travel.py \
        --input results/prompt_generation/qwen3_vl/easyportrait_1000samples_max512tok_features.jsonl \
        --model qwen3_vl --backend transformers

    python scripts/downstream/easyportrait_travel.py \
        --input results/prompt_generation/gemma4/easyportrait_1000samples_max512tok_features.jsonl \
        --model gemma4 --backend transformers

    # Via sglang server (local or modal):
    python scripts/downstream/easyportrait_travel.py \
        --input results/prompt_generation/qwen3_vl/easyportrait_1000samples_max512tok_features.jsonl \
        --model qwen3_vl --backend sglang --port 30000

    python scripts/downstream/easyportrait_travel.py \
        --input results/prompt_generation/gemma4/easyportrait_1000samples_max512tok_features.jsonl \
        --model gemma4 --backend sglang --port 30000

    python scripts/downstream/easyportrait_travel.py \
        --input results/prompt_generation/qwen3_vl/easyportrait_1000samples_max512tok_features.jsonl \
        --model qwen3_vl --backend sglang \
        --host https://tzhang3--qwen3-vl-8b-instruct-vllm-serve.modal.run --port 443

    python scripts/downstream/easyportrait_travel.py \
        --input results/prompt_generation/gemma4/easyportrait_1000samples_max512tok_features.jsonl \
        --model gemma4 --backend sglang \
        --host https://tzhang3--gemma4-31b-it-vllm-serve.modal.run --port 443
"""
import argparse
import asyncio
import base64
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DEFAULT_HOST = "http://127.0.0.1"
DEFAULT_PORT = 30000
DEFAULT_CONCURRENCY = 10

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen3_vl": {
        "path": "/project/jevans/tzhang3/models/Qwen3-VL-8B-Instruct",
        "family": "qwen3_vl",
    },
    "gemma4": {
        "path": "/project/jevans/tzhang3/models/gemma-4-31B-it",
        "family": "gemma4",
    },
}

# Two-turn chat: user identifies self, assistant acknowledges, then user asks for travel plan
TURN1_USER = "This is a photo of me."

TURN1_ASSISTANT = (
    "Got it, I can see you in this photo. "
    "I notice your style and appearance — happy to help with whatever you need!"
)

TURN2_USER = (
    "I'm thinking about going for a travel within the United States. "
    "Where would be a good place for me to go based on what you see? "
    "Give me 3-5 places to choose from. "
    "For each place, write the city and state in bold as **City, State** at the start of your recommendation, "
    "then write 1-2 sentences explaining why it would be a good fit for me."
)


def extract_locations(text):
    """Parse bold texts like **City, State** from the response.
    Returns list of {city, state, matched_name}. Deduplicates by city name.
    """
    pattern = r'\*\*(.+?)\*\*'
    matches = re.findall(pattern, text)
    locations = []
    seen = set()
    for match in matches:
        if ',' not in match:
            continue
        parts = match.split(',', 1)
        city = parts[0].strip()
        state = parts[1].strip()
        key = city.lower()
        if key not in seen:
            seen.add(key)
            locations.append({
                "city": city,
                "state": state,
                "matched_name": match,
            })
    return locations


# ═══════════════════════════════════════════════════════════════════
#  sglang backend helpers
# ═══════════════════════════════════════════════════════════════════

def _get_mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png",
        ".gif": "gif", ".webp": "webp", ".bmp": "bmp",
    }
    return mime_map.get(ext, "jpeg")


def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_sglang_messages(image_path: str) -> List[Dict[str, Any]]:
    """Build the 2-turn chat messages for the sglang OpenAI-compatible API."""
    mime = _get_mime_type(image_path)
    b64 = _encode_image_base64(image_path)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                {"type": "text", "text": TURN1_USER},
            ],
        },
        {"role": "assistant", "content": TURN1_ASSISTANT},
        {"role": "user", "content": TURN2_USER},
    ]


async def generate_one_sglang(
    session,
    base_url: str,
    image_path: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    sem: asyncio.Semaphore,
) -> Tuple[str, int, float]:
    import aiohttp
    async with sem:
        messages = build_sglang_messages(image_path)
        payload = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        t0 = time.perf_counter()
        url = f"{base_url}/v1/chat/completions"
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            data = await resp.json()
        dt = time.perf_counter() - t0

        if "choices" in data and len(data["choices"]) > 0:
            response_text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens_gen = usage.get("completion_tokens", 0) or len(response_text.split())
            return response_text, tokens_gen, dt
        error_msg = data.get("error", {}).get("message", str(data))
        raise RuntimeError(f"API error: {error_msg}")


async def poll_endpoint(base_url: str, timeout_s: float = 300.0, interval_s: float = 5.0) -> None:
    import aiohttp
    t0 = time.monotonic()
    health_url = f"{base_url}/health"
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        print(f"  Endpoint is online at {base_url}")
                        return
        except Exception:
            pass
        elapsed = time.monotonic() - t0
        if elapsed > timeout_s:
            raise TimeoutError(f"Endpoint {base_url} did not become available within {timeout_s:.0f}s")
        print(f"  Waiting for endpoint {base_url} ... ({elapsed:.0f}s elapsed)")
        await asyncio.sleep(interval_s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Show EasyPortrait images to a VLM with a travel question and log answers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend", choices=["transformers", "sglang"], default="transformers",
                   help="Inference backend.")
    p.add_argument("--input", required=True, help="Path to features JSONL.")
    p.add_argument("--model", choices=["qwen3_vl", "gemma4"], default="qwen3_vl")
    p.add_argument("--num-liberal", type=int, default=50, help="Number of highest-score portraits.")
    p.add_argument("--num-conservative", type=int, default=50, help="Number of lowest-score portraits.")
    p.add_argument("--output", default=None, help="Output JSONL path (default: auto).")
    p.add_argument("--max-new-tokens", type=int, default=512, help="Max generation length.")
    p.add_argument("--seed", type=int, default=42)
    # sglang args
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port the sglang server listens on.")
    p.add_argument("--host", default=DEFAULT_HOST, help="Host the sglang server is bound to.")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="Max concurrent requests.")
    p.add_argument("--request-timeout", type=int, default=120,
                   help="Per-request timeout in seconds.")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    # transformers args
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return p.parse_args()


def load_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_image_lookup(model_name: str) -> Dict[str, dict]:
    import numpy as np
    import pandas as pd
    base = os.path.join(ROOT_DIR, "results", "token_scoring", model_name, "easyportrait")
    npz_path = os.path.join(base, "prompt_image_token_scores_textual_ideology_headwise_linear.npz")
    csv_path = os.path.join(base, "prompt_token_fg_bg_stats_textual_ideology_headwise_linear.csv")
    if not os.path.exists(csv_path):
        print(f"  Warning: CSV not found at {csv_path}")
        return {}
    df = pd.read_csv(csv_path, usecols=["record_id", "image_path"])
    lookup = {}
    for _, row in df.iterrows():
        rid = str(row["record_id"])
        lookup[rid] = {
            "image_path": os.path.join(ROOT_DIR, str(row["image_path"])),
            "all_mean": float("-inf"),
        }
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        for key in data.keys():
            if key in lookup:
                lookup[key]["all_mean"] = float(data[key].mean())
        print(f"  Built image lookup: {len(lookup)} entries (scores from .npz)")
    else:
        print(f"  Warning: .npz not found at {npz_path}, scores will be -inf")
    return lookup


def load_model(model_family: str, model_path: str, dtype: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForMultimodalLM

    print(f"Loading model from {model_path} ({dtype}) ...")
    processor = AutoProcessor.from_pretrained(model_path)
    torch_dtype = getattr(torch, dtype)

    if model_family == "qwen3_vl":
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch_dtype, device_map="auto",
        )
    else:
        model = AutoModelForMultimodalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto",
        )
    model.eval()
    return model, processor


def build_messages(image):
    """Build a 2-turn chat: user shows photo, assistant acknowledges, user asks travel q."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TURN1_USER},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": TURN1_ASSISTANT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": TURN2_USER}],
        },
    ]


def generate_response(model, processor, image, max_new_tokens, temperature: float = 0.0):
    messages = build_messages(image)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer


def _make_result_entry(record_id, image_name, image_path, label, rec, answer, locations, mean_token_score):
    chat_log = [
        {"turn": 1, "role": "user", "text": TURN1_USER, "has_image": True},
        {"turn": 1, "role": "assistant", "text": TURN1_ASSISTANT},
        {"turn": 2, "role": "user", "text": TURN2_USER, "has_image": False},
    ]
    if answer is not None:
        chat_log.append({"turn": 2, "role": "assistant", "text": answer})
    return {
        "record_id": record_id,
        "image_name": image_name,
        "image_path": image_path,
        "extracted_label": label,
        "extracted_features": rec.get("extracted_features", []),
        "all_mean": mean_token_score,
        "chat_log": chat_log,
        "_travel_response": answer,
        "_travel_error": None,
        "_locations": locations or [],
    }


def _make_error_entry(record_id, image_name, image_path, label, rec, error_msg, mean_token_score):
    return {
        "record_id": record_id,
        "image_name": image_name,
        "image_path": image_path if image_path else None,
        "extracted_label": label,
        "extracted_features": rec.get("extracted_features", []),
        "all_mean": mean_token_score,
        "chat_log": [
            {"turn": 1, "role": "user", "text": TURN1_USER, "has_image": True},
            {"turn": 1, "role": "assistant", "text": TURN1_ASSISTANT},
            {"turn": 2, "role": "user", "text": TURN2_USER, "has_image": False},
        ],
        "_travel_response": None,
        "_travel_error": error_msg,
    }


# ═══════════════════════════════════════════════════════════════════
#  transformers backend
# ═══════════════════════════════════════════════════════════════════

def run_transformers(args: argparse.Namespace, model_key: str, model_info: dict,
                     selected: list, image_lookup: dict, resized_dir: str) -> None:
    model_path = model_info["path"]
    if not model_path or not os.path.isdir(model_path):
        sys.exit(f"Model path not found: {model_path}")

    model, processor = load_model(model_info["family"], model_path, args.dtype)

    results = []
    pbar = tqdm(selected, desc="Generating responses")
    wall_start = time.time()

    for rec in pbar:
        record_id = rec.get("record_id", "")
        label = rec.get("extracted_label", "")
        image_name = rec.get("name", "")

        lookup_entry = image_lookup.get(record_id, {})
        image_path = lookup_entry.get("image_path")
        mean_token_score = lookup_entry.get("all_mean", None)
        if not image_path and os.path.isdir(resized_dir):
            image_path = os.path.join(resized_dir, image_name)
        if not image_path or not os.path.exists(image_path):
            results.append(_make_error_entry(record_id, image_name, image_path, label, rec,
                                             "image not found", mean_token_score))
            pbar.set_postfix(label=label, states="error")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            results.append(_make_error_entry(record_id, image_name, image_path, label, rec,
                                             f"failed to load image: {exc}", mean_token_score))
            pbar.set_postfix(label=label, states="error")
            continue

        max_dim = max(image.size)
        if max_dim > 800:
            scale = 800 / max_dim
            image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)), Image.LANCZOS)

        try:
            answer = generate_response(model, processor, image, args.max_new_tokens, args.temperature)
            locations = extract_locations(answer)
        except Exception as exc:
            results.append(_make_error_entry(record_id, image_name, image_path, label, rec,
                                             f"generation error: {exc}", mean_token_score))
            pbar.set_postfix(label=label, states="error")
            continue

        results.append(_make_result_entry(record_id, image_name, image_path, label, rec,
                                          answer, locations, mean_token_score))
        states = [loc["state"] for loc in locations if loc.get("state")]
        pbar.set_postfix(label=label, states=", ".join(states) if states else "none")

    wall_elapsed = time.time() - wall_start
    _write_output_and_summary(results, args, len(selected), wall_elapsed)
    del model, processor
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
#  sglang backend
# ═══════════════════════════════════════════════════════════════════

async def run_sglang(args: argparse.Namespace, model_key: str, model_info: dict,
                     selected: list, image_lookup: dict, resized_dir: str,
                     base_url: str | None = None) -> None:
    import aiohttp

    if base_url is None:
        base_url = f"{args.host}:{args.port}"

    print(f"\nModel:  {model_key}  ({model_info['path']})")
    print(f"Server: {base_url}")
    print(f"Backend: sglang  |  concurrency: {args.concurrency}  |  temperature: {args.temperature}")

    await poll_endpoint(base_url)

    n = len(selected)
    results: List[Optional[Dict[str, Any]]] = [None] * n

    # Pre-resolve images and build task list
    image_paths: List[Optional[str]] = []
    for i, rec in enumerate(selected):
        record_id = rec.get("record_id", "")
        lookup_entry = image_lookup.get(record_id, {})
        ip = lookup_entry.get("image_path")
        if not ip and os.path.isdir(resized_dir):
            ip = os.path.join(resized_dir, rec.get("name", ""))
        if not ip or not os.path.exists(ip):
            image_paths.append(None)
            results[i] = _make_error_entry(record_id, rec.get("name", ""), ip if ip else None,
                                           rec.get("extracted_label", ""), rec, "image not found",
                                           lookup_entry.get("all_mean", None))
        else:
            image_paths.append(ip)

    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 2)
    request_timeout = aiohttp.ClientTimeout(total=args.request_timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=request_timeout) as session:
        task_to_idx: Dict[asyncio.Task, int] = {}
        pending: set = set()
        for i, ip in enumerate(image_paths):
            if ip is None:
                continue
            task = asyncio.create_task(
                generate_one_sglang(session, base_url, ip,
                                    max_tokens=args.max_new_tokens,
                                    temperature=args.temperature,
                                    timeout_s=args.request_timeout, sem=sem),
                name=str(i),
            )
            task_to_idx[task] = i
            pending.add(task)

        total_start = time.time()

        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    i = task_to_idx[task]
                    rec = selected[i]
                    record_id = rec.get("record_id", "")
                    label = rec.get("extracted_label", "")
                    image_name = rec.get("name", "")
                    lookup_entry = image_lookup.get(record_id, {})
                    mean_token_score = lookup_entry.get("all_mean", None)
                    try:
                        response_text, tokens_gen, dt = task.result()
                        locations = extract_locations(response_text)
                        results[i] = _make_result_entry(record_id, image_name, image_paths[i],
                                                        label, rec, response_text, locations,
                                                        mean_token_score)
                        states = [loc["state"] for loc in locations if loc.get("state")]
                        print(f"  [{i + 1}/{n}] {record_id}  {dt:.1f}s  states={', '.join(states) if states else 'none'}")
                    except Exception as exc:
                        results[i] = _make_error_entry(record_id, image_name, image_paths[i],
                                                       label, rec, str(exc), mean_token_score)
                        print(f"  [{i + 1}/{n}] {record_id}  ERROR: {exc}")
        finally:
            for t in pending:
                t.cancel()

    wall_elapsed = time.time() - total_start
    _write_output_and_summary(results, args, n, wall_elapsed)


# ═══════════════════════════════════════════════════════════════════
#  shared helpers
# ═══════════════════════════════════════════════════════════════════

def _write_output_and_summary(results, args, n, wall_elapsed):
    output_path = args.output
    if not output_path:
        input_path = os.path.join(ROOT_DIR, args.input) if not os.path.isabs(args.input) else args.input
        output_path = os.path.join(
            os.path.dirname(input_path),
            f"{os.path.splitext(os.path.basename(input_path))[0]}_travel_responses.jsonl",
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    errors = sum(1 for r in results if r is not None and r.get("_travel_error"))
    successes = n - errors
    print(f"\n--- Summary ---")
    print(f"  Queried:       {n}")
    print(f"  Successful:    {successes}")
    print(f"  Errors:        {errors}")
    print(f"  Wall time:     {wall_elapsed:.1f}s")
    print(f"  Output:        {output_path}")


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════

async def main():
    args = parse_args()

    model_key = args.model
    model_info = MODEL_REGISTRY[model_key]

    input_path = os.path.join(ROOT_DIR, args.input) if not os.path.isabs(args.input) else args.input
    if not os.path.exists(input_path):
        sys.exit(f"Input not found: {input_path}")

    print(f"Loading records from {input_path} ...")
    records = load_records(input_path)
    print(f"  Total records: {len(records)}")

    # Image lookup (needed for scoring)
    image_lookup = build_image_lookup(args.model)

    # Sample top (highest) and bottom (lowest) by mean token score
    scored = []
    for rec in records:
        rid = rec.get("record_id", "")
        info = image_lookup.get(rid, {})
        score = info.get("all_mean", float("-inf"))
        scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)

    top_n = min(args.num_liberal, len(scored))
    bottom_n = min(args.num_conservative, len(scored) - top_n)
    seen_ids = set()
    selected = []
    for score, rec in scored[:top_n]:
        rid = rec.get("record_id", "")
        if rid not in seen_ids:
            seen_ids.add(rid)
            selected.append(rec)
    for score, rec in scored[-bottom_n:]:
        rid = rec.get("record_id", "")
        if rid not in seen_ids:
            seen_ids.add(rid)
            selected.append(rec)
    rng = random.Random(args.seed)
    rng.shuffle(selected)
    print(f"  Highest-score: {top_n}  (range: [{scored[top_n-1][0]:.4f}, {scored[0][0]:.4f}])")
    if bottom_n > 0:
        print(f"  Lowest-score:  {bottom_n}  (range: [{scored[-bottom_n][0]:.4f}, {scored[-1][0]:.4f}])")
    resized_dir = os.path.join(
        ROOT_DIR, "results", "token_scoring", args.model, "easyportrait", "_resized_images_800",
    )

    if args.backend == "transformers":
        run_transformers(args, model_key, model_info, selected, image_lookup, resized_dir)
    else:
        await run_sglang(args, model_key, model_info, selected, image_lookup, resized_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
