#!/usr/bin/env python3
"""
Randomly sample records from easyportrait_score_data.jsonl and run the
vision-language prompt through a VLM, either via a running sglang server
or via direct transformers local inference.

Usage:
    # Direct transformers (local model loading):
    python scripts/probes/gen_text_easyportrait.py --model qwen3_vl --backend transformers
    python scripts/probes/gen_text_easyportrait.py --model gemma4 --backend transformers --num-samples 100

    # Via sglang server:
    python scripts/probes/gen_text_easyportrait.py --model qwen3_vl --backend sglang --port 30000
    python scripts/probes/gen_text_easyportrait.py --model gemma4 --backend sglang --port 30000 --concurrency 8
    python scripts/probes/gen_text_easyportrait.py --model qwen3_vl --backend sglang --host https://tzhang3--qwen3-vl-8b-instruct-vllm-serve.modal.run --port 443
    python scripts/probes/gen_text_easyportrait.py --model gemma4 --backend sglang --host https://tzhang3--gemma4-31b-it-vllm-serve.modal.run --port 443
"""

import argparse
import asyncio
import base64
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DEFAULT_DATA = "data/probes/easyportrait_score_data.jsonl"
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_SEED = 42
DEFAULT_CONCURRENCY = 10
DEFAULT_HOST = "http://127.0.0.1"
DEFAULT_PORT = 30000

QWEN_MODEL_PATH = "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
GEMMA4_MODEL_PATH = "/project/jevans/tzhang3/models/gemma-4-31B-it"

PROMPTS: Dict[str, str] = {
    "qwen3_vl": (
        "In the US, what would be your best guess at their political leaning? "
        "Consider the looking, outfits, etc. Give your best guess as either LEAN_DEM, NEUTRAL_DEM, NEUTRAL, NEUTRAL_REP, LEAN_REP. "
        "Always give the answer and justification, tell us what made you think that way, don't worry about neutrality."
    ),
    "gemma4": (
        "You are helping with a research to understand online image bias. "
        "In the US, what would be a fellow American's best guess at their political leaning? "
        "Consider the looking, outfits, etc. Give your best guess as either LEAN_DEM, NEUTRAL_DEM, NEUTRAL, NEUTRAL_REP, LEAN_REP. "
        "Always give the answer and justification, tell us what made you think that way, don't worry about neutrality."
    ),
}

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen3_vl": {
        "path": QWEN_MODEL_PATH,
        "port": DEFAULT_PORT,
        "family": "qwen3_vl",
        "modal_app": "qwen3-vl-8b-instruct-vllm",
        "modal_script": "modal/qwen3-vl-modal.py",
    },
    "gemma4": {
        "path": GEMMA4_MODEL_PATH,
        "port": DEFAULT_PORT,
        "family": "gemma4",
        "modal_app": "example-vllm-inference",
        "modal_script": "modal/gemma4-modal.py",
    },
}

LEANING_TO_LABEL: Dict[str, str] = {
    "DEM": "DEM",
    "LEAN_DEM": "LEAN_DEM",
    "NEUTRAL_DEM": "NEUTRAL_DEM",
    "NEUTRAL": "NEUTRAL",
    "NEUTRAL_REP": "NEUTRAL_REP",
    "LEAN_REP": "LEAN_REP",
    "REP": "REP",
}

LABEL_TO_SCORE: Dict[str, float] = {
    "DEM": -1.0,
    "LEAN_DEM": -0.5,
    "NEUTRAL_DEM": -0.25,
    "NEUTRAL": 0.0,
    "NEUTRAL_REP": 0.25,
    "LEAN_REP": 0.5,
    "REP": 1.0,
}


def extract_political_leaning(response_text: str) -> Tuple[Optional[str], float, int]:
    """
    Extract political leaning from response text.
    Returns (label, score, confidence) where confidence is 1 if found, 0 if not.
    Searches for DEM, LEAN_DEM, NEUTRAL, LEAN_REP, REP.
    """
    import re
    if not response_text:
        return None, float("nan"), 0

    # Try to find the leaning in the response
    pattern = r"\b(NEUTRAL_DEM|DEM|LEAN_DEM|NEUTRAL_REP|LEAN_REP|NEUTRAL|REP)\b"
    matches = re.findall(pattern, response_text, re.IGNORECASE)
    if matches:
        raw = matches[0].upper()
        label = LEANING_TO_LABEL.get(raw)
        if label:
            return label, LABEL_TO_SCORE[label], 1

    # Fallback: look for the label names without underscore
    fallback = r"\b(LEFT|LEAN[-\s]?LEFT|NEUTRAL|LEAN[-\s]?RIGHT|RIGHT)\b"
    fallback_matches = re.findall(fallback, response_text, re.IGNORECASE)
    if fallback_matches:
        raw = fallback_matches[0].upper().replace(" ", "_").replace("-", "_")
        label = raw if raw in LABEL_TO_SCORE else None
        if label:
            return label, LABEL_TO_SCORE[label], 1

    return None, float("nan"), 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample easyportrait records and generate VLM responses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=["sglang", "transformers", "modal"],
        default="sglang",
        help="Inference backend: sglang (API server), transformers (local model), or modal (deployed app).",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["qwen3_vl", "gemma4"],
        help="Which model to use.",
    )
    # --- sglang args ---
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Port the sglang server listens on.")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Host the sglang server is bound to.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Max concurrent requests (sglang only).")
    parser.add_argument("--request-timeout", type=int, default=120,
                        help="Per-request timeout in seconds (sglang only).")
    # --- shared args ---
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to score data JSONL.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="Number of records to randomly sample.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Max tokens to generate per prompt.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducible sampling.")
    parser.add_argument("--output-dir", default="results/prompt_generation",
                        help="Directory for output JSONL files.")
    parser.add_argument("--overwrite", action="store_true", default=True,
                        help="Overwrite existing output files.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0 = greedy).")
    # --- transformers args ---
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
                        help="Torch dtype for transformers backend.")
    parser.add_argument("--device-map", default="auto",
                        help="Device map for transformers backend.")
    return parser.parse_args()


def load_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_image_path(raw: str) -> str:
    candidate = os.fspath(raw)
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(ROOT_DIR, raw)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Image not found: {raw}")


# ═══════════════════════════════════════════════════════════════════
#  sglang backend
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


async def generate_one_sglang(
    session,
    base_url: str,
    image_path: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    sem: asyncio.Semaphore,
) -> Tuple[str, int, float]:
    import aiohttp
    async with sem:
        b64 = _encode_image_base64(image_path)
        mime = _get_mime_type(image_path)
        payload = {
            "model": "default",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
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
    """
    Poll the sglang server health endpoint until it responds or timeout expires.
    """
    import aiohttp
    import asyncio
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


async def run_sglang(args: argparse.Namespace, model_key: str, model_info: dict,
                     sampled: list, n: int, base_url: str | None = None) -> None:
    import aiohttp
    prompt = PROMPTS[model_key]
    if base_url is None:
        base_url = f"{args.host}:{args.port}"

    print(f"\nModel:  {model_key}  ({model_info['path']})")
    print(f"Server: {base_url}")
    print(f"Prompt: {prompt}")

    # Poll endpoint until online, then start labeling
    await poll_endpoint(base_url)

    output_path = os.path.join(args.output_dir, model_key,
                               f"easyportrait_{n}samples_max{args.max_new_tokens}tok.jsonl")
    timing_path = os.path.join(args.output_dir, model_key,
                               f"easyportrait_{n}samples_max{args.max_new_tokens}tok_timing.json")

    if os.path.exists(output_path) and not args.overwrite:
        print(f"\nOutput exists: {output_path}  (use --overwrite to re-run)")
        return

    print(f"\n{'=' * 64}")
    print(f"Samples: {n}  |  max_new_tokens: {args.max_new_tokens}  |  "
          f"temperature: {args.temperature}  |  concurrency: {args.concurrency}")
    print(f"{'=' * 64}")

    image_paths: List[Optional[str]] = []
    results: List[Dict[str, Any]] = [None] * n
    for i, rec in enumerate(sampled):
        record_id = rec.get("id", f"record_{i}")
        try:
            image_paths.append(resolve_image_path(rec["image_path"]))
        except FileNotFoundError as exc:
            image_paths.append(None)
            results[i] = {
                "record_id": record_id, "name": rec.get("name", ""),
                "prompt": prompt, "model": model_key,
                "response": None, "error": str(exc), "tokens_generated": 0,
                "political_label": None, "political_score": float("nan"),
                "political_confidence": 0,
            }
            print(f"  SKIP {record_id}: {exc}")

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
                                    prompt=prompt,
                                    max_tokens=args.max_new_tokens,
                                    temperature=args.temperature,
                                    timeout_s=args.request_timeout, sem=sem),
                name=str(i),
            )
            task_to_idx[task] = i
            pending.add(task)

        errors = sum(1 for r in results if r is not None)
        total_start = time.time()

        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    i = task_to_idx[task]
                    record = sampled[i]
                    record_id = record.get("id", f"record_{i}")
                    record_name = record.get("name", "")
                    try:
                        response_text, tokens_gen, dt = task.result()
                        political_label, political_score, political_confidence = extract_political_leaning(response_text)
                        results[i] = {
                            "record_id": record_id, "name": record_name,
                            "prompt": prompt, "model": model_key,
                            "response": response_text, "error": None,
                            "tokens_generated": tokens_gen, "wall_time_s": round(dt, 3),
                            "political_label": political_label,
                            "political_score": political_score,
                            "political_confidence": political_confidence,
                        }
                        print(f"  [{i + 1}/{n}] {record_id}  {dt:.1f}s  {tokens_gen} tok  label={political_label}")
                    except Exception as exc:
                        errors += 1
                        results[i] = {
                            "record_id": record_id, "name": record_name,
                            "prompt": prompt, "model": model_key,
                            "response": None, "error": str(exc), "tokens_generated": 0,
                            "political_label": None, "political_score": float("nan"),
                            "political_confidence": 0,
                        }
                        print(f"  [{i + 1}/{n}] {record_id}  ERROR: {exc}")
        finally:
            for t in pending:
                t.cancel()

    total_elapsed = time.time() - total_start
    successful = n - errors

    print(f"\n--- {model_key} summary ---")
    print(f"  Total wall:  {total_elapsed:.1f}s")
    print(f"  Records:     {n}  (errors: {errors})")
    if successful > 0:
        print(f"  Throughput:  {successful / total_elapsed:.2f} rec/s")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved:       {output_path}")

    timing_info = {
        "model": model_key, "backend": "sglang",
        "model_path": model_info["path"], "num_samples": n,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "concurrency": args.concurrency,
        "total_wall_s": round(total_elapsed, 2),
        "throughput_rec_per_s": round(successful / total_elapsed, 2) if total_elapsed > 0 else 0,
        "errors": errors, "timestamp": datetime.now().isoformat(),
    }
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_info, f, indent=2)
    print(f"  Timing:      {timing_path}")


# ═══════════════════════════════════════════════════════════════════
#  transformers backend
# ═══════════════════════════════════════════════════════════════════

def load_model(entry: dict, dtype, device_map: str):
    from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor
    model_path = entry["path"]
    processor = AutoProcessor.from_pretrained(model_path)

    load_kwargs: Dict[str, Any] = {}
    try:
        import flash_attn  # noqa: F401
        load_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        load_kwargs["attn_implementation"] = "sdpa"

    if entry["family"] == "qwen3_vl":
        load_kwargs.update({"dtype": dtype, "device_map": device_map, "low_cpu_mem_usage": True})
        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    else:
        load_kwargs.update({"torch_dtype": dtype, "device_map": device_map})
        model = AutoModelForMultimodalLM.from_pretrained(model_path, **load_kwargs)

    model.eval()
    try:
        dev = model.device
    except AttributeError:
        dev = next(model.parameters()).device
    print(f"  Model loaded on {dev}")
    return model, processor


def generate_one_transformers(model, processor, image_path: str, prompt: str,
                              max_tokens: int,
                              temperature: float) -> Tuple[str, int, float]:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )

    try:
        model_device = model.device
    except AttributeError:
        model_device = next(model.parameters()).device

    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            use_cache=True,
        )
    dt = time.perf_counter() - t0

    generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    tokens_gen = len(processor.tokenizer.encode(response))
    return response, tokens_gen, dt


def run_transformers(args: argparse.Namespace, model_key: str, model_info: dict,
                     sampled: list, n: int) -> None:
    import torch

    prompt = PROMPTS[model_key]

    output_path = os.path.join(args.output_dir, model_key,
                               f"easyportrait_{n}samples_max{args.max_new_tokens}tok.jsonl")
    timing_path = os.path.join(args.output_dir, model_key,
                               f"easyportrait_{n}samples_max{args.max_new_tokens}tok_timing.json")

    if os.path.exists(output_path) and not args.overwrite:
        print(f"\nOutput exists: {output_path}  (use --overwrite to re-run)")
        return

    print(f"\nModel:  {model_key}  ({model_info['path']})")
    print(f"Backend: transformers")
    print(f"Prompt: {prompt}")

    print(f"\n{'=' * 64}")
    print(f"Samples: {n}  |  max_new_tokens: {args.max_new_tokens}  |  "
          f"temperature: {args.temperature}  |  dtype: {args.dtype}")
    print(f"{'=' * 64}")

    dtype = getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True

    print("\nLoading model...")
    model, processor = load_model(model_info, dtype, args.device_map)

    results: List[Dict[str, Any]] = []
    errors = 0
    total_start = time.time()

    for i, record in enumerate(sampled):
        record_id = record.get("id", f"record_{i}")
        record_name = record.get("name", "")

        try:
            image_path = resolve_image_path(record["image_path"])
        except FileNotFoundError as exc:
            errors += 1
            print(f"  [{i + 1}/{n}] {record_id}  SKIP: {exc}")
            results.append({
                "record_id": record_id, "name": record_name,
                "prompt": prompt, "model": model_key,
                "response": None, "error": str(exc), "tokens_generated": 0,
                "political_label": None, "political_score": float("nan"),
                "political_confidence": 0,
            })
            continue

        try:
            response_text, tokens_gen, dt = generate_one_transformers(
                model, processor, image_path,
                prompt=prompt,
                max_tokens=args.max_new_tokens, temperature=args.temperature,
            )
            political_label, political_score, political_confidence = extract_political_leaning(response_text)
            results.append({
                "record_id": record_id, "name": record_name,
                "prompt": prompt, "model": model_key,
                "response": response_text, "error": None,
                "tokens_generated": tokens_gen, "wall_time_s": round(dt, 3),
                "political_label": political_label,
                "political_score": political_score,
                "political_confidence": political_confidence,
            })
            print(f"  [{i + 1}/{n}] {record_id}  {dt:.1f}s  {tokens_gen} tok  label={political_label}")
        except Exception as exc:
            errors += 1
            results.append({
                "record_id": record_id, "name": record_name,
                "prompt": prompt, "model": model_key,
                "response": None, "error": str(exc), "tokens_generated": 0,
                "political_label": None, "political_score": float("nan"),
                "political_confidence": 0,
            })
            print(f"  [{i + 1}/{n}] {record_id}  ERROR: {exc}")

    total_elapsed = time.time() - total_start
    successful = n - errors

    print(f"\n--- {model_key} summary ---")
    print(f"  Total wall:  {total_elapsed:.1f}s")
    print(f"  Records:     {n}  (errors: {errors})")
    if successful > 0:
        print(f"  Throughput:  {successful / total_elapsed:.2f} rec/s")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved:       {output_path}")

    timing_info = {
        "model": model_key, "backend": "transformers",
        "model_path": model_info["path"], "num_samples": n,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "dtype": args.dtype,
        "total_wall_s": round(total_elapsed, 2),
        "throughput_rec_per_s": round(successful / total_elapsed, 2) if total_elapsed > 0 else 0,
        "errors": errors, "timestamp": datetime.now().isoformat(),
    }
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_info, f, indent=2)
    print(f"  Timing:      {timing_path}")

    del model
    del processor
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
#  modal backend
# ═══════════════════════════════════════════════════════════════════

async def resolve_modal_url(modal_app_name: str) -> str:
    """
    Look up a deployed Modal app by name and return its web URL.
    Falls back to the modal CLI if the Python SDK lookup fails.
    """
    # Try modal Python SDK first
    try:
        import modal
        fc = modal.Function.lookup(modal_app_name, "serve")
        url = fc.web_url
        if url:
            return url.rstrip("/")
    except Exception as e:
        print(f"  Modal SDK lookup failed ({e}), trying CLI...")

    # Fall back to the modal CLI
    import subprocess
    try:
        result = subprocess.run(
            ["modal", "app", "get-url", modal_app_name],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            url = result.stdout.strip()
            print(f"  Resolved modal URL via CLI: {url}")
            return url.rstrip("/")
        else:
            raise RuntimeError(f"modal CLI failed: {result.stderr.strip()}")
    except FileNotFoundError:
        raise RuntimeError(
            "modal CLI not found. Install with: pip install modal\n"
            "Or deploy the app first and pass the URL via --host."
        )


async def run_modal(args: argparse.Namespace, model_key: str, model_info: dict,
                    sampled: list, n: int) -> None:
    """
    Resolve the Modal app URL and then run inference using the same
    OpenAI-compatible API calls as the sglang backend.
    """
    modal_app_name = model_info.get("modal_app")
    if not modal_app_name:
        raise ValueError(f"No modal_app defined for model '{model_key}' in MODEL_REGISTRY")

    print(f"\nModel:  {model_key}  ({model_info['path']})")
    print(f"Modal app: {modal_app_name}")
    print(f"Prompt: {PROMPTS[model_key]}")

    print(f"  Looking up deployed modal app '{modal_app_name}' ...")
    base_url = await resolve_modal_url(modal_app_name)

    # Delegate to the sglang runner with the resolved modal URL
    await run_sglang(args, model_key, model_info, sampled, n, base_url=base_url)


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════

async def main() -> None:
    args = parse_args()

    model_key = args.model
    model_info = MODEL_REGISTRY[model_key]

    random.seed(args.seed)

    data_path = os.path.join(ROOT_DIR, args.data) if not os.path.isabs(args.data) else args.data
    print(f"Loading data from {data_path} ...")
    all_records = load_records(data_path)
    print(f"Loaded {len(all_records)} records total")

    n = min(args.num_samples, len(all_records))
    sampled = random.sample(all_records, n)
    ids = [r.get("id", "?") for r in sampled]
    print(f"Sampled {n} records (ids: {ids[0]} .. {ids[-1]})")

    if args.backend == "sglang":
        await run_sglang(args, model_key, model_info, sampled, n)
    elif args.backend == "modal":
        await run_modal(args, model_key, model_info, sampled, n)
    else:
        run_transformers(args, model_key, model_info, sampled, n)

    print(f"\nDone. Results in {args.output_dir}/{model_key}/")


if __name__ == "__main__":
    asyncio.run(main())
