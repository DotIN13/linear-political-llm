"""Pilot: can an agentic tool transcript carry the stimulus image?

This answers five questions before any of it goes into the harness (docs/bench/08):

  Q1  does ``apply_chat_template`` accept assistant+tool_calls / role="tool"?
  Q2  can an image ride inside a tool_result, and where do its tokens land?
  Q3  does the model actually *read* an image delivered that way?
  Q4  does the ideology probe read the same on those image tokens?
  Q5  same story on Gemma?

Nothing in ``bench/`` is modified. Model loading, image prep, chat encoding, the
capture hooks and the probe runtime are all imported from
``scripts/probes/token_scoring.py`` via ``bench.adaptors.local_hf.token_scoring()``.
The one thing written here that mirrors an existing helper is ``encode()``:
``token_scoring.encode_prompts`` has no ``tools=`` parameter and the agentic
prompt needs one, so this is that call plus that keyword.

Usage (see bench/pilots/toolfmt.sbatch):

    python -m bench.pilots.tool_format_probe --phases items          # CPU
    python -m bench.pilots.tool_format_probe --phases template       # CPU
    python -m bench.pilots.tool_format_probe --phases behavior,probe # GPU
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bench.adaptors.local_hf import token_scoring, split_probe_id  # noqa: E402

OUT_DIR = os.path.join(ROOT_DIR, "runs", "pilot_toolfmt")
IMAGE_CACHE = os.path.join(OUT_DIR, "_image_cache")
LVIS_JSON = os.path.join(ROOT_DIR, "datasets", "lvis", "lvis_v1_train.json")
COCO_TRAIN = os.path.join(ROOT_DIR, "datasets", "lvis", "train", "train2017")
LVIS_RESIZED = os.path.join(
    ROOT_DIR, "results", "token_scoring", "qwen3_vl", "lvis", "_resized_images_800"
)
ITEMS_PATH = os.path.join(OUT_DIR, "items20.json")

QWEN_PATH = "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
GEMMA_PATH = "/home/tzhang3/jevans/models/gemma-4-E4B-it"
PROBE_ID = "combined_ideology_headwise_linear"
K_SWEEP = (8, 16, 32, 64, 96)
DEFAULT_K = 16  # probes/headwise_linear_probe.py score_samples(k=16); token_scoring --top-k 16

# ---------------------------------------------------------------------------
# the agentic prompt shape under test
# ---------------------------------------------------------------------------

SYSTEM_PLAIN = "You are a helpful assistant."
SYSTEM_AGENTIC = (
    "You are a helpful assistant. The user's personal files live at /memory/user. "
    "You may list that directory and open files in it when it helps you answer."
)
FILENAMES = ["note_2024_03.txt", "img_0417.jpg", "scan_0092.jpg"]
VIEWED = "img_0417.jpg"
OPENER = "Have a look through my stuff first."
ACK = "I've looked through your files."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory to list."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "Open an image file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Image file to open."}},
                "required": ["path"],
            },
        },
    },
]


def _text(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _call(name: str, path: str) -> Dict[str, Any]:
    # content must be a list even when empty: transformers' apply_chat_template
    # iterates message["content"] to collect visuals and a str crashes it.
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": {"path": path}}}],
    }


def _preamble(system: str, first_user: str, system_in_user: bool) -> List[Dict[str, Any]]:
    """A system turn plus the first user turn -- or the two folded together.

    Gemma's template runs ``messages[0]['content'] | trim`` on the system turn,
    which stringifies a content *list* into the prompt as a literal Python repr;
    a plain string would instead crash transformers' visual scan, which iterates
    ``message["content"]`` for every message. Folding it into the user turn is
    the only shape that is clean on both models.
    """
    if system_in_user:
        return [_text("user", f"{system}\n\n{first_user}")]
    return [_text("system", system), _text("user", first_user)]


def _agentic_prefix(image_path: Optional[str], system_in_user: bool) -> List[Dict[str, Any]]:
    """system -> user -> list_dir -> listing -> view_image -> (image?) -> ack."""
    viewed: List[Dict[str, Any]] = []
    if image_path is not None:
        viewed.append({"type": "image", "image": image_path})
    viewed.append({"type": "text", "text": VIEWED})
    return _preamble(SYSTEM_AGENTIC, OPENER, system_in_user) + [
        _call("list_dir", "/memory/user"),
        _text("tool", "\n".join(FILENAMES)),
        _call("view_image", f"/memory/user/{VIEWED}"),
        {"role": "tool", "content": viewed},
        _text("assistant", ACK),
    ]


TRANSCRIPT_AS_TEXT = (
    f"{OPENER}\n\n"
    "[tool transcript]\n"
    'assistant -> list_dir(path="/memory/user")\n'
    "tool -> " + "\n        ".join(FILENAMES) + "\n"
    f'assistant -> view_image(path="/memory/user/{VIEWED}")\n'
    f"tool -> {VIEWED} (the image itself is attached below)\n"
    "[end of tool transcript]"
)


def build_messages(mode: str, image_path: str, question: str,
                   system_in_user: bool = False) -> Tuple[List[Dict[str, Any]], Optional[List]]:
    """(messages, tools) for one delivery mode."""
    image_then_question = {"role": "user",
                           "content": [{"type": "image", "image": image_path},
                                       {"type": "text", "text": question}]}

    if mode == "M0_noimage":
        # control: no image at all. Measures how far the language prior alone
        # gets you on this question set -- the ceiling that M1/M2/M3 must beat.
        return _preamble(SYSTEM_PLAIN, question, system_in_user), None

    if mode == "M1_user":
        # baseline: how bench delivers images today
        if system_in_user:
            return [{"role": "user", "content": [{"type": "text", "text": SYSTEM_PLAIN + "\n\n"},
                                                 {"type": "image", "image": image_path},
                                                 {"type": "text", "text": question}]}], None
        return [_text("system", SYSTEM_PLAIN), image_then_question], None

    if mode == "M2_tool":
        return _agentic_prefix(image_path, system_in_user) + [_text("user", question)], TOOLS

    if mode == "M1b_user_in_agentic_ctx":
        # same context as M2, image moved to the final user turn: isolates
        # "which turn the image sits in" from "how much transcript precedes it".
        return _agentic_prefix(None, system_in_user) + [image_then_question], TOOLS

    if mode == "M3_text":
        return _preamble(SYSTEM_PLAIN, TRANSCRIPT_AS_TEXT, system_in_user) + [
            _text("assistant", ACK), image_then_question], None

    raise ValueError(f"unknown mode {mode!r}")


MODES = ["M1_user", "M2_tool", "M3_text", "M1b_user_in_agentic_ctx"]

# Behaviour arms: (arm name, delivery mode, which image is delivered).
# "other" delivers a *different* item's image with this item's question. If the
# model is reading the image it must stop naming this item's true category; if
# it keeps naming it, the answer was coming from the prior, not from the pixels.
ARMS: List[Tuple[str, str, Optional[str]]] = [
    ("M1_user", "M1_user", "self"),
    ("M2_tool", "M2_tool", "self"),
    ("M3_text", "M3_text", "self"),
    ("M1b_user_in_agentic_ctx", "M1b_user_in_agentic_ctx", "self"),
    ("C0_noimage_prior", "M0_noimage", None),
    ("C1_user_mismatch", "M1_user", "other"),
    ("C2_tool_mismatch", "M2_tool", "other"),
]
MISMATCH_OFFSET = 10


# ---------------------------------------------------------------------------
# Q3 items: 20 LVIS images, one large unambiguous category each
# ---------------------------------------------------------------------------

def clean_name(raw: str) -> str:
    """'beer_bottle' -> 'beer bottle'; 'crab_(animal)' -> 'crab'."""
    name = re.sub(r"_\([^)]*\)", "", raw)
    return name.replace("_", " ").strip().lower()


def head_noun(name: str) -> str:
    return name.split()[-1]


def phase_items(n: int, seed: int) -> None:
    print(f"[items] parsing {LVIS_JSON} ...", flush=True)
    started = time.time()
    with open(LVIS_JSON, "r", encoding="utf-8") as handle:
        lvis = json.load(handle)
    print(f"[items] parsed in {time.time() - started:.0f}s", flush=True)

    cats = {c["id"]: c for c in lvis["categories"]}
    imgs = {im["id"]: im for im in lvis["images"]}

    # biggest annotation per image, and the set of categories present
    biggest: Dict[int, Dict[str, Any]] = {}
    present: Dict[int, set] = {}
    for ann in lvis["annotations"]:
        iid = ann["image_id"]
        present.setdefault(iid, set()).add(ann["category_id"])
        cur = biggest.get(iid)
        if cur is None or ann["area"] > cur["area"]:
            biggest[iid] = ann

    rng = random.Random(seed)
    picked: List[Dict[str, Any]] = []
    used_cats: set = set()
    used_distractors: set = set()

    for iid in sorted(biggest):
        if len(picked) >= n:
            break
        ann = biggest[iid]
        meta = imgs[iid]
        area_frac = ann["area"] / float(meta["height"] * meta["width"])
        if area_frac < 0.25:
            continue
        true_cat = cats[ann["category_id"]]
        if true_cat["frequency"] != "f":            # common enough to be nameable
            continue
        true_name = clean_name(true_cat["name"])
        if true_cat["id"] in used_cats or " " in true_name and len(true_name.split()) > 2:
            continue

        file_name = os.path.basename(meta["coco_url"])
        src = os.path.join(COCO_TRAIN, file_name)
        if not os.path.exists(src):
            continue

        # distractor: a category LVIS *verified absent* from this image
        negs = [cats[c] for c in meta.get("neg_category_ids", []) if c in cats]
        cand = [c for c in negs
                if c["frequency"] == "f"
                and c["id"] not in present.get(iid, set())
                and c["id"] not in used_distractors
                and head_noun(clean_name(c["name"])) != head_noun(true_name)
                and not set(clean_name(c["name"]).split()) & set(true_name.split())]
        if not cand:
            continue
        dist = rng.choice(sorted(cand, key=lambda c: c["id"]))
        dist_name = clean_name(dist["name"])

        used_cats.add(true_cat["id"])
        used_distractors.add(dist["id"])
        true_first = len(picked) % 2 == 0        # balanced option order
        a, b = (true_name, dist_name) if true_first else (dist_name, true_name)
        picked.append({
            "image_id": iid,
            "file_name": file_name,
            "source_path": src,
            "true_name": true_name,
            "distractor_name": dist_name,
            "true_first": true_first,
            "area_frac": round(area_frac, 3),
            "question": f"Which of these is in the photo: a {a} or a {b}? Answer with one word.",
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(ITEMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(picked, handle, indent=2)
    print(f"[items] wrote {len(picked)} items -> {ITEMS_PATH}")
    for row in picked:
        print(f"  {row['file_name']}  {row['true_name']!r} vs {row['distractor_name']!r} "
              f"(area {row['area_frac']})")


def load_items() -> List[Dict[str, Any]]:
    with open(ITEMS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# encoding / grading helpers
# ---------------------------------------------------------------------------

def encode(processor, messages: List[Dict[str, Any]], tools: Optional[List] = None) -> Dict[str, Any]:
    """token_scoring.encode_prompts + a tools= keyword (which it does not expose)."""
    import torch

    kwargs: Dict[str, Any] = {}
    if tools is not None:
        kwargs["tools"] = tools
    encoded = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        processor_kwargs={"padding": True},
        return_dict=True,
        return_tensors="pt",
        **kwargs,
    )
    return {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in encoded.items()}


def render(processor, messages: List[Dict[str, Any]], tools: Optional[List] = None) -> str:
    kwargs: Dict[str, Any] = {"tools": tools} if tools is not None else {}
    out = processor.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True, **kwargs
    )
    return out[0] if isinstance(out, list) else out


def image_token_spans(input_ids, image_ids: Sequence[int]) -> Tuple[List[int], List[Tuple[int, int]]]:
    idx = [i for i, t in enumerate(input_ids) if t in set(image_ids)]
    spans: List[Tuple[int, int]] = []
    for i in idx:
        if spans and i == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], i)
        else:
            spans.append((i, i))
    return idx, spans


PUNCT = re.compile(r"[^a-z0-9 ]+")


def grade(output: str, true_name: str, distractor_name: str) -> Dict[str, Any]:
    text = PUNCT.sub(" ", (output or "").lower())
    words = set(text.split())
    t_head, d_head = head_noun(true_name), head_noun(distractor_name)
    hit_true = t_head in words or true_name in text
    hit_dist = d_head in words or distractor_name in text
    if hit_true and not hit_dist:
        verdict = "correct"
    elif hit_dist and not hit_true:
        verdict = "wrong"
    elif hit_true and hit_dist:
        verdict = "both"
    else:
        verdict = "neither"
    return {"verdict": verdict, "correct": verdict == "correct"}


def load_model(model_path: str, model_family: str, device_map: str = "auto"):
    import torch
    from transformers import AutoProcessor

    ts = token_scoring()
    processor = AutoProcessor.from_pretrained(model_path)
    model_cls = ts.select_model_loader(model_family, model_path)
    dtype = ts.resolve_torch_dtype("auto")
    if model_family == "qwen3-vl":
        load_kwargs = {"dtype": dtype, "low_cpu_mem_usage": True, "device_map": device_map}
    else:
        load_kwargs = {"torch_dtype": "auto" if torch.cuda.is_available() else dtype,
                       "device_map": device_map}
    print(f"[load] {model_path} ({model_family}) {load_kwargs}", flush=True)
    model = model_cls.from_pretrained(model_path, **load_kwargs)
    model.eval()
    return processor, model


def resolve(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ts = token_scoring()
    out = ts.resolve_messages_images(messages, image_root=None, cache_dir=IMAGE_CACHE)
    if out is None:
        raise RuntimeError("image preparation failed")
    return out


# ---------------------------------------------------------------------------
# Q1 / Q2
# ---------------------------------------------------------------------------

def phase_template(model_path: str, model_family: str, tag: str, system_in_user: bool) -> None:
    from transformers import AutoProcessor

    ts = token_scoring()
    os.makedirs(OUT_DIR, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_path)
    image_ids = sorted(ts.gather_candidate_image_token_ids(processor.tokenizer))

    items = load_items()
    item = items[0]
    report: Dict[str, Any] = {"tag": tag, "model_path": model_path,
                              "system_in_user": system_in_user,
                              "image_token_ids": image_ids, "modes": {}}
    lines: List[str] = []

    for mode in MODES:
        messages, tools = build_messages(mode, item["source_path"], item["question"], system_in_user)
        messages = resolve(messages)
        entry: Dict[str, Any] = {}
        try:
            text = render(processor, messages, tools)
            entry["rendered_ok"] = True
        except Exception as exc:                     # noqa: BLE001 - reporting the failure is the point
            entry["rendered_ok"] = False
            entry["render_error"] = f"{type(exc).__name__}: {exc}"
            text = ""
        try:
            encoded = encode(processor, messages, tools)
            ids = [int(i) for i in encoded["input_ids"][0].tolist()]
            idx, spans = image_token_spans(ids, image_ids)
            entry.update({
                "encoded_ok": True,
                "n_all_tokens": len(ids),
                "num_image_tokens": len(idx),
                "image_token_spans": spans,
                "frac_before_image": round(spans[0][0] / len(ids), 3) if spans else None,
                "pixel_values_shape": list(encoded["pixel_values"].shape)
                if "pixel_values" in encoded else None,
            })
            # where does the image sit relative to the tool block?
            if spans:
                head = processor.tokenizer.decode(ids[max(0, spans[0][0] - 24):spans[0][0]])
                tail = processor.tokenizer.decode(ids[spans[-1][1] + 1:spans[-1][1] + 25])
                entry["context_before_image"] = head
                entry["context_after_image"] = tail
        except Exception as exc:                     # noqa: BLE001
            entry["encoded_ok"] = False
            entry["encode_error"] = f"{type(exc).__name__}: {exc}"

        report["modes"][mode] = entry
        lines.append(f"\n{'=' * 78}\n=== {tag} / {mode}  (tools={'yes' if tools else 'no'})\n{'=' * 78}\n")
        lines.append(text)

    with open(os.path.join(OUT_DIR, f"q1_render_{tag}.txt"), "w", encoding="utf-8") as handle:
        handle.write("".join(lines))
    with open(os.path.join(OUT_DIR, f"q2_tokens_{tag}.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Q3
# ---------------------------------------------------------------------------

def phase_behavior(model_path: str, model_family: str, tag: str, max_new_tokens: int,
                   system_in_user: bool) -> None:
    import torch

    ts = token_scoring()
    os.makedirs(OUT_DIR, exist_ok=True)
    processor, model = load_model(model_path, model_family)
    image_ids = sorted(ts.gather_candidate_image_token_ids(processor.tokenizer))
    items = load_items()

    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        other = items[(i + MISMATCH_OFFSET) % len(items)]
        for arm, mode, which in ARMS:
            delivered = None if which is None else (item if which == "self" else other)
            path = delivered["source_path"] if delivered else ""
            messages, tools = build_messages(mode, path, item["question"], system_in_user)
            encoded = encode(processor, resolve(messages), tools)
            ids = [int(t) for t in encoded["input_ids"][0].tolist()]
            idx, spans = image_token_spans(ids, image_ids)
            with torch.no_grad():
                out = model.generate(**ts.move_to_device(encoded, model),
                                     max_new_tokens=max_new_tokens, do_sample=False)
            text = processor.tokenizer.decode(out[0][encoded["input_ids"].shape[1]:],
                                              skip_special_tokens=True).strip()
            row = {"image_id": item["image_id"], "file_name": item["file_name"],
                   "arm": arm, "mode": mode, "delivered_image": which,
                   "delivered_file": delivered["file_name"] if delivered else None,
                   "delivered_true_name": delivered["true_name"] if delivered else None,
                   "true_name": item["true_name"], "distractor_name": item["distractor_name"],
                   "true_first": item["true_first"], "question": item["question"],
                   "output": text, "n_all_tokens": len(ids), "num_image_tokens": len(idx),
                   "image_token_spans": spans}
            row.update(grade(text, item["true_name"], item["distractor_name"]))
            words = set(PUNCT.sub(" ", text.lower()).split())
            row["names_asked_true"] = row["verdict"] in ("correct", "both")
            row["names_delivered_true"] = bool(
                delivered and (head_noun(delivered["true_name"]) in words
                               or delivered["true_name"] in text.lower()))
            rows.append(row)
            print(f"[{i + 1:2d}/{len(items)}] {arm:24s} {row['verdict']:8s} {text!r}", flush=True)

    acc: Dict[str, Dict[str, Any]] = {}
    for arm, _mode, _which in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        acc[arm] = {
            "n": len(sub),
            "accuracy": round(sum(r["correct"] for r in sub) / max(len(sub), 1), 3),
            "verdicts": {v: sum(r["verdict"] == v for r in sub)
                         for v in ("correct", "wrong", "both", "neither")},
            "names_asked_true": sum(r["names_asked_true"] for r in sub),
            "names_delivered_true": sum(r["names_delivered_true"] for r in sub),
            "mean_image_tokens": round(sum(r["num_image_tokens"] for r in sub) / max(len(sub), 1), 1),
        }
    with open(os.path.join(OUT_DIR, f"q3_behavior_{tag}.json"), "w", encoding="utf-8") as handle:
        json.dump({"tag": tag, "model_path": model_path, "accuracy": acc, "rows": rows}, handle, indent=2)
    print(json.dumps(acc, indent=2))


# ---------------------------------------------------------------------------
# Q4
# ---------------------------------------------------------------------------

def phase_probe(model_path: str, model_family: str, tag: str, probe_id: str,
                mode_pair: Sequence[str], system_in_user: bool) -> None:
    import numpy as np

    ts = token_scoring()
    os.makedirs(OUT_DIR, exist_ok=True)
    processor, model = load_model(model_path, model_family)
    image_ids = sorted(ts.gather_candidate_image_token_ids(processor.tokenizer))

    prefix, probe_type = split_probe_id(probe_id)
    probe_cls = ts.PROBE_CLASSES[probe_type]
    probe = probe_cls(model_path=model_path, prefix=prefix, mode="vision",
                      model_family=model_family, data_dir="results/probes").load()
    if probe.metadata_ is not None:
        meta_paths = probe.metadata_.extra.get("module_paths")
        if isinstance(meta_paths, dict):
            from probes.base import resolve_module_paths
            probe.module_paths = resolve_module_paths(model_family, meta_paths)

    runtimes = {k: ts.build_probe_runtime(model=model, probe=probe, top_k=k,
                                          mode="vision", model_family=model_family)
                for k in K_SWEEP}
    union = sorted({name for rt in runtimes.values() for name in rt["module_names"]})
    print(f"[probe] {probe_id} k={list(K_SWEEP)} union modules={len(union)}", flush=True)

    items = load_items()
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        row: Dict[str, Any] = {"image_id": item["image_id"], "file_name": item["file_name"], "modes": {}}
        for mode in mode_pair:
            messages, tools = build_messages(mode, item["source_path"], item["question"], system_in_user)
            encoded = encode(processor, resolve(messages), tools)
            ids = [int(t) for t in encoded["input_ids"][0].tolist()]
            idx, spans = image_token_spans(ids, image_ids)
            captured = ts.capture_module_outputs(model=model, encoded=encoded, module_names=union)
            entry: Dict[str, Any] = {"n_all_tokens": len(ids), "num_image_tokens": len(idx),
                                     "image_token_spans": spans, "s_img": {}, "s_txt": {}}
            for k, rt in runtimes.items():
                scores = ts.score_from_captured(captured, rt)[0].cpu().numpy()
                entry["s_img"][str(k)] = float(scores[idx].mean()) if idx else None
                entry["s_txt"][str(k)] = float(scores[-1])
            row["modes"][mode] = entry
        rows.append(row)
        a, b = mode_pair
        print(f"[{i + 1:2d}/{len(items)}] s_img(k={DEFAULT_K}) "
              f"{a}={row['modes'][a]['s_img'][str(DEFAULT_K)]:+.4f} "
              f"{b}={row['modes'][b]['s_img'][str(DEFAULT_K)]:+.4f}", flush=True)

    a, b = mode_pair
    summary: Dict[str, Any] = {"probe_id": probe_id, "default_k": DEFAULT_K,
                               "k_sweep": list(K_SWEEP), "mode_pair": list(mode_pair), "by_k": {}}
    for k in K_SWEEP:
        va = np.array([r["modes"][a]["s_img"][str(k)] for r in rows], dtype=float)
        vb = np.array([r["modes"][b]["s_img"][str(k)] for r in rows], dtype=float)
        summary["by_k"][str(k)] = {
            "n": len(va),
            "n_heads": int(sum(len(v) for v in runtimes[k]["groups"].values()))
            if isinstance(next(iter(runtimes[k]["groups"].values())), list) else len(runtimes[k]["groups"]),
            "pearson_r": float(np.corrcoef(va, vb)[0, 1]),
            "spearman_r": float(np.corrcoef(np.argsort(np.argsort(va)),
                                            np.argsort(np.argsort(vb)))[0, 1]),
            "mean_abs_diff": float(np.mean(np.abs(va - vb))),
            f"mean_s_img_{a}": float(va.mean()),
            f"mean_s_img_{b}": float(vb.mean()),
            f"sd_s_img_{a}": float(va.std(ddof=1)),
            f"sd_s_img_{b}": float(vb.std(ddof=1)),
        }
    with open(os.path.join(OUT_DIR, f"q4_probe_{tag}.json"), "w", encoding="utf-8") as handle:
        json.dump({"tag": tag, "summary": summary, "rows": rows}, handle, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", default="template",
                        help="comma list of: items,template,behavior,probe")
    parser.add_argument("--model-path", default=QWEN_PATH)
    parser.add_argument("--model-family", default="qwen3-vl")
    parser.add_argument("--tag", default=None, help="output filename tag (default: model dir name)")
    parser.add_argument("--probe", default=PROBE_ID)
    parser.add_argument("--n-items", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--probe-modes", default="M1_user,M2_tool")
    parser.add_argument("--system-in-user", action="store_true",
                        help="fold the system prompt into the first user turn (needed for gemma4)")
    args = parser.parse_args()

    tag = args.tag or os.path.basename(args.model_path.rstrip("/")).lower()
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    os.makedirs(OUT_DIR, exist_ok=True)

    for phase in phases:
        print(f"\n########## phase: {phase} ({tag}) ##########", flush=True)
        if phase == "items":
            phase_items(args.n_items, args.seed)
        elif phase == "template":
            phase_template(args.model_path, args.model_family, tag, args.system_in_user)
        elif phase == "behavior":
            phase_behavior(args.model_path, args.model_family, tag, args.max_new_tokens, args.system_in_user)
        elif phase == "probe":
            phase_probe(args.model_path, args.model_family, tag, args.probe,
                        [m.strip() for m in args.probe_modes.split(",")], args.system_in_user)
        else:
            raise SystemExit(f"unknown phase {phase!r}")


if __name__ == "__main__":
    main()
