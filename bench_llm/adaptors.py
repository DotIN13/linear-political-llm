"""Model backends.

* ``OpenAICompat``: any OpenAI-compatible chat endpoint, a local vLLM server (the
  sbatch starts one) and hosted APIs alike. Decoding is greedy (``temperature=0``)
  and, when ``top_logprobs`` is asked for, the first generated token's top tokens
  come back as ``{token: logprob}``.
* ``HFProbe``: the model loaded in transformers in this process, so a trained probe
  from the repo's ``probes`` package can read its activations. A server cannot hand
  those back. torch, transformers and ``probes`` are imported in ``setup``, so the
  rest of this package still runs without them.
"""

from __future__ import annotations

import base64
import contextlib
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Sequence

from bench_llm.types import Response, Trial


def data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as handle:
        return f"data:{mime};base64,{base64.b64encode(handle.read()).decode('ascii')}"


def to_openai(messages: List[Dict[str, Any]], image_loader: Callable[[str], str] = data_uri) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        parts = []
        for p in content:
            if p["type"] == "text":
                parts.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                parts.append({"type": "image_url", "image_url": {"url": image_loader(p["image"])}})
            else:
                raise ValueError(f"unsupported part type {p['type']!r}")
        if all(p["type"] == "text" for p in parts):
            out.append({"role": m["role"], "content": "".join(p["text"] for p in parts)})
        else:
            out.append({"role": m["role"], "content": parts})
    return out


def payload(trial: Trial, model: str, seed: int, image_loader=data_uri) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model, "messages": to_openai(trial.conversation.messages, image_loader),
        "max_tokens": int(trial.max_new_tokens), "temperature": 0.0, "seed": int(seed),
    }
    if trial.top_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = int(trial.top_logprobs)
    return body


def first_token_logprobs(choice: Dict[str, Any]) -> Dict[str, float] | None:
    content = (choice.get("logprobs") or {}).get("content") or []
    if not content:
        return None
    top = content[0].get("top_logprobs") or []
    return {str(t["token"]): float(t["logprob"]) for t in top if t.get("token") is not None} or None


class OpenAICompat:
    """``name`` and ``model`` go into every trial key."""

    name = "openai_compat"

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8000/v1",
                 api_key: str | None = None, seed: int = 42, timeout: int = 600,
                 health_timeout: int = 900) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.seed = seed
        self.timeout = timeout
        self.health_timeout = health_timeout

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "model": self.model, "base_url": self.base_url, "seed": self.seed}

    def setup(self) -> None:
        """Wait for a local server's /health; a hosted API has none and is skipped."""
        if not self.base_url.startswith(("http://127.0.0.1", "http://localhost")):
            return
        url = self.base_url.removesuffix("/v1") + "/health"
        deadline, last = time.time() + self.health_timeout, ""
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    if 200 <= resp.status < 300:
                        return
                    last = f"HTTP {resp.status}"
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(3)
        raise RuntimeError(f"server not ready at {url} after {self.health_timeout}s ({last})")

    def run(self, trial: Trial) -> Response:
        started = time.time()
        ms = lambda: (time.time() - started) * 1000.0  # noqa: E731
        try:
            body = payload(trial, self.model, self.seed)
        except Exception as exc:  # noqa: BLE001
            return Response(error=f"payload: {type(exc).__name__}: {exc}", timing_ms=ms())
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return Response(error=f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:600]}",
                            timing_ms=ms())
        except Exception as exc:  # noqa: BLE001
            return Response(error=f"{type(exc).__name__}: {exc}", timing_ms=ms())
        choice = (out.get("choices") or [{}])[0]
        usage = dict(out.get("usage") or {})
        usage["finish_reason"] = choice.get("finish_reason")
        return Response(text=(choice.get("message") or {}).get("content") or "",
                        logprobs=first_token_logprobs(choice), usage=usage, timing_ms=ms())

    def teardown(self) -> None:
        pass


def _with_image(conv: Any, path: str) -> Any:
    """The conversation with every image part pointing at ``path``."""
    msgs = []
    for m in conv.messages:
        c = m["content"]
        if isinstance(c, list):
            c = [dict(p, image=path) if p.get("type") == "image" else p for p in c]
        msgs.append({**m, "content": c})
    return type(conv)(messages=msgs)


HF_LOADERS = {"qwen3-vl": "AutoModelForImageTextToText", "gemma4": "AutoModelForMultimodalLM"}
PROBE_CLASSES = {"headwise_linear": "HeadwiseLinearProbe", "layerwise_linear": "LayerwiseLinearProbe",
                 "layerwise_rfm": "LayerwiseRFM"}


class HFProbe:
    """Greedy generation in transformers, plus one probe's score at the last prompt token.

    The prompt is the chat template with the generation prompt appended, so its last
    token is the position whose output is the answer's first token. ``extract_features``
    from ``probes.base`` captures every attention head there in one forward pass, and
    the probe's ``score_samples`` averages its top ``top_k`` heads: positive reads
    conservative, negative liberal, as in ``scripts/probes/token_scoring.py``.

    One trial at a time (``workers=1``): the model holds the GPU.
    """

    name = "hf_probe"

    def __init__(self, model: str, model_path: str, model_family: str,
                 probe_prefix: str = "textual_ideology", probe_type: str = "headwise_linear",
                 top_k: int = 16, mode: str = "vision", data_dir: str = "results/probes",
                 seed: int = 42, dtype: str = "bfloat16", device_map: str = "auto",
                 candidates: Sequence[str] = (), rank_prefix: str = "combined_ideology", **_: Any) -> None:
        if model_family not in HF_LOADERS:
            raise ValueError(f"model_family must be one of {sorted(HF_LOADERS)}")
        if probe_type not in PROBE_CLASSES:
            raise ValueError(f"probe_type must be one of {sorted(PROBE_CLASSES)}")
        self.model = model
        self.model_path = model_path
        self.model_family = model_family
        self.probe_prefix = probe_prefix
        self.probe_type = probe_type
        self.top_k = int(top_k)
        self.mode = mode
        self.data_dir = data_dir
        self.seed = seed
        self.dtype = dtype
        self.device_map = device_map
        self.candidates = tuple(candidates)
        self.rank_prefix = rank_prefix   # the image-token probe a ranked splice orders cells by
        self._hf = None
        self._scorer = None

    def describe(self) -> Dict[str, Any]:
        out = {"name": self.name, "model": self.model, "model_path": self.model_path,
               "model_family": self.model_family, "probe": self.probe_id(), "top_k": self.top_k,
               "mode": self.mode, "seed": self.seed, "dtype": self.dtype}
        if self.candidates:
            out["candidates"] = list(self.candidates)
        return out

    def probe_id(self) -> str:
        return f"{self.probe_prefix}_{self.probe_type}"

    def setup(self) -> None:
        import torch
        import transformers
        import probes
        from probes.base import extract_features, resolve_module_paths

        torch.manual_seed(self.seed)
        loader = getattr(transformers, HF_LOADERS[self.model_family])
        processor = transformers.AutoProcessor.from_pretrained(self.model_path)
        # The same load as scripts/probes/token_scoring.py, which scored the patch maps.
        dtype = getattr(torch, self.dtype)
        if self.model_family == "qwen3-vl":
            kwargs = {"dtype": dtype, "low_cpu_mem_usage": True, "device_map": self.device_map}
        else:
            kwargs = {"torch_dtype": dtype, "device_map": self.device_map}
        net = loader.from_pretrained(self.model_path, **kwargs)
        net.eval()
        probe = getattr(probes, PROBE_CLASSES[self.probe_type])(
            model_path=self.model_path, prefix=self.probe_prefix, mode=self.mode,
            model_family=self.model_family, data_dir=self.data_dir).load()
        saved = (probe.metadata_.extra or {}).get("module_paths") if probe.metadata_ else None
        if isinstance(saved, dict):
            probe.module_paths = resolve_module_paths(self.model_family, saved)
        self._hf = {"torch": torch, "processor": processor, "net": net, "probe": probe,
                    "extract": extract_features}
        self._tap = None   # a bench_llm.splice.VisionTap, made on the first spliced trial

    def encode(self, trial: Trial) -> Dict[str, Any]:
        hf = self._hf
        encoded = hf["processor"].apply_chat_template(
            [trial.conversation.messages], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        device = hf["net"].device
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}

    def run(self, trial: Trial) -> Response:
        started = time.time()
        ms = lambda: (time.time() - started) * 1000.0  # noqa: E731
        if self._hf is None:
            return Response(error="setup() was not called", timing_ms=ms())
        hf, torch = self._hf, self._hf["torch"]
        stack = contextlib.ExitStack()
        try:
            encoded = self.encode(trial)
            spliced = self.splice(trial, encoded, stack) if trial.meta.get("splice") else None
            n_prompt = int(encoded["input_ids"].shape[-1])
            last_id = int(encoded["input_ids"][0, -1])
            feats = hf["extract"](hf["net"], [encoded], mode=self.mode, model_family=self.model_family,
                                  module_paths=hf["probe"].module_paths)
            score = float(hf["probe"].score_samples(feats, k=self.top_k)[0])
            with torch.no_grad():
                out = hf["net"].generate(**encoded, max_new_tokens=int(trial.max_new_tokens), do_sample=False)
            new = out[0, n_prompt:]
            text = hf["processor"].tokenizer.decode(new, skip_special_tokens=True)
            words = trial.meta.get("candidates") or self.candidates   # a trial may name its own
            cands = {c: self.continuation_logprob(encoded, c) for c in words}
        except Exception as exc:  # noqa: BLE001
            stack.close()
            return Response(error=f"{type(exc).__name__}: {exc}", timing_ms=ms())
        stack.close()
        tok = hf["processor"].tokenizer
        probe = {"probe": self.probe_id(), "top_k": self.top_k, "position": "last_prompt_token",
                 "last_prompt_token": tok.decode([last_id]), "score": score}
        if spliced:
            probe["splice"] = spliced
        if cands:
            probe["candidates"] = cands
        return Response(
            text=text, timing_ms=ms(),
            usage={"prompt_tokens": n_prompt, "completion_tokens": int(new.numel())}, probe=probe)

    def splice(self, trial: Trial, encoded: Dict[str, Any], stack: contextlib.ExitStack) -> Dict[str, Any]:
        """Swap image tokens for the rest of this trial, as ``trial.meta["splice"]`` says:
        ``{"mode": "swap" | "fill_check" | "normal", "donor_image": path, "base_mask": path,
        "donor_mask": path, "aspect": height / width}``, and for ``"ranked"`` also ``share``, ``how``
        (top | random), ``direction`` (+1: the donor's most conservative tokens; -1: most liberal) and
        ``seed``: that share of the cells that are person in both photos moves from the donor, at the
        same positions. The trial's own image is the base;
        the donor is encoded with the same prompt. See ``bench_llm/splice.py``."""
        import numpy as np
        from PIL import Image
        from bench_llm import splice as sp
        spec = trial.meta["splice"]
        if self._tap is None:
            self._tap = sp.VisionTap(self._hf["net"], self.model_family)
        base = self._tap.capture(encoded)
        donor_trial = Trial(item_id=trial.item_id, conversation=_with_image(trial.conversation, spec["donor_image"]),
                            variant=trial.variant, max_new_tokens=trial.max_new_tokens, meta={})
        donor_encoded = self.encode(donor_trial)
        donor = self._tap.capture(donor_encoded)
        h, w = sp.infer_grid(base.n, float(spec["aspect"]))
        bp = sp.person_cells(np.asarray(Image.open(spec["base_mask"])) > 127, h, w)
        dp = sp.person_cells(np.asarray(Image.open(spec["donor_mask"])) > 127, h, w)
        extra: Dict[str, Any] = {}
        if spec["mode"] == "ranked":
            scores = self.image_token_scores(donor_encoded)
            if len(scores) != base.n:
                raise RuntimeError(f"{len(scores)} scored image tokens, the encoder gave {base.n}")
            eligible = [a and b for a, b in zip(bp, dp)]
            cells = sp.ranked_cells(scores, eligible, float(spec["share"]), spec.get("how", "top"),
                                    int(spec.get("direction", 1)), int(spec.get("seed", 0)))
            plan = sp.ranked_plan(base.n, cells)
            extra = {"share": spec["share"], "how": spec.get("how", "top"), "direction": int(spec.get("direction", 1)),
                     "eligible": sum(eligible), "cells": cells, "rank_probe": self._scorer.probe_id,
                     "moved_score_mean": float(scores[cells].mean()) if cells else None,
                     "donor_eligible_score_mean": float(scores[[i for i, e in enumerate(eligible) if e]].mean()) if any(eligible) else None}
        else:
            plan = sp.splice_plan(bp, dp, spec["mode"])
        stack.enter_context(self._tap.override(sp.apply_plan(base, donor, plan, bp, dp)))
        return {"mode": spec["mode"], "grid": [h, w], "sources": sp.plan_counts(plan),
                "deepstack_layers": len(base.deepstack), **extra}

    def image_token_scores(self, encoded: Dict[str, Any]):
        """The donor's own per-image-token probe scores (``bench_llm/token_scores.py``), with the
        ranking probe of the token scoring: combined_ideology_headwise_linear, top 16 heads."""
        if getattr(self, "_scorer", None) is None:
            from bench_llm.token_scores import ImageTokenScorer
            self._scorer = ImageTokenScorer(self._hf["net"], self._hf["processor"], self.model_path,
                                            self.model_family, prefix=self.rank_prefix, top_k=self.top_k,
                                            mode=self.mode, data_dir=self.data_dir)
        return self._scorer.image_scores(encoded)

    def continuation_logprob(self, encoded: Dict[str, Any], word: str) -> Dict[str, Any]:
        """log P(word | prompt): the word's tokens appended to the prompt, scored by teacher forcing,
        their log-probabilities summed, so a word split into pieces counts in full."""
        hf, torch = self._hf, self._hf["torch"]
        ids = hf["processor"].tokenizer.encode(word, add_special_tokens=False)
        n = int(encoded["input_ids"].shape[-1])
        extra = torch.tensor([ids], device=encoded["input_ids"].device, dtype=encoded["input_ids"].dtype)
        longer = dict(encoded)
        for key, value in encoded.items():
            # every per-token tensor ([1, n]) grows with the word: ids get the word, masks get ones,
            # token-type ids get zeros (text)
            if hasattr(value, "shape") and value.dim() == 2 and int(value.shape[0]) == 1 and int(value.shape[-1]) == n:
                if key == "input_ids":
                    fill = extra
                elif key == "attention_mask":
                    fill = torch.ones_like(extra, dtype=value.dtype)
                else:
                    fill = torch.zeros_like(extra, dtype=value.dtype)
                longer[key] = torch.cat([value, fill], dim=-1)
        with torch.no_grad():
            logits = hf["net"](**longer).logits[0, n - 1:n - 1 + len(ids)].float()
        lp = torch.log_softmax(logits, dim=-1)
        per = [float(lp[i, t]) for i, t in enumerate(ids)]
        return {"logprob": sum(per), "tokens": [hf["processor"].tokenizer.decode([t]) for t in ids],
                "per_token": per}

    def teardown(self) -> None:
        if self._hf is not None:
            torch = self._hf["torch"]
            self._hf = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


ADAPTORS = {"openai_compat": OpenAICompat, "vllm": OpenAICompat, "hf_probe": HFProbe}
