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
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List

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
                 seed: int = 42, dtype: str = "bfloat16", device_map: str = "auto", **_: Any) -> None:
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
        self._hf = None

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "model": self.model, "model_path": self.model_path,
                "model_family": self.model_family, "probe": self.probe_id(), "top_k": self.top_k,
                "mode": self.mode, "seed": self.seed, "dtype": self.dtype}

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
        try:
            encoded = self.encode(trial)
            n_prompt = int(encoded["input_ids"].shape[-1])
            last_id = int(encoded["input_ids"][0, -1])
            feats = hf["extract"](hf["net"], [encoded], mode=self.mode, model_family=self.model_family,
                                  module_paths=hf["probe"].module_paths)
            score = float(hf["probe"].score_samples(feats, k=self.top_k)[0])
            with torch.no_grad():
                out = hf["net"].generate(**encoded, max_new_tokens=int(trial.max_new_tokens), do_sample=False)
            new = out[0, n_prompt:]
            text = hf["processor"].tokenizer.decode(new, skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001
            return Response(error=f"{type(exc).__name__}: {exc}", timing_ms=ms())
        tok = hf["processor"].tokenizer
        return Response(
            text=text, timing_ms=ms(),
            usage={"prompt_tokens": n_prompt, "completion_tokens": int(new.numel())},
            probe={"probe": self.probe_id(), "top_k": self.top_k, "position": "last_prompt_token",
                   "last_prompt_token": tok.decode([last_id]), "score": score})

    def teardown(self) -> None:
        if self._hf is not None:
            torch = self._hf["torch"]
            self._hf = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


ADAPTORS = {"openai_compat": OpenAICompat, "vllm": OpenAICompat, "hf_probe": HFProbe}
