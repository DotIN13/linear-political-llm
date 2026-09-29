"""Model backends. One for now: any OpenAI-compatible chat endpoint.

That covers a local vLLM server (the sbatch starts one) and hosted APIs alike.
Decoding is greedy (``temperature=0``) and, when ``top_logprobs`` is asked for,
the first generated token's top tokens come back as ``{token: logprob}``.
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


ADAPTORS = {"openai_compat": OpenAICompat, "vllm": OpenAICompat}
