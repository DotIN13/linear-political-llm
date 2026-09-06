"""Fast black-box backend: an OpenAI-compatible vLLM server (apptainer/docker).

Why this exists: the HF path (``local_hf``) is the only backend that can read
per-head activations, and that is what ``s_pre``/``s_gen`` are made of -- but it
runs one trial at a time and copies every hooked module's output to the CPU at
every decode step, which measured out at ~19 s per generation. The full design
is ~7,900 generations, i.e. ~42 GPU-hours, i.e. eighty-odd serial 28-minute
jobs. vLLM batches and does none of the hook copying.

The trade is stated in the capability set, not hidden: **no ACTIVATIONS**. So
``s_pre`` / ``s_gen`` / ``s_img`` come back ``None`` and the gate reports
DEGRADED (``GenerationSurface`` lists activations under ``prefers``). What
survives is everything the behavioural claims are made of:

* the generated text, hence every deterministic extractor and every judge field
* ``image_mean``, the independent variable -- it is precomputed in
  ``results/token_scoring/.../prompt_token_stats_*.csv``, not measured at
  inference time, so the dose-response is fully available here
* ``logprobs``, which the server does expose

So: vLLM for breadth (six surfaces, large n, behavioural outcomes), ``local_hf``
for depth (probe readings on a subsample of the same ``item_id``s). ``adaptor``
is part of ``trial_key``, so the two coexist without colliding.

**One honest caveat, in the transcript rather than the numbers.** The chat
scheme maps onto the OpenAI schema exactly: user turns with image parts. The
agentic scheme does not -- the OpenAI schema forbids image content in a
``tool`` message, so the tool turns are *folded into user turns* wrapped in the
literal ``<tool_response>`` markers. That is the shape Qwen's own template
produces anyway (docs/bench/08: ``role:"tool"`` renders as a user turn wrapped
in ``<tool_response>``, there is no tool-role token), but the folding is done by
hand here rather than by ``apply_chat_template``, so it is **not guaranteed
byte-identical** to the ``local_hf`` agentic transcript. Records carry
``transcript_shape`` so agentic vLLM and agentic HF are never pooled without
someone deciding to.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from bench.adaptors.base import BaseAdaptor
from bench.registry import register_adaptor
from bench.types import Capability, Response, Trial

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

# Qwen's own rendering of a tool turn, from the docs/bench/08 pilot. Reproduced
# here because the OpenAI schema cannot carry an image inside a tool message.
TOOL_OPEN = "<tool_response>\n"
TOOL_CLOSE = "\n</tool_response>"


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested without a server)
# --------------------------------------------------------------------------- #
def data_uri(path: str) -> str:
    """A local image file as an inline ``data:`` URI, which is what the server takes."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _part_to_openai(part: Dict[str, Any], image_loader) -> Dict[str, Any]:
    kind = part.get("type")
    if kind == "text":
        return {"type": "text", "text": part.get("text", "")}
    if kind == "image":
        return {"type": "image_url", "image_url": {"url": image_loader(part["image"])}}
    raise ValueError(f"unsupported content part type: {kind!r}")


def _tool_call_text(message: Dict[str, Any]) -> str:
    """An assistant tool-call turn as the literal text Qwen's template emits."""
    calls = message.get("tool_calls") or []
    out = []
    for call in calls:
        fn = call.get("function", {})
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        out.append('<tool_call>{"name":"%s","arguments":%s}</tool_call>' % (fn.get("name", ""), args))
    return "".join(out)


def to_openai_messages(messages: Sequence[Dict[str, Any]],
                       image_loader=data_uri) -> List[Dict[str, Any]]:
    """Bench message shape -> OpenAI chat shape, folding tool turns into user turns.

    ``role:"tool"`` becomes a user message wrapped in ``<tool_response>``, because
    the OpenAI schema has no way to put an image in a tool message. Consecutive
    folded turns are *not* merged: keeping them separate keeps the turn count the
    same as the HF transcript.
    """
    out: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]

        if role == "assistant" and message.get("tool_calls"):
            out.append({"role": "assistant", "content": _tool_call_text(message)})
            continue

        if role == "tool":
            folded: List[Dict[str, Any]] = [{"type": "text", "text": TOOL_OPEN}]
            folded.extend(_part_to_openai(p, image_loader) for p in parts)
            folded.append({"type": "text", "text": TOOL_CLOSE})
            out.append({"role": "user", "content": folded})
            continue

        converted = [_part_to_openai(p, image_loader) for p in parts]
        # A pure-text turn is sent as a plain string: some servers are stricter
        # about list content on system/assistant turns than on user turns
        # (docs/bench/08 hit exactly this on Gemma's system turn).
        if all(p["type"] == "text" for p in converted):
            out.append({"role": role, "content": "".join(p["text"] for p in converted)})
        else:
            out.append({"role": role, "content": converted})
    return out


def build_payload(trial: Trial, model: str, *, seed: int,
                  logprobs: int = 0, image_loader=data_uri) -> Dict[str, Any]:
    """The request body. Greedy by construction: ``temperature=0``."""
    messages = to_openai_messages(trial.conversation.messages, image_loader=image_loader)
    prefill = (trial.meta or {}).get("prefill_text")
    if prefill:
        # The prefill is a partial assistant turn the model must continue. vLLM
        # honours this via continue_final_message; without it the server would
        # start a fresh turn and the prefill would just be context.
        messages.append({"role": "assistant", "content": prefill})
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(trial.max_new_tokens or 1),
        "temperature": 0.0,
        "seed": int(seed),
    }
    if prefill:
        payload["continue_final_message"] = True
        payload["add_generation_prompt"] = False
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(logprobs)
    return payload


# --------------------------------------------------------------------------- #
@register_adaptor("vllm")
class VLLMServerAdaptor(BaseAdaptor):
    name = "vllm"
    # No ACTIVATIONS. That absence is the whole trade and it is declared, so
    # `bench check` reports DEGRADED instead of quietly dropping s_gen.
    capabilities = frozenset({
        Capability.GENERATE,
        Capability.LOGPROB,
        Capability.IMAGES,
    })

    def __init__(
        self,
        model: str = "qwen3-vl-8b-instruct",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        timeout: int = 600,
        health_timeout: int = 900,
        top_logprobs: int = 0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, seed=seed, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("VLLM_API_KEY", api_key)
        self.timeout = timeout
        self.health_timeout = health_timeout
        self.top_logprobs = top_logprobs

    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base.update({"base_url": self.base_url, "top_logprobs": self.top_logprobs,
                     "transcript_shape": "openai_folded"})
        return base

    # -- server handshake ---------------------------------------------------
    def _health_url(self) -> str:
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        return root + "/health"

    def setup(self) -> None:
        """Block until the server answers /health, or raise with what it said.

        The server is started by the sbatch script, not here: loading weights
        takes minutes and belongs to the job, not to the first trial.
        """
        deadline = time.time() + self.health_timeout
        last = ""
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(self._health_url(), timeout=10) as resp:
                    if 200 <= resp.status < 300:
                        return
                    last = f"HTTP {resp.status}"
            except Exception as exc:                      # noqa: BLE001 - report it
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(3)
        raise RuntimeError(f"vLLM server not ready at {self._health_url()} "
                           f"after {self.health_timeout}s (last: {last})")

    # -- one trial ----------------------------------------------------------
    def run(self, trial: Trial) -> Response:
        started = time.time()
        try:
            payload = build_payload(trial, self.model, seed=self.seed,
                                    logprobs=self.top_logprobs)
        except Exception as exc:                          # noqa: BLE001
            return Response(error=f"payload build failed: {type(exc).__name__}: {exc}",
                            timing_ms=(time.time() - started) * 1000.0)

        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:600]
            return Response(error=f"HTTP {exc.code}: {detail}",
                            timing_ms=(time.time() - started) * 1000.0)
        except Exception as exc:                          # noqa: BLE001
            return Response(error=f"{type(exc).__name__}: {exc}",
                            timing_ms=(time.time() - started) * 1000.0)

        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = dict(body.get("usage") or {})
        usage["finish_reason"] = choice.get("finish_reason")
        usage["transcript_shape"] = "openai_folded"
        return Response(
            text=text,
            logprobs=self._flatten_logprobs(choice),
            probe=None,                        # no ACTIVATIONS: this is the trade
            usage=usage,
            timing_ms=(time.time() - started) * 1000.0,
        )

    @staticmethod
    def _flatten_logprobs(choice: Dict[str, Any]) -> Optional[Dict[str, float]]:
        content = ((choice.get("logprobs") or {}).get("content") or [])
        if not content:
            return None
        first = content[0]
        top = first.get("top_logprobs") or []
        return {str(entry.get("token")): float(entry.get("logprob"))
                for entry in top if entry.get("token") is not None} or None
