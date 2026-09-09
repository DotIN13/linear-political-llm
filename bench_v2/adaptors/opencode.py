"""Black-box backend: opencode via the agent-bridge `ab` CLI.

It is text in, text out. It explicitly does NOT declare `logprob` or
`activations`, which is what makes ``bench check --surface vote2020 --adaptor
opencode`` report a degraded run instead of quietly producing a different
number under the same column name (docs/bench/01).

Its real jobs are judging, stimulus generation, a black-box replication arm and
agentic surfaces -- not the primary measurement.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from bench_v2.adaptors.base import BaseAdaptor
from bench_v2.registry import register_adaptor
from bench_v2.types import Capability, Response, Trial


@register_adaptor("opencode")
class OpenCodeAdaptor(BaseAdaptor):
    name = "opencode"
    # No LOGPROB. No ACTIVATIONS. That absence is load-bearing.
    capabilities = frozenset({
        Capability.GENERATE,
        Capability.IMAGES,
        Capability.SESSION,
    })

    def __init__(
        self,
        model: str = "deepseek/deepseek-v4-flash",
        agent: str = "opencode",
        gateway: Optional[str] = None,
        ab_binary: str = "ab",
        timeout: int = 300,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, seed=seed, **kwargs)
        self.agent = agent
        self.gateway = gateway
        self.ab_binary = shutil.which(ab_binary) or ab_binary
        self.timeout = timeout

    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base.update({"agent": self.agent, "gateway": self.gateway,
                     "ab_binary": self.ab_binary, "timeout": self.timeout})
        return base

    def setup(self) -> None:
        if not os.path.exists(self.ab_binary):
            raise RuntimeError(
                f"agent-bridge CLI not found at {self.ab_binary!r}; opencode adaptor unavailable"
            )

    def _command(self, trial: Trial) -> List[str]:
        cmd = [self.ab_binary, "run", "--json", "--agent", self.agent,
               "--model", self.model, "--timeout", str(self.timeout), "--prompt-stdin"]
        if self.gateway:
            cmd[1:1] = ["--gateway", self.gateway]
        for image in trial.conversation.images:
            cmd += ["--upload", image]
        return cmd

    def run(self, trial: Trial) -> Response:
        started = time.time()
        prompt = self._render_prompt(trial)
        cmd = self._command(trial)
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                timeout=self.timeout + 60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Response(error=f"{type(exc).__name__}: {exc}",
                            timing_ms=(time.time() - started) * 1000.0)

        if proc.returncode != 0:
            return Response(error=f"ab exited {proc.returncode}: {proc.stderr[-500:]}",
                            timing_ms=(time.time() - started) * 1000.0)

        text, payload = _parse_ab_json(proc.stdout)
        return Response(
            text=text,
            logprobs=None,        # capability absent by construction
            probe=None,           # capability absent by construction
            session_log=[{"session_id": payload.get("session_id"),
                          "job_id": payload.get("job_id")}],
            usage=payload.get("usage", {}),
            timing_ms=(time.time() - started) * 1000.0,
            cost_usd=float(payload.get("cost_usd") or 0.0),
        )

    @staticmethod
    def _render_prompt(trial: Trial) -> str:
        """Flatten the same conversation the local backend prefills.

        The prior assistant turns are handed over as transcript text so the two
        arms see the same words even though only one of them can be prefilled.
        """
        header = (
            "Continue this conversation. Reply as ASSISTANT to the final USER turn, "
            "in a single word where the question asks for one.\n\n"
        )
        return header + trial.conversation.render_text() + "\nASSISTANT:"


def _parse_ab_json(stdout: str) -> tuple:
    payload: Dict[str, Any] = {}
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout.strip(), {}
    for key in ("text", "result", "output", "assistant_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), payload
    return stdout.strip(), payload
