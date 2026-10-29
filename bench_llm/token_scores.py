"""Per-image-token probe scores, computed exactly as ``scripts/probes/token_scoring.py`` does.

The token scoring script reads a probe at every prompt token in one forward pass; this module
reuses its own functions (loaded from the script, not copied) so a score computed during a
bench_llm trial is the same number the scoring run wrote to disk for the same prompt.

    scorer = ImageTokenScorer(net, processor, model_path, "qwen3-vl")   # combined_ideology, top 16
    scores = scorer.image_scores(encoded)    # numpy [n_image_tokens], raster order, + conservative

The image tokens come first in every bench_llm portrait prompt, so with a causal model their
scores do not depend on the question that follows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probes" / "token_scoring.py"
_TS = None


def token_scoring():
    """The token scoring script, imported once as a module."""
    global _TS
    if _TS is None:
        spec = importlib.util.spec_from_file_location("token_scoring", SCRIPT)
        _TS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_TS)
    return _TS


class ImageTokenScorer:
    def __init__(self, net: Any, processor: Any, model_path: str, model_family: str,
                 prefix: str = "combined_ideology", probe_type: str = "headwise_linear", top_k: int = 16,
                 mode: str = "vision", data_dir: str = "results/probes") -> None:
        ts = token_scoring()
        self.net, self.processor, self.ts = net, processor, ts
        self.probe_id = f"{prefix}_{probe_type}"
        probe = ts.PROBE_CLASSES[probe_type](model_path=model_path, prefix=prefix, mode=mode,
                                             model_family=model_family, data_dir=data_dir).load()
        saved = (probe.metadata_.extra or {}).get("module_paths") if probe.metadata_ else None
        if isinstance(saved, dict):
            probe.module_paths = ts.resolve_module_paths(model_family, saved)
        self.runtime = ts.build_probe_runtime(net, probe, top_k=top_k, mode=mode, model_family=model_family)
        self.image_ids = sorted(ts.gather_candidate_image_token_ids(processor.tokenizer))

    def image_scores(self, encoded: Dict[str, Any]):
        import numpy as np
        captured = self.ts.capture_module_outputs(self.net, encoded, sorted(self.runtime["module_names"]))
        scores = self.ts.score_from_captured(captured, self.runtime)[0].cpu().numpy()
        ids = encoded["input_ids"][0].cpu().numpy()
        return scores[np.isin(ids, self.image_ids)].astype(np.float32)
