"""Local white-box backend.

Everything expensive here is imported from ``scripts/probes/token_scoring.py``
rather than rewritten: model selection (``select_model_loader``), image prep
(``prepare_image_for_scoring``), chat encoding (``encode_prompts``), the module
hooks (``capture_module_outputs``), and probe scoring (``build_probe_runtime`` /
``score_from_captured``).

Per docs/bench/02 the only genuinely new pieces are "read at a given position"
and "take the next-token logits". The latter is done by hanging one extra
forward hook on ``lm_head`` *around* ``capture_module_outputs``, so the whole
trial is still a single prefill.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from bench.adaptors.base import BaseAdaptor
from bench.registry import register_adaptor
from bench.types import Capability, Response, Trial

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOKEN_SCORING_PATH = os.path.join(ROOT_DIR, "scripts", "probes", "token_scoring.py")

DEFAULT_MODEL_PATH = "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
DEFAULT_PROBE = "combined_ideology_headwise_linear"
KNOWN_PROBE_TYPES = ("headwise_linear", "layerwise_linear", "layerwise_rfm")

_TS = None


def token_scoring():
    """Import scripts/probes/token_scoring.py by path (scripts/ is not a package)."""
    global _TS
    if _TS is not None:
        return _TS
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    spec = importlib.util.spec_from_file_location("bench_token_scoring", TOKEN_SCORING_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {TOKEN_SCORING_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_token_scoring"] = module
    spec.loader.exec_module(module)
    _TS = module
    return _TS


def split_probe_id(probe_id: str) -> Tuple[str, str]:
    """'combined_ideology_headwise_linear' -> ('combined_ideology', 'headwise_linear')."""
    for probe_type in KNOWN_PROBE_TYPES:
        suffix = "_" + probe_type
        if probe_id.endswith(suffix):
            return probe_id[: -len(suffix)], probe_type
    raise ValueError(
        f"Cannot split probe id {probe_id!r}; expected it to end with one of {KNOWN_PROBE_TYPES}"
    )


@register_adaptor("local_hf")
class LocalHFAdaptor(BaseAdaptor):
    name = "local_hf"
    capabilities = frozenset({
        Capability.GENERATE,
        Capability.LOGPROB,
        Capability.ACTIVATIONS,
        Capability.IMAGES,
    })

    def __init__(
        self,
        model: str = DEFAULT_MODEL_PATH,
        model_family: str = "qwen3-vl",
        mode: str = "vision",
        probe: str = DEFAULT_PROBE,
        top_k: int = 16,
        data_dir: str = "results/probes",
        dtype: str = "auto",
        device_map: str = "auto",
        seed: int = 42,
        resized_cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, seed=seed, **kwargs)
        self.model_path = model
        self.model_family = model_family
        self.mode = mode
        self.probe_id = probe
        self.top_k = top_k
        self.data_dir = data_dir
        self.dtype = dtype
        self.device_map = device_map
        # bench-owned cache. Deliberately NOT the LVIS _resized_images_800 directory:
        # prepare_image_for_scoring keys on abs_path+mtime, so pointing it there would
        # drop extra files into an existing results/ artifact.
        self.resized_cache_dir = resized_cache_dir or os.path.join(
            ROOT_DIR, "items", "_image_cache"
        )
        self._ready = False
        self.processor = None
        self.hf_model = None   # the nn.Module; self.model stays the model *id* string
        self.runtime = None
        self.module_names: List[str] = []
        self.image_token_ids: set = set()
        self._lm_head = None

    # -- describe ------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base.update({
            "model_path": self.model_path,
            "model_family": self.model_family,
            "mode": self.mode,
            "probe": self.probe_id,
            "top_k": self.top_k,
            "dtype": self.dtype,
            "device_map": self.device_map,
            "probe_weights": self.probe_weights_path(),
        })
        return base

    # -- setup ---------------------------------------------------------------
    def setup(self) -> None:
        if self._ready:
            return
        import torch
        from transformers import AutoProcessor

        ts = token_scoring()
        prefix, probe_type = split_probe_id(self.probe_id)

        print(f"[local_hf] loading processor+model: {self.model_path}", flush=True)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        model_cls = ts.select_model_loader(self.model_family, self.model_path)
        default_dtype = ts.resolve_torch_dtype(self.dtype)

        # Same load kwargs token_scoring.main() uses per family.
        if self.model_family == "qwen3-vl":
            load_kwargs = {"dtype": default_dtype, "low_cpu_mem_usage": True, "device_map": self.device_map}
        else:
            load_kwargs = {"torch_dtype": default_dtype, "device_map": self.device_map}

        self.hf_model = model_cls.from_pretrained(self.model_path, **load_kwargs)
        self.hf_model.eval()

        probe_cls = ts.PROBE_CLASSES[probe_type]
        probe = probe_cls(
            model_path=self.model_path,
            prefix=prefix,
            mode=self.mode,
            model_family=self.model_family,
            data_dir=self.data_dir,
        ).load()
        if probe.metadata_ is not None:
            meta_paths = probe.metadata_.extra.get("module_paths")
            if isinstance(meta_paths, dict):
                from probes.base import resolve_module_paths
                probe.module_paths = resolve_module_paths(self.model_family, meta_paths)

        self.runtime = ts.build_probe_runtime(
            model=self.hf_model, probe=probe, top_k=self.top_k,
            mode=self.mode, model_family=self.model_family,
        )
        self.module_names = sorted(self.runtime["module_names"])
        self.image_token_ids = ts.gather_candidate_image_token_ids(self.processor.tokenizer)
        self._lm_head = _find_lm_head(self.hf_model)

        torch.manual_seed(self.seed)
        print(f"[local_hf] probe={self.probe_id} modules={len(self.module_names)} "
              f"image_token_ids={sorted(self.image_token_ids)}", flush=True)
        self._ready = True

    def teardown(self) -> None:
        self.hf_model = None
        self.processor = None
        self._ready = False

    # -- tokenizer access (used by the candidate gate) -----------------------
    def tokenize(self, text: str) -> List[int]:
        """Token ids for a literal string. Loads the processor only, not the model."""
        if self.processor is None:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        return [int(i) for i in self.processor.tokenizer.encode(text, add_special_tokens=False)]

    def probe_weights_path(self) -> str:
        return os.path.join(
            self.data_dir, os.path.basename(self.model_path).lower(),
            f"{self.probe_id}_weights.pkl",
        )

    # -- run -----------------------------------------------------------------
    def run(self, trial: Trial) -> Response:
        import numpy as np
        import torch

        if not self._ready:
            self.setup()
        ts = token_scoring()
        started = time.time()

        messages = ts.resolve_messages_images(
            trial.conversation.messages, image_root=None, cache_dir=self.resized_cache_dir,
        )
        if messages is None:
            return Response(error="image preparation failed", timing_ms=(time.time() - started) * 1000.0)

        encoded = ts.encode_prompts(self.processor, [messages])

        # The only new piece: grab next-token logits from the same forward pass.
        logits_box: Dict[str, Any] = {}

        def lm_hook(_module, _inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            logits_box["last"] = tensor[0, -1, :].detach().to(dtype=torch.float32).cpu()

        handle = self._lm_head.register_forward_hook(lm_hook)
        try:
            captured = ts.capture_module_outputs(
                model=self.hf_model, encoded=encoded, module_names=self.module_names,
            )
        finally:
            handle.remove()

        token_scores = ts.score_from_captured(captured, self.runtime)[0].cpu().numpy()
        input_ids = encoded["input_ids"][0].cpu().numpy()

        image_mask = np.isin(input_ids, np.asarray(sorted(self.image_token_ids), dtype=np.int64)) \
            if self.image_token_ids else np.zeros_like(input_ids, dtype=bool)
        n_image_tokens = int(image_mask.sum())

        # "last text position" == the final prefill position, i.e. the assistant
        # prefix the model is about to continue from. Batch size is 1, so there
        # is no padding to step over.
        last_idx = len(input_ids) - 1
        probe: Dict[str, Any] = {
            "probe_id": self.probe_id,
            "s_txt": float(token_scores[last_idx]),
            "s_img": float(token_scores[image_mask].mean()) if n_image_tokens else None,
            "s_obj": None,
            "n_image_tokens": n_image_tokens,
            "n_all_tokens": int(len(input_ids)),
            "last_text_token": self.processor.tokenizer.decode([int(input_ids[last_idx])]),
        }

        logprobs: Optional[Dict[str, float]] = None
        cand_meta: Dict[str, Any] = {}
        if trial.candidates:
            last_logits = logits_box.get("last")
            if last_logits is None:
                raise RuntimeError("lm_head hook produced no logits; cannot score candidates")
            logprob_vec = torch.log_softmax(last_logits, dim=-1)
            logprobs = {}
            token_ids: Dict[str, int] = {}
            multi_token: List[str] = []
            for candidate in trial.candidates:
                ids = self.processor.tokenizer.encode(candidate, add_special_tokens=False)
                if not ids:
                    raise ValueError(f"Candidate {candidate!r} tokenized to nothing")
                if len(ids) > 1:
                    multi_token.append(candidate)
                token_ids[candidate] = int(ids[0])
                logprobs[candidate] = float(logprob_vec[ids[0]].item())
            if len(set(token_ids.values())) != len(token_ids):
                raise ValueError(f"Candidates share a first token: {token_ids}")
            top_id = int(torch.argmax(logprob_vec).item())
            top_lp, top_ids = torch.topk(logprob_vec, k=5)
            cand_meta = {
                "candidate_first_token_ids": token_ids,
                "candidates_multi_token": multi_token,
                "argmax_token": self.processor.tokenizer.decode([top_id]),
                "argmax_logprob": float(logprob_vec[top_id].item()),
                "top_tokens": [[self.processor.tokenizer.decode([int(i)]), float(v)]
                               for v, i in zip(top_lp.tolist(), top_ids.tolist())],
            }

        text = None
        if trial.max_new_tokens > 0:
            with torch.no_grad():
                out = self.hf_model.generate(
                    **ts.move_to_device(encoded, self.hf_model),
                    max_new_tokens=trial.max_new_tokens,
                    do_sample=False,
                )
            new_tokens = out[0][encoded["input_ids"].shape[1]:]
            text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return Response(
            text=text,
            logprobs=logprobs,
            probe=probe,
            usage={"prefill_tokens": int(len(input_ids)),
                   "image_tokens": n_image_tokens,
                   **cand_meta},
            timing_ms=(time.time() - started) * 1000.0,
            cost_usd=0.0,
        )


def _find_lm_head(model):
    for attr_path in ("lm_head", "language_model.lm_head", "model.lm_head", "model.language_model.lm_head"):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj
    for name, module in reversed(list(model.named_modules())):
        if name.endswith("lm_head"):
            return module
    raise RuntimeError("Could not locate lm_head on this model; next-token logits unavailable")
