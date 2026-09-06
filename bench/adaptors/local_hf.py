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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

        # Load on a single device directly. ``device_map="auto"`` (accelerate
        # dispatch) wraps every module in device-transfer hooks that make
        # autoregressive generation ~20x slower here (4 tok/s vs 83 tok/s); a
        # plain ``.to(device)`` is fast. The 8B/15B models fit on one H200.
        if self.model_family == "qwen3-vl":
            load_kwargs = {"dtype": default_dtype}
        else:
            load_kwargs = {"torch_dtype": default_dtype}

        self.hf_model = model_cls.from_pretrained(self.model_path, **load_kwargs)
        target = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_model = self.hf_model.to(target)
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
        self.probe = probe                    # kept so generation can build k=8 too
        self.probe_type = probe_type
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
    def _is_generation(self, trial: Trial) -> bool:
        return bool(trial.max_new_tokens > 0 and any(
            p.kind in ("prefix_end", "generated_tokens") for p in trial.probe_points))

    def _encode(self, messages: List[Dict[str, Any]], tools: Optional[List] = None,
                add_generation_prompt: bool = True) -> Dict[str, Any]:
        """token_scoring.encode_prompts + a ``tools=`` entry and gen-prompt control.

        ``encode_prompts`` has no ``tools=`` parameter (docs/bench/08), and the
        agentic scheme needs it. ``add_generation_prompt=False`` is used to find
        the end of the shared prefix for ``s_pre``.
        """
        import torch

        kwargs: Dict[str, Any] = {}
        if tools is not None:
            kwargs["tools"] = tools
        encoded = self.processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            processor_kwargs={"padding": True},
            return_dict=True,
            return_tensors="pt",
            **kwargs,
        )
        return {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in encoded.items()}

    def _generation_runtimes(self, ks: Sequence[int]) -> Dict[int, Any]:
        ts = token_scoring()
        return {int(k): ts.build_probe_runtime(
            model=self.hf_model, probe=self.probe, top_k=int(k),
            mode=self.mode, model_family=self.model_family,
        ) for k in ks}

    def _run_generation(self, trial: Trial) -> Response:
        """Generate + read s_pre (prefix end), s_img (image tokens), s_gen (each
        generated token). Reads both k=top_k (primary) and k=8 (robustness)."""
        import numpy as np
        import torch

        if not self._ready:
            self.setup()
        ts = token_scoring()
        started = time.time()

        meta = trial.meta or {}
        tools = meta.get("tools")
        prefix_n = int(meta.get("prefix_n_messages", len(trial.conversation.messages) - 1))

        messages = ts.resolve_messages_images(
            trial.conversation.messages, image_root=None, cache_dir=self.resized_cache_dir)
        if messages is None:
            return Response(error="image preparation failed",
                            timing_ms=(time.time() - started) * 1000.0)

        prefix = messages[:prefix_n]
        full = self._encode(messages, tools=tools, add_generation_prompt=True)
        pref = self._encode(prefix, tools=tools, add_generation_prompt=False)
        k = int(pref["input_ids"][0].numel()) - 1          # end of the shared prefix

        # R1 prefill (docs/bench/12): append the prefill tokens after the
        # generation prompt, so they are *input* (excluded from s_gen below), not
        # generated. Qwen3-VL's M-RoPE requires mm_token_type_ids to stay the same
        # length as input_ids, so it is padded with zeros alongside input_ids.
        prefill = meta.get("prefill")
        if prefill:
            pids = self.processor.tokenizer.encode(prefill, add_special_tokens=False)
            full["input_ids"] = torch.cat(
                [full["input_ids"], torch.tensor([pids], dtype=full["input_ids"].dtype)], dim=1
            )
            if "attention_mask" in full:
                full["attention_mask"] = torch.cat(
                    [full["attention_mask"],
                     torch.ones((1, len(pids)), dtype=full["attention_mask"].dtype)], dim=1
                )
            if "mm_token_type_ids" in full:
                full["mm_token_type_ids"] = torch.cat(
                    [full["mm_token_type_ids"],
                     torch.zeros((1, len(pids)), dtype=full["mm_token_type_ids"].dtype)], dim=1
                )

        runtimes = self._generation_runtimes((self.top_k, 8))
        union = sorted({n for rt in runtimes.values() for n in rt["module_names"]})
        named_modules = dict(self.hf_model.named_modules())
        missing = [n for n in union if n not in named_modules]
        if missing:
            return Response(error=f"missing probe modules: {missing[:3]}",
                            timing_ms=(time.time() - started) * 1000.0)

        # hook every probe module for the whole generate() call: the first fire is
        # the prefill (seq_len = full), each later fire is one decode step
        # (seq_len = 1 with KV cache). Scoring always reads the last position, so
        # it is correct either way.
        logs: Dict[str, List[torch.Tensor]] = {n: [] for n in union}

        def make_hook(name: str):
            def hook_fn(_m, _i, out):
                tensor = out[0] if isinstance(out, tuple) else out
                logs[name].append(tensor.detach().to(dtype=torch.float32).cpu())
            return hook_fn

        hooks = [named_modules[n].register_forward_hook(make_hook(n)) for n in union]
        try:
            with torch.no_grad():
                out = self.hf_model.generate(
                    **ts.move_to_device(full, self.hf_model),
                    max_new_tokens=trial.max_new_tokens,
                    do_sample=False,
                )
        finally:
            for hook in hooks:
                hook.remove()

        input_ids = full["input_ids"][0].cpu().numpy()
        prefill_len = int(len(input_ids))
        generated = out[0][prefill_len:]
        text = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

        image_mask = np.isin(input_ids, np.asarray(sorted(self.image_token_ids), dtype=np.int64)) \
            if self.image_token_ids else np.zeros_like(input_ids, dtype=bool)
        n_image_tokens = int(image_mask.sum())

        def score(captured: Dict[str, torch.Tensor], runtime: Dict[str, Any]) -> torch.Tensor:
            return ts.score_from_captured(captured, runtime)[0]

        def last_scalar(scores: torch.Tensor, position: Optional[int] = None) -> Optional[float]:
            arr = scores.cpu().numpy()
            if position is not None:
                return float(arr[position])
            return float(arr[-1])

        prefill_captured = {n: logs[n][0] for n in union}
        decode_captured = [{n: logs[n][j] for n in union} for j in range(1, len(logs[union[0]]))]

        primary = runtimes[self.top_k]
        robust = runtimes[8]
        prefill_primary = score(prefill_captured, primary).cpu().numpy()
        prefill_robust = score(prefill_captured, robust).cpu().numpy()

        gen_primary = [float(score(dc, primary).cpu().numpy()[-1]) for dc in decode_captured]
        gen_robust = [float(score(dc, robust).cpu().numpy()[-1]) for dc in decode_captured]

        def segments(values: List[float]) -> Dict[str, Optional[float]]:
            if not values:
                return {"mean": None, "first25": None, "last25": None}
            arr = np.asarray(values, dtype=float)
            q = max(1, len(arr) // 4)
            return {"mean": float(arr.mean()), "first25": float(arr[:q].mean()),
                    "last25": float(arr[-q:].mean()), "n": int(len(arr))}

        probe = {
            "probe_id": self.probe_id,
            "top_k": self.top_k,
            "s_pre": float(prefill_primary[k]),
            "s_img": float(prefill_primary[image_mask].mean()) if n_image_tokens else None,
            "s_gen": segments(gen_primary)["mean"],
            "s_gen_first25": segments(gen_primary)["first25"],
            "s_gen_last25": segments(gen_primary)["last25"],
            "n_generated_tokens": int(len(generated)),
            "n_image_tokens": n_image_tokens,
            "n_all_tokens": prefill_len + int(len(generated)),
            "k8": {
                "s_pre": float(prefill_robust[k]),
                "s_img": float(prefill_robust[image_mask].mean()) if n_image_tokens else None,
                "s_gen": segments(gen_robust)["mean"],
                "s_gen_first25": segments(gen_robust)["first25"],
                "s_gen_last25": segments(gen_robust)["last25"],
            },
        }

        return Response(
            text=text,
            logprobs=None,
            probe=probe,
            usage={"prefill_tokens": prefill_len, "image_tokens": n_image_tokens,
                   "generated_tokens": int(len(generated))},
            timing_ms=(time.time() - started) * 1000.0,
            cost_usd=0.0,
        )

    def run(self, trial: Trial) -> Response:
        import numpy as np
        import torch

        if self._is_generation(trial):
            return self._run_generation(trial)

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
        n_generated_tokens = 0
        truncated = False
        if trial.max_new_tokens > 0:
            with torch.no_grad():
                out = self.hf_model.generate(
                    **ts.move_to_device(encoded, self.hf_model),
                    max_new_tokens=trial.max_new_tokens,
                    do_sample=False,
                )
            new_tokens = out[0][encoded["input_ids"].shape[1]:]
            n_generated_tokens = int(len(new_tokens))
            # Greedy decode with no EOS emitted inside the budget == the model was
            # cut off at max_new_tokens. Recorded so the round-5 report can state
            # the truncation rate under the 1200-token cap honestly.
            truncated = n_generated_tokens >= trial.max_new_tokens
            text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return Response(
            text=text,
            logprobs=logprobs,
            probe=probe,
            usage={"prefill_tokens": int(len(input_ids)),
                   "image_tokens": n_image_tokens,
                   "n_generated_tokens": n_generated_tokens,
                   "truncated": truncated,
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
