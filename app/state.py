"""AppState and probe runtime building."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from probes.headwise_linear_probe import HeadwiseLinearProbe
from probes.layerwise_linear_probe import LayerwiseLinearProbe
from probes.base import get_head_module_names, resolve_module_paths

MODEL_REGISTRY = {
    "qwen3_vl": "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct",
    "gemma4": "/home/tzhang3/jevans/models/gemma-4-31B-it",
}


class AppState:
    def __init__(self, model_paths: Dict[str, str], data_dir: str, top_k: int):
        self.model_paths = model_paths
        self.data_dir = data_dir
        self.top_k = top_k
        self.current_model_name: Optional[str] = None
        self.model = None
        self.processor = None
        self._runtimes: dict = {}
        self._loaded = False

    @property
    def model_family(self) -> str:
        if self.current_model_name == "gemma4":
            return "gemma4"
        return "qwen3-vl"

    @property
    def model_base(self) -> str:
        if self.current_model_name == "gemma4":
            return "gemma-4-31b-it"
        return "qwen3-vl-8b-instruct"

    def _ensure_model_loaded(self, progress=None):
        if self._loaded and self.current_model_name is not None:
            return
        if self.current_model_name is None:
            self.current_model_name = list(self.model_paths.keys())[0]
        _load_model(self, self.current_model_name, progress)

    def switch_model(self, name: str, progress=None):
        if self.current_model_name == name and self._loaded:
            return
        # Clear old state
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()
        self.model = None
        self.processor = None
        self._loaded = False
        self._runtimes = {}
        self.current_model_name = name
        _load_model(self, name, progress)

    def get_runtime(self, prefix: str, probe_name: str, progress=None):
        self._ensure_model_loaded(progress)
        key = (self.current_model_name, prefix, probe_name)
        if key in self._runtimes:
            return self._runtimes[key]
        if progress is not None:
            progress(0.6, desc=f"Loading probe: {prefix}/{probe_name} ...")
        print(f"Loading probe: {prefix}/{probe_name} (model={self.current_model_name}) ...")
        probe_cls = {"headwise_linear": HeadwiseLinearProbe, "layerwise_linear": LayerwiseLinearProbe}[probe_name]
        probe = probe_cls(
            model_path=self.model_paths[self.current_model_name],
            prefix=prefix,
            mode="vision",
            model_family=self.model_family,
            data_dir=self.data_dir,
        ).load()
        runtime = build_runtime(self.model, probe, self.top_k)
        self._runtimes[key] = runtime
        print(f"Probe loaded. Using top-{self.top_k} {'heads' if probe_name == 'headwise_linear' else 'layers'}.")
        if progress is not None:
            progress(1.0, desc="Ready")
        return runtime


def _load_model(state: AppState, name: str, progress=None):
    model_path = state.model_paths[name]
    if progress is not None:
        progress(0, desc=f"Loading model: {name} ...")
    print(f"Loading model from {model_path} ...")
    state.processor = AutoProcessor.from_pretrained(model_path)
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    try:
        import flash_attn
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print("FlashAttention2 enabled.")
    except ImportError:
        load_kwargs["attn_implementation"] = "sdpa"
        print("FlashAttention2 not found, falling back to SDPA.")
    state.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    state.model.eval()
    try:
        dev = state.model.device
    except AttributeError:
        dev = next(state.model.parameters()).device
    print(f"Model loaded on {dev}.")
    state._loaded = True


_state: Optional[AppState] = None


def build_runtime(model, probe, top_k: int) -> dict:
    if probe.weights_ is None or probe.scores_ is None:
        raise ValueError("Probe must be loaded before building runtime.")

    head_names = get_head_module_names(
        model, mode=probe.mode, model_family=probe.model_family, module_paths=probe.module_paths,
    )

    runtime: dict = {
        "probe_type": probe.probe_type,
        "head_names": head_names,
        "groups": {},
        "normalizer": 1.0,
        "module_names": set(),
    }

    if probe.probe_type == "headwise_linear":
        top_units = probe.topk_indices(k=top_k)
        groups: Dict[str, List[Tuple[int, torch.Tensor]]] = {}
        for li, hi in top_units:
            li = int(li)
            hi = int(hi)
            module_name = head_names[li]
            coef = np.asarray(probe.weights_[li][hi].coef_, dtype=np.float32)
            groups.setdefault(module_name, []).append((hi, torch.from_numpy(coef)))
        runtime["groups"] = groups
        runtime["normalizer"] = float(max(len(top_units), 1))
        runtime["module_names"] = set(groups.keys())
        return runtime

    if probe.probe_type == "layerwise_linear":
        top_layers = [int(x) for x in probe.topk_layers(k=top_k)]
        groups_layer: Dict[str, torch.Tensor] = {}
        for li in top_layers:
            module_name = head_names[li]
            vec = np.asarray(probe.weights_[li].coef_, dtype=np.float32)
            groups_layer[module_name] = torch.from_numpy(vec)
        runtime["groups"] = groups_layer
        runtime["normalizer"] = float(max(len(top_layers), 1))
        runtime["module_names"] = set(groups_layer.keys())
        return runtime

    raise ValueError(f"Unsupported probe type: {probe.probe_type}")
