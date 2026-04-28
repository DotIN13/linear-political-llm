import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm


# =========================
# Metadata
# =========================


@dataclass
class ProbeMetadata:
    probe_type: str
    model_path: str
    model_base_name: str
    prefix: str
    mode: str
    data_dir: str
    num_samples: int
    created_at_utc: str
    score_metric: str
    score_shape: List[int]
    label_mean: float
    label_std: float
    extra: Dict[str, Any]


@dataclass(frozen=True)
class ModulePaths:
    layer_prefix: str
    head_out_suffix: str = "self_attn.head_out"
    head_module_template: Optional[str] = None
    text_config_path: str = "config.text_config"


ModulePathsLike = Optional[Union[ModulePaths, Dict[str, str]]]


DEFAULT_MODULE_PATHS: Dict[str, ModulePaths] = {
    "gemma4": ModulePaths(layer_prefix="model.language_model.layers"),
    "mllama": ModulePaths(layer_prefix="model.language_model.layers"),
    "qwen3-vl": ModulePaths(layer_prefix="model.language_model.layers"),
}


# =========================
# Generic Helpers
# =========================


def model_base_name(model_path: str) -> str:
    return model_path.split("/")[-1].lower()


def _resolve_attr_path(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            raise AttributeError(f"Could not resolve '{path}' at '{part}'.")
        current = getattr(current, part)
    return current


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def resolve_module_paths(model_family: str, module_paths: ModulePathsLike = None) -> ModulePaths:
    if model_family not in DEFAULT_MODULE_PATHS:
        raise ValueError(f"Unknown model_family '{model_family}'.")

    default_paths = DEFAULT_MODULE_PATHS[model_family]
    if module_paths is None:
        return default_paths
    if isinstance(module_paths, ModulePaths):
        return module_paths
    if not isinstance(module_paths, dict):
        raise TypeError("module_paths must be a ModulePaths instance or a dictionary of overrides.")

    valid_fields = {field.name for field in fields(ModulePaths)}
    unknown_fields = sorted(set(module_paths) - valid_fields)
    if unknown_fields:
        raise ValueError(f"Unknown module_paths keys: {unknown_fields}.")

    return replace(default_paths, **module_paths)


# =========================
# Model Structure Parsing
# =========================


def _get_text_cfg(model, model_family: str, module_paths: ModulePathsLike = None):
    resolved_paths = resolve_module_paths(model_family, module_paths)
    try:
        cfg = _resolve_attr_path(model, resolved_paths.text_config_path)
        if cfg is not None:
            return cfg
    except AttributeError:
        pass
    return model.config


def get_num_layers(model, model_family: str, module_paths: ModulePathsLike = None) -> int:
    cfg = _get_text_cfg(model, model_family, module_paths)
    if hasattr(cfg, "num_hidden_layers"):
        return int(cfg.num_hidden_layers)
    resolved_paths = resolve_module_paths(model_family, module_paths)
    layers = _resolve_attr_path(model, resolved_paths.layer_prefix)
    return int(len(layers))


def get_num_heads(model, model_family: str, module_paths: ModulePathsLike = None) -> int:
    cfg = _get_text_cfg(model, model_family, module_paths)
    if not hasattr(cfg, "num_attention_heads"):
        raise ValueError("Model config missing num_attention_heads.")
    return int(cfg.num_attention_heads)


def get_head_module_names(
    model,
    mode: str,
    model_family: str,
    module_paths: ModulePathsLike = None,
) -> List[str]:
    resolved_paths = resolve_module_paths(model_family, module_paths)
    n_layers = get_num_layers(model, model_family, module_paths)

    template = resolved_paths.head_module_template
    if template is None:
        template = f"{resolved_paths.layer_prefix}.{{layer_idx}}.{resolved_paths.head_out_suffix}"

    # Build list of actual module names, handling mixed attention architectures
    # For mllama models with both self_attn and cross_attn layers, only include
    # layers with self_attn since cross_attn requires vision features that may not
    # be available during text-only feature extraction.
    head_names = []
    named_modules = dict(model.named_modules())
    for i in range(n_layers):
        # Always prefer self_attn for text-only probing
        self_attn_path = f"{resolved_paths.layer_prefix}.{i}.self_attn.head_out"

        if self_attn_path in named_modules:
            head_names.append(self_attn_path)
        else:
            # Include cross-attn only for mllama when mode is explicitly vision.
            cross_attn_path = f"{resolved_paths.layer_prefix}.{i}.cross_attn.head_out"
            if cross_attn_path in named_modules and model_family == "mllama" and mode == "vision":
                head_names.append(cross_attn_path)
            else:
                # Fall back to a generic template when provided by module_paths overrides.
                generic_path = template.format(layer_idx=i)
                if generic_path in named_modules:
                    head_names.append(generic_path)
            # Otherwise skip this layer (it doesn't have self_attn and we're not in vision mode)
    
    return head_names


# =========================
# Feature Extraction
# =========================


def _capture_module_outputs(model, encoded_inputs: Dict[str, Any], module_names: List[str]) -> Dict[str, torch.Tensor]:
    named_modules = dict(model.named_modules())
    missing = [name for name in module_names if name not in named_modules]
    if missing:
        raise ValueError(
            "Could not find expected module(s) for feature extraction: "
            f"{missing[:3]}{'...' if len(missing) > 3 else ''}. "
            "Provide model-specific paths via module_paths."
        )

    outputs: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(_module, _inp, out):
            tensor_out = out[0] if isinstance(out, tuple) else out
            outputs[name] = tensor_out.detach()

        return hook_fn

    for name in module_names:
        hooks.append(named_modules[name].register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(**encoded_inputs)

    for h in hooks:
        h.remove()

    return outputs


def _extract_last_token_head_output(layer_out: torch.Tensor, num_heads: int) -> torch.Tensor:
    # Convert to [H, D] for batch-size 1 probing.
    # Common cases:
    # - [B, S, H, D] -> take last token -> [H, D]
    # - [B, S, H*D]  -> take last token and reshape -> [H, D]
    if layer_out.dim() == 4:
        return layer_out[:, -1, :, :].squeeze(0)
    if layer_out.dim() == 3:
        last_token = layer_out[:, -1, :].squeeze(0)
        if last_token.numel() % num_heads != 0:
            raise ValueError(
                f"Cannot reshape head output of size {last_token.numel()} into {num_heads} heads."
            )
        head_dim = last_token.numel() // num_heads
        return last_token.reshape(num_heads, head_dim)
    raise ValueError(f"Unexpected head output shape: {tuple(layer_out.shape)}")


def extract_features(
    model,
    prompts: List,
    device: Optional[Union[str, torch.device]] = None,
    mode: str = "text",
    model_family: str = "qwen3-vl",
    module_paths: ModulePathsLike = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Extract attention head outputs (last-token) as either:
    - uniform ndarray [N, 1, L, H, D], or
    - ragged list of per-layer arrays [N, 1, H, D] when head dims vary by layer.
    """
    dev = _resolve_device(device)
    encoded_list = [{k: v.to(dev) for k, v in p.items() if isinstance(v, torch.Tensor)} for p in prompts]
    num_heads = get_num_heads(model, model_family, module_paths)

    head_names = get_head_module_names(model, mode=mode, model_family=model_family, module_paths=module_paths)
    if len(head_names) == 0:
        raise ValueError("No probeable attention head_out modules were found for the given mode/model_family.")

    # First pass determines whether head dimensions vary by layer.
    first_outputs = _capture_module_outputs(model, encoded_list[0], head_names)
    head_dims_per_layer = [
        int(_extract_last_token_head_output(first_outputs[name], num_heads=num_heads).shape[-1]) for name in head_names
    ]
    all_same_dim = len(set(head_dims_per_layer)) == 1

    if all_same_dim:
        features: List[np.ndarray] = []
        for enc in tqdm(encoded_list, total=len(encoded_list), desc="Extracting features"):
            out_map = _capture_module_outputs(model, enc, head_names)
            per_layer = []
            for name in head_names:
                last = _extract_last_token_head_output(out_map[name], num_heads=num_heads).to(dtype=torch.float32).cpu()
                per_layer.append(last)
            per_layer_last = torch.stack(per_layer, dim=0).numpy()  # [L, H, D]
            features.append(np.expand_dims(per_layer_last, axis=0))  # [1, L, H, D]
        return np.stack(features, axis=0)  # [N, 1, L, H, D]

    n_layers = len(head_names)
    features_per_layer: List[List[np.ndarray]] = [[] for _ in range(n_layers)]
    for enc in tqdm(encoded_list, total=len(encoded_list), desc="Extracting features"):
        out_map = _capture_module_outputs(model, enc, head_names)
        for li, name in enumerate(head_names):
            last = _extract_last_token_head_output(out_map[name], num_heads=num_heads).to(
                dtype=torch.float32
            ).cpu().numpy()  # [H, D_i]
            features_per_layer[li].append(last)

    return [np.stack(layer_feats, axis=0)[:, np.newaxis, :, :] for layer_feats in features_per_layer]


# =========================
# Base Probe
# =========================


class BaseDimensionProbe(ABC):
    """Shared interface for probing and steering across text and vision models."""

    probe_type: str = "base"

    def __init__(
        self,
        model_path: str,
        prefix: str,
        mode: str = "text",
        model_family: str = "qwen3-vl",
        data_dir: str = "results/probes",
        seed: int = 42,
        module_paths: ModulePathsLike = None,
    ) -> None:
        self.model_path = model_path
        self.model_base_name = model_base_name(model_path)
        self.prefix = prefix
        self.mode = mode
        self.model_family = model_family
        self.data_dir = data_dir
        self.seed = seed
        self.module_paths = resolve_module_paths(model_family, module_paths)
        self.base_dir = os.path.join(self.data_dir, self.model_base_name)

        self.weights_: Optional[Any] = None
        self.scores_: Optional[np.ndarray] = None
        self.metadata_: Optional[ProbeMetadata] = None

    @property
    def weights_path(self) -> str:
        return os.path.join(self.base_dir, f"{self.prefix}_{self.probe_type}_weights.pkl")

    @property
    def scores_path(self) -> str:
        return os.path.join(self.base_dir, f"{self.prefix}_{self.probe_type}_scores.npy")

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.base_dir, f"{self.prefix}_{self.probe_type}_metadata.json")

    def _ensure_base_dir(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)

    def _prepare_labels(self, labels: Union[List[float], np.ndarray]) -> np.ndarray:
        y = np.asarray(labels, dtype=np.float32).reshape(-1)
        if y.ndim != 1:
            raise ValueError(f"Expected 1D labels, got shape {y.shape}.")
        if len(y) == 0:
            raise ValueError("Labels are empty.")
        return y

    def _prepare_prompts(self, prompts: List[Any], tokenizer=None, device=None) -> List[Any]:
        if len(prompts) == 0:
            raise ValueError("Prompts are empty.")

        if isinstance(prompts[0], str):
            if self.mode != "text":
                raise ValueError("String prompts are only supported when mode='text'.")
            if tokenizer is None:
                raise ValueError("Tokenizer is required for string prompts.")
            tokenized = [tokenizer(p, return_tensors="pt") for p in prompts]
            return [{k: v.to(device) for k, v in p.items()} for p in tokenized]

        prepared = []
        for p in prompts:
            if hasattr(p, "items"):
                prepared.append({k: v.to(device) if hasattr(v, "to") else v for k, v in p.items()})
            else:
                prepared.append(p)
        return prepared

    def extract_probe_features(self, model, prompts: List[Any], tokenizer=None, device=None):
        encoded = self._prepare_prompts(prompts=prompts, tokenizer=tokenizer, device=device)
        return extract_features(
            model=model,
            prompts=encoded,
            device=device,
            mode=self.mode,
            model_family=self.model_family,
            module_paths=self.module_paths,
        )

    def _build_metadata(
        self,
        labels: np.ndarray,
        score_metric: str,
        score_shape: List[int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> ProbeMetadata:
        return ProbeMetadata(
            probe_type=self.probe_type,
            model_path=self.model_path,
            model_base_name=self.model_base_name,
            prefix=self.prefix,
            mode=self.mode,
            data_dir=self.data_dir,
            num_samples=int(labels.shape[0]),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            score_metric=score_metric,
            score_shape=score_shape,
            label_mean=float(np.mean(labels)),
            label_std=float(np.std(labels)),
            extra=extra or {},
        )

    @staticmethod
    def _json_default(obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def save(self) -> None:
        if self.weights_ is None or self.scores_ is None or self.metadata_ is None:
            raise ValueError("Cannot save probe before fitting/loading weights, scores, and metadata.")

        self._ensure_base_dir()
        with open(self.weights_path, "wb") as f:
            pickle.dump(self.weights_, f)
        np.save(self.scores_path, self.scores_)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.metadata_), f, indent=2, default=self._json_default)

    def load(self) -> "BaseDimensionProbe":
        with open(self.weights_path, "rb") as f:
            self.weights_ = pickle.load(f)
        self.scores_ = np.load(self.scores_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata_ = ProbeMetadata(**json.load(f))
        return self

    @abstractmethod
    def fit(self, model, prompts: List[Any], labels: Union[List[float], np.ndarray], tokenizer=None, device=None):
        raise NotImplementedError


# =========================
# Feature Format Helpers
# =========================


def get_layer_tensor(features, layer_idx: int) -> np.ndarray:
    """Return layer features with shape [N, H, D] for both uniform and ragged feature formats."""
    if isinstance(features, list):
        return features[layer_idx][:, 0, :, :]
    return features[:, 0, layer_idx, :, :]


def get_head_tensor(features, layer_idx: int, head_idx: int) -> np.ndarray:
    """Return head features with shape [N, D] for both uniform and ragged feature formats."""
    if isinstance(features, list):
        return features[layer_idx][:, 0, head_idx, :]
    return features[:, 0, layer_idx, head_idx, :]


def features_is_ragged(features) -> bool:
    return isinstance(features, list)
