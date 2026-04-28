from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from probes.base import BaseDimensionProbe, features_is_ragged, get_layer_tensor
from probes.rfm.direction_utils import train_rfm_probe_on_concept


class LayerwiseRFM(BaseDimensionProbe):
    """Per-layer RFM probe over flattened [H, D] representations."""

    probe_type = "rfm_layerwise"

    def __init__(
        self,
        model_path: str,
        prefix: str,
        mode: str = "text",
        model_family: str = "qwen3-vl",
        data_dir: str = "results/probes",
        seed: int = 42,
        train_fraction: float = 0.8,
        bandwidths: Optional[List[int]] = None,
        module_paths: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            prefix=prefix,
            mode=mode,
            model_family=model_family,
            data_dir=data_dir,
            seed=seed,
            module_paths=module_paths,
        )
        self.train_fraction = train_fraction
        self.bandwidths = bandwidths or [1, 10, 100]

    def _split_indices(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1).")
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n_samples)
        n_train = int(self.train_fraction * n_samples)
        return perm[:n_train], perm[n_train:]

    def fit(self, model, prompts: List[Any], labels: Union[List[float], np.ndarray], tokenizer=None, device=None):
        y_np = self._prepare_labels(labels)
        features = self.extract_probe_features(model=model, prompts=prompts, tokenizer=tokenizer, device=device)

        n_layers = len(features) if features_is_ragged(features) else features.shape[2]
        n_samples = len(features[0]) if features_is_ragged(features) else features.shape[0]
        if n_samples != len(y_np):
            raise ValueError(f"Feature/label mismatch: {n_samples} features vs {len(y_np)} labels.")

        train_idx, val_idx = self._split_indices(n_samples)
        if len(val_idx) < 2:
            raise ValueError("Validation split is too small for correlation metrics.")

        dev = device
        if dev is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(dev, str):
            dev = torch.device(dev)

        y = torch.as_tensor(y_np, dtype=torch.float32, device=dev).reshape(-1, 1)
        y_train = y[torch.as_tensor(train_idx, device=dev)]
        y_val = y[torch.as_tensor(val_idx, device=dev)]

        directions: Dict[int, np.ndarray] = {}
        scores = np.zeros((n_layers,), dtype=np.float32)
        per_layer_metrics: Dict[int, Dict[str, float]] = {}

        for li in range(n_layers):
            layer = get_layer_tensor(features, li)
            x_all = torch.as_tensor(layer.reshape(layer.shape[0], -1), dtype=torch.float32, device=dev)
            x_train = x_all[torch.as_tensor(train_idx, device=dev)]
            x_val = x_all[torch.as_tensor(val_idx, device=dev)]

            direction = train_rfm_probe_on_concept(
                train_X=x_train,
                train_y=y_train,
                val_X=x_val,
                val_y=y_val,
                hyperparams={"probe_type": self.probe_type},
                bws=self.bandwidths,
            )

            direction = direction.reshape(-1)
            norm = torch.norm(direction) + 1e-12
            direction = direction / norm

            preds = (x_val @ direction.reshape(-1, 1)).detach().cpu().numpy().reshape(-1)
            y_val_np = y_val.detach().cpu().numpy().reshape(-1)
            p = pearsonr(y_val_np, preds).statistic
            s = spearmanr(y_val_np, preds).statistic
            p = 0.0 if np.isnan(p) else float(p)
            s = 0.0 if np.isnan(s) else float(s)

            directions[li] = direction.detach().cpu().numpy().astype(np.float32)
            scores[li] = p
            per_layer_metrics[li] = {"pearsonr": p, "spearmanr": s}

        self.weights_ = directions
        self.scores_ = scores
        self.metadata_ = self._build_metadata(
            labels=y_np,
            score_metric="pearsonr",
            score_shape=[n_layers],
            extra={
                "seed": self.seed,
                "train_fraction": self.train_fraction,
                "bandwidths": self.bandwidths,
                "is_ragged": features_is_ragged(features),
                "per_layer_metrics": per_layer_metrics,
                "module_paths": self.module_paths,
            },
        )
        return self

    def topk_layers(self, k: int = 8) -> np.ndarray:
        if self.scores_ is None:
            raise ValueError("Fit or load the probe before requesting top-k layers.")
        return np.argsort(self.scores_)[::-1][:k]

    def build_steering(
        self,
        features: Union[np.ndarray, List[np.ndarray]],
        k: int = 8,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        if self.weights_ is None or self.scores_ is None:
            raise ValueError("Fit or load the probe before building steering vectors.")

        top_layers = set(self.topk_layers(k=k).tolist())
        dev = torch.device(device) if isinstance(device, str) else device

        if features_is_ragged(features):
            steering: Dict[int, torch.Tensor] = {}
            for li, layer_features in enumerate(features):
                h, d = layer_features.shape[2], layer_features.shape[3]
                vec = self.weights_[li].reshape(h, d) if li in top_layers else np.zeros((h, d), dtype=np.float32)
                steering[li] = torch.as_tensor(vec, device=dev)
            return steering

        _, _, h, d = features.shape[1:]
        l = features.shape[2]
        steering = np.zeros((l, h, d), dtype=np.float32)
        for li in top_layers:
            steering[int(li)] = self.weights_[int(li)].reshape(h, d)
        return torch.as_tensor(steering, device=dev)

    def score_samples(self, features: Union[np.ndarray, List[np.ndarray]], k: int = 8) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Fit or load the probe before scoring samples.")

        top_layers = self.topk_layers(k=k)
        n_samples = len(features[0]) if features_is_ragged(features) else features.shape[0]
        out = np.zeros(n_samples, dtype=np.float32)

        for n in range(n_samples):
            vals = []
            for li in top_layers:
                layer = get_layer_tensor(features, int(li))[n].reshape(-1)
                vals.append(float(np.dot(layer, self.weights_[int(li)])))
            out[n] = float(np.mean(vals)) if len(vals) > 0 else 0.0
        return out
