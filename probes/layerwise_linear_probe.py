from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from probes.base import BaseDimensionProbe, features_is_ragged, get_layer_tensor, get_num_layers


class LayerwiseLinearProbe(BaseDimensionProbe):
    """Per-layer linear probe over flattened [H, D] representations."""

    probe_type = "layerwise_linear"

    def __init__(
        self,
        model_path: str,
        prefix: str,
        mode: str = "text",
        data_dir: str = "results/probes",
        alpha: float = 1.0,
        n_splits: int = 2,
        seed: int = 42,
        module_paths: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            prefix=prefix,
            mode=mode,
            data_dir=data_dir,
            seed=seed,
            module_paths=module_paths,
        )
        self.alpha = alpha
        self.n_splits = n_splits

    def fit(self, model, prompts: List[Any], labels: Union[List[float], np.ndarray], tokenizer=None, device=None):
        y = self._prepare_labels(labels)
        features = self.extract_probe_features(model=model, prompts=prompts, tokenizer=tokenizer, device=device)

        n_layers = get_num_layers(model, self.mode, self.module_paths)
        n_samples = len(features[0]) if features_is_ragged(features) else features.shape[0]

        if n_samples != len(y):
            raise ValueError(f"Feature/label mismatch: {n_samples} features vs {len(y)} labels.")

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        ridge_models: Dict[int, Ridge] = {}
        scores = np.zeros((n_layers,), dtype=np.float32)

        for li in range(n_layers):
            fold_scores = []
            layer = get_layer_tensor(features, li)
            x_all = layer.reshape(layer.shape[0], -1)
            last_model = None

            for train_idx, val_idx in kf.split(np.arange(n_samples)):
                x_train = x_all[train_idx]
                x_val = x_all[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]

                ridge = Ridge(alpha=self.alpha, fit_intercept=False)
                ridge.fit(x_train, y_train)
                y_pred = ridge.predict(x_val)
                corr = spearmanr(y_val, y_pred).statistic
                fold_scores.append(0.0 if np.isnan(corr) else float(corr))
                last_model = ridge

            ridge_models[li] = last_model
            scores[li] = float(np.mean(fold_scores))

        self.weights_ = ridge_models
        self.scores_ = scores
        self.metadata_ = self._build_metadata(
            labels=y,
            score_metric="spearmanr",
            score_shape=[n_layers],
            extra={
                "alpha": self.alpha,
                "n_splits": self.n_splits,
                "seed": self.seed,
                "is_ragged": features_is_ragged(features),
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
                coef = np.asarray(self.weights_[li].coef_, dtype=np.float32).reshape(h, d)
                std = np.std(layer_features[:, 0], axis=0)
                layer_coef = coef * std if li in top_layers else np.zeros((h, d), dtype=np.float32)
                steering[li] = torch.as_tensor(layer_coef, device=dev)
            return steering

        l, _, h, d = features.shape
        _ = l
        std_tensor = np.std(features[:, 0], axis=0)  # [L, H, D]
        steering = np.zeros_like(std_tensor, dtype=np.float32)
        for li in top_layers:
            coef = np.asarray(self.weights_[li].coef_, dtype=np.float32).reshape(h, d)
            steering[li] = coef * std_tensor[li]
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
                layer = get_layer_tensor(features, int(li))[n].reshape(1, -1)
                vals.append(float(self.weights_[int(li)].predict(layer)[0]))
            out[n] = float(np.mean(vals)) if len(vals) > 0 else 0.0
        return out
