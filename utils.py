import os
import pickle
import random
import gc
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from baukit import Trace, TraceDict
from tqdm.auto import tqdm
from IPython.display import HTML, display

# Plotting
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# HF / sklearn / stats
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42) -> None:
    """
    Set seeds and configurations for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set seed {seed}")


def model_base_name(model_path: str) -> str:
    """Return the normalized trailing name from a model path."""
    return model_path.split("/")[-1].lower()


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def load_tokenizer_and_model(
    model_path: str,
    device: Optional[Union[str, torch.device]] = None,
    hf_token: Optional[str] = None
):
    """
    Loads a tokenizer and model for causal LM generation.
    """
    device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        use_auth_token=hf_token
    ).to(device).eval()
    return tokenizer, model


def get_top_indices(performance: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Return top-k (layer, head) indices from a performance matrix [L, H].
    """
    L, H = performance.shape
    flat_idx = np.argsort(performance.ravel())
    top_flat = flat_idx[-k:]
    return np.dstack(np.unravel_index(top_flat, (L, H)))[0]


def load_ridge_models(model_path: str, prefix: str):
    """
    Load ridge models and performance scores from disk.
    """
    base_name = model_base_name(model_path)
    base_dir = os.path.join("results", base_name)
    performance = pickle.load(open(os.path.join(base_dir, f"{prefix}_performance.pkl"), 'rb'))
    ridge_models = pickle.load(open(os.path.join(base_dir, f"{prefix}_ridge.pkl"), 'rb'))
    return ridge_models, performance


# =========================
# Model Introspection Helpers (text & vision)
# =========================

def _get_text_cfg(model):
    """Return the text config object for either a pure text model or a VLM with text_config."""
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        return model.config.text_config
    return model.config

def get_num_layers(model, mode: str) -> int:
    cfg = _get_text_cfg(model) if mode == "vision" else model.config
    if not hasattr(cfg, "num_hidden_layers"):
        raise ValueError("Model config missing num_hidden_layers.")
    return int(cfg.num_hidden_layers)

def get_num_heads(model, mode: str) -> int:
    cfg = _get_text_cfg(model) if mode == "vision" else model.config
    if not hasattr(cfg, "num_attention_heads"):
        raise ValueError("Model config missing num_attention_heads.")
    return int(cfg.num_attention_heads)

def get_hidden_size(model, mode: str) -> int:
    cfg = _get_text_cfg(model) if mode == "vision" else model.config
    if not hasattr(cfg, "hidden_size"):
        raise ValueError("Model config missing hidden_size.")
    return int(cfg.hidden_size)

def get_head_dim(model, mode: str) -> int:
    return get_hidden_size(model, mode) // get_num_heads(model, mode)

def get_layer_module_prefix(mode: str) -> str:
    """Return the module path prefix that contains transformer block layers."""
    if mode == "text":
        return "model.layers"
    if mode == "vision":
        return "model.language_model.layers"
    raise ValueError(f"Unknown mode '{mode}', expected 'text' or 'vision'.")

def get_head_out_key(layer_idx: int, mode: str) -> str:
    """Return the full named_module key to the head_out tensor for the given layer."""
    return f"{get_layer_module_prefix(mode)}.{layer_idx}.self_attn.head_out"

def get_all_head_out_keys(model, mode: str) -> List[str]:
    """Return all head_out keys across layers for tracing."""
    return [get_head_out_key(i, mode) for i in range(get_num_layers(model, mode))]


# Back-compat internal helpers (kept but routed through new ones)
def _head_names(num_layers: int, mode: str) -> List[str]:
    return [get_head_out_key(i, mode) for i in range(num_layers)]

def _get_layer_head_counts(model, mode: str) -> Tuple[int, int]:
    return get_num_layers(model, mode), get_num_heads(model, mode)


def clean_up(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Clean up CPU/GPU memory.
    """
    gc.collect()
    dev = _resolve_device(device) if device is not None else None
    if dev and dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =========================
# Feature Extraction & Probing
# =========================

def extract_and_save_features(
    model,
    model_path: str,
    prompts: List,
    labels: List,
    output_dir: str = "results",
    prefix: str = "politician",
    device: Optional[Union[str, torch.device]] = None,
    mode: str = "text"
):
    """
    Given a DataFrame with columns ['prompt', 'label'], extract attention head outputs
    (last-token slice) and save to pickle:
      (features: np.ndarray [N, 1, L, H, D], labels: np.ndarray [N])
    """
    device = _resolve_device(device)

    # Tokenize
    encoded_list = [p['input_ids'].to(device) for p in prompts]

    num_layers, _ = _get_layer_head_counts(model, mode)
    heads = get_all_head_out_keys(model, mode)

    head_wise_hidden_states_list: List[np.ndarray] = []

    for enc in tqdm(encoded_list, total=len(encoded_list)):
        with torch.no_grad():
            with TraceDict(model, heads) as ret:
                _ = model(enc.to(device))
                # ret[head].output -> shape [B, S, H, D] or similar (we squeeze)
                per_head = []
                for head in heads:
                    out = ret[head].output.squeeze().detach().to(dtype=torch.float32).cpu()
                    per_head.append(out)
                # Stack across layers -> [L, S, H, D], keep all tokens (slice later)
                per_head = torch.stack(per_head, dim=0).numpy()
                head_wise_hidden_states_list.append(per_head)

    # Select last token across sequences for each (L, H, D)
    # Convert to target shape: [N, 1, L, H, D]
    features = []
    for arr in head_wise_hidden_states_list:  # arr: [L, S, H, D]
        last_tok = arr[:, -1, :, :]           # [L, H, D]
        features.append([last_tok])           # add T=1 dimension
    features = np.stack(features, axis=0)     # [N, 1, L, H, D]

    # Save
    base_name = model_base_name(model_path)
    base_dir = os.path.join(output_dir, base_name)
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, f"{prefix}_features.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump((features, labels), f)
    print(f"Saved features to {out_path}")

    return features


def train_ridge_models(
    model,
    model_path: str,
    data_dir: str = "results",
    prefix: str = "politician",
    alpha: float = 1.0,
    n_splits: int = 2,
    seed: int = 42,
    mode: str = "text"
):
    """
    Load saved features and labels, train per-head Ridge regressors with KFold CV,
    returning (performance [L,H], ridge_models: dict[layer][head] -> model).
    """
    # Load
    base_name = model_base_name(model_path)
    data_path = os.path.join(data_dir, base_name, f"{prefix}_features.pkl")
    with open(data_path, 'rb') as f:
        features, labels = pickle.load(f)

    n_layers, n_heads = _get_layer_head_counts(model, mode)
    performance = np.zeros((n_layers, n_heads), dtype=np.float32)
    ridge_dict: Dict[int, Dict[int, Ridge]] = {}

    # Train per head
    for i in tqdm(range(n_layers), desc="Layers"):
        ridge_dict[i] = {}
        for j in range(n_heads):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            score_sum = 0.0
            for train_idx, test_idx in kf.split(range(features.shape[0])):
                X_train = features[train_idx, 0, i, j, :]  # [N_train, D]
                X_test  = features[test_idx, 0, i, j, :]   # [N_test, D]
                y_train = np.asarray(labels)[train_idx]
                y_test  = np.asarray(labels)[test_idx]

                model_ridge = Ridge(alpha=alpha, fit_intercept=False)
                model_ridge.fit(X_train, y_train)
                y_pred = model_ridge.predict(X_test)

                corr = spearmanr(y_test, y_pred).statistic
                if np.isnan(corr):
                    corr = 0.0
                score_sum += corr
                ridge_dict[i][j] = model_ridge

            performance[i, j] = score_sum / float(n_splits)

    # Save
    base_dir = os.path.join(data_dir, base_name)
    os.makedirs(base_dir, exist_ok=True)
    pickle.dump(performance, open(os.path.join(base_dir, f"{prefix}_performance.pkl"), 'wb'))
    pickle.dump(ridge_dict, open(os.path.join(base_dir, f"{prefix}_ridge.pkl"), 'wb'))

    return performance, ridge_dict


def probe_heads(
    model_path: str,
    prompts: List,
    labels: List,
    prefix: str = "politician",
    device: Union[str, torch.device] = "cpu",
    alpha: float = 1.0,
    n_splits: int = 2,
    seed: int = 42,
    data_dir: str = "results",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 150,
    cmap: str = 'viridis',
    vmin: int = 0,
    vmax: int = 100,
    model=None,
    mode: str = "text"
):
    """
    Load/prepare a model, extract features from df_prompts, train ridge models,
    then plot a heatmap of head performance sorted (row-wise) high-to-low.
    """
    device = _resolve_device(device)

    # Load tokenizer and model (if needed)
    delete_model = False
    if model is None:
        _, model = load_tokenizer_and_model(model_path, device=device)
        delete_model = True
    else:
        model.to(device)

    # Extract features and save to disk
    extract_and_save_features(
        model, model_path, prompts=prompts, labels=labels,
        prefix=prefix, device=device, mode=mode
    )

    # Train ridge models
    _perf, _models = train_ridge_models(
        model, model_path,
        data_dir=data_dir, prefix=prefix,
        alpha=alpha, n_splits=n_splits, seed=seed, mode=mode
    )

    # Free model memory before plotting/return
    if delete_model:
        del model
        clean_up(device=device)

    # Load to ensure consistency with downstream APIs
    ridge_models, performance = load_ridge_models(model_path, prefix)

    # Prepare performance as integer percent and sort each row desc
    perf_percent = (performance * 100).astype(int)
    perf_sorted = -np.sort(-perf_percent, axis=1)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.heatmap(
        perf_sorted,
        annot=True,
        cmap=cmap,
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )
    ax.invert_yaxis()
    ax.set_title('Heatmap (Rows Sorted High to Low)')
    ax.set_xlabel('Sorted Head Index')
    ax.set_ylabel('Layer Index')
    plt.tight_layout()
    plt.show()

    return ax, performance, ridge_models


# =========================
# Token-level extraction & scoring
# =========================

def extract_head_out_per_token(
    model,
    tokenizer,
    prompts: List,
    candidate_answers: Optional[List[str]] = None,
    max_new_tokens: int = 32,
    output_only: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    mode: str = "text"
):
    """
    Extract attention head outputs per generated token and (optionally) first-step probabilities
    over candidate answer tokens.

    Returns a list of dicts:
      - 'features': np.ndarray [T, L, H, D]
      - 'token_ids': np.ndarray [seq_len]
      - 'answer': str
      - optional: 'candidate_logits'/'candidate_probs': np.ndarray [len(candidates)]
    """
    device = _resolve_device(device)

    num_layers = get_num_layers(model, mode)
    num_heads = get_num_heads(model, mode)
    head_dim = get_head_dim(model, mode)
    heads = get_all_head_out_keys(model, mode)

    # Prepare candidate token IDs if applicable
    candidate_token_ids = None
    if candidate_answers is not None:
        candidate_token_ids = [tokenizer.convert_tokens_to_ids(ans) for ans in candidate_answers]
        candidate_token_ids = torch.tensor(candidate_token_ids, device=device)

    results = []

    for prompt in tqdm(prompts):
        prompt = prompt.to(device)

        token_outputs_by_layer: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]

        def make_hook(layer_idx: int):
            def hook_fn(module, inp, out):
                # out shape: [B, S, H, D]
                # capture either only the last step or the whole sequence
                if output_only:
                    token_outputs_by_layer[layer_idx].append(out.squeeze(0)[-1].detach().cpu().to(dtype=torch.float32))
                else:
                    token_outputs_by_layer[layer_idx].extend(out.squeeze(0).detach().cpu().to(dtype=torch.float32))
            return hook_fn

        # Register hooks
        hooks = []
        named = dict(model.named_modules())
        for i, name in enumerate(heads):
            module = named[name]
            hooks.append(module.register_forward_hook(make_hook(i)))

        # Generate tokens & (optionally) return scores for first generated step
        with torch.no_grad():
            outputs = model.generate(
                **prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=(candidate_answers is not None),
                output_attentions=False
            )

        # Remove hooks
        for h in hooks:
            h.remove()

        # Sequence outputs
        generated_ids = outputs.sequences.squeeze(0).cpu().numpy()
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Build [T, L, H, D] from captured
        layerwise_outputs = []
        for layer_outputs in token_outputs_by_layer:
            if not layer_outputs:
                # In rare cases with very short generations, ensure shape consistency
                layerwise_outputs.append(torch.zeros((0, num_heads, head_dim)))
                continue
            stacked = torch.stack(layer_outputs, dim=0)             # [T, H*D] or [T, H, D]
            if stacked.dim() == 2:
                # If hooks return [T, hidden_size], reshape to [T, H, D]
                stacked = stacked.view(stacked.size(0), num_heads, head_dim)
            layerwise_outputs.append(stacked)

        if len(layerwise_outputs) == 0:
            feats = torch.zeros((0, num_layers, num_heads, head_dim))
        else:
            layerwise_outputs = torch.stack(layerwise_outputs, dim=0)      # [L, T, H, D]
            feats = rearrange(layerwise_outputs, 'l t h d -> t l h d')     # [T, L, H, D]

        result = {
            "features": feats.numpy(),
            "token_ids": generated_ids,
            "answer": decoded,
        }

        # Candidate token probabilities from the first generation step
        if candidate_token_ids is not None and len(outputs.scores) > 0:
            first_step_logits = outputs.scores[0].squeeze(0)               # [vocab]
            probs = F.softmax(first_step_logits, dim=-1)                   # [vocab]
            result["candidate_logits"] = first_step_logits[candidate_token_ids].cpu().numpy()
            result["candidate_probs"] = probs[candidate_token_ids].cpu().numpy()

        results.append(result)

    return results


def predict_per_token_scores(
    results: List[Dict],
    ridge_models: Dict[int, Dict[int, Ridge]],
    performance: np.ndarray,
    k: int
):
    """
    For each result with 'features' [T, L, H, D], predict a per-token scalar using
    the average of the top-k heads' ridge predictions.
    """
    top_indices = get_top_indices(performance, k=k)

    for res in results:
        feats = res['features']  # [T, L, H, D]
        if feats.size == 0:
            res['scores'] = np.array([])
            continue

        T, _, _, _ = feats.shape
        scores = np.zeros(T, dtype=np.float32)

        for t in range(T):
            s = 0.0
            for i, j in top_indices:
                model_ridge = ridge_models[int(i)][int(j)]
                x = feats[t, int(i), int(j), :].reshape(1, -1)
                s += float(model_ridge.predict(x)[0])
            scores[t] = s / len(top_indices)

        res['scores'] = scores

    return results


def visualize_token_scores(
    results: List[Dict],
    tokenizer,
    title_prefix: str = "Prompt",
    vmin: float = -1.0,
    vmax: float = 1.0
) -> None:
    """
    Displays tokens with per-token scores using color-coding.
    """
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('RdBu_r')

    for idx, res in enumerate(results):
        ids = res.get('token_ids', [])
        scores = res.get('scores', np.array([]))
        tokens = tokenizer.convert_ids_to_tokens(ids) if len(ids) else []

        spans = []
        # Align the shorter of tokens[1:] and scores
        n = min(len(tokens) - 1 if len(tokens) > 0 else 0, len(scores))
        for tok, sc in zip(tokens[1: 1 + n], scores[:n]):  # skip BOS if present
            color = mpl.colors.to_hex(cmap(norm(float(sc))))
            safe_tok = (
                tok.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('Ġ', ' ')
            )
            spans.append(
                f"<span style='background-color:{color};padding:2px;margin:1px;border-radius:3px;color:#313131;'>{safe_tok}</span>"
            )
        html = '<div style="line-height:1.6;font-family:monospace;">' + ' '.join(spans) + '</div>'
        display(HTML(f"<h4>{title_prefix} {idx + 1}:</h4>{html}"))


def plot_candidate_probabilities(
    results: List[Dict],
    candidate_answers: List[str],
    title_prefix: str = "Prompt",
    figsize: Tuple[int, int] = (8, 4)
) -> None:
    """
    Visualizes the candidate probability distributions for each prompt result.
    """
    for idx, result in enumerate(results):
        if "candidate_probs" not in result:
            print(f"Skipping prompt {idx+1}: no candidate_probs found.")
            continue

        probs = result["candidate_probs"]

        plt.figure(figsize=figsize)
        bars = plt.bar(candidate_answers, probs)
        plt.title(f"{title_prefix} {idx + 1}")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.xticks(rotation=0)

        # Annotate bar values
        for bar, prob in zip(bars, probs):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{prob:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.show()


def generate_and_score_tokens(
    model_path: str,
    prompts: List,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 512,
    hf_token: Optional[str] = None,
    k: int = 8,
    candidate_answers: Optional[List[str]] = None,
    output_only: bool = False,
    ridge_prefix: Optional[str] = None,
    visualize: bool = False,
    mode: str = "text",
    device: Optional[Union[str, torch.device]] = None
):
    """
    Generate tokens, extract head features, and compute token-level scores from top-k ridge heads.
    """
    device = _resolve_device(device)
    delete_model = False

    if model is None or tokenizer is None:
        tokenizer, model = load_tokenizer_and_model(model_path, device=device, hf_token=hf_token)
        delete_model = True
    else:
        model.to(device)

    # 1) Extract features per generated token
    results = extract_head_out_per_token(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        candidate_answers=candidate_answers,
        output_only=output_only,
        device=device, mode=mode
    )

    # 2) Load trained ridge models & performance
    ridge_models, performance = load_ridge_models(model_path, ridge_prefix)

    # 3) Predict token-level scores
    results = predict_per_token_scores(results, ridge_models, performance, k)

    if delete_model:
        del model
        clean_up(device=device)

    if visualize:
        visualize_token_scores(results, tokenizer)

    return results



# =========================
# Steering Helpers
# =========================

def load_feats(
    model_path: str,
    prefix: str,
    results_dir: str = "results",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Load features from: {results_dir}/{model_base_name(model_path)}/{prefix}_features.pkl

    Returns:
      feats: torch.Tensor [N, T, L, H, D] (on device if provided)
    """
    safe_name = model_base_name(model_path)
    feats_cpu, _ = pickle.load(open(f"{results_dir}/{safe_name}/{prefix}_features.pkl", "rb"))
    feats = torch.as_tensor(feats_cpu, device=device)
    if feats.dim() != 5:
        raise ValueError(f"Expected feats with shape [N, T, L, H, D], got {feats.shape}")
    return feats


def topk_head_indices(
    performance: Union[np.ndarray, torch.Tensor],
    k: int,
    device: Optional[torch.device] = None,
) -> torch.LongTensor:
    """
    Return the indices of the top-k heads by performance as a [k, 2] tensor (layer, head).
    """
    perf = torch.as_tensor(performance, device=device)  # [L, H]
    L, H = perf.shape
    flat = perf.view(-1)
    k = min(k, flat.numel())
    topk_flat = torch.argsort(flat, descending=True)[:k]  # [k]
    top_idx = torch.stack([topk_flat // H, topk_flat % H], dim=1)  # [k, 2]
    return top_idx


def build_combined_coefs_from_indices(
    feats: torch.Tensor,             # [N, T, L, H, D]
    ridge_models: dict,              # ridge_models[layer][head].coef_
    top_idx: torch.LongTensor,       # [k, 2] (li, hi)
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build [L, H, D] combined coefficients:
      - per-head std at timestep 0 from feats
      - fill ridge coefs for selected heads
      - return ridge * std
    """
    assert feats.dim() == 5, "feats must be [N, T, L, H, D]"
    if device is not None and feats.device != device:
        feats = feats.to(device)

    _, _, L, H, D = feats.shape

    # Per-head std at timestep 0 → [L, H, D]
    std_tensor = torch.std(feats[:, 0], dim=0)  # [L, H, D]

    # Assemble ridge coefs
    ridge_t = torch.zeros((L, H, D), device=feats.device, dtype=std_tensor.dtype)
    for li, hi in top_idx.tolist():
        coef = ridge_models[int(li)][int(hi)].coef_
        ridge_t[int(li), int(hi)] = torch.as_tensor(coef, device=feats.device, dtype=std_tensor.dtype)

    return ridge_t * std_tensor  # [L, H, D]


def compute_combined_coefs(
    feats: Union[torch.Tensor, np.ndarray],
    ridge_models: dict,
    performance: Optional[Union[np.ndarray, torch.Tensor]] = None,
    k: Optional[int] = None,
    top_idx: Optional[torch.LongTensor] = None,  # overrides k/performance if given
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute combined_coefs [L, H, D] given preloaded feats.

    Selection logic:
      - If top_idx is provided -> use it (overrides k/performance).
      - Else requires both k and performance to pick top-k heads.
    """
    feats_t = feats if isinstance(feats, torch.Tensor) else torch.as_tensor(feats, device=device)
    if device is not None and feats_t.device != device:
        feats_t = feats_t.to(device)

    if top_idx is None:
        if k is None or performance is None:
            raise ValueError("Provide either top_idx OR both k and performance.")
        top_idx = topk_head_indices(performance=performance, k=k, device=device)

    return build_combined_coefs_from_indices(
        feats=feats_t,
        ridge_models=ridge_models,
        top_idx=top_idx,
        device=device,
    )


def generate_with_head_intervention_gpu(
    model,
    tokenizer,
    prompts: Union[str, List[str]],
    alpha: float,
    max_new_tokens: int = 200,
    combined_coefs: torch.Tensor = None,  # [L, H, D]
    return_features: bool = False,
    device: Optional[torch.device] = None
) -> Union[List[str], List[Dict]]:
    """
    Fully‐GPU head‐intervention + feature capture on one/many prompts, via a single hook.

    Returns:
      - if return_features=False: List[str] (answers)
      - else: List[dict] with "features"[T,L,H,D], "token_ids", "answer"
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if combined_coefs is None:
        raise ValueError("combined_coefs [L, H, D] must be provided.")

    # Infer dims from combined_coefs and model for robustness
    L, H, D = combined_coefs.shape
    heads = [get_head_out_key(i, "text") for i in range(L)]  # intervention currently on text path

    # Buffer for captured head outputs
    captured: Dict[int, List[torch.Tensor]] = {li: [] for li in range(L)}

    def hook_fn(output, module_name):
        # output: [B, S, H, D]
        h = output
        li = int(module_name.split('.')[2])  # 'model.layers.{i}.self_attn.head_out'
        # apply intervention on last token logits of heads
        h[:, -1] = h[:, -1] + alpha * combined_coefs[li]
        # capture (whole sequence)
        captured[li].extend(h[:, :].detach().cpu().squeeze(0))
        return h

    results = []
    with TraceDict(model, heads, edit_output=hook_fn):
        for prompt in prompts:
            # reset captures
            for li in range(L):
                captured[li].clear()

            inp = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            out = model.generate(inp, max_new_tokens=max_new_tokens, do_sample=False)

            tokens = out[0].cpu().numpy()
            answer = tokenizer.decode(out[0], skip_special_tokens=True).strip()

            if not return_features:
                results.append(answer)
                continue

            # Build [T, L, H, D]
            T = len(captured[0]) if L > 0 else 0
            if T == 0:
                feats_arr = torch.zeros((0, L, H, D))
            else:
                layer_feats = [torch.stack(captured[li], dim=0) for li in range(L)]  # each [T, H, D]
                feats_arr = torch.stack(layer_feats, dim=0).permute(1, 0, 2, 3)     # [T, L, H, D]

            results.append({
                "features": feats_arr.cpu().numpy(),
                "token_ids": tokens,
                "answer": answer
            })

    return results
