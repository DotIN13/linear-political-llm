import math
import numpy as np
import torch
import pygad
from functools import lru_cache
from scipy.stats import pearsonr
import pandas as pd
import os
import json
import pickle
import random
import warnings
import gc
from itertools import product
import pickle
from typing import Union, List, Dict, Optional

import torch
from einops import rearrange
from baukit import TraceDict
import numpy as np

import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from einops import rearrange
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm
from IPython.display import HTML, display

from baukit import Trace, TraceDict

warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

transformers.logging.set_verbosity_error()


def set_seed(seed: int = 42):
    """
    Set seeds and configurations for reproducibility in
    random, numpy, pandas (via numpy), and PyTorch.
    
    Args:
        seed (int): Random seed value.
    """

    # Python's built-in random
    random.seed(seed)

    # NumPy (Pandas uses NumPy under the hood)
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set seed {seed}")

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

def model_base_name(model_path):
    return model_path.split("/")[-1].lower()

def load_tokenizer_and_model(model_path, device=None, hf_token=None):
    """
    Loads a tokenizer and model for causal LM generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_auth_token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        use_auth_token=hf_token
    ).to(device).eval()
    return tokenizer, model


def extract_attention_head_activations(model, prompts, device=None):
    """
    Extracts prefill attention head activations for a list of prompts.
    """
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    head_wise_hidden_states_list = []

    for prompt in tqdm(prompts, total=len(prompts)):
        with torch.no_grad():
            with TraceDict(model, HEADS) as ret:
                output = model(prompt.to(device))
                head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
                head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
                head_wise_hidden_states_list.append(head_wise_hidden_states[:, :, :])

    features = [
        [states[:, -1, :]] for states in head_wise_hidden_states_list
    ]
    return np.stack(features, axis=0)


def extract_and_save_features(model, tokenizer, model_path, df_prompts, output_dir="results", prefix="politician", device=None):
    """
    Given a DataFrame with columns ['prompt', 'label'], extract attention features and save to pickle.
    """
    # Tokenize all prompts
    encoded = [tokenizer(p, return_tensors='pt')['input_ids'].to(device) for p in df_prompts['prompt']]
    labels = df_prompts['label'].values

    # Extract features
    features = extract_attention_head_activations(model, encoded, device=device)

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
    model_path,
    data_dir="results",
    prefix="politician",
    alpha=1.0,
    n_splits=2,
    seed=SEED
):
    # Load features and labels
    base_name = model_base_name(model_path)
    data_path = os.path.join(data_dir, base_name, f"{prefix}_features.pkl")
    with open(data_path, 'rb') as f:
        features, labels = pickle.load(f)

    # Initialize containers
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    performance = np.zeros((n_layers, n_heads))
    ridge_dict = {}

    # Train/test per attention head
    for i in tqdm(range(n_layers), desc="Layers"):
        ridge_dict[i] = {}
        for j in range(n_heads):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for train_idx, test_idx in kf.split(range(features.shape[0])):
                X_train = features[train_idx, 0, i, j, :]
                X_test = features[test_idx, 0, i, j, :]
                y_train = np.array(labels)[train_idx]
                y_test = np.array(labels)[test_idx]

                model_ridge = Ridge(alpha=alpha, fit_intercept=False)
                model_ridge.fit(X_train, y_train)
                ridge_dict[i][j] = model_ridge

                y_pred = model_ridge.predict(X_test)
                performance[i, j] += spearmanr(y_test, y_pred).statistic

    # Average across folds
    performance /= n_splits

    # Save results
    base_dir = os.path.join(data_dir, base_name)
    os.makedirs(base_dir, exist_ok=True)
    pickle.dump(performance, open(os.path.join(base_dir, f"{prefix}_performance.pkl"), 'wb'))
    pickle.dump(ridge_dict, open(os.path.join(base_dir, f"{prefix}_ridge.pkl"), 'wb'))

    return performance, ridge_dict


def clean_up(device=None):
    """
    Clean up the model and GPU memory.
    """
    gc.collect()
    if device:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def extract_head_out_per_token(
    model, tokenizer, prompts,
    candidate_answers=None,
    max_new_tokens=32,
    output_only=False,
    device=None):
    """
    Extracts attention head outputs per token and optionally probabilities for candidate answers.

    Args:
        model: HuggingFace transformer model
        tokenizer: Corresponding tokenizer
        prompts: List of prompt strings
        candidate_answers: List of string candidates (e.g., ["1", "2", ..., "7"]) or None
        max_new_tokens: Number of tokens to generate

    Returns:
        List of dictionaries, each with:
            - 'features': np.array (T, L, H, D)
            - 'token_ids': np.array of generated token IDs
            - 'answer': decoded generated string
            - 'candidate_logits': np.array (len(candidate_answers),) if applicable
            - 'candidate_probs': np.array (len(candidate_answers),) if applicable
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads

    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(num_layers)]

    # Prepare candidate token IDs
    candidate_token_ids = None
    if candidate_answers is not None:
        candidate_token_ids = [tokenizer.convert_tokens_to_ids(ans) for ans in candidate_answers]
        candidate_token_ids = torch.tensor(candidate_token_ids, device=device)

    results = []

    for prompt in tqdm(prompts):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

        token_outputs_by_layer = [[] for _ in range(num_layers)]

        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                if output_only:
                    token_outputs_by_layer[layer_idx].append(out.squeeze(0)[-1].detach().cpu())
                else:
                    token_outputs_by_layer[layer_idx].extend(out.squeeze(0).detach().cpu())
            return hook_fn

        # Register hooks
        hooks = []
        for i, name in enumerate(HEADS):
            module = dict(model.named_modules())[name]
            hooks.append(module.register_forward_hook(make_hook(i)))

        # Generate tokens and optionally return scores
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=(candidate_answers is not None),
                output_attentions=False
            )

        for h in hooks:
            h.remove()

        # Token output
        generated_ids = outputs.sequences.squeeze(0).cpu().numpy()
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Attention head features
        layerwise_outputs = []
        for layer_outputs in token_outputs_by_layer:
            stacked = torch.stack(layer_outputs, dim=0)  # (T, hidden_size)
            reshaped = stacked.view(stacked.size(0), num_heads, head_dim)  # (T, H, D)
            layerwise_outputs.append(reshaped)

        layerwise_outputs = torch.stack(layerwise_outputs, dim=0)  # (L, T, H, D)
        layerwise_outputs = rearrange(layerwise_outputs, 'l t h d -> t l h d')  # (T, L, H, D)

        # Package result
        result = {
            "features": layerwise_outputs.numpy(),  # (T, L, H, D)
            "token_ids": generated_ids,
            "answer": decoded,
        }

        # Extract candidate logits & probs
        if candidate_token_ids is not None and len(outputs.scores) > 0:
            first_step_logits = outputs.scores[0].squeeze(0)  # (vocab_size,)
            probs = F.softmax(first_step_logits, dim=-1)  # (vocab_size,)
            candidate_logits = first_step_logits[candidate_token_ids].cpu().numpy()
            candidate_probs = probs[candidate_token_ids].cpu().numpy()
            result["candidate_logits"] = candidate_logits
            result["candidate_probs"] = candidate_probs

        results.append(result)

    return results


def get_top_indices(performance, k=8):
    # Top-k (i, j) heads
    return np.dstack(np.unravel_index(np.argsort(performance.ravel()), (32, 32)))[0][-k:, :]


def predict_per_token_scores(results, ridge_models, performance, k):
    """
    Predict per-token scores for voter samples with variable-length inputs.

    Args:
        results: List of dictionaries with keys 'features' and 'token_ids'
        models: dict mapping (i,j) -> trained Ridge model
        performance: dict mapping (i,j) -> performance score
        k: int, number of top ridge models to use

    Returns:
        scores: List[np.ndarray] where each element has shape (T,)
    """
    top_indices = get_top_indices(performance, k=k)

    for res in results:  # sample_features: (T, L, H, D)
        sample_features = res['features']  # Shape: (T, L, H, D)
        T, L, H, D = sample_features.shape
        scores = np.zeros(T)

        for t in range(T):
            for i, j in top_indices:
                model = ridge_models[i][j]
                x = sample_features[t, i, j, :]  # Shape: (D,)
                scores[t] += model.predict(x.reshape(1, -1))[0]

        scores /= len(top_indices)
        res['scores'] = scores

    return results


def visualize_token_scores(results, tokenizer, title_prefix="Prompt", vmin=-1.0, vmax=1.0):
    """
    Displays tokens with per-token scores using color-coding.

    Args:
        results (List[Dict]): List of dictionaries with keys 'token_ids' and 'features'.
        scores (np.ndarray): Array of shape (N_prompts, T) with score values per token.
        tokenizer (transformers.PreTrainedTokenizer): HuggingFace tokenizer to decode tokens.
        title_prefix (str): Optional prefix for prompt titles.
    """    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('RdBu_r')

    for idx, res in enumerate(results):
        ids = res['token_ids']
        score_arr = res['scores']
        tokens = tokenizer.convert_ids_to_tokens(ids)
        spans = []
        for tok, sc in zip(tokens[1:], score_arr[: len(tokens)]):  # skip BOS token
            color = mpl.colors.to_hex(cmap(norm(sc)))
            safe_tok = (
                tok.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('Ġ', ' ')  # optional: replace GPT-style space token if BPE
            )
            spans.append(
                f"<span style='background-color:{color};padding:2px;margin:1px;border-radius:3px;color:#313131;'>{safe_tok}</span>"
            )
        html = (
            '<div style="line-height:1.6;font-family:monospace;">'
            + ' '.join(spans)
            + '</div>'
        )
        display(HTML(f"<h4>{title_prefix} {idx + 1}:</h4>{html}"))
        

def plot_candidate_probabilities(results, candidate_answers, title_prefix="Prompt", figsize=(8, 4)):
    """
    Visualizes the candidate probability distributions for each prompt result.

    Args:
        results: List of result dictionaries returned by extract_head_out_per_token
        candidate_answers: List of string labels corresponding to candidate tokens
        title_prefix: Prefix used in subplot titles (e.g., "Prompt 1", "Prompt 2", ...)
        figsize: Size of each subplot figure
    """
    num_plots = len(results)

    for idx, result in enumerate(results):
        if "candidate_probs" not in result:
            print(f"Skipping prompt {idx+1}: no candidate_probs found.")
            continue

        probs = result["candidate_probs"]
        decoded_answer = result["answer"]

        plt.figure(figsize=figsize)
        bars = plt.bar(candidate_answers, probs)

        plt.title(f"{title_prefix} {idx + 1}")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.xticks(rotation=0)

        # Annotate bar values
        for bar, prob in zip(bars, probs):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{prob:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.show()


def load_ridge_models(model_path, prefix):
    """
    Load ridge models and performance scores.
    """
    base_name = model_base_name(model_path)
    performance = pickle.load(open(os.path.join("results", base_name, f"{prefix}_performance.pkl"), 'rb'))
    ridge_models = pickle.load(open(os.path.join("results", base_name, f"{prefix}_ridge.pkl"), 'rb'))
    
    return ridge_models, performance


def generate_and_score_tokens(
    model_path,
    prompts,
    model=None,
    tokenizer=None,
    max_new_tokens=512,
    hf_token=None,
    k=8,
    candidate_answers=None,
    output_only=False,
    ridge_prefix=None,
    visualize=False,
    device=None
):
    """
    Generate and visualize token-level bias scores for a list of prompts.

    Args:
        model_path (str): Name of the HuggingFace model.
        prompts (List[str]): List of user prompts.
        max_new_tokens (int): Max tokens to generate during inference.
        hf_token (str): HuggingFace token if authentication is needed.
        k (int): Number of top ridge models to use.
    """
    delete_model = False
    
    if model and tokenizer:
        model.to(device)
    else:
        tokenizer, model = load_tokenizer_and_model(model_path, device=device, hf_token=hf_token)
        delete_model = True

    # Step 1: Extract features
    results = extract_head_out_per_token(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        candidate_answers=candidate_answers,
        output_only=output_only,
        device=device
    )

    ridge_models, performance = load_ridge_models(model_path, ridge_prefix)

    # Step 4: Predict token-level scores
    results = predict_per_token_scores(results, ridge_models, performance, k)

    if delete_model:
        del model
        clean_up(device=device)

    if visualize:
        visualize_token_scores(
            results, tokenizer
        )

    return results


def probe_heads(
    model_path,
    df_prompts,
    prefix="politician",
    device="cpu",
    alpha=1.0,
    n_splits=2,
    seed=42,
    data_dir="results",
    figsize=(10, 8),
    dpi=150,
    cmap='viridis',
    vmin=0,
    vmax=100
):
    """
    Loads a model, extracts features, trains ridge models, and plots a heatmap of head performance.

    Args:
        model_path (str): Path to the model directory.
        df_prompts (pd.DataFrame): DataFrame of prompts.
        prefix (str): Prefix for saving and loading features/models.
        device (str): Device specifier for model (e.g., 'cpu' or 'cuda').
        alpha (float): Regularization strength for ridge regression.
        n_splits (int): Number of cross-validation splits.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory for saving results.
        figsize (tuple): Figure size for the plot.
        dpi (int): Resolution of the figure.
        cmap (str): Colormap for the heatmap.
        vmin (int): Minimum value for color scaling.
        vmax (int): Maximum value for color scaling.

    Returns:
        ax: Matplotlib Axes object of the heatmap.
        performance (np.ndarray): Raw performance array [layers x heads].
        ridge_models (dict): Trained ridge models.
    """
    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(model_path, device=device)

    # Extract features and save to disk
    extract_and_save_features(
        model,
        tokenizer,
        model_path,
        df_prompts,
        prefix=prefix,
        device=device
    )

    # Train ridge models
    _perf, _models = train_ridge_models(
        model,
        model_path,
        data_dir=data_dir,
        prefix=prefix,
        alpha=alpha,
        n_splits=n_splits,
        seed=seed
    )

    del model
    clean_up(device=device)

    # Load trained models and performance metrics
    ridge_models, performance = load_ridge_models(model_path, prefix)

    # Prepare performance for plotting
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
    device: torch.device = None,
) -> torch.LongTensor:
    """
    Return the indices of the top-k heads by performance.

    Args:
      performance: [L, H] (np or torch)
      k: number of heads to pick
      device: torch device

    Returns:
      top_idx: LongTensor of shape [k, 2] with (layer_index, head_index)
    """
    perf = torch.as_tensor(performance, device=device)  # [L, H]
    L, H = perf.shape

    flat = perf.view(-1)
    k = min(k, flat.numel())
    topk_flat = torch.argsort(flat, descending=True)[:k]  # [k]

    # use H from performance.shape instead of passing explicitly
    top_idx = torch.stack([topk_flat // H, topk_flat % H], dim=1)  # [k, 2]
    return top_idx


def build_combined_coefs_from_indices(
    feats: torch.Tensor,             # [N, T, L, H, D]
    ridge_models: dict,              # ridge_models[layer][head].coef_
    top_idx: torch.LongTensor,       # [k, 2] (li, hi)
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build [L, H, D] combined coefficients:
      - infer (L,H,D) from feats
      - compute per-head std at timestep 0 from feats
      - fill ridge coefs for selected heads
      - return ridge * std
    """
    assert feats.dim() == 5, "feats must be [N, T, L, H, D]"
    # Ensure feats on device (if provided)
    if device is not None and feats.device != device:
        feats = feats.to(device)

    # Get dims from feats
    _, _, L, H, D = feats.shape

    # Per-head std at timestep 0 → [L, H, D]
    std_tensor = torch.std(feats[:, 0], dim=0)  # [L, H, D]

    # Assemble ridge coefs
    ridge_t = torch.zeros((L, H, D), device=feats.device, dtype=std_tensor.dtype)
    for li, hi in top_idx.tolist():
        coef = ridge_models[li][hi].coef_.astype(np.float32)  # [D]
        ridge_t[li, hi] = torch.as_tensor(coef, device=feats.device)

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

    Args:
      feats: [N, T, L, H, D] (torch or numpy)
      ridge_models: dict of ridge models per layer/head
      top_idx: optional LongTensor [[li, hi], ...]
      k: number of heads to select (used if top_idx is None)
      performance: [L, H] scores (used if top_idx is None)
      device: torch device

    Returns:
      combined_coefs: [L, H, D] tensor on device (if provided)
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
    combined_coefs: torch.Tensor = None,
    return_features: bool = False,
    device: torch.device = None
) -> Union[List[str], List[Dict]]:
    """
    Fully‐GPU head‐intervention + full‐layer feature capture on one or many prompts,
    with a single hook registration.

    Args:
      model, tokenizer: HF model + tokenizer
      prompts: single str or list of str
      alpha: intervention strength
      max_new_tokens: generation length
      combined_coefs: precomputed [L, H, D] tensor
      return_features: if True returns a list of dicts with "features","token_ids","answer"
      device: torch.device, defaults to cuda

    Returns:
      List[str] (if return_features=False) or List[dict] (if True)
    """
    # normalize prompts to list
    if isinstance(prompts, str):
        prompts = [prompts]

    if device is None:
        device = torch.device("cuda")
    model = model.to(device)

    # dims
    L, H, D = combined_coefs.shape

    # prepare one shared captured‐buffer
    captured = {li: [] for li in range(L)}

    # hook function
    def hook_fn(output, module_name):
        # output: [B, S, H, D]
        B, S, _, _ = output.shape
        h = output
        li = int(module_name.split('.')[2])
        # apply intervention on last token
        h[:, -1] += alpha * combined_coefs[li]
        # capture all tokens (you can restrict to last if you prefer)
        captured[li].extend(h[:, :].detach().cpu().squeeze(0))
        return h

    # hook every layer once
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(L)]
    results = []

    with TraceDict(model, HEADS, edit_output=hook_fn):
        for prompt in prompts:
            # clear previous captures
            for li in range(L):
                captured[li].clear()

            # generate
            inp = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            out = model.generate(inp, max_new_tokens=max_new_tokens, do_sample=False)

            tokens = out[0].cpu().numpy()
            answer = tokenizer.decode(out[0], skip_special_tokens=True).strip()

            result = {"answer": answer}

            if return_features:
                T = len(captured[0])
                # build [T, L, H, D]
                layer_feats = [ torch.stack(captured[li], dim=0) for li in range(L) ]  # each [T, H, D]
                feats_arr   = torch.stack(layer_feats, dim=0).permute(1, 0, 2, 3)      # → [T, L, H, D]
                result = {
                    "features": feats_arr.cpu().numpy(),
                    "token_ids": tokens,
                    "answer":   answer
                }
            
            results.append(result)

    return results


MODELS = [
    "/project/jevans/tzhang3/models/Llama-2-7b-chat-hf",
    "/project/jevans/tzhang3/models/Llama-3.1-8B-Instruct",
    "/project/jevans/tzhang3/models/Qwen2.5-7B-Instruct",
    "/project/jevans/tzhang3/models/Qwen2.5-14B-Instruct"
]

def build_prompt(query):
    return (
        f"USER: In the context of 2019 U.S. politics, is the following statement biased and leaning toward a liberal or conservative viewpoint? "
        f"{query.strip()}\n"
        f"ASSISTANT: The statement is"
    )

df_bias = pd.read_csv("data/policy_statements_240.csv")

prompts_bias = []
for _, row in df_bias.iterrows():
    text = row['response']
    prompts_bias.append(build_prompt(text))

len(prompts_bias)

model_path = MODELS[0]
prefix = "politician"
results_dir = "results"
prompts = prompts_bias
max_new_tokens = 12

base_name = model_base_name(model_path)

# 1) Load tokenizer/model once
tokenizer, model = load_tokenizer_and_model(model_path, device=device)

# 2) Load feats, ridge models, and performance once
feats = load_feats(model_path=model_path, prefix=prefix, results_dir=results_dir, device=device)
ridge_models, performance = load_ridge_models(model_path, prefix)

# Assumes you already have these from your setup:
# tokenizer, model
# feats = [N, T, L, H, D] torch tensor (on device)
# ridge_models
# prompts, max_new_tokens
# device = torch.device("cuda") (or cpu)
# IMPORTANT: you need at least 2 alphas for a correlation
alphas_for_fitness = [-30, 0, 30]  # example; customize as you like

# --- Helper & Fitness Functions --------

def _parse_label(answer: str) -> int:
    """Reuse your label parser."""
    answer = answer.split("ASSISTANT", 2)[-1]
    if any(s in answer for s in ["neutral", "neither", "moderate", "centrist", "center", "balanced", "not biased"]):
        return 0
    if any(s in answer for s in ["left", "liberal", "progressive"]):
        return -1 if "slight" in answer else -2
    if any(s in answer for s in ["right", "conservative"]):
        return 1 if "slight" in answer else 2
    return 0

def decode_flat_indices(flat_idxs: np.ndarray, H: int) -> torch.LongTensor:
    """Decode flat indices -> top_idx tensor [[li, hi], ...]."""
    li = flat_idxs // H
    hi = flat_idxs %  H
    pairs = np.stack([li, hi], axis=1)  # [k, 2]
    return torch.as_tensor(pairs, dtype=torch.long, device=device)

@lru_cache(maxsize=256)
def _cached_combined_coefs(chromosome_tuple):
    """
    Cache combined coefs per chromosome to avoid recomputing across alphas.
    The cache key is a tuple of flat head indices.
    """
    flat_indices = np.array(chromosome_tuple, dtype=int)
    _, _, L, H, D = feats.shape
    
    if len(flat_indices) == 0:
        return None

    top_idx = decode_flat_indices(flat_indices, H)
    coefs = compute_combined_coefs(
        feats=feats,
        ridge_models=ridge_models,
        top_idx=top_idx,
        device=device,
    )
    return coefs

def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness function for a chromosome with k flat head indices.
    solution: ndarray of shape [k] with unique flat head indices in [0, L*H).
    Returns a scalar fitness (higher is better).
    """
    # The 'solution' is now an array of k flat indices.
    # The cache key is a tuple of these indices, which is hashable.
    coefs = _cached_combined_coefs(tuple(solution))
    if coefs is None:
        return 0.0

    # For each alpha, run generation and compute avg parsed label
    sample_prompts = random.sample(prompts, k=30)
    avg_by_alpha = []
    for alpha in alphas_for_fitness:
        results = generate_with_head_intervention_gpu(
            model=model,
            tokenizer=tokenizer,
            prompts=sample_prompts,
            alpha=float(alpha),
            max_new_tokens=max_new_tokens,
            combined_coefs=coefs,
            return_features=False,
            device=device,
        )
        labels = [_parse_label(r["answer"]) for r in results]
        print("Labels: ", labels)
        avg_by_alpha.append(float(np.mean(labels)))

    if len(alphas_for_fitness) < 2:
        return 0.0

    # If all y are constant, correlation is undefined; treat as 0
    if np.std(avg_by_alpha) == 0 or np.std(alphas_for_fitness) == 0:
        return 0.0

    try:
        r, _ = pearsonr(alphas_for_fitness, avg_by_alpha)
        if math.isnan(r):
            r = 0.0
        fitness = -float(r)
    except Exception:
        fitness = 0.0

    print(solution)
    print(avg_by_alpha)
    print(solution_idx, float(r))
    return fitness

# --- Main GA Runner (Reverted to fixed k) -----------------------------------

def on_gen(ga_instance):
    """Callback function to show progress after each generation."""
    print(f"Generation {ga_instance.generations_completed:3d} | Best Fitness = {ga_instance.best_solution()[1]:.4f}")

def run_ga_for_top_idx(k: int, num_generations: int = 20, sol_per_pop: int = 20, num_parents_mating: int = 10):
    """
    Evolves k head indices that maximize |corr(alpha, avg_label)|.
    Returns (ga_instance, best_solution_flat, best_fitness, decoded_top_idx_tensor).
    """
    _, _, L, H, _ = feats.shape
    gene_space = list(range(L * H))  # All possible flat head indices

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=k,
        gene_space=gene_space,
        gene_type=int,
        allow_duplicate_genes=False, # Crucial: ensure unique heads
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_num_genes=max(1, k // 5), # Mutate ~20% of genes
        on_generation=on_gen,
    )

    print("--- Starting Genetic Algorithm ---")
    ga_instance.run()
    print("\n--- Genetic Algorithm Finished ---")
    
    best_solution_flat, best_fitness, _ = ga_instance.best_solution()
    best_solution_flat = np.asarray(best_solution_flat, dtype=int)

    # Decode the best flat indices to get the (li, hi) tensor
    top_idx_tensor = decode_flat_indices(best_solution_flat, H)

    return ga_instance, best_solution_flat, best_fitness, top_idx_tensor

# ---------------------- Example usage ----------------------------------------
k_opt = 32 # Set the number of heads to optimize for

ga, best_flat, best_fit, best_top_idx = run_ga_for_top_idx(
    k=k_opt,
    num_generations=15,
    sol_per_pop=24,
    num_parents_mating=12,
)

print(f"\nGA found an optimal set of {len(best_flat)} heads.")
print(f"Best fitness (|pearson r|): {best_fit:.4f}")
print("Best flat head indices:", best_flat.tolist())
print("Best top_idx (li, hi):", best_top_idx.tolist())
