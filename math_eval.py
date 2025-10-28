import os
import json
import pickle
import random
import warnings
import gc
from itertools import product

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
from utils import *

# %%
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

transformers.logging.set_verbosity_error()

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# %%
MODELS = [
    "/project/jevans/tzhang3/models/Llama-2-7b-chat-hf",
    "/project/jevans/tzhang3/models/Llama-3.1-8B-Instruct",
    "/project/jevans/tzhang3/models/Qwen2.5-7B-Instruct",
    "/project/jevans/tzhang3/models/Qwen2.5-14B-Instruct"
]

# %% [markdown]
# ## MATH

# %%
from grading.grader import grade_answer

# %%
# ===============================================
# MATH evaluation with DW-vote steering
# ===============================================
import os
import re
import json
import glob
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Dict, Optional, Tuple

# ---------- Config ----------
MATH_PATHS_TRY = [
    "data/math500.jsonl",     # JSONL with {"problem": "...", "solution": "...", "answer": "..."}
]
DW_PREFIX = "politician"        # or "dw_vote" if you want those heads
ALPHAS = [-20, 0, 20]
K_LIST = [32]
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.2
TOP_P = 0.95
SEED = 42
LIMIT = 100

def extract_predicted_answer(text: str) -> Optional[str]:
    """
    Extract model final answer:
    - Prefer \boxed{...} if present
    - Else look for 'Final Answer:' pattern
    - Else fallback to last inline mathy token snippet
    """
    if text is None:
        return None

    # Common explicit pattern
    m = re.search(r"Final Answer:\s*(.+?)\s*$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)

    return None

def _load_math_from_jsonl(path: str) -> List[Dict]:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            # Accept several common field names
            q = obj.get("problem") or obj.get("question") or obj.get("prompt") or ""
            a = obj.get("answer") or obj.get("final_answer") or obj.get("target") or ""
            recs.append({"qid": i, "question": str(q).strip(), "answer": str(a).strip()})
    return recs

def _load_math_from_csv(path: str) -> List[Dict]:
    df = pd.read_csv(path)
    # tolerate different header spellings
    qcol = "problem" if "problem" in df.columns else ("question" if "question" in df.columns else "prompt")
    acol = "answer"  if "answer"  in df.columns else ("final_answer" if "final_answer" in df.columns else None)
    if acol is None:
        raise ValueError("CSV must include an 'answer' (or 'final_answer') column.")
    recs = []
    for i, row in df.iterrows():
        recs.append({"qid": i, "question": str(row[qcol]).strip(), "answer": str(row[acol]).strip()})
    return recs

def _load_math_from_dir(path: str) -> List[Dict]:
    """
    Hendrycks MATH layout often has many JSON files with fields {"problem", "solution", "answer"}.
    """
    files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
    recs = []
    for i, p in enumerate(files):
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            q = obj.get("problem") or obj.get("question") or ""
            a = obj.get("answer") or obj.get("final_answer") or ""
            recs.append({"qid": i, "question": str(q).strip(), "answer": str(a).strip()})
        except Exception:
            continue
    return recs

def load_math_dataset(paths: List[str]) -> List[Dict]:
    """
    Try JSONL, then CSV, then directory-of-JSON files. If nothing is found,
    fall back to a tiny mock set (so the pipeline still runs).
    """
    for p in paths:
        if not os.path.exists(p):
            continue
        if os.path.isdir(p):
            recs = _load_math_from_dir(p)
            if recs:
                print(f"Loaded MATH from dir {p} (N={len(recs)})")
                return recs
        else:
            ext = os.path.splitext(p)[1].lower()
            if ext == ".jsonl":
                recs = _load_math_from_jsonl(p)
                print(f"Loaded MATH from {p} (N={len(recs)})")
                return recs
            elif ext == ".csv":
                recs = _load_math_from_csv(p)
                print(f"Loaded MATH from {p} (N={len(recs)})")
                return recs

    print("MATH file not found. Using a tiny mock dataset (2 items) for a dry run.")
    return [
        {"qid": 0, "question": "Compute 7^2 - 4^2.", "answer": r"\boxed{33}"},
        {"qid": 1, "question": "What is \\frac{3}{4} + \\frac{5}{6}? Give your answer as a fraction.", "answer": r"\boxed{\frac{19}{12}}"},
    ]

# ---------- Prompting ----------
def build_math_chat_prompt(tokenizer, question: str) -> str:
    """
    Chat-formatted prompt to elicit one final line as \\boxed{...}.
    """
    messages = [
        {"role": "system",
         "content": ("You are a careful competition mathematician. Solve step by step, "
                     "then put the final result on the last line as:\n\nFinal Answer: {answer}\n")},
        {"role": "user",
         "content": (f"{question}")}
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    sys = messages[0]["content"]
    usr = messages[1]["content"]
    return f"<|system|>\n{sys}\n<|user|>\n{usr}\n<|assistant|>\n"

# ---------- Generation ----------
def answer_batch_unsteered(model, tokenizer, problems: List[str], max_new_tokens=512) -> List[str]:
    import torch
    outs = []
    eos_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id > 0:
            eos_ids.append(eot_id)
    except Exception:
        pass

    model.eval()
    with torch.no_grad():
        for q in problems:
            prompt = build_math_chat_prompt(tokenizer, q)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            text = text.split("Final Answer: {answer}")[-1]
            outs.append(text.strip())
    return outs

def answer_batch_steered_dw(model, tokenizer, problems: List[str], combined_coefs, alpha: float, max_new_tokens=512) -> List[str]:
    outs = []
    for q in problems:
        prompt = build_math_chat_prompt(tokenizer, q)
        res = generate_with_head_intervention_gpu(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            alpha=alpha,
            max_new_tokens=max_new_tokens,
            combined_coefs=combined_coefs,
            return_features=False,
            device=model.device if hasattr(model, "device") else None,
        )
        text = res[0]["answer"]
        text = text.split("Final Answer: {answer}")[-1]
        outs.append(text)
    return outs

# ---------- Eval ----------
def evaluate_run(gold: List[str], outputs: List[str]) -> Dict:
    preds = [extract_predicted_answer(t) for t in outputs]
    correct = [grade_answer(g, p) for g, p in zip(gold, preds)]
    for i, t in enumerate(outputs):
        print(preds[i], gold[i], correct[i])

    df = pd.DataFrame({"pred": preds, "gold": gold, "correct": correct, "output": outputs})
    df.to_csv("./results.csv")

    return {
        "n": len(gold),
        "correct": int(sum(correct)),
        "acc": float(np.mean(correct)) if len(correct) else 0.0,
        "preds": preds,
        "raw": outputs,
        "is_correct": correct,
    }

# ---------- Experiment ----------
def run_math_dw_compare(model_path: str,
                        math_records: List[Dict],
                        ks: List[int] = K_LIST,
                        alphas: List[int] = ALPHAS) -> pd.DataFrame:
    """
    Baseline vs DW-steered evaluation on MATH.
    Saves CSV and a simple matplotlib plot under ./results/<model_base>/
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_tokenizer_and_model(model_path, device=device)
    base_name = model_base_name(model_path)

    out_dir = os.path.join("results", base_name)
    os.makedirs(out_dir, exist_ok=True)

    questions = [r["question"] for r in math_records]
    answers   = [r["answer"]   for r in math_records]

    # Load ridge/perf (DW_PREFIX can be 'politician' or 'dw_vote')
    print("\n[DW-steering] Loading ridge models & performance...")
    # Use feats shortcut if you have them; else k+performance path works too
    try:
        feats = load_feats(model_path=model_path, prefix=DW_PREFIX, results_dir="./results", device=device)
    except Exception:
        feats = None
    ridge_models, performance = load_ridge_models(model_path, DW_PREFIX)

    rows = []
    for k in ks:
        print(f"\nBuilding combined_coefs for k={k} ...")
        coefs = compute_combined_coefs(
            feats=feats,
            ridge_models=ridge_models,
            k=k,
            performance=performance,
            device=model.device if hasattr(model, "device") else None,
        )

        for alpha in alphas:
            print(f"[DW] alpha={alpha}, k={k} → generating ...")
            t0 = time.time()
            steered_text = answer_batch_steered_dw(
                model, tokenizer, questions,
                combined_coefs=coefs,
                alpha=alpha,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            ev = evaluate_run(answers, steered_text)
            dt = time.time() - t0
            print(f"   acc={ev['acc']:.3f} ({ev['correct']}/{ev['n']})  time={dt:.1f}s")
            rows.append({
                "model": base_name,
                "mode": "steered_dw",
                "alpha": alpha,
                "k": k,
                "acc": ev["acc"],
                "correct": ev["correct"],
                "n": ev["n"],
            })

    # Baseline
    print("\n[Baseline] Generating without steering...")
    t0 = time.time()
    base_text = answer_batch_unsteered(model, tokenizer, questions, max_new_tokens=MAX_NEW_TOKENS)
    base_eval = evaluate_run(answers, base_text)
    print(f"Baseline MATH acc = {base_eval['acc']:.3f} ({base_eval['correct']}/{base_eval['n']})  time={time.time()-t0:.1f}s")

    rows.append({
        "model": base_name,
        "mode": "baseline",
        "alpha": 0,
        "k": 0,
        "acc": base_eval["acc"],
        "correct": base_eval["correct"],
        "n": base_eval["n"],
    })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "math_dw_vote_eval.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV → {csv_path}")

    # Plot (single figure, default colors)
    fig = plt.figure(figsize=(7, 4.5))
    plt.axhline(y=base_eval["acc"], linestyle="--", label=f"Baseline acc = {base_eval['acc']:.3f}")
    for k in sorted(set(df[df["mode"] == "steered_dw"]["k"])):
        sub = df[(df["mode"] == "steered_dw") & (df["k"] == k)].sort_values("alpha")
        plt.plot(sub["alpha"].tolist(), sub["acc"].tolist(), marker="o", label=f"Steered (k={k})")
    plt.xlabel("alpha (DW-vote steering strength)")
    plt.ylabel("MATH accuracy")
    plt.title(f"MATH: Baseline vs DW-steered • {base_name}")
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(out_dir, "math_dw_vote_eval.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot PNG → {png_path}")

    del model
    clean_up(device=None)
    return df

def run_all_models_on_math(models: List[str] = None):
    recs = load_math_dataset(MATH_PATHS_TRY)[:LIMIT]
    used = models if models is not None else MODELS
    outs = []
    for m in used:
        print(f"\n===== Evaluating {m} on MATH =====")
        df = run_math_dw_compare(model_path=m, math_records=recs, ks=K_LIST, alphas=ALPHAS)
        outs.append(df.assign(model_path=m))
    big = pd.concat(outs, ignore_index=True)
    out_csv = "./results/math_dw_vote_eval_all_models.csv"
    os.makedirs("./results", exist_ok=True)
    big.to_csv(out_csv, index=False)
    print(f"\nWrote combined results: {out_csv}")
    return big

# ---------- Example calls ----------
# big_df = run_all_models_on_math(MODELS)
df_single = run_math_dw_compare(MODELS[1], load_math_dataset(MATH_PATHS_TRY)[:LIMIT], ks=[32, 64], alphas=[-20, -10, 10, 20])
