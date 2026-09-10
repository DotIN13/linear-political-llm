# linear-political-llm

Linear probes on political images and language-model outputs.

## Dependency

The required Python packages are:
- `Python 3.10.13`
  - `baukit==0.0.1`
  - `einops==0.8.0`
  - `scipy==1.12.0`
  - `tqdm==4.66.2`
  - `transformers==4.38.2`
  - `matplotlib==3.8.3`
  - `numpy==1.26.4`
  - `pandas==2.2.1`
  - `pyvene==0.1.1`
  - `torch==2.3.0`
  - `seaborn==0.13.2`

Install the shared library in editable mode so `lpl` imports work from anywhere:

```
pip install -e .
```

## Layout

- `lpl/` — the shared library (`utils.py`, `ga.py`, `math_eval.py`, `grading/`).
- `bench_v2/` — the active experiment harness (pilot-first; each task self-contained).
- `probes/` — probe trainings and the vendored `rfm/` toolkit.
- `scripts/` — runnable scripts: `data_gen/`, `downstream/`, `probes/`, `media/`, `sam3/`.
- `notebooks/` — exploratory notebooks (`replication_results.ipynb` replicates the draft).
- `app/` — Gradio app.
- `data/` — committed data products.
- `figs/`, `artifacts/` — figures and writeups.
- `runs/`, `items/`, `conversations/`, `judge_cache/` — uncommitted run outputs.

## Tests

```
pytest
```

## Files

- `notebooks/replication_results.ipynb`: Code to replicate the results in the draft.
- `scripts/data_gen/create_statements.py`: Code to create the statements used in the experiments.
- `lpl/utils.py`: Attention-head access, probing and scoring helpers.
- `data/`: Data for experiments.
