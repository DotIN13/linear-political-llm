# Probes Module

This folder contains three probe classes with a shared API:

- `HeadwiseLinearProbe`
- `LayerwiseLinearProbe`
- `LayerwiseRFM`

## Shared API

Each class exposes the same primary methods:

1. `fit(model, prompts, labels, tokenizer=None, device=None)`
2. `save()`
3. `load()`
4. `build_steering(features, k=...)`
5. `score_samples(features, k=...)`

`prompts` can be either:

- plain strings (text mode, requires `tokenizer`), or
- pre-tokenized dictionaries from `processor.apply_chat_template(..., tokenize=True, return_dict=True)` for vision/combined settings.

Model parsing and feature extraction are implemented directly in `probes` (no dependency on `utils.py`).

## Model-specific module paths

All probe classes accept an optional `module_paths` value in `__init__` for model-specific internals.

Default module path presets are stored in `DEFAULT_MODULE_PATHS` as `ModulePaths` dataclasses for:

- `text`
- `vision`
- `qwen3-vl`

You can either pass a full `ModulePaths` object or a dictionary of field overrides.

Supported keys:

- `layer_prefix`: path to the transformer layer list (e.g. `model.layers`)
- `head_out_suffix`: module suffix under each layer (default `self_attn.head_out`)
- `head_module_template`: full template string with `{layer_idx}` (overrides prefix/suffix)
- `text_config_path`: path to text config for VLMs (default `config.text_config`)

Example:

```python
from dataclasses import replace

from probes import DEFAULT_MODULE_PATHS, LayerwiseRFM

probe = LayerwiseRFM(
	model_path=MODEL_PATH,
	prefix="combined_ideology",
	mode="vision",
	module_paths=replace(
		DEFAULT_MODULE_PATHS["vision"],
		head_out_suffix="self_attn.head_out",
	),
)
```

Or with a lightweight override dictionary:

```python
probe = LayerwiseRFM(
	model_path=MODEL_PATH,
	prefix="combined_ideology",
	mode="vision",
	module_paths={"head_module_template": "model.language_model.layers.{layer_idx}.self_attn.head_out"},
)
```

## Artifact format

All probes save into:

`results/probes/<model_base_name>/`

With consistent names:

- `<prefix>_<probe_type>_weights.pkl`
- `<prefix>_<probe_type>_scores.npy`
- `<prefix>_<probe_type>_metadata.json`

`prefix` should represent your dimension, for example `textual_ideology` or `combined_ideology`.
