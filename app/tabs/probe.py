"""Interactive probe tab — model loading, image scoring, heatmap, text generation."""

import os
from datetime import datetime
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from app import state as _app_state
from app.common import MAX_IMAGE_DIM, RESULTS_DIR, make_heatmap_overlay
from app.state import AppState, MODEL_REGISTRY

AVAILABLE_MODELS = sorted(MODEL_REGISTRY.keys())
AVAILABLE_PREFIXES = ["combined_ideology", "textual_ideology"]
AVAILABLE_PROBES = ["headwise_linear", "layerwise_linear"]

PROMPT_TEMPLATES = {
    "News article": (
        "write a possible news article that would best accompany this image, from the best "
        "possible media outlet. Use your knowledge about American political culture and the "
        "media landscape in doing so. If the image seems slanted, write a slanted article. "
        "Do not worry about neutrality: write the article in the style of the media outlet "
        "most likely to have used this image."
    ),
    "Twitter analysis": (
        "Analyze this image. What political perspective or bias might be present? "
        "If this image were shared on Twitter, what political leaning would be most likely to share it? "
        "Consider visual elements, context, and any accompanying sentiment."
    ),
    "Congress position": "What's his/her position on US politics?",
    "Simple describe": "Describe this image.",
}


def compute_grid_hw(state: AppState, thumb_h: int, thumb_w: int, encoded) -> Tuple[int, int]:
    """Compute vision grid dimensions for the current model."""
    if state.model_family == "qwen3-vl":
        image_processor = state.processor.image_processor
        min_pixels = image_processor.size["shortest_edge"]
        max_pixels = image_processor.size["longest_edge"]
        patch = image_processor.patch_size
        merge = image_processor.merge_size
        factor = patch * merge
        resized_h, resized_w = smart_resize(
            thumb_h, thumb_w, factor, min_pixels=min_pixels, max_pixels=max_pixels,
        )
        grid_h = (resized_h // patch) // merge
        grid_w = (resized_w // patch) // merge
        return int(grid_h), int(grid_w)

    if state.model_family == "gemma4":
        image_pos = encoded.get("image_position_ids")
        if image_pos is None:
            return -1, -1
        pos0 = image_pos[0]
        valid = (pos0[:, 0] >= 0) & (pos0[:, 1] >= 0)
        if not valid.any():
            return 0, 0
        patch_grid_w = int(pos0[valid, 0].max().item() + 1)
        patch_grid_h = int(pos0[valid, 1].max().item() + 1)
        pool_k = getattr(state.processor.image_processor, "pooling_kernel_size", 1)
        grid_w = patch_grid_w // int(pool_k)
        grid_h = patch_grid_h // int(pool_k)
        return int(grid_h), int(grid_w)

    return -1, -1


def build_messages(pil_image: Image.Image, prompt_text: str) -> list:
    img_copy = pil_image.copy()
    img_copy.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
    if img_copy.mode in {"RGBA", "LA", "P"}:
        img_copy = img_copy.convert("RGB")
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text.strip() or "Describe this image."},
            {"type": "image", "image": img_copy},
        ],
    }]


def score_image(
    pil_image: Image.Image,
    state: AppState,
    runtime: dict,
    prompt_text: str = "Describe this image.",
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
    model = state.model
    processor = state.processor

    img_copy = pil_image.copy()
    img_copy.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
    if img_copy.mode in {"RGBA", "LA", "P"}:
        img_copy = img_copy.convert("RGB")

    thumb_h, thumb_w = img_copy.height, img_copy.width

    messages = build_messages(pil_image, prompt_text)

    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    try:
        model_device = model.device
    except AttributeError:
        model_device = next(model.parameters()).device

    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}

    named_modules = dict(model.named_modules())
    captured: dict = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(_module, _inp, out):
            tensor_out = out[0] if isinstance(out, tuple) else out
            captured[name] = tensor_out.detach().to(dtype=torch.float32).cpu()
        return hook_fn

    for module_name in runtime["module_names"]:
        if module_name in named_modules:
            hooks.append(named_modules[module_name].register_forward_hook(make_hook(module_name)))

    try:
        with torch.inference_mode():
            _ = model(**inputs)
    finally:
        for hook in hooks:
            hook.remove()

    probe_type = runtime["probe_type"]
    normalizer = runtime["normalizer"]
    token_scores = None

    if probe_type == "headwise_linear":
        for module_name, head_entries in runtime["groups"].items():
            tensor_out = captured[module_name]
            module_score = None
            for head_idx, coef in head_entries:
                coef_t = coef.to(device=tensor_out.device, dtype=tensor_out.dtype)
                head_scores = torch.einsum("b s d, d -> b s", tensor_out[:, :, head_idx, :], coef_t)
                module_score = head_scores if module_score is None else module_score + head_scores
            token_scores = module_score if token_scores is None else token_scores + module_score

    elif probe_type == "layerwise_linear":
        for module_name, vec in runtime["groups"].items():
            tensor_out = captured[module_name]
            flat = tensor_out.reshape(tensor_out.size(0), tensor_out.size(1), -1) if tensor_out.dim() == 4 else tensor_out
            vec_t = vec.to(device=flat.device, dtype=flat.dtype)
            layer_scores = torch.einsum("b s d, d -> b s", flat, vec_t)
            token_scores = layer_scores if token_scores is None else token_scores + layer_scores

    if token_scores is None:
        raise RuntimeError("Failed to compute token scores.")

    token_scores = (token_scores / normalizer).to(dtype=torch.float32).cpu().numpy()
    input_ids = encoded["input_ids"].cpu().numpy().astype(np.int64)

    image_token_ids = set()
    tokenizer = processor.tokenizer
    for attr in ("image_token_id", "image_pad_token_id"):
        if hasattr(tokenizer, attr):
            v = getattr(tokenizer, attr)
            if isinstance(v, int) and v >= 0:
                image_token_ids.add(int(v))
    for tok in ["<|image_pad|>", "<image_soft_token>", "<image>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            image_token_ids.add(int(tid))

    image_mask = np.isin(input_ids[0], sorted(image_token_ids))
    image_scores = token_scores[0][image_mask]

    grid_h, grid_w = compute_grid_hw(state, thumb_h, thumb_w, encoded)

    return image_scores, token_scores[0], (grid_h, grid_w), (thumb_h, thumb_w)


def generate_text_fn(
    image: Optional[Image.Image],
    prompt_text: str,
    progress=gr.Progress(),
) -> str:
    if image is None:
        return "Please upload an image first."

    _app_state._state._ensure_model_loaded(progress)
    model = _app_state._state.model
    processor = _app_state._state.processor

    for module in model.modules():
        module._forward_hooks.clear()

    messages = build_messages(image, prompt_text)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    try:
        model_device = model.device
    except AttributeError:
        model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)

    generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def save_result(
    pil_image: Image.Image,
    image_scores: np.ndarray,
    grid_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    all_scores: np.ndarray,
    probe_name: str,
    prefix: str,
) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(RESULTS_DIR, f"probe_{prefix}_{probe_name}_{ts}")

    overlay = make_heatmap_overlay(pil_image, image_scores, grid_hw, orig_hw)
    overlay.save(f"{base}_heatmap.png")

    torch.save(
        {
            "image_scores": torch.from_numpy(image_scores.astype(np.float32)),
            "all_token_scores": torch.from_numpy(all_scores.astype(np.float32)),
            "grid_hw": torch.tensor([int(grid_hw[0]), int(grid_hw[1])], dtype=torch.int32),
            "orig_hw": torch.tensor([int(orig_hw[0]), int(orig_hw[1])], dtype=torch.int32),
            "image_mean": float(np.mean(image_scores)),
            "image_std": float(np.std(image_scores)),
        },
        f"{base}_scores.pt",
    )

    return f"{base}_heatmap.png"


def process_image(
    model_name: str,
    image: Optional[Image.Image],
    prefix: str,
    probe_name: str,
    prompt_text: str,
    progress=gr.Progress(),
) -> Tuple[Optional[Image.Image], str]:
    if image is None:
        return None, "Please upload an image."

    try:
        state = _app_state._state
        if state.current_model_name != model_name:
            state.switch_model(model_name, progress)
        else:
            state._ensure_model_loaded(progress)

        runtime = state.get_runtime(prefix, probe_name, progress)

        image_scores, all_scores, grid_hw, orig_hw = score_image(
            image, state, runtime,
            prompt_text=prompt_text.strip() or "Describe this image.",
        )

        overlay = make_heatmap_overlay(image, image_scores, grid_hw, orig_hw)

        save_result(
            image, image_scores, grid_hw, orig_hw, all_scores,
            probe_name=probe_name, prefix=prefix,
        )

        mean = float(np.mean(image_scores))
        stats = (
            f"**{prefix} / {probe_name}**\n"
            f"Model: {model_name}  |  "
            f"Mean: {mean:.4f}  |  "
            f"Min: {np.min(image_scores):.4f}  |  "
            f"Max: {np.max(image_scores):.4f}  |  "
            f"Std: {np.std(image_scores):.4f}\n"
            f"Grid: {grid_hw[0]}x{grid_hw[1]}"
        )

        return overlay, stats

    except Exception as e:
        return None, f"Error: {e}"


def _build_probe_tab(args):
    """Build the interactive probe tab UI."""
    gr.Markdown(
        "# Political Ideology Image Probe\n"
        "Upload an image to see which regions are associated with "
        "political ideology as predicted by a pre-trained linear probe. "
        "Change the probe settings to re-score the same image."
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload an image")

            prompt_text = gr.Textbox(
                value=PROMPT_TEMPLATES["Simple describe"], label="Prompt text", lines=3,
            )

            with gr.Row():
                for label, template in PROMPT_TEMPLATES.items():
                    gr.Button(label).click(
                        fn=lambda t=template: t,
                        outputs=prompt_text,
                    )

            with gr.Row():
                gen_text_output = gr.Textbox(label="Generated text", lines=10, scale=3)
                gen_btn = gr.Button("Generate Text", variant="secondary", scale=1, min_width=120)
                gen_btn.click(
                    fn=generate_text_fn,
                    inputs=[image_input, prompt_text],
                    outputs=gen_text_output,
                )

        with gr.Column(scale=1):
            heatmap_output = gr.Image(type="pil", label="Heatmap overlay")
            stats_text = gr.Markdown()

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    AVAILABLE_MODELS, value=AVAILABLE_MODELS[0], label="Model",
                )
                prefix_dropdown = gr.Dropdown(AVAILABLE_PREFIXES, value=args.prefix, label="Direction")
                probe_dropdown = gr.Dropdown(AVAILABLE_PROBES, value=args.probe, label="Probe type")

            gr.Button("Score", variant="primary").click(
                fn=process_image,
                inputs=[model_dropdown, image_input, prefix_dropdown, probe_dropdown, prompt_text],
                outputs=[heatmap_output, stats_text],
            )

    img_inputs = [model_dropdown, image_input, prefix_dropdown, probe_dropdown, prompt_text]
    image_input.change(
        fn=process_image,
        inputs=img_inputs,
        outputs=[heatmap_output, stats_text],
    )
    prompt_text.submit(
        fn=process_image,
        inputs=img_inputs,
        outputs=[heatmap_output, stats_text],
    )
    model_dropdown.change(
        fn=process_image,
        inputs=img_inputs,
        outputs=[heatmap_output, stats_text],
    )
    prefix_dropdown.change(
        fn=process_image,
        inputs=img_inputs,
        outputs=[heatmap_output, stats_text],
    )
    probe_dropdown.change(
        fn=process_image,
        inputs=img_inputs,
        outputs=[heatmap_output, stats_text],
    )
