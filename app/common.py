"""Shared utilities used by both tabs."""

from typing import Tuple

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MAX_IMAGE_DIM = 800
RESULTS_DIR = "app/results"


def make_heatmap_overlay(
    pil_image: Image.Image,
    image_scores: np.ndarray,
    grid_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    alpha: float = 0.45,
    cmap_name: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> Image.Image:
    grid_h, grid_w = grid_hw
    orig_h, orig_w = orig_hw

    expected = grid_h * grid_w
    actual = len(image_scores)
    if actual != expected:
        pad = expected - actual
        if pad > 0:
            image_scores = np.concatenate([image_scores, np.zeros(pad, dtype=image_scores.dtype)])
        else:
            image_scores = image_scores[:expected]
    heatmap_2d = image_scores.reshape(grid_h, grid_w)

    img = np.asarray(pil_image.convert("RGB")).astype(np.float32) / 255.0
    h_img, w_img = img.shape[:2]

    from PIL import Image as PILImage
    hm_img = PILImage.fromarray(
        (np.clip((heatmap_2d - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
    ).convert("L").resize((w_img, h_img), PILImage.BILINEAR)
    hm_up = np.array(hm_img) / 255.0

    cm = plt.get_cmap(cmap_name)
    hm_rgb = cm(hm_up)[..., :3]

    out = (1 - alpha) * img + alpha * hm_rgb
    out = np.clip(out * 255, 0, 255).astype(np.uint8)

    plt.close("all")
    return Image.fromarray(out)
