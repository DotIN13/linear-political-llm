#!/usr/bin/env python
"""
Segment objects in an image using SAM3 Promptable Concept Segmentation (PCS)
with a text prompt. Saves an overlaid mask visualization.

Usage:
    python scripts/sam3_detect.py --image path/to/image.jpg --prompt "person"
    python scripts/sam3_detect.py --image path/to/image.jpg --prompt "chair" --threshold 0.4

If no image is provided, downloads a sample COCO image.
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

# Force matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 object detection with text prompt")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="person", help="Text prompt for segmentation")
    parser.add_argument("--output", type=str, default=None, help="Output path for visualization (default: <image_stem>_sam3_<prompt>.png)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for filtering masks")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Threshold for mask binarization")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (auto-detected if not set)")
    return parser.parse_args()


def load_image(image_path=None):
    """Load an image from path, or download a sample COCO image if not provided."""
    if image_path is not None:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    import requests
    url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    print(f"No image provided, downloading sample: {url}")
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")


def _to_float_tensor(t):
    """Convert tensor to float32 numpy array, handling bfloat16."""
    if isinstance(t, torch.Tensor):
        return t.float().cpu().numpy()
    return t


def overlay_masks(image, masks, boxes=None, scores=None):
    """Overlay colored masks and bounding boxes on the image."""
    image = image.convert("RGBA")
    masks = _to_float_tensor(masks) if isinstance(masks, torch.Tensor) else masks.astype(np.uint8)
    n_masks = masks.shape[0]

    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(max(n_masks, 1))
    colors = [tuple(int(c * 255) for c in cmap(i % cmap.N)[:3]) for i in range(n_masks)]

    # Overlay masks
    for mask, color in zip(masks, colors):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)

    # Draw bounding boxes and scores using matplotlib
    if boxes is not None and scores is not None:
        image_np = np.array(image.convert("RGB"))
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(image_np)
        for i, (box, score) in enumerate(zip(boxes, scores)):
            box = _to_float_tensor(box)
            score_val = score.item() if isinstance(score, torch.Tensor) else score
            x1, y1, x2, y2 = box
            color = [c / 255.0 for c in colors[i]]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{score_val:.2f}", color="white", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.7))
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        result = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)
        plt.close(fig)
        return result.convert("RGB")

    return image.convert("RGB")


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image = load_image(args.image)
    print(f"Image size: {image.size}")

    from transformers import Sam3Model, Sam3Processor

    MODEL_DIR = "/project/jevans/tzhang3/models/sam3"

    print(f"Loading SAM3 model from {MODEL_DIR}...")
    model = Sam3Model.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to(device)
    processor = Sam3Processor.from_pretrained(MODEL_DIR)
    print("Model loaded.")

    print(f"Running segmentation with prompt: '{args.prompt}'")
    inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    n_found = len(results["masks"])
    print(f"Found {n_found} objects.")

    if n_found == 0:
        print("No objects found. Try lowering --threshold or changing the prompt.")
        sys.exit(0)

    # Generate visualization
    result_image = overlay_masks(image, results["masks"], results["boxes"], results["scores"])

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.image:
        base = os.path.splitext(args.image)[0]
        output_path = f"{base}_sam3_{args.prompt.replace(' ', '_')}.png"
    else:
        output_path = f"sam3_{args.prompt.replace(' ', '_')}.png"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result_image.save(output_path)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
