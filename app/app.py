#!/usr/bin/env python3
"""
Interactive webapp for probing images with pre-trained linear probes.

Supports multiple models (Qwen3-VL, Gemma 4) with hot-switching via the UI.
Upload images, score them, view heatmap overlays, and save results.

Usage:
    python app/app.py
    python app/app.py --port 7860 --share
"""

import argparse
import os

import gradio as gr

from app import state as _app_state
from app.state import AppState, MODEL_REGISTRY
from app.tabs.probe import _build_probe_tab, AVAILABLE_PROBES
from app.tabs.gallery import _build_gallery_tab
from app.tabs.feature_masks import _build_feature_masks_tab
from app.common import RESULTS_DIR

DEFAULT_DATA_DIR = "results/probes"
DEFAULT_PREFIX = "combined_ideology"
DEFAULT_PROBE = "headwise_linear"
DEFAULT_TOP_K = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive probe webapp")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--probe", default=DEFAULT_PROBE, choices=AVAILABLE_PROBES)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--server-name", default="0.0.0.0")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _app_state._state = AppState(
        model_paths=MODEL_REGISTRY,
        data_dir=args.data_dir,
        top_k=args.top_k,
    )

    with gr.Blocks(title="Political Ideology Image Probe") as demo:
        with gr.Tabs():
            with gr.TabItem("Interactive Probe"):
                _build_probe_tab(args)
            with gr.TabItem("EasyPortrait Gallery"):
                _build_gallery_tab()
            with gr.TabItem("Feature Masks"):
                _build_feature_masks_tab()

    print(f"\nLaunching webapp on http://{args.server_name}:{args.port}")
    demo.launch(server_name=args.server_name, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
