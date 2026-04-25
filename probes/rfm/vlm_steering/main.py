"""
VLM Steering: Vision Language Model Steering

This module provides tools for steering vision-language models using
precomputed directions. Supports text-only models (Llama, Gemma) and
multimodal models (Llama-Vision, LLaVA).

Usage:
    python -m vlm_steering.main
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    MllamaForConditionalGeneration, AutoProcessor,
    LlavaForConditionalGeneration
)
from probes.rfm.utils import harmful_dataset, LLMType
from probes.rfm.neural_controllers import NeuralController
from probes.rfm.generation_utils import extract_image
from collections import namedtuple
from PIL import Image

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

LLM = namedtuple('LLM', ['language_model', 'tokenizer', 'processor', 'name', 'model_type'])


def select_llm(model_type):
    """
    Load and configure a language model for VLM steering.

    Args:
        model_type: Model type ('llama', 'gemma', 'llama-vision', 'llava')

    Returns:
        LLM namedtuple containing the model, tokenizer, processor, and metadata
    """
    if model_type == 'llama':
        model_id = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda",
        )

        use_fast_tokenizer = "LlamaForCausalLM" not in language_model.config.architectures
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
        tokenizer.pad_token_id = 0
        model_name = "llama_3_70b_it"
        processor = None
        llm_type = LLMType.TEXT

    elif model_type == 'gemma':
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        language_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-9b-it",
            device_map="auto",
        )
        model_name = 'gemma_2_9b_it'
        processor = None
        llm_type = LLMType.TEXT

    elif model_type == 'llama-vision':
        model_id = "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit"

        language_model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", legacy=False)
        tokenizer.pad_token_id = 0
        processor = AutoProcessor.from_pretrained(model_id)
        model_name = 'llama_3_90b_4bit_it'
        llm_type = LLMType.MULTIMODAL

    elif model_type == 'llava':
        model_id = "llava-hf/llava-1.5-7b-hf"

        language_model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", legacy=False)
        tokenizer.pad_token_id = 0
        processor = AutoProcessor.from_pretrained(model_id)
        model_name = 'llava-1.5-7b'
        llm_type = LLMType.MULTIMODAL

    llm = LLM(language_model, tokenizer, processor, model_name, llm_type)
    return llm


def compute_save_directions(llm, dataset, concept, control_method='rfm'):
    """
    Compute and save steering directions for various concept types.

    Args:
        llm: The language model
        dataset: Training data with inputs and labels
        concept: The concept name
        control_method: Algorithm for direction computation
    """
    if concept in ('creativity', 'biology_expert', 'hallucination', 'harmful'):
        controller = NeuralController(
            llm, llm.tokenizer,
            rfm_iters=8, control_method=control_method, n_components=1
        )
        if concept in ('hallucination', 'harmful'):
            controller.compute_directions(dataset['train']['inputs'], np.concatenate(dataset['train']['labels']).tolist())
        else:
            controller.compute_directions(dataset['train']['inputs'], dataset['train']['labels'])
        controller.save(concept=concept, model_name=llm.name, path='directions/')

    elif concept == 'poetry':
        for concept_type in ['prose']:
            controller = NeuralController(
                llm, llm.tokenizer,
                rfm_iters=8, control_method=control_method, n_components=1
            )
            controller.compute_directions(dataset[concept_type]['train']['inputs'],
                                          np.array(dataset[concept_type]['train']['labels']).tolist())
            controller.save(concept=concept_type, model_name=llm.name, path='directions/')

    elif concept == 'politics':
        for concept_type in ['Republican']:
            controller = NeuralController(
                llm, llm.tokenizer,
                rfm_iters=8, control_method=control_method, n_components=1
            )
            controller.compute_directions(dataset[concept_type]['train']['inputs'],
                                          dataset[concept_type]['train']['labels'])
            controller.save(concept=concept_type, model_name=llm.name, path='directions/')

    elif concept == 'shakespeare':
        for concept_type in ['english', 'shakespeare']:
            controller = NeuralController(
                llm, llm.tokenizer,
                rfm_iters=8, control_method=control_method, n_components=1
            )
            controller.compute_directions(dataset[concept_type]['train']['inputs'],
                                          dataset[concept_type]['train']['labels'])
            controller.save(concept=concept_type, model_name=llm.name, path='directions/')

    elif concept == 'conspiracy':
        controller = NeuralController(
            llm, llm.tokenizer,
            rfm_iters=8, control_method=control_method, n_components=1
        )
        controller.compute_directions(dataset['conspiracy']['train']['inputs'],
                                      dataset['conspiracy']['train']['labels'])
        controller.save(concept='conspiracy', model_name=llm.name, path='directions/')

    else:
        controller = NeuralController(
            llm, llm.tokenizer,
            rfm_iters=8, control_method=control_method, n_components=1
        )
        controller.compute_directions(dataset[concept]['train']['inputs'],
                                      dataset[concept]['train']['labels'])
        controller.save(concept=concept, model_name=llm.name, path='directions/')


def generate(concept, llm, prompt, image=None, coefs=[0.4], control_method='rfm', max_tokens=100):
    """
    Generate steered outputs for a concept.

    Args:
        concept: The concept to steer towards
        llm: The language model
        prompt: Input prompt
        image: Optional image input for multimodal models
        coefs: List of steering coefficients
        control_method: Steering algorithm
        max_tokens: Maximum tokens to generate
    """
    controller = NeuralController(
        llm, llm.tokenizer,
        rfm_iters=8, control_method=control_method, n_components=1
    )

    controller.load(concept=concept, model_name=llm.name, path='directions/')

    # Unsteered baseline
    original_output = controller.generate(prompt, image=image, max_new_tokens=max_tokens, do_sample=False)
    print(original_output)

    # Steered outputs at different coefficients
    for coef in coefs:
        print(f"Coeff: {coef} " + "=" * 60)
        steered_output = controller.generate(
            prompt, image=image,
            layers_to_control=controller.hidden_layers,
            control_coef=coef,
            max_new_tokens=max_tokens,
            do_sample=False
        )
        print(steered_output)


def combine_directions(dirs1, dirs2, a=0.5, b=0.5):
    """Combine two direction dictionaries with weighted sum."""
    return {k: a * dirs1[k] + b * dirs2[k] for k in dirs1.keys()}


def generate_combined(concept1, concept2, llm, prompt, image=None, coefs=[0.4],
                      control_method='rfm', max_tokens=100, a=0.5, b=0.5, show_original=True):
    """
    Generate outputs using combined steering directions from two concepts.

    Args:
        concept1, concept2: The two concepts to combine
        llm: The language model
        prompt: Input prompt
        image: Optional image for multimodal models
        coefs: List of steering coefficients
        control_method: Steering algorithm
        max_tokens: Maximum tokens to generate
        a, b: Weights for combining directions
        show_original: Whether to show unsteered output
    """
    controller1 = NeuralController(
        llm, llm.tokenizer,
        rfm_iters=8, control_method=control_method, n_components=1
    )
    controller1.load(concept=concept1, model_name=llm.name, path='directions/')

    controller2 = NeuralController(
        llm, llm.tokenizer,
        rfm_iters=8, control_method=control_method, n_components=1
    )
    controller2.load(concept=concept2, model_name=llm.name, path='directions/')

    controller2.directions = combine_directions(controller1.directions, controller2.directions, a=a, b=b)

    if show_original:
        original_output = controller2.generate(prompt, image=image, max_new_tokens=max_tokens, do_sample=False)
        print(original_output)

    for coef in coefs:
        print(f"Coeff: {coef} " + "=" * 60)
        steered_output = controller2.generate(
            prompt, image=image,
            layers_to_control=controller2.hidden_layers,
            control_coef=coef,
            max_new_tokens=max_tokens,
            do_sample=False
        )
        print(steered_output)


def main():
    """Example usage of VLM steering."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load model
    model_type = 'llama-vision'
    llm = select_llm(model_type)

    # Example: Conspiracy steering with vaccine image
    image = Image.open("images/1024px-Covid-Vaccine-41_(50753217957)_(cropped).jpg")
    prompt = "Take on the role of a conspiracy theorist. Is it safe to take this? Be concise in your answer."
    generate('conspiracy', llm, prompt, image=image, coefs=[-.2, -.25, -.3, -.35, -.4],
             control_method='rfm', max_tokens=100)


if __name__ == "__main__":
    main()
