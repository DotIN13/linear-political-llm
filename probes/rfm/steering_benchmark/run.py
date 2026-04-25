"""
Steering Benchmark: Direction Computation

This script computes steering directions for various concepts (personalities, moods,
fears, places, personas) using different LLM models. Directions are saved to disk
for later use in steering generation.

Usage:
    python -m steering_benchmark.run --model_set phi --model_version 3-medium-4k-instruct
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from probes.rfm.neural_controllers import NeuralController
from probes.rfm.utils import LLMType
from collections import namedtuple
from tqdm import tqdm
import gc
import os
from probes.rfm import utils

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

LLM = namedtuple('LLM', ['language_model', 'tokenizer', 'processor', 'name', 'model_type'])


def select_llm(model_type, MODEL_VERSION='3.1', MODEL_SIZE='8B'):
    """
    Load and configure a language model for steering direction computation.

    Args:
        model_type: Model family ('llama', 'phi', 'falcon', 'mistral')
        MODEL_VERSION: Specific version within the family
        MODEL_SIZE: Model size variant

    Returns:
        LLM namedtuple containing the model, tokenizer, and metadata
    """
    if model_type == 'llama':
        if MODEL_VERSION == '3.1' and MODEL_SIZE == '8B':
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif MODEL_VERSION == '3.1' and MODEL_SIZE == '70B':
            model_id = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
        elif MODEL_VERSION == '3.3' and MODEL_SIZE == '70B':
            model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

        language_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda",
        )

        use_fast_tokenizer = "LlamaForCausalLM" not in language_model.config.architectures
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
        tokenizer.pad_token_id = 0

        if MODEL_VERSION == '3.1' and MODEL_SIZE == '8B':
            model_name = 'llama_3_8b_it_eng_only'
        elif MODEL_VERSION == '3.1' and MODEL_SIZE == '70B':
            model_name = "llama_3.1_70b_it_eng_only"
        elif MODEL_VERSION == '3.3' and MODEL_SIZE == '70B':
            model_name = "llama_3.3_70b_it_eng_only"

        processor = None
        llm_type = LLMType.TEXT

    elif model_type == 'phi':
        if MODEL_VERSION == '4':
            model_id = "microsoft/phi-4"
            model_name = 'phi_4'
        elif MODEL_VERSION == '3':
            model_id = "microsoft/phi-3"
            model_name = 'phi_3'
        elif MODEL_VERSION == '3-medium-4k-instruct':
            model_id = "microsoft/Phi-3-medium-4k-instruct"
            model_name = 'phi_3_medium_4k_instruct'

        print(f"Loading Phi {MODEL_VERSION} model")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        processor = None
        llm_type = LLMType.TEXT

    elif model_type == 'falcon':
        if MODEL_SIZE == '10B' and MODEL_VERSION == '3':
            model_id = "tiiuae/Falcon3-10B-Instruct"
            model_name = 'falcon3_10b_it'
        if MODEL_SIZE == '3B' and MODEL_VERSION == '3':
            model_id = "tiiuae/Falcon3-3B-Instruct"
            model_name = 'falcon3_3b_it'
        if MODEL_SIZE == '40B' and MODEL_VERSION == '1':
            model_id = "tiiuae/falcon-40B-instruct"
            model_name = 'falcon_40b_it'

        print(f"Loading Falcon {MODEL_SIZE} model")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        processor = None
        llm_type = LLMType.TEXT

    elif model_type == 'mistral':
        if MODEL_VERSION == 'Small-Instruct-2409':
            model_id = "mistralai/Mistral-Small-Instruct-2409"
            model_name = 'mistral_small_2409_it'
        elif MODEL_VERSION == 'Large-Instruct-2407-4bit':
            model_id = "unsloth/Mistral-Large-Instruct-2407-bnb-4bit"
            model_name = 'mistral_large_2407_4bit_it'

        print(f"Loading Mistral {MODEL_VERSION} model")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        processor = None
        llm_type = LLMType.TEXT

    llm = LLM(language_model, tokenizer, processor, model_name, llm_type)
    return llm


def compute_save_directions(llm, dataset, concept, control_method='rfm'):
    """
    Compute and save steering directions for a concept.

    Args:
        llm: The language model
        dataset: Training data with inputs and labels
        concept: The concept name to steer towards
        control_method: Algorithm for direction computation ('rfm', 'linear', etc.)
    """
    controller = NeuralController(
        llm,
        llm.tokenizer,
        rfm_iters=8,
        control_method=control_method,
        n_components=1,
        batch_size=8,
    )
    controller.compute_directions(dataset[concept]['train']['inputs'], dataset[concept]['train']['labels'])
    controller.save(concept=concept, model_name=llm.name, path='directions/')


def read_file(fname, lower=True):
    """Read concepts from a text file, one per line."""
    concepts = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            if lower:
                concepts.append(line.strip().lower())
            else:
                concepts.append(line.strip())
    concepts = sorted(list(set(concepts)))
    return concepts


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute steering directions for concepts")
    parser.add_argument("--model_set", type=str, default='phi')
    parser.add_argument("--model_version", type=str, default='3-medium-4k-instruct')
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--concepts_to_steer", type=str, default='all')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    fnames = {
        'fears': 'data/fears/fears.txt',
        'personalities': 'data/personalities/personalities.txt',
        'moods': 'data/moods/moods.txt',
        'places': 'data/places/places.txt',
        'personas': 'data/personas/personas.txt'
    }
    lowers = {
        'fears': True,
        'personalities': True,
        'moods': True,
        'places': False,
        'personas': False
    }

    MODEL_SET = args.model_set
    MODEL_VERSION = args.model_version
    MODEL_SIZE = args.model_size

    if MODEL_SIZE is None:
        if MODEL_SET == 'phi':
            MODEL_VERSION = '3-medium-4k-instruct'
            MODEL_SIZE = '14B'
        elif MODEL_SET == 'mistral':
            if MODEL_VERSION == 'Small-Instruct-2409':
                MODEL_SIZE = '7B'
            elif MODEL_VERSION == 'Large-Instruct-2407-4bit':
                MODEL_SIZE = '120B'

    llm = select_llm(MODEL_SET, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)
    METHOD = 'rfm'

    if args.concepts_to_steer == 'all':
        concepts_to_steer = ['fears', 'personas', 'places', 'personalities', 'moods']
    else:
        concepts_to_steer = [args.concepts_to_steer]
    number_of_concepts_to_steer = 120

    for concept_label in concepts_to_steer:
        fname = fnames[concept_label]
        concepts = read_file(fname, lower=lowers[concept_label])

        if number_of_concepts_to_steer < len(concepts):
            import random
            random.seed(0)
            subconcepts_to_steer = random.sample(concepts, number_of_concepts_to_steer)
        else:
            subconcepts_to_steer = concepts

        for concept in tqdm(subconcepts_to_steer):
            directions_file = f'directions/{METHOD}_{concept}_{llm.name}.pkl'
            if os.path.exists(directions_file):
                print(f"Skipping {concept} because directions file already exists")
                continue
            else:
                print(f"Computing directions to file: {directions_file}")

            print(f"===== CONCEPT={concept} =====")

            if concept_label == 'fears':
                dataset = utils.pca_fears_dataset(llm, concept)
            elif concept_label == 'personalities':
                dataset = utils.pca_personalities_dataset(llm, concept)
            elif concept_label == 'personas':
                dataset = utils.pca_persona_dataset(llm, concept)
            elif concept_label == 'moods':
                dataset = utils.pca_mood_dataset(llm, concept)
            elif concept_label == 'places':
                dataset = utils.pca_places_dataset(llm, concept)

            compute_save_directions(llm, dataset, concept, control_method=METHOD)
            del dataset
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
