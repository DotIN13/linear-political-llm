"""
Steering Benchmark: Result Parsing

This script evaluates steered generations using GPT-4o as a judge.
It scores how well each generation reflects the target concept.

Usage:
    python -m steering_benchmark.parse_results --model_set mistral --model_version Large-Instruct-2407-4bit
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import csv
from openai import OpenAI
import pickle
import os
from tqdm import tqdm
import time
import random

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Model-specific tags for parsing responses
ASSISTANT_TAGS = {
    'llama': '<|start_header_id|>assistant<|end_header_id|>',
    'mistral': '[/INST]',
    'falcon': '<|assistant|>',
    'phi': '<|assistant|>',
}

EOT_STRS = {
    'mistral': '</s>',
    'falcon': '<|endoftext|>',
    'phi': '<|end|>',
}


def parse_response(response, model_type):
    """
    Extract the assistant's response from the full model output.

    Args:
        response: Tuple of (coefficient, full_output)
        model_type: The model family name

    Returns:
        Parsed assistant response string
    """
    try:
        passage = response[1].split(ASSISTANT_TAGS[model_type], 1)[1].lstrip()
        if model_type in EOT_STRS:
            passage = passage.strip(EOT_STRS[model_type])
    except Exception as e:
        print(f"Error parsing response for model: {model_type}, response: {response}")
        return ""
    return "".join(passage)


def load_prompt(label, version):
    """Load the evaluation prompt template for a concept category."""
    prompt_dir = 'evaluation_prompts/'
    version_label = '' if version == 1 else f'_v{version}'

    prompt_files = {
        'fears': f'phobia_eval{version_label}.txt',
        'personalities': f'personality_eval{version_label}.txt',
        'moods': f'mood_eval{version_label}.txt',
        'places': f'topophile_eval{version_label}.txt',
        'personas': f'persona_eval{version_label}.txt'
    }

    with open(prompt_dir + prompt_files[label], "r") as f:
        return f.read()


def evaluate_with_gpt4(prompt, max_retries=6):
    """
    Evaluate a response using GPT-4o with retry logic.

    Args:
        prompt: The evaluation prompt
        max_retries: Maximum number of retry attempts

    Returns:
        GPT-4o response content
    """
    for retry_idx in range(max_retries):
        try:
            output = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.,
                max_tokens=20,
                model='gpt-4o-2024-11-20'
            )
            return output.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            if retry_idx == max_retries - 1:
                raise
            sleep_seconds = (2 ** retry_idx) + random.uniform(0, 0.5)
            time.sleep(sleep_seconds)
    raise RuntimeError("OpenAI chat completion failed after retries")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate steered generations with GPT-4o")
    parser.add_argument("--model_set", type=str, default='mistral')
    parser.add_argument("--model_version", type=str, default='Large-Instruct-2407-4bit')
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--concepts_to_steer", type=str, default='all')
    args = parser.parse_args()

    METHOD = 'rfm'
    VERSIONS = [1, 2, 3, 4, 5]

    MODEL_SIZE = args.model_size
    MODEL_NAME = args.model_set
    MODEL_VERSION = args.model_version

    print(f"Evaluating generations for {MODEL_NAME} {MODEL_VERSION} {MODEL_SIZE}")

    if args.concepts_to_steer == 'all':
        CONCEPT_CLASSES = ['personalities', 'moods', 'places', 'personas', 'fears']
    else:
        CONCEPT_CLASSES = [args.concepts_to_steer]

    for VERSION in VERSIONS:
        VERSION_LABEL = '' if VERSION == 1 else f'_v{VERSION}'

        for CONCEPT_CLASS in tqdm(CONCEPT_CLASSES):
            output_csv = f"csvs/{METHOD}_{CONCEPT_CLASS}_gpt4o_outputs_500_concepts_{MODEL_NAME}_{MODEL_VERSION}_{MODEL_SIZE}_english_only{VERSION_LABEL}.csv"

            if os.path.exists(output_csv):
                print(f"Skipping {output_csv} - already exists")
                continue

            file_path = f'cached_outputs/{METHOD}_{CONCEPT_CLASS}_steered_500_concepts_{MODEL_NAME}_{MODEL_VERSION}_{MODEL_SIZE}_english_only{VERSION_LABEL}.pkl'

            os.makedirs('cached_outputs', exist_ok=True)
            os.makedirs('csvs', exist_ok=True)

            results = pickle.load(open(file_path, 'rb'))

            # Load existing cache for resume support
            outputs_cache_path = f"cached_outputs/{METHOD}_{CONCEPT_CLASS}_gpt4o_outputs_500_concepts_{MODEL_NAME}_{MODEL_VERSION}_{MODEL_SIZE}_english_only{VERSION_LABEL}.pkl"
            try:
                outputs = pickle.load(open(outputs_cache_path, 'rb'))
                if not isinstance(outputs, dict):
                    outputs = {}
            except FileNotFoundError:
                outputs = {}

            # Support both dict and list formats
            results_iter = results.items() if isinstance(results, dict) else results

            for personality, responses in tqdm(results_iter):
                print(f"===== PERSONALITY: {personality} =====")

                if personality in outputs:
                    continue

                best_score = 0
                for response in responses:
                    parsed_response = parse_response(response, MODEL_NAME)
                    if parsed_response == "":
                        parsed_response = "None"

                    prompt_template = load_prompt(CONCEPT_CLASS, VERSION)
                    prompt = prompt_template.format(personality=personality, parsed_response=parsed_response)

                    content = evaluate_with_gpt4(prompt)

                    score = 0
                    if "Score: " in content:
                        try:
                            score = int(content.split("Score: ")[1][0])
                        except (IndexError, ValueError):
                            score = 0
                    best_score = max(score, best_score)

                print(f"{personality} - Best score: {best_score}")
                outputs[personality] = best_score

                # Persist cache after each personality
                with open(outputs_cache_path, 'wb') as cache_f:
                    pickle.dump(outputs, cache_f)

            # Write final CSV
            with open(output_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["GPT-4o responses"])
                writer.writerows(outputs.items())


if __name__ == "__main__":
    main()
