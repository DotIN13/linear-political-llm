"""
Steering Benchmark: Result Aggregation

This script reads evaluation CSV files and computes aggregate steering success rates.

Usage:
    python -m steering_benchmark.read_csv --model_set falcon --model_version 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def readfile(fname):
    """Read a CSV file and count steered successes."""
    total = 0
    steered = 0
    with open(fname) as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            t = int(line.strip().split(',')[-1])
            steered += t
            total += 1
    print(f"Steered: {steered} out of {total}")
    return steered, total


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate steering evaluation results")
    parser.add_argument("--model_set", type=str, default='falcon')
    parser.add_argument("--model_version", type=str, default='3')
    args = parser.parse_args()

    methods = ['rfm']
    concepts = ['fears', 'personalities', 'moods', 'places', 'personas']

    # Configure model parameters
    if args.model_set == 'llama':
        MODEL_SET = 'llama'
        MODEL_VERSION = '3.1'
        MODEL_SIZE = '70B'
    elif args.model_set == 'falcon':
        MODEL_SET = 'falcon'
        MODEL_VERSION = args.model_version
        MODEL_SIZE = '10B' if args.model_version == '3' else None
    elif args.model_set == 'mistral':
        MODEL_SET = 'mistral'
        MODEL_VERSION = args.model_version
        if args.model_version == 'Small-Instruct-2409':
            MODEL_SIZE = '7B'
        elif args.model_version == 'Large-Instruct-2407-4bit':
            MODEL_SIZE = '120B'

    VERSIONS = [1, 2, 3, 4, 5]

    for VERSION in VERSIONS:
        VERSION_LABEL = f'_v{VERSION}' if VERSION >= 2 else ''

        results = {}
        for method in methods:
            results[method] = []
            for concept in concepts:
                fname = f'csvs/{method}_{concept}_gpt4o_outputs_500_concepts_{MODEL_SET}_{MODEL_VERSION}_{MODEL_SIZE}_english_only{VERSION_LABEL}.csv'
                try:
                    steered, total = readfile(fname)
                    results[method].append((concept, steered, total))
                except FileNotFoundError:
                    print(f"File not found: {fname}")

        for method in results:
            total = 0
            steered = 0
            for c, s, t in results[method]:
                total += t
                steered += s
            if total > 0:
                print(f"Version {VERSION} - Method: {method}, Steered: {steered}/{total} ({steered/total:.2%})")


if __name__ == "__main__":
    main()
