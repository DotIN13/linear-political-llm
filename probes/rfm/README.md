# Universal Steering

Code for steering language models using learned activation directions. This repository contains two main components:

1. **Steering Benchmark** - Evaluates steering across hundreds of concepts (personalities, moods, fears, places, personas)
2. **VLM Steering** - Applies steering to vision-language models

## Repository Structure

```
universal_steering/
├── steering_benchmark/       # Benchmark evaluation pipeline
│   ├── run.py               # Compute steering directions
│   ├── eval_generations.py  # Generate steered outputs
│   ├── parse_results.py     # Evaluate with GPT-4o judge
│   └── read_csv.py          # Aggregate results
├── vlm_steering/            # Vision-language model steering
│   └── main.py              # VLM steering examples
├── neural_controllers.py    # Core controller class
├── control_toolkits.py      # Direction computation algorithms
├── direction_utils.py       # Hidden state extraction utilities
├── generation_utils.py      # Text generation with hooks
├── rfm.py                   # Random Feature Model algorithm
├── utils.py                 # Dataset loaders and utilities
├── data/                    # Concept lists (personalities, moods, etc.)
└── evaluation_prompts/      # GPT-4o evaluation templates
```

## Steering Benchmark

The benchmark evaluates steering effectiveness across 5 concept categories using multiple LLMs.

### Pipeline

1. **Compute Directions**: Extract steering directions from training data
```bash
python -m steering_benchmark.run --model_set phi --model_version 3-medium-4k-instruct
```

2. **Generate Steered Outputs**: Generate text with steering applied
```bash
python -m steering_benchmark.eval_generations --model_set phi --model_version 3-medium-4k-instruct
```

3. **Evaluate with GPT-4o**: Score generations for concept alignment
```bash
python -m steering_benchmark.parse_results --model_set phi --model_version 3-medium-4k-instruct
```

4. **Aggregate Results**: Compute success rates
```bash
python -m steering_benchmark.read_csv --model_set phi --model_version 3-medium-4k-instruct
```

### Supported Models

- **Llama**: 3.1-8B, 3.1-70B, 3.3-70B
- **Phi**: 3, 4, 3-medium-4k-instruct
- **Falcon**: 3-3B, 3-10B
- **Mistral**: Small-Instruct-2409, Large-Instruct-2407

## VLM Steering

Steer vision-language models on multimodal inputs.

```python
from vlm_steering.main import select_llm, generate

# Load model
llm = select_llm('llama-vision')

# Generate with steering
generate('conspiracy', llm, prompt, image=image, coefs=[0.3, 0.4])
```

### Supported VLM Models

- **Llama-Vision**: 3.2-90B
- **LLaVA**: 1.5-7B
- **Gemma**: 2-9B (text-only)

## Control Methods

- **RFM** (default): Random Feature Model with metric learning
- **Linear**: Linear regression probe
- **Logistic**: Logistic regression probe
- **PCA**: Principal component analysis
- **Mean Difference**: Simple difference between class means

## Requirements

- PyTorch
- Transformers (Hugging Face)
- OpenAI API (for GPT-4o evaluation)
- CUDA-capable GPU
