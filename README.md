# Tulu LLM: Prompting Framework for Low-Resource Languages

A comprehensive framework for teaching LLMs to generate text in Tulu, an endangered Dravidian language. This implementation uses a 5-layer prompt architecture to prevent contamination from related languages (Kannada) and ensure grammatical accuracy.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key"  # or GEMINI_API_KEY

# Run all experiments
python run_all.py
```

## Overview

This framework implements a 5-layer prompt architecture designed to:
- Prevent language contamination (Kannada words in Tulu output)
- Ensure grammatical correctness
- Support low-resource language learning for LLMs

### Key Features

- **55+ Negative Constraints**: Kannada→Tulu word mappings to prevent contamination
- **Complete Grammar System**: 15 verbs with full paradigms, 8 cases with allomorph rules
- **Standardized Romanization**: Efficient tokenization (1.4 tokens/word vs 3.2 for Kannada script)
- **Multi-Model Support**: OpenAI GPT, Google Gemini, and Llama
- **Comprehensive Evaluation**: Contamination detection, grammar checking, vocabulary analysis

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── config.py                          # Configuration
├── run_all.py                         # Main orchestration script
├── generate_results.py                # Generate graphs and tables
│
├── prompts/
│   ├── tulu_prompt.py                 # 5-layer prompt builder
│   └── comprehensive_tulu_prompts.py  # Detailed teaching prompts
│
├── romanization/
│   └── romanization_rules.py          # Standardized transliteration
│
├── grammar/
│   └── tulu_grammar.py                # Complete grammar documentation
│
├── constraints/
│   └── negative_constraints.py        # 50+ Kannada→Tulu mappings
│
├── generation/
│   └── self_play.py                   # Self-play Q/A generation
│
├── evaluation/
│   ├── contamination.py               # Contamination detection
│   ├── grammar_accuracy.py            # Grammar checking
│   ├── vocabulary_coverage.py         # Vocabulary analysis
│   └── tokenization_efficiency.py     # Tokenization metrics
│
├── experiments/
│   ├── baseline.py                    # Baseline system
│   ├── full_system.py                 # Complete 5-layer system
│   ├── ablation.py                    # Ablation studies
│   └── falsification.py               # Falsification experiment
│
├── models/
│   ├── base_model.py                  # Abstract base class
│   ├── openai_model.py                # GPT-3.5/4 wrapper
│   ├── gemini_model.py                # Gemini wrapper
│   └── llama_model.py                 # Llama wrapper
│
├── results/
│   ├── visualize_results.py           # Visualization module
│   └── README.md                      # Results documentation
│
└── utils/
    ├── logging_utils.py               # Experiment logging
    └── tokenization.py                # Tokenization utilities
```

## Architecture Overview

| Component | Module | Description |
|-----------|--------|-------------|
| Romanization | `romanization/romanization_rules.py` | Standardized transliteration system |
| Grammar | `grammar/tulu_grammar.py` | Complete grammar (15 verbs, 8 cases) |
| Constraints | `constraints/negative_constraints.py` | 50+ Kannada→Tulu mappings |
| Prompts | `prompts/tulu_prompt.py` | 5-layer prompt builder (~2,800 tokens) |
| Generation | `generation/self_play.py` | Self-play Q/A generation with quality control |
| Evaluation | `evaluation/*.py` | 4 evaluation metrics |
| Experiments | `experiments/*.py` | Baseline, full system, ablation, falsification |

## Setup

### Prerequisites

- Python 3.9+
- API key for at least one LLM:
  - OpenAI API key (for GPT-3.5/4)
  - Google Gemini API key (for Gemini 2.0 Flash)

### Installation

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

## Running Experiments

### All Experiments

```bash
python run_all.py
```

### Individual Experiments

```python
from models.openai_model import OpenAIModel
from experiments.full_system import FullSystemExperiment
import json

# Initialize model
model = OpenAIModel(model_name="gpt-4o")

# Load test set
with open("data/tulu_test.json") as f:
    test_set = json.load(f)

# Run experiment
experiment = FullSystemExperiment(model)
results = experiment.run(test_set[:10])

print(f"Contamination: {results['contamination_rate']:.1%}")
print(f"Grammar Accuracy: {results['grammar_accuracy']:.1%}")
```

## Generating Results Visualizations

After running experiments, generate publication-ready graphs and tables:

```bash
python generate_results.py
```

This creates:
- Graphs: Contamination comparison, grammar accuracy, ablation study, model comparison
- Tables: All results in LaTeX and CSV formats

See `results/README.md` for details.

## Implementation Details

### 5-Layer Prompt Architecture

1. **Identity Establishment** (200 tokens): Native Tulu speaker persona
2. **Negative Constraints** (600 tokens): 50+ Kannada→Tulu mappings **MUST come before grammar**
3. **Grammar Rules** (1200 tokens): 15 verbs, 8 cases, pronouns, syntax
4. **Few-Shot Examples** (600 tokens): 10-15 Q&A pairs
5. **Self-Verification** (200 tokens): Pre-generation checklist

**Total: ~2,800 tokens**

### Key Components

- **55 Negative Constraints**: Kannada→Tulu word mappings
- **15 Verbs**: Complete conjugation paradigms (48 forms each)
- **8 Cases**: With morphophonological allomorph rules
- **3-Judge Quality Control**: 1-5 Likert scale, 64% retention
- **4 Evaluation Metrics**: Contamination, grammar, vocabulary, tokenization

