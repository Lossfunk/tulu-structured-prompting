<p align="center">
  <img src="lossfunk-logo.jpg" alt="Lossfunk Logo" width="350"/>
</p>


# Making Large Language Models Speak Tulu: Structured Prompting for an Extremely Low-Resource Language

A framework for teaching LLMs to generate text in Tulu, a low-resource Dravidian language, using a 5-layer prompt architecture. No fine-tuning required.

Accepted at the **LoResLM Workshop at EACL, 2026**. [[arXiv]](https://arxiv.org/abs/2602.15378)

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

When asked to "respond in Tulu," LLMs default to Kannada (a related, higher-resource language). This framework uses a ~2,800-token structured prompt to prevent that contamination and produce grammatically correct Tulu output.

The prompt has 5 layers, applied in order:

1. **Identity** (~200 tokens): Native Tulu speaker persona with romanization rules
2. **Negative Constraints** (~600 tokens): 50+ Kannada words to never use, each paired with the Tulu alternative
3. **Grammar** (~1,200 tokens): Verb conjugation tables, 8 case markers, SOV word order
4. **Few-Shot Examples** (~600 tokens): 10-15 Tulu Q&A pairs
5. **Self-Verification** (~200 tokens): Pre-response checklist

## Repository Structure

```
.
├── run_all.py                         # Main experiment runner
├── config.py                          # Configuration
├── requirements.txt                   # Dependencies
│
├── prompts/
│   ├── tulu_prompt.py                 # 5-layer prompt builder
│   └── comprehensive_tulu_prompts.py  # Extended prompt templates
│
├── romanization/
│   └── romanization_rules.py          # Standardized transliteration
│
├── grammar/
│   └── tulu_grammar.py                # Tulu grammar (15 verbs, 8 cases)
│
├── constraints/
│   └── negative_constraints.py        # 50+ Kannada→Tulu word mappings
│
├── generation/
│   └── self_play.py                   # Self-play Q&A generation with 3-judge filter
│
├── evaluation/
│   ├── contamination.py               # Kannada contamination detection
│   ├── grammar_accuracy.py            # Rule-based grammar checking
│   ├── vocabulary_coverage.py         # Vocabulary analysis
│   └── tokenization_efficiency.py     # Tokenization metrics
│
├── experiments/
│   ├── baseline.py                    # V1 baseline ("respond in Tulu")
│   ├── full_system.py                 # V4 full 5-layer system
│   ├── ablation.py                    # Ablation studies (V1-V4 + component removal)
│   ├── falsification.py               # Falsification with incorrect grammar
│   ├── finetuning.py                  # Fine-tuning comparison (Llama 3.2 3B)
│   ├── evaluate_finetuned.py          # Evaluation for fine-tuned model
│   ├── compare_finetuning.py          # Prompt vs fine-tuning comparison
│   └── prepare_training_data.py       # Training data preparation
│
├── models/
│   ├── base_model.py                  # Abstract base class
│   ├── openai_model.py                # GPT wrapper
│   ├── gemini_model.py                # Gemini wrapper
│   └── llama_model.py                 # Llama wrapper
│
├── utils/
│   ├── logging_utils.py               # Experiment logging
│   └── tokenization.py                # Tokenization utilities
│
└── data/
    ├── tulu_train.json                # Training set
    ├── tulu_dev.json                  # Dev set
    ├── tulu_test.json                 # Test set (100 sentences)
    ├── seed_examples.json             # Few-shot seed examples
    └── tulu_grammar_rules.json        # Grammar rules (JSON)
```

## Running Experiments

### Full pipeline

```bash
python run_all.py
```

This runs: baseline (V1), full system (V4), ablation (V1-V4 + component removal), paper versions, and falsification.

### Individual experiments

```python
from models.openai_model import OpenAIModel
from experiments.full_system import FullSystemExperiment
import json

model = OpenAIModel(model_name="gpt-4o")

with open("data/tulu_test.json") as f:
    test_data = json.load(f)
    test_set = test_data["sentences"] if "sentences" in test_data else test_data

experiment = FullSystemExperiment(model)
results = experiment.run(test_set[:10])

print(f"Contamination: {results['contamination_rate']:.1%}")
print(f"Grammar accuracy: {results['grammar_accuracy']:.1%}")
```

## Key Results

| Version | Contamination | Grammar Accuracy |
|---------|--------------|-----------------|
| V1 (baseline) | ~72% | ~35% |
| V2 (+ grammar) | ~58% | ~48% |
| V3 (+ constraints) | ~22% | ~62% |
| V4 (full system) | ~14% | ~74% |

Negative constraints are the single most impactful layer. Removing them from the full system increases contamination by ~38 percentage points.

## Setup

**Prerequisites:**
- Python 3.9+
- API key for at least one of: OpenAI (GPT-4o), Google Gemini, or a local Llama setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
# or
export GEMINI_API_KEY="your-key-here"
```

## Citation

If you use this code, please cite our paper:

**Preprint:** [arxiv.org/abs/2602.15378](https://arxiv.org/abs/2602.15378)

```bibtex
@inproceedings{devadiga2026tulu,
  title={Making Large Language Models Speak Tulu: Structured Prompting for an Extremely Low-Resource Language},
  author={Devadiga, Prathamesh and Chopra, Paras},
  booktitle={Proceedings of the LoResLM Workshop at EACL 2026},
  year={2026},
  url={https://arxiv.org/abs/2602.15378}
}
```

## License

See [LICENSE](LICENSE).
