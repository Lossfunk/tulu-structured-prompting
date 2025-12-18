# Data Directory

This directory contains the datasets and prompts used for Tulu language learning experiments.

## Required Files

### Test Set
- **`tulu_test.json`** - Test set for evaluation (100 Q&A pairs)
  - Format: `[{"english": "...", "tulu": "..."}, ...]`

### Few-Shot Examples
- **`seed_examples.json`** - Seed examples for few-shot learning (10-15 examples)
  - Format: `[{"english": "...", "tulu": "..."}, ...]`

## Optional Files

- **`tulu_train.json`** - Training dataset
- **`tulu_dev.json`** - Development/validation set
- **`tulu_grammar_rules.json`** - Grammar rules in JSON format
- Other dataset files as needed

## Data Format

All JSON files should follow this structure:

```json
[
  {
    "english": "Hello, how are you?",
    "tulu": "namaskara, encha ullaru?"
  },
  {
    "english": "What is your name?",
    "tulu": "ninna pudar encha?"
  }
]
```

## Usage

The code automatically loads:
- Test set from `data/tulu_test.json` (used in `run_all.py`)
- Few-shot examples from `data/seed_examples.json` (used in experiments)

## License

All data files in this directory are part of the open source release and follow the repository's LICENSE.


