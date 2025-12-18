# Fine-tuning Experiment: Llama 3.2 3B with Unsloth

This directory contains code for fine-tuning Llama 3.2 3B on Tulu question-answer pairs using Unsloth, and comparing performance against the prompt-based approach.

## Overview

This experiment fine-tunes **Llama 3.2 3B** on 520 Tulu question-answer pairs using **Unsloth** for efficient training (2-5× faster than standard LoRA), then evaluates performance using the same metrics as the prompt-based approach.

## Budget Constraints

- **Platform**: Lightning AI with 4.5 credits
- **GPU**: Single A10G or L4 (cheapest options)
- **Estimated cost**: ~2-3 credits for complete experiment
- **Training time**: ~15-30 minutes with Unsloth

## Installation

### Step 1: Install Unsloth and Dependencies

```bash
# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install additional dependencies
pip install --no-deps xformers trl peft accelerate bitsandbytes

# Install core dependencies (if not already installed)
pip install transformers>=4.30.0 torch>=2.0.0 datasets>=2.0.0
```

### Step 2: Verify Installation

```python
from unsloth import FastLanguageModel
print("Unsloth installed successfully!")
```

## Quick Start

### 0. Verify Setup (Recommended)

Before starting, verify that everything is set up correctly:

```bash
python experiments/verify_setup.py
```

This will check:
- Unsloth installation
- Required dependencies
- Data files
- GPU availability
- Output directory permissions

### 0.5. Prepare Training Data (Optional)

If you need to combine multiple dataset files to get 520 examples:

```bash
python experiments/prepare_training_data.py
```

This will combine datasets from:
- `data/tulu_train.json`
- `data/tulu_training_dataset.json`
- `data/expanded_tulu_dataset.json`
- `data/enhanced_tulu_lessons.json`
- `data/corrected_tulu_lessons.json`

And save to `data/tulu_train_520.json`.

### 1. Train the Model

```bash
python experiments/finetuning.py
```

This will:
- Load training data from `data/tulu_train.json`
- Load validation data from `data/tulu_dev.json` (or split from training)
- Fine-tune Llama 3.2 3B with Unsloth
- Save model to `outputs/finetuned_llama3.2_3b`

### 2. Evaluate the Model

```bash
python experiments/evaluate_finetuned.py
```

This will:
- Load the fine-tuned model
- Evaluate on test set (`data/tulu_test.json`)
- Compute metrics: contamination, grammar accuracy, vocabulary coverage
- Save results to `outputs/finetuning_evaluation_results.json`

### 3. Compare with Baseline

```bash
python experiments/compare_finetuning.py
```

This will:
- Load fine-tuning results
- Compare with prompt-based baseline
- Generate comparison table (CSV, Markdown, LaTeX)
- Print analysis for paper

## Hyperparameters

Default hyperparameters (optimized for budget):

- **Model**: `unsloth/Llama-3.2-3B-Instruct`
- **LoRA rank**: 16
- **LoRA alpha**: 16
- **LoRA dropout**: 0.0
- **Batch size**: 2 (with gradient accumulation = 4, effective batch = 8)
- **Epochs**: 3
- **Learning rate**: 2e-4
- **Max sequence length**: 512
- **Warmup ratio**: 0.03
- **4-bit quantization**: Enabled

## Data Format

The scripts expect data in JSON format with one of these structures:

**Format 1: Direct list**
```json
[
  {"english": "Hello, how are you?", "tulu": "namaskara, encha ullaru?"},
  ...
]
```

**Format 2: With metadata**
```json
{
  "metadata": {...},
  "sentences": [
    {"english": "Hello, how are you?", "tulu": "namaskara, encha ullaru?"},
    ...
  ]
}
```

The scripts automatically convert `english`/`tulu` fields to `question`/`tulu_response` format for training.

## File Structure

```
experiments/
├── finetuning.py              # Main training script
├── evaluate_finetuned.py      # Evaluation script
├── compare_finetuning.py      # Comparison script
└── README_FINETUNING.md       # This file

outputs/
├── finetuned_llama3.2_3b/     # Saved model checkpoints
├── finetuning_evaluation_results.json  # Evaluation results
└── finetuning_comparison.*    # Comparison tables (CSV, MD, TEX)
```

## Usage Examples

### Custom Model Path

```bash
python experiments/finetuning.py --model_path outputs/my_custom_model
```

### Custom Hyperparameters

Edit `experiments/finetuning.py` and modify the `config` dictionary in the `main()` function:

```python
config = {
    "num_epochs": 5,  # Increase epochs
    "learning_rate": 1e-4,  # Lower learning rate
    "lora_r": 32,  # Increase LoRA rank
    # ... other parameters
}
```

### Evaluate with Custom Settings

```bash
python experiments/evaluate_finetuned.py \
    --model_path outputs/finetuned_llama3.2_3b \
    --test_data data/tulu_test.json \
    --max_tokens 512 \
    --temperature 0.5
```

### Compare with Custom Baseline

```bash
python experiments/compare_finetuning.py \
    --finetuning_results outputs/finetuning_evaluation_results.json \
    --baseline_results outputs/paper_results.json \
    --baseline_grammar 78.0 \
    --baseline_contamination 6.0
```

## Lightning AI Setup

### Step-by-Step Execution

1. **Create Studio** (Lightning AI):
   - Go to lightning.ai
   - Click "Create Studio"
   - Select: "A10G" GPU (24GB, cheapest option ~$1.50/hour)
   - Or: "L4" GPU if available (cheaper)

2. **Open Terminal/Notebook**:
   - Create new Jupyter notebook or terminal
   - Clone repository or upload files

3. **Install Dependencies** (5 minutes):
   ```bash
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   !pip install --no-deps xformers trl peft accelerate bitsandbytes
   ```

4. **Upload Data**:
   - Ensure `data/tulu_train.json` and `data/tulu_test.json` are available

5. **Run Training** (15-30 minutes):
   ```bash
   python experiments/finetuning.py
   ```

6. **Run Evaluation** (5-10 minutes):
   ```bash
   python experiments/evaluate_finetuned.py
   ```

7. **Compare Results** (1 minute):
   ```bash
   python experiments/compare_finetuning.py
   ```

8. **Download Results**:
   - Download `outputs/finetuning_evaluation_results.json`
   - Download comparison tables

9. **Stop Studio**: Don't forget to stop the studio to save credits!

## Budget Breakdown

| Activity | Time | Cost (credits) |
|----------|------|----------------|
| Setup & install | 5 min | ~0.1 |
| Training (3 epochs) | 20 min | ~0.5 |
| Evaluation | 10 min | ~0.25 |
| Analysis & export | 10 min | ~0.25 |
| **Buffer** | 10 min | ~0.25 |
| **Total** | ~55 min | **~1.35 credits** |

You'll have 3+ credits remaining for:
- Re-running with different hyperparameters
- Trying 5 epochs instead of 3
- Testing on additional examples
- Debugging if needed

## Evaluation Metrics

The evaluation uses the same metrics as the prompt-based approach:

1. **Contamination Rate**: Percentage of responses containing prohibited Kannada words
2. **Grammar Accuracy**: Rule-based grammar checking (verb conjugation, case markers, word order)
3. **Vocabulary Coverage**: Vocabulary size and type-token ratio
4. **Tokenization Efficiency**: Average tokens per word

## Expected Results

### Scenario 1: Fine-tuning underperforms (most likely)
- Fine-tuned 3B: 60-70% grammar, 15-25% contamination
- Prompt-based 70B: 78% grammar, 6% contamination
- **Analysis**: Smaller model struggles with low-resource language, even with fine-tuning

### Scenario 2: Fine-tuning competitive
- Fine-tuned 3B: 70-75% grammar, 8-12% contamination
- **Analysis**: Fine-tuning can partially compensate for model size

### Scenario 3: Fine-tuning excels (unlikely with 3B)
- Fine-tuned 3B: >80% grammar, <8% contamination
- **Analysis**: Data-specific fine-tuning beats general prompting

## Troubleshooting

### Import Error: Unsloth not found
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

### CUDA Out of Memory
- Reduce `per_device_batch_size` to 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Ensure `load_in_4bit=True`

### Model Not Found
- Check that model path exists: `outputs/finetuned_llama3.2_3b`
- Run training first: `python experiments/finetuning.py`

### Data Format Error
- Ensure JSON files have `english` and `tulu` fields (or `question` and `tulu_response`)
- Check file encoding is UTF-8

## Output Files

### Model Checkpoints
- `outputs/finetuned_llama3.2_3b/`: Model weights, tokenizer, config

### Evaluation Results
- `outputs/finetuning_evaluation_results.json`: Complete evaluation results
  - Per-example results
  - Aggregate metrics
  - Detailed contamination/grammar analysis

### Comparison Tables
- `outputs/finetuning_comparison.csv`: CSV format
- `outputs/finetuning_comparison.md`: Markdown format
- `outputs/finetuning_comparison.tex`: LaTeX format (for paper)

## For Your Paper

After running the experiment, add this to your Results section:

```latex
\subsection{Fine-tuning Comparison}

We compared our prompt-based approach against fine-tuning Llama 3.2 3B 
on our 520 training examples using Unsloth for efficient training. 
Fine-tuning achieved X\% grammar accuracy and Y\% contamination on 
the same 100-example test set (Table~\ref{tab:finetuning_comparison}).

[Choose based on results:]
- [If worse]: While fine-tuning provides task-specific adaptation, our 
  prompt-based approach with larger models (70B) substantially outperforms 
  fine-tuned smaller models (3B). This suggests that for extremely 
  low-resource languages, access to larger base models via prompting may 
  be more effective than fine-tuning smaller models, particularly when 
  training data is limited (<1000 examples).

- [If similar]: Fine-tuning achieves comparable performance to our 
  prompt-based approach, demonstrating that both strategies are viable. 
  However, prompt-based approaches require no training infrastructure 
  and can leverage the latest models via APIs, while fine-tuning offers 
  deployment control and lower inference costs.

Training completed in approximately 20 minutes on a single A10G GPU 
using Unsloth's optimized implementation.
```

## References

- **Unsloth**: https://github.com/unslothai/unsloth
- **Llama 3.2**: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- **Lightning AI**: https://lightning.ai

## License

This code follows the same license as the main repository.

