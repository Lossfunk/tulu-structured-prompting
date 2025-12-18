# Fine-tuning Experiment Code Summary

This document summarizes the fine-tuning experiment code that has been created.

## Files Created

### Main Scripts

1. **`finetuning.py`** - Main training script
   - Loads Llama 3.2 3B with Unsloth optimizations
   - Formats data for instruction-following
   - Trains model with LoRA
   - Saves checkpoints

2. **`evaluate_finetuned.py`** - Evaluation script
   - Loads fine-tuned model
   - Generates responses on test set
   - Computes all metrics (contamination, grammar, vocabulary, tokenization)
   - Saves detailed results

3. **`compare_finetuning.py`** - Comparison script
   - Loads fine-tuning and baseline results
   - Creates comparison table (CSV, Markdown, LaTeX)
   - Generates analysis text for paper

### Utility Scripts

4. **`prepare_training_data.py`** - Data preparation utility
   - Combines multiple dataset files
   - Deduplicates examples
   - Creates 520-example training set

5. **`verify_setup.py`** - Setup verification
   - Checks Unsloth installation
   - Verifies dependencies
   - Checks data files
   - Verifies GPU availability

### Documentation

6. **`README_FINETUNING.md`** - Comprehensive guide
   - Installation instructions
   - Usage examples
   - Lightning AI setup guide
   - Troubleshooting

## Key Features

### Data Handling
- Automatically handles different JSON formats (list vs. metadata wrapper)
- Converts `english`/`tulu` to `question`/`tulu_response` format
- Supports validation split from dev set or automatic split

### Model Configuration
- Uses Unsloth for 2-5× faster training
- 4-bit quantization for memory efficiency
- LoRA with rank 16, alpha 16
- Optimized hyperparameters for budget constraints

### Evaluation
- Integrates with existing evaluators:
  - `ContaminationEvaluator`
  - `GrammarAccuracyEvaluator`
  - `VocabularyCoverageEvaluator`
  - `TokenizationEfficiencyEvaluator`
- Generates per-example and aggregate results

### Comparison
- Creates publication-ready tables
- Generates LaTeX for paper inclusion
- Provides analysis text based on results

## Usage Workflow

```bash
# 1. Verify setup
python experiments/verify_setup.py

# 2. Prepare data (if needed)
python experiments/prepare_training_data.py

# 3. Train model
python experiments/finetuning.py

# 4. Evaluate model
python experiments/evaluate_finetuned.py

# 5. Compare with baseline
python experiments/compare_finetuning.py
```

## Integration with Existing Codebase

The fine-tuning experiment integrates seamlessly with existing code:

- **Uses existing evaluators** from `evaluation/` directory
- **Uses existing data format** from `data/` directory
- **Follows same structure** as other experiments in `experiments/`
- **Outputs to same directory** (`outputs/`) as other experiments

## Dependencies Added

Updated `requirements.txt` with:
- Unsloth (install via git)
- xformers, trl, peft, accelerate, bitsandbytes
- transformers, torch, datasets (for fine-tuning)

## Expected Outputs

After running the full workflow:

1. **Model checkpoints**: `outputs/finetuned_llama3.2_3b/`
2. **Evaluation results**: `outputs/finetuning_evaluation_results.json`
3. **Comparison tables**: 
   - `outputs/finetuning_comparison.csv`
   - `outputs/finetuning_comparison.md`
   - `outputs/finetuning_comparison.tex`

## Next Steps

1. **Install Unsloth** on Lightning AI or local GPU machine
2. **Run verification** to ensure setup is correct
3. **Train model** (takes ~15-30 minutes)
4. **Evaluate** on test set
5. **Compare** with prompt-based baseline
6. **Add results** to paper using generated LaTeX

## Notes

- The code is designed to work with the existing data format
- All hyperparameters are optimized for budget constraints (4.5 credits)
- The evaluation uses the same metrics as the prompt-based approach for fair comparison
- Error handling is included for common issues (missing files, import errors, etc.)

