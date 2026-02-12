# Evaluation script for fine-tuned Llama 3.2 3B model.

import os
import json
import sys
from pathlib import Path
from typing import List, Dict
import torch

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Change to project root to fix relative imports
original_cwd = os.getcwd()
os.chdir(project_root)

from experiments.finetuning import (
    load_model_with_unsloth,
    load_tulu_dataset,
    generate_response,
    format_llama3_prompt,
    UNSLOTH_AVAILABLE
)

import importlib
import importlib.util

# Load constraints.negative_constraints first (needed by evaluation.contamination)
negative_constraints_path = project_root / "constraints" / "negative_constraints.py"
if negative_constraints_path.exists():
    if "constraints" not in sys.modules:
        constraints_pkg = type(sys)("constraints")
        constraints_pkg.__path__ = [str(project_root / "constraints")]
        sys.modules["constraints"] = constraints_pkg

    spec = importlib.util.spec_from_file_location(
        "constraints.negative_constraints",
        negative_constraints_path
    )
    negative_constraints_mod = importlib.util.module_from_spec(spec)
    negative_constraints_mod.__package__ = "constraints"
    sys.modules["constraints.negative_constraints"] = negative_constraints_mod
    spec.loader.exec_module(negative_constraints_mod)

if "evaluation" not in sys.modules:
    evaluation_pkg = type(sys)("evaluation")
    evaluation_pkg.__path__ = [str(project_root / "evaluation")]
    sys.modules["evaluation"] = evaluation_pkg

if "grammar" not in sys.modules:
    grammar_pkg = type(sys)("grammar")
    grammar_pkg.__path__ = [str(project_root / "grammar")]
    sys.modules["grammar"] = grammar_pkg
    tulu_grammar_path = project_root / "grammar" / "tulu_grammar.py"
    if tulu_grammar_path.exists():
        spec = importlib.util.spec_from_file_location(
            "grammar.tulu_grammar",
            tulu_grammar_path
        )
        tulu_grammar_mod = importlib.util.module_from_spec(spec)
        tulu_grammar_mod.__package__ = "grammar"
        sys.modules["grammar.tulu_grammar"] = tulu_grammar_mod
        spec.loader.exec_module(tulu_grammar_mod)
        setattr(grammar_pkg, "tulu_grammar", tulu_grammar_mod)

contamination_path = project_root / "evaluation" / "contamination.py"
if contamination_path.exists():
    with open(contamination_path, 'r', encoding='utf-8') as f:
        contamination_code = f.read()
    contamination_code = contamination_code.replace(
        "from ..constraints.negative_constraints import NegativeConstraints",
        "from constraints.negative_constraints import NegativeConstraints"
    )
    contamination_mod = type(sys)("evaluation.contamination")
    contamination_mod.__package__ = "evaluation"
    contamination_mod.__file__ = str(contamination_path)
    exec(compile(contamination_code, str(contamination_path), 'exec'), contamination_mod.__dict__)
    sys.modules["evaluation.contamination"] = contamination_mod
    ContaminationEvaluator = contamination_mod.ContaminationEvaluator
else:
    raise ImportError(f"Could not find contamination.py at {contamination_path}")

grammar_path = project_root / "evaluation" / "grammar_accuracy.py"
if grammar_path.exists():
    with open(grammar_path, 'r', encoding='utf-8') as f:
        grammar_code = f.read()
    grammar_code = grammar_code.replace(
        "from ..grammar.tulu_grammar import TuluGrammar",
        "from grammar.tulu_grammar import TuluGrammar"
    )
    grammar_mod = type(sys)("evaluation.grammar_accuracy")
    grammar_mod.__package__ = "evaluation"
    grammar_mod.__file__ = str(grammar_path)
    exec(compile(grammar_code, str(grammar_path), 'exec'), grammar_mod.__dict__)
    sys.modules["evaluation.grammar_accuracy"] = grammar_mod
    GrammarAccuracyEvaluator = grammar_mod.GrammarAccuracyEvaluator
else:
    raise ImportError(f"Could not find grammar_accuracy.py at {grammar_path}")

vocab_path = project_root / "evaluation" / "vocabulary_coverage.py"
if vocab_path.exists():
    spec = importlib.util.spec_from_file_location(
        "evaluation.vocabulary_coverage",
        vocab_path
    )
    vocab_mod = importlib.util.module_from_spec(spec)
    vocab_mod.__package__ = "evaluation"
    sys.modules["evaluation.vocabulary_coverage"] = vocab_mod
    spec.loader.exec_module(vocab_mod)
    VocabularyCoverageEvaluator = vocab_mod.VocabularyCoverageEvaluator
else:
    raise ImportError(f"Could not find vocabulary_coverage.py at {vocab_path}")

tokenization_path = project_root / "evaluation" / "tokenization_efficiency.py"
if tokenization_path.exists():
    with open(tokenization_path, 'r', encoding='utf-8') as f:
        tokenization_code = f.read()
    if "from .." in tokenization_code:
        tokenization_code = tokenization_code.replace(
            "from ..romanization.romanization_rules import RomanizationSystem",
            "from romanization.romanization_rules import RomanizationSystem"
        )
        tokenization_mod = type(sys)("evaluation.tokenization_efficiency")
        tokenization_mod.__package__ = "evaluation"
        tokenization_mod.__file__ = str(tokenization_path)
        exec(compile(tokenization_code, str(tokenization_path), 'exec'), tokenization_mod.__dict__)
        sys.modules["evaluation.tokenization_efficiency"] = tokenization_mod
    else:
        spec = importlib.util.spec_from_file_location(
            "evaluation.tokenization_efficiency",
            tokenization_path
        )
        tokenization_mod = importlib.util.module_from_spec(spec)
        tokenization_mod.__package__ = "evaluation"
        sys.modules["evaluation.tokenization_efficiency"] = tokenization_mod
        spec.loader.exec_module(tokenization_mod)
    TokenizationEfficiencyEvaluator = tokenization_mod.TokenizationEfficiencyEvaluator
else:
    raise ImportError(f"Could not find tokenization_efficiency.py at {tokenization_path}")


def evaluate_finetuned_model(
    model_path: str = "outputs/finetuned_llama3.2_3b",
    test_data_path: str = "data/tulu_test.json",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 1,
) -> Dict:
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth is not available. Please install it first.")

    print("=" * 70)
    print("Evaluating Fine-tuned Llama 3.2 3B")
    print("=" * 70)

    print(f"\nLoading test data from {test_data_path}...")
    test_data = load_tulu_dataset(test_data_path)
    print(f"  Test examples: {len(test_data)}")

    print(f"\nLoading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first using:\n"
            "  python experiments/finetuning.py"
        )

    try:
        from unsloth import FastLanguageModel
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
            )
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
            )
        FastLanguageModel.for_inference(model)
        print("  Model loaded successfully!")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        print(f"  Make sure the model was trained and saved correctly.")
        raise

    contamination_eval = ContaminationEvaluator()
    grammar_eval = GrammarAccuracyEvaluator()
    vocab_eval = VocabularyCoverageEvaluator()
    tokenization_eval = TokenizationEfficiencyEvaluator()

    print(f"\nGenerating responses for {len(test_data)} test examples...")
    generated_responses = []
    reference_responses = []
    questions = []

    for i, example in enumerate(test_data):
        question = example["question"]
        reference = example["tulu_response"]

        print(f"  [{i+1}/{len(test_data)}] Generating response...", end="\r")

        try:
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            generated_responses.append(response)
            reference_responses.append(reference)
            questions.append(question)
        except Exception as e:
            print(f"\n  ERROR generating response for example {i+1}: {e}")
            generated_responses.append("")
            reference_responses.append(reference)
            questions.append(question)

    print(f"\n  Generated {len(generated_responses)} responses")

    print("\nEvaluating contamination...")
    contamination_results = contamination_eval.compute_contamination_rate(generated_responses)
    contamination_rate = contamination_results["contamination_percentage"]
    print(f"  Contamination rate: {contamination_rate:.2f}%")
    print(f"  Contaminated responses: {contamination_results['contaminated_responses']}/{contamination_results['total_responses']}")

    print("\nEvaluating grammar accuracy...")
    grammar_results = grammar_eval.compute_grammar_accuracy(generated_responses)
    grammar_accuracy = grammar_results["grammar_accuracy_percentage"]
    print(f"  Grammar accuracy: {grammar_accuracy:.2f}%")
    print(f"  Correct responses: {grammar_results['correct_responses']}/{grammar_results['total_responses']}")

    print("\nEvaluating vocabulary coverage...")
    vocab_results = vocab_eval.compute_vocabulary_coverage(generated_responses)
    vocab_size = vocab_results["vocabulary_size"]
    ttr = vocab_results["type_token_ratio"]
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Type-token ratio: {ttr:.4f}")

    print("\nEvaluating tokenization efficiency...")
    tokenization_results = tokenization_eval.evaluate_corpus(generated_responses)
    avg_tokens_per_word = tokenization_results.get("average_tokens_per_word", 0.0)
    print(f"  Average tokens per word: {avg_tokens_per_word:.2f}")

    results = {
        "model": "Llama-3.2-3B-Instruct (Fine-tuned)",
        "test_examples": len(test_data),
        "contamination_rate": contamination_rate,
        "grammar_accuracy": grammar_accuracy,
        "vocabulary_size": vocab_size,
        "type_token_ratio": ttr,
        "average_tokens_per_word": avg_tokens_per_word,
        "detailed_results": {
            "contamination": contamination_results,
            "grammar": grammar_results,
            "vocabulary": vocab_results,
            "tokenization": tokenization_results,
        },
        "per_example_results": [
            {
                "question": q,
                "reference": ref,
                "generated": gen,
                "contamination": contamination_eval.evaluate_single(gen),
                "grammar": grammar_eval.evaluate_single(gen),
            }
            for q, ref, gen in zip(questions, reference_responses, generated_responses)
        ],
    }

    return results


def print_results_summary(results: Dict):
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nModel: {results['model']}")
    print(f"Test examples: {results['test_examples']}")
    print(f"\nMetrics:")
    print(f"  Grammar Accuracy:        {results['grammar_accuracy']:.2f}%")
    print(f"  Contamination Rate:     {results['contamination_rate']:.2f}%")
    print(f"  Vocabulary Size:        {results['vocabulary_size']}")
    print(f"  Type-Token Ratio:       {results['type_token_ratio']:.4f}")
    print(f"  Avg Tokens/Word:        {results['average_tokens_per_word']:.2f}")
    print("\n" + "=" * 70)


def save_results(results: Dict, output_path: str = "outputs/finetuning_evaluation_results.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


def main():
    global original_cwd

    if not UNSLOTH_AVAILABLE:
        print("ERROR: Unsloth is not available. Please install it first.")
        os.chdir(original_cwd)
        return

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Llama 3.2 3B model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/finetuned_llama3.2_3b",
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/tulu_test.json",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/finetuning_evaluation_results.json",
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p"
    )

    args = parser.parse_args()

    try:
        results = evaluate_finetuned_model(
            model_path=args.model_path,
            test_data_path=args.test_data,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print_results_summary(results)
        save_results(results, args.output)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_cwd)
        return 1

    os.chdir(original_cwd)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if 'original_cwd' in globals():
            os.chdir(original_cwd)
