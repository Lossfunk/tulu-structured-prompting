#!/usr/bin/env python3
# Main orchestration script for running all experiments

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from models.openai_model import OpenAIModel
from models.gemini_model import GeminiModel
from experiments.baseline import BaselineExperiment
from experiments.full_system import FullSystemExperiment
from experiments.ablation import AblationExperiment
from experiments.falsification import FalsificationExperiment


def load_test_set(test_set_path: str = "data/tulu_test.json"):
    if not os.path.exists(test_set_path):
        print(f"Warning: Test set not found at {test_set_path}")
        return []

    with open(test_set_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif "test" in data:
        return data["test"]
    elif "examples" in data:
        return data["examples"]
    elif "sentences" in data:
        return data["sentences"]
    else:
        return []


def load_few_shot_examples(seed_path: str = "data/seed_examples.json"):
    if not os.path.exists(seed_path):
        print(f"Warning: Seed examples not found at {seed_path}")
        return []

    with open(seed_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data[:15]
    elif "examples" in data:
        return data["examples"][:15]
    else:
        return []


def main():
    print("=" * 60)
    print("Tulu LLM Experiments")
    print("=" * 60)

    if not Config.validate():
        print("Error: API keys not configured")
        return

    print("\nLoading data...")
    test_set = load_test_set()
    few_shot = load_few_shot_examples()

    if not test_set:
        print("Error: No test set found")
        return

    print(f"  Test set: {len(test_set)} examples")
    print(f"  Few-shot examples: {len(few_shot)} examples")

    model_type = os.getenv("MODEL_TYPE", "openai").lower()
    model_name = os.getenv("MODEL_NAME", Config.DEFAULT_MODEL)

    print(f"\nInitializing model: {model_type} ({model_name})...")

    if model_type == "openai":
        model = OpenAIModel(model_name=model_name)
    elif model_type == "gemini":
        model = GeminiModel(model_name=model_name)
    else:
        print(f"Error: Unknown model type {model_type}")
        return

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    results = {}

    # 1. Baseline experiment
    print("\n" + "=" * 60)
    print("Experiment 1: Baseline")
    print("=" * 60)
    baseline = BaselineExperiment(model)
    results["baseline"] = baseline.run(test_set[:20])

    # 2. Full system experiment
    print("\n" + "=" * 60)
    print("Experiment 2: Full System")
    print("=" * 60)
    full_system = FullSystemExperiment(model, few_shot_examples=few_shot)
    results["full_system"] = full_system.run(test_set[:20])

    # 3. Ablation study
    print("\n" + "=" * 60)
    print("Experiment 3: Ablation Study")
    print("=" * 60)
    ablation = AblationExperiment(model, few_shot_examples=few_shot)
    results["ablation"] = ablation.run_all_ablations(test_set[:20])

    # 4. Paper versions (V1–V4)
    print("\n" + "=" * 60)
    print("Experiment 4: Paper Versions (V1–V4)")
    print("=" * 60)
    paper_v234 = ablation.run_paper_versions(test_set[:20])
    results["paper_versions"] = {
        "V1": {
            "description": "Simple 'respond in Tulu' baseline",
            "contamination_rate": results["baseline"]["contamination_rate"],
            "grammar_accuracy": results["baseline"]["grammar_accuracy"],
            "total_examples": len(test_set[:20]),
            "note": "From BaselineExperiment"
        },
        "V2": paper_v234["versions"]["V2"],
        "V3": paper_v234["versions"]["V3"],
        "V4": paper_v234["versions"]["V4"],
        "model": paper_v234["model"]
    }

    # 5. Falsification experiment
    print("\n" + "=" * 60)
    print("Experiment 5: Falsification")
    print("=" * 60)
    falsification = FalsificationExperiment(model, few_shot_examples=few_shot)
    results["falsification"] = falsification.run_comparison(test_set[:20])

    output_path = os.path.join(Config.OUTPUT_DIR, "all_experiments_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print("\nSummary:")
    print(f"  Baseline (V1) - Contamination: {results['baseline']['contamination_rate']:.1%}, "
          f"Grammar: {results['baseline']['grammar_accuracy']:.1%}")
    print(f"  Full System (V4) - Contamination: {results['full_system']['contamination_rate']:.1%}, "
          f"Grammar: {results['full_system']['grammar_accuracy']:.1%}")
    pv = results.get("paper_versions", {})
    if pv:
        print("  Paper versions: V1 (baseline), V2 (+grammar), V3 (+constraints), V4 (full)")
        for v in ["V2", "V3", "V4"]:
            if v in pv:
                print(f"    {v} - Contamination: {pv[v]['contamination_rate']:.1%}, Grammar: {pv[v]['grammar_accuracy']:.1%}")


if __name__ == "__main__":
    main()
