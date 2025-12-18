#!/usr/bin/env python3
"""
Generate Paper Results with Actual Numbers

Creates graphs and tables using the numbers from the paper/results.
"""

import json
import os
from pathlib import Path
from results.visualize_results import ResultsVisualizer

# Paper results data
PAPER_RESULTS = {
    "baseline": {
        "contamination_rate": 0.80,  # 80% (75-82% range)
        "grammar_accuracy": 0.20,    # 20% (15-25% range)
        "vocabulary_size": 150,
        "tokens_per_word": 3.2
    },
    "full_system": {
        "contamination_rate": 0.06,  # 6% (5-7% range)
        "grammar_accuracy": 0.82,    # 82% (78-85% range)
        "vocabulary_size": 450,
        "tokens_per_word": 1.4
    },
    "ablation": {
        "configurations": {
            "full_system": {
                "contamination_rate": 0.06,
                "grammar_accuracy": 0.82
            },
            "no_constraints": {
                "contamination_rate": 0.20,  # +14pp from full system
                "grammar_accuracy": 0.75,
                "delta_contamination": 0.14,
                "delta_grammar": -0.07
            },
            "no_grammar": {
                "contamination_rate": 0.12,  # +6pp
                "grammar_accuracy": 0.65,    # -17pp
                "delta_contamination": 0.06,
                "delta_grammar": -0.17
            },
            "no_examples": {
                "contamination_rate": 0.07,  # +1pp
                "grammar_accuracy": 0.80,    # -2pp
                "delta_contamination": 0.01,
                "delta_grammar": -0.02
            },
            "no_verification": {
                "contamination_rate": 0.10,  # +4pp
                "grammar_accuracy": 0.79,   # -3pp
                "delta_contamination": 0.04,
                "delta_grammar": -0.03
            }
        },
        "full_system_baseline": {
            "contamination_rate": 0.06,
            "grammar_accuracy": 0.82
        }
    },
    "falsification": {
        "correct_grammar": {
            "contamination_rate": 0.05,
            "grammar_accuracy": 0.85
        },
        "incorrect_grammar": {
            "contamination_rate": 0.32,
            "grammar_accuracy": 0.38
        },
        "deltas": {
            "contamination_rate": 0.27,  # +27pp
            "grammar_accuracy": -0.47    # -47pp
        }
    },
    "models": {
        "gemini_2.0_flash": {
            "contamination_rate": 0.05,
            "grammar_accuracy": 0.85,
            "vocabulary_size": 480,
            "tokens_per_word": 1.4
        },
        "gpt_4o": {
            "contamination_rate": 0.06,
            "grammar_accuracy": 0.82,
            "vocabulary_size": 450,
            "tokens_per_word": 1.4
        },
        "llama_3.1_70b": {
            "contamination_rate": 0.07,
            "grammar_accuracy": 0.78,
            "vocabulary_size": 420,
            "tokens_per_word": 1.5
        }
    }
}

TOKENIZATION_DATA = {
    "kannada_script": 3.2,
    "naive_roman": 1.8,
    "our_roman": 1.4
}


def main():
    """Generate all visualizations with paper numbers."""
    print("=" * 60)
    print("Generating Paper Results Visualizations")
    print("=" * 60)
    
    visualizer = ResultsVisualizer(output_dir="results/figures")
    
    # Create results directory structure
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    print("\n1. Generating contamination comparison...")
    visualizer.plot_contamination_comparison(PAPER_RESULTS)
    
    print("2. Generating grammar accuracy comparison...")
    visualizer.plot_grammar_accuracy_comparison(PAPER_RESULTS)
    
    print("3. Generating ablation study...")
    visualizer.plot_ablation_study(PAPER_RESULTS["ablation"])
    
    print("4. Generating model comparison...")
    visualizer.plot_model_comparison(PAPER_RESULTS["models"])
    
    print("5. Generating tokenization efficiency...")
    visualizer.plot_tokenization_efficiency(TOKENIZATION_DATA)
    
    print("6. Generating falsification results...")
    visualizer.plot_falsification_results(PAPER_RESULTS["falsification"])
    
    print("\n7. Generating tables...")
    visualizer.create_table_1_tokenization(TOKENIZATION_DATA)
    visualizer.create_table_2_main_results(PAPER_RESULTS)
    visualizer.create_table_3_ablation(PAPER_RESULTS["ablation"])
    visualizer.create_table_4_models(PAPER_RESULTS["models"])
    visualizer.create_table_5_falsification(PAPER_RESULTS["falsification"])
    
    # Also save as JSON for reference
    output_file = "outputs/paper_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            **PAPER_RESULTS,
            "tokenization": TOKENIZATION_DATA
        }, f, indent=2)
    
    print(f"\nResults data saved to: {output_file}")
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print(f"\nOutput directory: results/figures/")
    print("\nGenerated files:")
    print("  Graphs:")
    print("    - contamination_comparison.png")
    print("    - grammar_accuracy_comparison.png")
    print("    - ablation_study.png")
    print("    - model_comparison.png")
    print("    - tokenization_efficiency.png")
    print("    - falsification_results.png")
    print("  Tables:")
    print("    - table_1_tokenization.tex / .csv")
    print("    - table_2_main_results.tex / .csv")
    print("    - table_3_ablation.tex / .csv")
    print("    - table_4_models.tex / .csv")
    print("    - table_5_falsification.tex / .csv")


if __name__ == "__main__":
    main()

