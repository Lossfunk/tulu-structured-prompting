#!/usr/bin/env python3
"""
Generate Results Visualizations

Generates all graphs and tables from experiment results.
Run this after running experiments to create publication-ready visualizations.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from results.visualize_results import ResultsVisualizer


def main():
    """Generate all visualizations from results."""
    print("=" * 60)
    print("Generating Results Visualizations")
    print("=" * 60)
    
    results_file = "outputs/all_experiments_results.json"
    
    if not os.path.exists(results_file):
        print(f"\nError: Results file not found: {results_file}")
        print("Please run experiments first using: python run_all.py")
        return
    
    visualizer = ResultsVisualizer(output_dir="results/figures")
    visualizer.generate_all_visualizations(results_file)
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"\nAll figures saved to: results/figures/")
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

