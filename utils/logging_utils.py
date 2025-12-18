"""Logging utilities for experiments."""

import json
import os
from datetime import datetime
from typing import Dict
from ..config import Config


def save_experiment_results(results: Dict, experiment_name: str):
    """
    Save experiment results to JSON file.
    
    Args:
        results: Experiment results dictionary
        experiment_name: Name of experiment
    """
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    output_path = os.path.join(
        Config.OUTPUT_DIR,
        f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")
    return output_path

