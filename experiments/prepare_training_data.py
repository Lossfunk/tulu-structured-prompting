# Utility script to prepare training data for fine-tuning.

import os
import json
from typing import List, Dict
from pathlib import Path


def load_dataset_file(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        examples = data
    elif "sentences" in data:
        examples = data["sentences"]
    elif "examples" in data:
        examples = data["examples"]
    else:
        print(f"Warning: Unknown format in {file_path}")
        return []

    formatted = []
    for ex in examples:
        if "english" in ex and "tulu" in ex:
            formatted.append({
                "english": ex["english"],
                "tulu": ex["tulu"]
            })
        elif "question" in ex and "tulu_response" in ex:
            formatted.append({
                "english": ex["question"],
                "tulu": ex["tulu_response"]
            })
        else:
            print(f"Warning: Skipping example missing required fields")

    return formatted


def combine_datasets(
    data_files: List[str],
    target_size: int = 520,
    output_path: str = "data/tulu_train_520.json",
    deduplicate: bool = True
) -> List[Dict]:
    all_examples = []
    seen = set()

    print(f"Loading datasets from {len(data_files)} files...")
    for file_path in data_files:
        examples = load_dataset_file(file_path)
        print(f"  {file_path}: {len(examples)} examples")

        for ex in examples:
            key = (ex["english"].lower().strip(), ex["tulu"].lower().strip())

            if deduplicate and key in seen:
                continue

            seen.add(key)
            all_examples.append(ex)

    print(f"\nTotal unique examples: {len(all_examples)}")

    if len(all_examples) > target_size:
        print(f"Trimming to {target_size} examples...")
        all_examples = all_examples[:target_size]
    elif len(all_examples) < target_size:
        print(f"Warning: Only {len(all_examples)} examples available, target is {target_size}")

    output_data = {
        "metadata": {
            "total_sentences": len(all_examples),
            "split": "train",
            "description": f"Combined training set for fine-tuning ({len(all_examples)} examples)",
            "source_files": data_files,
            "target_size": target_size,
        },
        "sentences": all_examples
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nCombined dataset saved to: {output_path}")
    print(f"  Total examples: {len(all_examples)}")

    return all_examples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare training data for fine-tuning")
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        default=[
            "data/tulu_train.json",
            "data/tulu_training_dataset.json",
            "data/expanded_tulu_dataset.json",
            "data/enhanced_tulu_lessons.json",
            "data/corrected_tulu_lessons.json",
        ],
        help="List of dataset files to combine"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=520,
        help="Target number of training examples"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tulu_train_520.json",
        help="Output path for combined dataset"
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Don't remove duplicate examples"
    )

    args = parser.parse_args()

    existing_files = [f for f in args.data_files if os.path.exists(f)]
    if not existing_files:
        print("ERROR: No existing data files found!")
        print(f"  Looked for: {args.data_files}")
        return 1

    print("=" * 70)
    print("Preparing Training Data for Fine-tuning")
    print("=" * 70)

    combine_datasets(
        data_files=existing_files,
        target_size=args.target_size,
        output_path=args.output,
        deduplicate=not args.no_deduplicate,
    )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
