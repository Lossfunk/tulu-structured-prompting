"""
Verification script to check if fine-tuning setup is correct.

Checks:
- Unsloth installation
- Required data files
- GPU availability
- Dependencies
"""

import os
import sys
from pathlib import Path


def check_unsloth():
    """Check if Unsloth is installed."""
    print("Checking Unsloth installation...")
    try:
        from unsloth import FastLanguageModel
        print("  ✓ Unsloth is installed")
        return True
    except ImportError:
        print("  ✗ Unsloth is NOT installed")
        print("    Install with:")
        print('      pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
        print("      pip install --no-deps xformers trl peft accelerate bitsandbytes")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("trl", "TRL"),
        ("peft", "PEFT"),
        ("accelerate", "Accelerate"),
        ("bitsandbytes", "BitsAndBytes"),
    ]
    
    all_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name} is installed")
        except ImportError:
            print(f"  ✗ {name} is NOT installed")
            all_ok = False
    
    return all_ok


def check_data_files():
    """Check if required data files exist."""
    print("\nChecking data files...")
    data_files = [
        ("data/tulu_train.json", "Training data"),
        ("data/tulu_test.json", "Test data"),
    ]
    
    all_ok = True
    for file_path, description in data_files:
        if os.path.exists(file_path):
            # Try to load and check format
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"  ✓ {description}: {file_path}")
            except Exception as e:
                print(f"  ✗ {description}: {file_path} (error loading: {e})")
                all_ok = False
        else:
            print(f"  ✗ {description}: {file_path} (not found)")
            all_ok = False
    
    return all_ok


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU available: {gpu_name}")
            print(f"    Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("  ✗ No GPU available (CUDA not available)")
            print("    Fine-tuning requires a GPU (A10G, L4, or similar)")
            return False
    except ImportError:
        print("  ✗ PyTorch not installed, cannot check GPU")
        return False


def check_output_directory():
    """Check if output directory is writable."""
    print("\nChecking output directory...")
    output_dir = "outputs"
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, ".test_write")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"  ✓ Output directory is writable: {output_dir}")
        return True
    except Exception as e:
        print(f"  ✗ Cannot write to output directory: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 70)
    print("Fine-tuning Setup Verification")
    print("=" * 70)
    
    checks = [
        ("Unsloth", check_unsloth),
        ("Dependencies", check_dependencies),
        ("Data Files", check_data_files),
        ("GPU", check_gpu),
        ("Output Directory", check_output_directory),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    if all_passed:
        print("\n✓ All checks passed! You're ready to run fine-tuning.")
        print("\nNext steps:")
        print("  1. Prepare data (if needed): python experiments/prepare_training_data.py")
        print("  2. Train model: python experiments/finetuning.py")
        print("  3. Evaluate: python experiments/evaluate_finetuned.py")
        print("  4. Compare: python experiments/compare_finetuning.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

