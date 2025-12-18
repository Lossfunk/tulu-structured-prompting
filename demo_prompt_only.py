#!/usr/bin/env python3
"""
Demo: Show Generated Prompt (No API Call)

Shows what the full 5-layer prompt looks like without calling the API.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from constraints.negative_constraints import NegativeConstraints
from grammar.tulu_grammar import TuluGrammar
from prompts.tulu_prompt import TuluPromptBuilder


def demo_prompt():
    """Show the generated prompt structure."""
    print("=" * 60)
    print("Tulu LLM System - Prompt Demo")
    print("=" * 60)
    
    # Load components
    print("\n1. Loading components...")
    constraints = NegativeConstraints()
    grammar = TuluGrammar()
    prompt_builder = TuluPromptBuilder()
    
    print(f"   - Negative constraints: {len(constraints.get_all_constraints())} mappings")
    print(f"   - Grammar: {grammar.verb_conjugation['total_verbs']} verbs, {grammar.case_marking['total_cases']} cases")
    
    # Create test input
    test_input = "Hello, how are you?"
    
    print(f"\n2. Building 5-layer prompt for: '{test_input}'")
    print("   " + "-" * 56)
    
    # Build prompt
    prompt = prompt_builder.build_full_prompt(
        user_input=test_input,
        negative_constraints=constraints.get_all_constraints(),
        grammar_rules=grammar.get_all_grammar(),
        few_shot_examples=[
            {"english": "What is your name?", "tulu": "ninna pudar encha?"},
            {"english": "I am going to school", "tulu": "yAn SAlege pOten"},
            {"english": "How much does this cost?", "tulu": "I encina dubbu?"}
        ],
        include_self_verification=True
    )
    
    print(f"\n3. Prompt Statistics:")
    print(f"   - Total tokens: {prompt_builder.token_count}")
    print(f"   - Target: ~2800 tokens")
    print(f"   - Characters: {len(prompt)}")
    
    # Show prompt preview
    print("\n4. Prompt Preview (first 2000 characters):")
    print("=" * 60)
    print(prompt[:2000])
    print("...")
    print("=" * 60)
    
    # Show last part
    print("\n5. Prompt Ending (last 500 characters):")
    print("=" * 60)
    print("..." + prompt[-500:])
    print("=" * 60)
    
    # Show contamination check example
    print("\n6. Contamination Detection Example:")
    test_responses = [
        "yAn soukhyana ullena",  # Good Tulu
        "nAnu soukhyana iddene",  # Contains Kannada "nAnu"
        "yAn bari ullena"  # Good Tulu
    ]
    
    for resp in test_responses:
        result = constraints.detect_contamination(resp)
        status = "CONTAMINATED" if result["is_contaminated"] else "CLEAN"
        print(f"   '{resp}' -> {status}")
        if result["is_contaminated"]:
            print(f"      Found: {result['contaminated_words']}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    print("\nTo test with actual API:")
    print("  python3 demo.py")
    print("\nNote: Your Gemini API key has quota limits.")


if __name__ == "__main__":
    demo_prompt()


