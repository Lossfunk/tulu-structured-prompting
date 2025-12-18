#!/usr/bin/env python3
"""
Quick Demo: Test Tulu LLM System with Gemini

This script demonstrates the full system with a simple example.
"""

import os
import sys
from pathlib import Path

# Set API key in environment
os.environ["GEMINI_API_KEY"] = "AIzaSyBPCPdCrio-UlaVIlAzGv7Ej_-30XhACMo"

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import using absolute imports
import config
from models.gemini_model import GeminiModel
from constraints.negative_constraints import NegativeConstraints
from grammar.tulu_grammar import TuluGrammar
from prompts.tulu_prompt import TuluPromptBuilder


def demo():
    """Run a quick demo of the Tulu LLM system."""
    print("=" * 60)
    print("Tulu LLM System Demo")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Initializing Gemini model...")
    try:
        model = GeminiModel(model_name="gemini-2.0-flash-exp")
        print("   Model initialized successfully!")
    except Exception as e:
        print(f"   Error initializing model: {e}")
        return
    
    # Show constraints
    print("\n2. Loading negative constraints...")
    constraints = NegativeConstraints()
    print(f"   Loaded {len(constraints.get_all_constraints())} Kannada→Tulu mappings")
    
    # Show grammar
    print("\n3. Loading grammar rules...")
    grammar = TuluGrammar()
    print(f"   Loaded {grammar.verb_conjugation['total_verbs']} verbs, {grammar.case_marking['total_cases']} cases")
    
    # Create a simple test example
    print("\n4. Creating test prompt...")
    test_input = "Hello, how are you?"
    
    prompt_builder = TuluPromptBuilder()
    prompt = prompt_builder.build_full_prompt(
        user_input=test_input,
        negative_constraints=constraints.get_all_constraints(),
        grammar_rules=grammar.get_all_grammar(),
        few_shot_examples=[
            {"english": "What is your name?", "tulu": "ninna pudar encha?"},
            {"english": "I am going to school", "tulu": "yAn SAlege pOten"}
        ],
        include_self_verification=True
    )
    
    print(f"   Prompt length: {prompt_builder.token_count} tokens")
    
    # Generate response
    print("\n5. Generating Tulu response...")
    print(f"   Input: {test_input}")
    
    try:
        response = model.generate(prompt, temperature=0.7, max_tokens=100)
        print(f"   Output: {response}")
        
        # Check for contamination
        print("\n6. Checking for Kannada contamination...")
        contamination_result = constraints.detect_contamination(response)
        if contamination_result["is_contaminated"]:
            print(f"   WARNING: Found Kannada words: {contamination_result['contaminated_words']}")
        else:
            print("   No Kannada contamination detected!")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError generating response: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Make sure your Gemini API key is valid and has quota available.")


if __name__ == "__main__":
    demo()
