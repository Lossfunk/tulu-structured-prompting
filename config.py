"""Centralized configuration for experiments."""

import os
from typing import Optional


class Config:
    """Configuration class for all experiments."""
    
    # API Keys (from environment)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Model Settings
    DEFAULT_MODEL = "gpt-3.5-turbo"
    TEMPERATURE = 0.7  # Standard temperature for generation
    MAX_TOKENS = 512  # Maximum tokens per response
    
    # Experiment Parameters
    NUM_SYNTHETIC_SAMPLES = 1000  # Number of synthetic Q/A pairs to generate
    JUDGE_THRESHOLD = 3.5  # Average score threshold for retaining samples (1-5 scale)
    NUM_JUDGES = 3  # Number of independent judges for quality assessment
    
    # Prompt Parameters
    PROMPT_TARGET_LENGTH = 2800
    NUM_FEW_SHOT_EXAMPLES = 10
    
    # Paths
    DATA_DIR = "data/"
    OUTPUT_DIR = "outputs/"
    PROMPTS_DIR = "prompts/"
    
    # Evaluation Parameters
    CONTAMINATION_DETECTION_METHOD = "exact_string_match"
    GRAMMAR_CHECK_STRICTNESS = "strict"  # Strict rule-based checking
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required API keys are set."""
        if not cls.OPENAI_API_KEY and not cls.GEMINI_API_KEY:
            print("Warning: No API keys found. Set OPENAI_API_KEY or GEMINI_API_KEY")
            return False
        return True

