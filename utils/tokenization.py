"""Tokenization efficiency analysis utilities."""

import tiktoken
from typing import Dict


def analyze_tokenization(text: str, model_name: str = "gpt-3.5-turbo") -> Dict:
    """Analyze tokenization efficiency."""
    encoding = tiktoken.encoding_for_model(model_name)
    words = text.split()
    tokens = encoding.encode(text)
    
    return {
        "word_count": len(words),
        "token_count": len(tokens),
        "tokens_per_word": len(tokens) / len(words) if words else 0.0,
        "compression_ratio": len(words) / len(tokens) if tokens else 0.0
    }

