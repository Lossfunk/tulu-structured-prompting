"""Tokenization Efficiency Evaluation - analyzes tokenization efficiency across different scripts."""

import tiktoken
from typing import List, Dict
from ..romanization.romanization_rules import RomanizationSystem


class TokenizationEfficiencyEvaluator:
    """Evaluates tokenization efficiency across different scripts."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.romanization = RomanizationSystem()
    
    def compute_tokenization_efficiency(self, text: str) -> Dict:
        """
        Compute tokenization efficiency metrics.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dict with tokenization metrics
        """
        words = text.split()
        tokens = self.encoding.encode(text)
        token_count = len(tokens)
        
        tokens_per_word = token_count / len(words) if words else 0.0
        compression_ratio = len(words) / token_count if token_count > 0 else 0.0
        
        return {
            "word_count": len(words),
            "token_count": token_count,
            "tokens_per_word": tokens_per_word,
            "compression_ratio": compression_ratio,
            "text_length": len(text),
            "characters_per_token": len(text) / token_count if token_count > 0 else 0.0
        }
    
    def compare_scripts(self, kannada_text: str, roman_text: str) -> Dict:
        """Compare tokenization efficiency between Kannada script and romanization."""
        kannada_metrics = self.compute_tokenization_efficiency(kannada_text)
        roman_metrics = self.compute_tokenization_efficiency(roman_text)
        
        improvement = (
            kannada_metrics["tokens_per_word"] - roman_metrics["tokens_per_word"]
        ) / kannada_metrics["tokens_per_word"] if kannada_metrics["tokens_per_word"] > 0 else 0.0
        
        return {
            "kannada_script": kannada_metrics,
            "romanization": roman_metrics,
            "improvement_percentage": improvement * 100,
            "tokens_saved_per_word": (
                kannada_metrics["tokens_per_word"] - roman_metrics["tokens_per_word"]
            )
        }
    
    def evaluate_corpus(self, corpus: List[str]) -> Dict:
        """
        Evaluate tokenization efficiency across corpus.
        
        Returns:
            Average metrics across all texts
        """
        all_metrics = [self.compute_tokenization_efficiency(text) for text in corpus]
        
        if not all_metrics:
            return {}
        
        avg_tokens_per_word = sum(m["tokens_per_word"] for m in all_metrics) / len(all_metrics)
        avg_compression = sum(m["compression_ratio"] for m in all_metrics) / len(all_metrics)
        
        return {
            "corpus_size": len(corpus),
            "average_tokens_per_word": avg_tokens_per_word,
            "average_compression_ratio": avg_compression,
            "total_tokens": sum(m["token_count"] for m in all_metrics),
            "total_words": sum(m["word_count"] for m in all_metrics)
        }

