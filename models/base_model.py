"""Abstract base class for LLM models."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseModel(ABC):
    """Abstract base class for all LLM models."""
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional model-specific parameters
        
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

