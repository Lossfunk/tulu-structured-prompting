# Llama model wrapper (Llama 3.1 70B)

from typing import Optional
try:
    from .base_model import BaseModel
except ImportError:
    from models.base_model import BaseModel


class LlamaModel(BaseModel):
    def __init__(self, model_name: str = "llama-3.1-70b", api_endpoint: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_endpoint = api_endpoint

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(
            "Llama API implementation depends on deployment."
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())
