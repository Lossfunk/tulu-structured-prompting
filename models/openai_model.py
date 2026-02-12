# OpenAI API wrapper (GPT-3.5, GPT-4o)

import openai
from typing import Optional
try:
    from .base_model import BaseModel
except ImportError:
    from models.base_model import BaseModel
try:
    from ..config import Config
except ImportError:
    from config import Config


class OpenAIModel(BaseModel):
    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        super().__init__(model_name, **kwargs)
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""

    def count_tokens(self, text: str) -> int:
        import tiktoken
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(text))
