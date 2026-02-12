# Google Gemini API wrapper (Gemini 2.0 Flash)

import os
import google.generativeai as genai
from typing import Optional
try:
    from .base_model import BaseModel
    from ..config import Config
except ImportError:
    from models.base_model import BaseModel
    import config as Config


class GeminiModel(BaseModel):
    def __init__(self, model_name: str = "gemini-2.0-flash-exp", **kwargs):
        super().__init__(model_name, **kwargs)
        api_key = Config.GEMINI_API_KEY if hasattr(Config, 'GEMINI_API_KEY') else os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", self.temperature),
                    max_output_tokens=kwargs.get("max_tokens", self.max_tokens)
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return ""

    def count_tokens(self, text: str) -> int:
        return len(text) // 4
