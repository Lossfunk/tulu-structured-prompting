from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name, temperature=0.7, max_tokens=512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt, **kwargs):
        pass

    @abstractmethod
    def count_tokens(self, text):
        pass
