import os


class Config:
    # API keys (from environment)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Model settings
    DEFAULT_MODEL = "gpt-3.5-turbo"
    TEMPERATURE = 0.7
    MAX_TOKENS = 512

    # Experiment params
    NUM_SYNTHETIC_SAMPLES = 1000
    JUDGE_THRESHOLD = 3.5  # 1-5 scale
    NUM_JUDGES = 3

    # Prompt params
    PROMPT_TARGET_LENGTH = 2800
    NUM_FEW_SHOT_EXAMPLES = 10

    # Paths
    DATA_DIR = "data/"
    OUTPUT_DIR = "outputs/"
    PROMPTS_DIR = "prompts/"

    # Evaluation
    CONTAMINATION_DETECTION_METHOD = "exact_string_match"
    GRAMMAR_CHECK_STRICTNESS = "strict"

    RANDOM_SEED = 42

    @classmethod
    def validate(cls) -> bool:
        if not cls.OPENAI_API_KEY and not cls.GEMINI_API_KEY:
            print("Warning: No API keys found. Set OPENAI_API_KEY or GEMINI_API_KEY")
            return False
        return True
