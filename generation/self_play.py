# Self-play dialogue generation with quality filtering (multi-judge, 1-5 scale)

import json
import random
from typing import List, Dict, Optional, Tuple

try:
    from ..config import Config
    from ..models.base_model import BaseModel
except ImportError:
    from config import Config
    from models.base_model import BaseModel


class SelfPlayGenerator:
    def __init__(self, model: BaseModel, judges: List[BaseModel] = None):
        self.model = model
        self.judges = judges or [model] * 3
        if len(self.judges) != 3:
            raise ValueError("Requires exactly 3 judges for quality control")

        self.domains = self._load_domains()

    def _load_domains(self) -> List[str]:
        return [
            "family relationships",
            "food",
            "education",
            "travel",
            "festivals",
            "daily routines",
            "weather",
            "health",
            "work",
            "shopping",
            "sports",
            "housing",
            "agriculture",
            "coastal activities",
            "traditional arts",
            "religion",
            "technology",
            "childhood memories",
            "future plans",
            "community events"
        ]

    def generate_question(self, domain: str, prompt_template: str) -> str:
        prompt = (
            f"Generate a question in Tulu about {domain}.\n"
            "The question should be natural and conversational.\n"
            "Respond ONLY with the question in Tulu (no English, no explanation)."
        )
        if prompt_template:
            prompt = prompt_template + "\n\n" + prompt

        response = self.model.generate(prompt)
        return response.strip()

    def generate_answer(self, question: str, prompt_template: str) -> str:
        prompt = prompt_template + f"\n\n## Current User Input\n\nUser said: \"{question}\"\n\nRespond in Tulu:"

        response = self.model.generate(prompt)
        return response.strip()

    def judge_quality(self, question: str, answer: str) -> List[float]:
        evaluation_prompt = (
            f"Rate the quality of this Tulu Q&A pair on a scale of 1-5:\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Rate on these dimensions (1-5 each):\n"
            "1. Grammar correctness\n"
            "2. Vocabulary purity (no Kannada contamination)\n"
            "3. Naturalness\n"
            "4. Relevance\n"
            "5. Cultural appropriateness\n\n"
            "Provide a single overall score (1-5) as a number only."
        )

        scores = []
        for judge in self.judges:
            response = judge.generate(evaluation_prompt)
            try:
                score = float(response.strip().split()[0])
                scores.append(max(1.0, min(5.0, score)))
            except (ValueError, IndexError):
                scores.append(3.0)

        return scores

    def generate_pairs(self,
                      num_pairs: int = 500,
                      prompt_template: str = "",
                      seed_examples: List[Dict] = None) -> List[Dict]:
        generated_pairs = []

        print(f"Generating {num_pairs} raw Q&A pairs via self-play...")

        for i in range(num_pairs):
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_pairs} pairs...")

            domain = random.choice(self.domains)
            question = self.generate_question(domain, prompt_template)
            answer = self.generate_answer(question, prompt_template)
            scores = self.judge_quality(question, answer)
            avg_score = sum(scores) / len(scores)

            pair = {
                "question": question,
                "answer": answer,
                "domain": domain,
                "judge_scores": scores,
                "average_score": avg_score,
                "meets_threshold": avg_score >= Config.JUDGE_THRESHOLD
            }

            generated_pairs.append(pair)

        validated_pairs = [
            p for p in generated_pairs
            if p["meets_threshold"]
        ]

        print(f"\nQuality filtering results:")
        print(f"  Raw pairs: {len(generated_pairs)}")
        print(f"  Validated pairs: {len(validated_pairs)}")
        print(f"  Retention rate: {len(validated_pairs)/len(generated_pairs)*100:.1f}%")
        print(f"  Threshold: {Config.JUDGE_THRESHOLD}")

        return validated_pairs

    def save_pairs(self, pairs: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_pairs": len(pairs),
                "generation_method": "self-play",
                "judge_threshold": Config.JUDGE_THRESHOLD,
                "num_judges": len(self.judges),
                "pairs": pairs
            }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("Self-Play Generator")
    print(f"  Domains: {len(SelfPlayGenerator(None).domains)}")
    print(f"  Judge threshold: {Config.JUDGE_THRESHOLD}")
    print(f"  Num judges: 3")
