"""
Self-Play Data Generation

Self-play dialogue generation with quality filtering.
- Multi-judge quality control: 3 independent model instances, 1-5 scale
- Generate 500 raw pairs
- 3 judges rate each pair (1-5 Likert scale)
- Threshold: average score >= 3.5 (from Config)
- Final: 320 high-quality examples + 200 manual seeds = 520 total
"""

import json
import random
from typing import List, Dict, Optional, Tuple
from ..config import Config
from ..models.base_model import BaseModel


class SelfPlayGenerator:
    """Self-play dialogue generation with multi-judge quality control."""
    
    def __init__(self, model: BaseModel, judges: List[BaseModel] = None):
        """
        Initialize self-play generator.
        
        Args:
            model: Model for generating Q&A pairs
            judges: List of 3 independent model instances for quality assessment
        """
        self.model = model
        self.judges = judges or [model] * 3
        if len(self.judges) != 3:
            raise ValueError("Requires exactly 3 judges for quality control")
        
        self.domains = self._load_domains()
    
    def _load_domains(self) -> List[str]:
        """Load 20 conversational domains."""
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
        """
        Generate a question in Tulu on given topic.
        
        Paper Reference: Appendix C.2 - Step 1
        """
        prompt = f"""Generate a question in Tulu about {domain}.
The question should be natural and conversational.
Respond ONLY with the question in Tulu (no English, no explanation)."""
        
        # Combine with full prompt template if provided
        if prompt_template:
            prompt = prompt_template + "\n\n" + prompt
        
        response = self.model.generate(prompt)
        return response.strip()
    
    def generate_answer(self, question: str, prompt_template: str) -> str:
        """
        Generate corresponding Tulu answer.
        
        Paper Reference: Appendix C.2 - Step 2
        """
        prompt = prompt_template + f"\n\n## Current User Input\n\nUser said: \"{question}\"\n\nRespond in Tulu:"
        
        response = self.model.generate(prompt)
        return response.strip()
    
    def judge_quality(self, question: str, answer: str) -> List[float]:
        """
        Multi-judge quality assessment (1-5 Likert scale).
        
        Paper Reference: Section 3.5 - 3 independent judges, 1-5 scale
        
        Returns:
            List of 3 scores (one per judge)
        """
        evaluation_prompt = f"""Rate the quality of this Tulu Q&A pair on a scale of 1-5:

Question: {question}
Answer: {answer}

Rate on these dimensions (1-5 each):
1. Grammar correctness
2. Vocabulary purity (no Kannada contamination)
3. Naturalness
4. Relevance
5. Cultural appropriateness

Provide a single overall score (1-5) as a number only."""

        scores = []
        for judge in self.judges:
            response = judge.generate(evaluation_prompt)
            # Extract numeric score
            try:
                score = float(response.strip().split()[0])
                scores.append(max(1.0, min(5.0, score)))  # Clamp to 1-5
            except (ValueError, IndexError):
                # Default to 3 if parsing fails
                scores.append(3.0)
        
        return scores
    
    def generate_pairs(self, 
                      num_pairs: int = 500,
                      prompt_template: str = "",
                      seed_examples: List[Dict] = None) -> List[Dict]:
        """
        Generate synthetic Q&A pairs via self-play.
        
        Paper Reference: Section 3.5
        - Generate 500 raw pairs
        - Filter via 3-judge quality control
        - Retention rate: 64% (320/500)
        
        Args:
            num_pairs: Number of raw pairs to generate (paper: 500)
            prompt_template: Full 5-layer prompt template
            seed_examples: Seed examples for few-shot learning
        
        Returns:
            List of validated Q&A pairs
        """
        generated_pairs = []
        
        print(f"Generating {num_pairs} raw Q&A pairs via self-play...")
        
        for i in range(num_pairs):
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_pairs} pairs...")
            
            # Select random domain
            domain = random.choice(self.domains)
            
            # Generate question
            question = self.generate_question(domain, prompt_template)
            
            # Generate answer
            answer = self.generate_answer(question, prompt_template)
            
            # Judge quality
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
        
        # Filter by threshold
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
        """Save generated pairs to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_pairs": len(pairs),
                "generation_method": "self-play",
                "judge_threshold": Config.JUDGE_THRESHOLD,
                "num_judges": len(self.judges),
                "pairs": pairs
            }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage (requires model implementation)
    print("Self-Play Generator")
    print(f"  Domains: {len(SelfPlayGenerator(None).domains)}")
    print(f"  Judge threshold: {Config.JUDGE_THRESHOLD}")
    print(f"  Num judges: 3")

