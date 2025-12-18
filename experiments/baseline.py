"""Baseline System Experiment - simple instruction only (no grammar, no constraints, no examples)."""

from typing import List, Dict
from ..models.base_model import BaseModel
from ..evaluation.contamination import ContaminationEvaluator
from ..evaluation.grammar_accuracy import GrammarAccuracyEvaluator
from ..evaluation.vocabulary_coverage import VocabularyCoverageEvaluator
from ..evaluation.tokenization_efficiency import TokenizationEfficiencyEvaluator


class BaselineExperiment:
    """Baseline system: Simple instruction only."""
    
    def __init__(self, model: BaseModel):
        self.model = model
        self.contamination_eval = ContaminationEvaluator()
        self.grammar_eval = GrammarAccuracyEvaluator()
        self.vocab_eval = VocabularyCoverageEvaluator()
        self.token_eval = TokenizationEfficiencyEvaluator()
    
    def create_baseline_prompt(self, user_input: str) -> str:
        """Create baseline prompt: simple instruction only."""
        return f"""Respond in Tulu language.

User input: {user_input}

Respond in Tulu:"""
    
    def run(self, test_set: List[Dict[str, str]]) -> Dict:
        """
        Run baseline experiment.
        
        Args:
            test_set: List of {english, tulu} pairs
        
        Returns:
            Dict with evaluation results
        """
        print("Running baseline experiment...")
        
        responses = []
        for i, example in enumerate(test_set):
            if (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(test_set)}...")
            
            prompt = self.create_baseline_prompt(example["english"])
            response = self.model.generate(prompt)
            
            responses.append({
                "input": example["english"],
                "reference": example["tulu"],
                "prediction": response
            })
        
        # Evaluate
        prediction_texts = [r["prediction"] for r in responses]
        
        contamination = self.contamination_eval.compute_contamination_rate(prediction_texts)
        grammar = self.grammar_eval.compute_grammar_accuracy(prediction_texts)
        vocabulary = self.vocab_eval.compute_vocabulary_coverage(prediction_texts)
        tokenization = self.token_eval.evaluate_corpus(prediction_texts)
        
        return {
            "experiment": "baseline",
            "model": self.model.model_name,
            "total_examples": len(test_set),
            "contamination_rate": contamination["contamination_rate"],
            "grammar_accuracy": grammar["grammar_accuracy"],
            "vocabulary_size": vocabulary["vocabulary_size"],
            "tokens_per_word": tokenization.get("average_tokens_per_word", 0.0),
            "detailed_results": responses,
            "contamination_details": contamination,
            "grammar_details": grammar,
            "vocabulary_details": vocabulary,
            "tokenization_details": tokenization
        }

