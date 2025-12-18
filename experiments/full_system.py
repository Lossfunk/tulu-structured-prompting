"""Full System Experiment - full 5-layer system implementation."""

from typing import List, Dict
from ..models.base_model import BaseModel
from ..prompts.tulu_prompt import TuluPromptBuilder
from ..constraints.negative_constraints import NegativeConstraints
from ..grammar.tulu_grammar import TuluGrammar
from ..evaluation.contamination import ContaminationEvaluator
from ..evaluation.grammar_accuracy import GrammarAccuracyEvaluator
from ..evaluation.vocabulary_coverage import VocabularyCoverageEvaluator
from ..evaluation.tokenization_efficiency import TokenizationEfficiencyEvaluator


class FullSystemExperiment:
    """Full 5-layer system experiment."""
    
    def __init__(self, model: BaseModel, few_shot_examples: List[Dict] = None):
        self.model = model
        self.prompt_builder = TuluPromptBuilder()
        self.constraints = NegativeConstraints()
        self.grammar = TuluGrammar()
        self.few_shot_examples = few_shot_examples or []
        
        self.contamination_eval = ContaminationEvaluator()
        self.grammar_eval = GrammarAccuracyEvaluator()
        self.vocab_eval = VocabularyCoverageEvaluator()
        self.token_eval = TokenizationEfficiencyEvaluator()
    
    def run(self, test_set: List[Dict[str, str]]) -> Dict:
        """
        Run full system experiment.
        
        Args:
            test_set: List of {english, tulu} pairs
        
        Returns:
            Dict with evaluation results
        """
        print("Running full system experiment...")
        print(f"  Using {len(self.few_shot_examples)} few-shot examples")
        
        responses = []
        for i, example in enumerate(test_set):
            if (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(test_set)}...")
            
            # Build full 5-layer prompt
            prompt = self.prompt_builder.build_full_prompt(
                user_input=example["english"],
                negative_constraints=self.constraints.get_all_constraints(),
                grammar_rules=self.grammar.get_all_grammar(),
                few_shot_examples=self.few_shot_examples,
                include_self_verification=True
            )
            
            response = self.model.generate(prompt)
            
            responses.append({
                "input": example["english"],
                "reference": example["tulu"],
                "prediction": response,
                "prompt_length": self.prompt_builder.token_count
            })
        
        # Evaluate
        prediction_texts = [r["prediction"] for r in responses]
        
        contamination = self.contamination_eval.compute_contamination_rate(prediction_texts)
        grammar = self.grammar_eval.compute_grammar_accuracy(prediction_texts)
        vocabulary = self.vocab_eval.compute_vocabulary_coverage(prediction_texts)
        tokenization = self.token_eval.evaluate_corpus(prediction_texts)
        
        return {
            "experiment": "full_system",
            "model": self.model.model_name,
            "total_examples": len(test_set),
            "contamination_rate": contamination["contamination_rate"],
            "grammar_accuracy": grammar["grammar_accuracy"],
            "vocabulary_size": vocabulary["vocabulary_size"],
            "tokens_per_word": tokenization.get("average_tokens_per_word", 0.0),
            "prompt_length_tokens": self.prompt_builder.token_count,
            "detailed_results": responses,
            "contamination_details": contamination,
            "grammar_details": grammar,
            "vocabulary_details": vocabulary,
            "tokenization_details": tokenization
        }

