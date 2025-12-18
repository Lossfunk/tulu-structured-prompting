"""Ablation Studies - systematic experiments removing components to measure impact."""

from typing import List, Dict, Optional
from ..models.base_model import BaseModel
from ..prompts.tulu_prompt import TuluPromptBuilder
from ..constraints.negative_constraints import NegativeConstraints
from ..grammar.tulu_grammar import TuluGrammar
from ..evaluation.contamination import ContaminationEvaluator
from ..evaluation.grammar_accuracy import GrammarAccuracyEvaluator


class AblationExperiment:
    """Ablation studies: test impact of removing each component."""
    
    def __init__(self, model: BaseModel, few_shot_examples: List[Dict] = None):
        self.model = model
        self.prompt_builder = TuluPromptBuilder()
        self.constraints = NegativeConstraints()
        self.grammar = TuluGrammar()
        self.few_shot_examples = few_shot_examples or []
        
        self.contamination_eval = ContaminationEvaluator()
        self.grammar_eval = GrammarAccuracyEvaluator()
    
    def run_ablation(self, 
                     test_set: List[Dict[str, str]],
                     remove_grammar: bool = False,
                     remove_constraints: bool = False,
                     remove_examples: bool = False,
                     remove_verification: bool = False) -> Dict:
        """
        Run ablation experiment with specified components removed.
        
        Args:
            test_set: Test examples
            remove_grammar: Remove Layer 3 (Grammar Rules)
            remove_constraints: Remove Layer 2 (Negative Constraints)
            remove_examples: Remove Layer 4 (Few-Shot Examples)
            remove_verification: Remove Layer 5 (Self-Verification)
        
        Returns:
            Dict with results
        """
        config_name = self._get_config_name(
            remove_grammar, remove_constraints, remove_examples, remove_verification
        )
        
        print(f"Running ablation: {config_name}...")
        
        responses = []
        for i, example in enumerate(test_set):
            if (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(test_set)}...")
            
            # Build prompt with specified components
            negative_constraints = [] if remove_constraints else self.constraints.get_all_constraints()
            grammar_rules = {} if remove_grammar else self.grammar.get_all_grammar()
            few_shot = [] if remove_examples else self.few_shot_examples
            include_verification = not remove_verification
            
            prompt = self.prompt_builder.build_full_prompt(
                user_input=example["english"],
                negative_constraints=negative_constraints,
                grammar_rules=grammar_rules,
                few_shot_examples=few_shot,
                include_self_verification=include_verification
            )
            
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
        
        return {
            "experiment": "ablation",
            "configuration": config_name,
            "model": self.model.model_name,
            "removed_components": {
                "grammar": remove_grammar,
                "constraints": remove_constraints,
                "examples": remove_examples,
                "verification": remove_verification
            },
            "contamination_rate": contamination["contamination_rate"],
            "grammar_accuracy": grammar["grammar_accuracy"],
            "total_examples": len(test_set),
            "detailed_results": responses
        }
    
    def _get_config_name(self, no_grammar, no_constraints, no_examples, no_verification) -> str:
        """Generate configuration name."""
        parts = []
        if no_grammar:
            parts.append("no_grammar")
        if no_constraints:
            parts.append("no_constraints")
        if no_examples:
            parts.append("no_examples")
        if no_verification:
            parts.append("no_verification")
        return "_".join(parts) if parts else "full_system"
    
    def run_all_ablations(self, test_set: List[Dict[str, str]]) -> Dict:
        """
        Run all ablation configurations.
        
        """
        results = {}
        
        # Full system (baseline for comparison)
        results["full_system"] = self.run_ablation(
            test_set, remove_grammar=False, remove_constraints=False,
            remove_examples=False, remove_verification=False
        )
        
        # Remove each component
        results["no_constraints"] = self.run_ablation(
            test_set, remove_constraints=True
        )
        
        results["no_grammar"] = self.run_ablation(
            test_set, remove_grammar=True
        )
        
        results["no_examples"] = self.run_ablation(
            test_set, remove_examples=True
        )
        
        results["no_verification"] = self.run_ablation(
            test_set, remove_verification=True
        )
        
        # Calculate deltas from full system
        full_contamination = results["full_system"]["contamination_rate"]
        full_grammar = results["full_system"]["grammar_accuracy"]
        
        for config_name, result in results.items():
            if config_name != "full_system":
                result["delta_contamination"] = (
                    result["contamination_rate"] - full_contamination
                )
                result["delta_grammar"] = (
                    result["grammar_accuracy"] - full_grammar
                )
        
        return {
            "experiment": "ablation_study",
            "model": self.model.model_name,
            "configurations": results,
            "full_system_baseline": {
                "contamination_rate": full_contamination,
                "grammar_accuracy": full_grammar
            }
        }

