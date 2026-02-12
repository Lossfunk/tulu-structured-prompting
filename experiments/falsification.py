# Falsification Experiment - intentionally incorrect grammar to test rule utilization.

from typing import List, Dict

try:
    from ..models.base_model import BaseModel
    from ..prompts.tulu_prompt import TuluPromptBuilder
    from ..constraints.negative_constraints import NegativeConstraints
    from ..grammar.tulu_grammar import TuluGrammar
    from ..evaluation.contamination import ContaminationEvaluator
    from ..evaluation.grammar_accuracy import GrammarAccuracyEvaluator
    from ..evaluation.vocabulary_coverage import VocabularyCoverageEvaluator
except ImportError:
    from models.base_model import BaseModel
    from prompts.tulu_prompt import TuluPromptBuilder
    from constraints.negative_constraints import NegativeConstraints
    from grammar.tulu_grammar import TuluGrammar
    from evaluation.contamination import ContaminationEvaluator
    from evaluation.grammar_accuracy import GrammarAccuracyEvaluator
    from evaluation.vocabulary_coverage import VocabularyCoverageEvaluator


class FalsificationExperiment:
    def __init__(self, model: BaseModel, few_shot_examples: List[Dict] = None):
        self.model = model
        self.prompt_builder = TuluPromptBuilder()
        self.constraints = NegativeConstraints()
        self.grammar = TuluGrammar()
        self.few_shot_examples = few_shot_examples or []

        self.contamination_eval = ContaminationEvaluator()
        self.grammar_eval = GrammarAccuracyEvaluator()
        self.vocab_eval = VocabularyCoverageEvaluator()

    def create_incorrect_grammar(self) -> Dict:
        # Intentionally incorrect: swapped dative/accusative, reversed gender agreement
        correct_grammar = self.grammar.get_all_grammar()

        incorrect_grammar = correct_grammar.copy()

        # Swap dative and accusative
        if "case_marking" in incorrect_grammar:
            cases = incorrect_grammar["case_marking"]["cases"]
            dative_marker = cases["dative"]["marker"]
            accusative_marker = cases["accusative"]["marker"]
            cases["dative"]["marker"] = accusative_marker
            cases["accusative"]["marker"] = dative_marker

        # Reverse gender agreement
        if "verb_conjugation" in incorrect_grammar:
            verbs = incorrect_grammar["verb_conjugation"]["verbs"]
            if "povuni" in verbs and "present_tense" in verbs["povuni"]:
                present = verbs["povuni"]["present_tense"]
                masc = present.get("3sg_masc", "")
                fem = present.get("3sg_fem", "")
                present["3sg_masc"] = fem
                present["3sg_fem"] = masc

        return incorrect_grammar

    def run(self, test_set: List[Dict[str, str]], use_incorrect: bool = True) -> Dict:
        experiment_type = "incorrect_grammar" if use_incorrect else "correct_grammar"
        print(f"Running falsification experiment: {experiment_type}...")

        grammar_rules = self.create_incorrect_grammar() if use_incorrect else self.grammar.get_all_grammar()

        responses = []
        for i, example in enumerate(test_set):
            if (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(test_set)}...")

            prompt = self.prompt_builder.build_full_prompt(
                user_input=example["english"],
                negative_constraints=self.constraints.get_all_constraints(),
                grammar_rules=grammar_rules,
                few_shot_examples=self.few_shot_examples,
                include_self_verification=True
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
        vocabulary = self.vocab_eval.compute_vocabulary_coverage(prediction_texts)

        return {
            "experiment": "falsification",
            "grammar_type": experiment_type,
            "model": self.model.model_name,
            "total_examples": len(test_set),
            "contamination_rate": contamination["contamination_rate"],
            "grammar_accuracy": grammar["grammar_accuracy"],
            "vocabulary_size": vocabulary["vocabulary_size"],
            "detailed_results": responses,
            "contamination_details": contamination,
            "grammar_details": grammar,
            "vocabulary_details": vocabulary
        }

    def run_comparison(self, test_set: List[Dict[str, str]]) -> Dict:
        correct_results = self.run(test_set, use_incorrect=False)
        incorrect_results = self.run(test_set, use_incorrect=True)

        delta_grammar = incorrect_results["grammar_accuracy"] - correct_results["grammar_accuracy"]
        delta_contamination = incorrect_results["contamination_rate"] - correct_results["contamination_rate"]
        delta_vocab = incorrect_results["vocabulary_size"] - correct_results["vocabulary_size"]

        return {
            "experiment": "falsification_comparison",
            "model": self.model.model_name,
            "correct_grammar": correct_results,
            "incorrect_grammar": incorrect_results,
            "deltas": {
                "grammar_accuracy": delta_grammar,
                "contamination_rate": delta_contamination,
                "vocabulary_size": delta_vocab
            },
            "expected_deltas": {
                "grammar_drop": -0.47,
                "contamination_increase": 0.27
            }
        }
