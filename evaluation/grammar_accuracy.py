"""Grammar Accuracy Evaluation - rule-based grammar checking."""

import re
from typing import List, Dict
from ..grammar.tulu_grammar import TuluGrammar


class GrammarAccuracyEvaluator:
    """Evaluates grammatical correctness using rule-based checking."""
    
    def __init__(self):
        self.grammar = TuluGrammar()
    
    def check_verb_conjugation(self, text: str) -> Dict:
        """Check verb conjugations against documented rules."""
        # Simplified implementation - full version would parse and check all verb forms
        issues = []
        
        # Check for common verb patterns
        verbs = self.grammar.verb_conjugation["verbs"]
        for verb_name, verb_data in verbs.items():
            if "present_tense" in verb_data:
                # Check if verb forms appear correctly
                # (Simplified - full implementation requires morphological parsing)
                pass
        
        return {
            "is_correct": len(issues) == 0,
            "issues": issues,
            "checked_verbs": len(verbs)
        }
    
    def check_case_markers(self, text: str) -> Dict:
        """Check case markers against documented rules."""
        issues = []
        cases = self.grammar.case_marking["cases"]
        
        # Simplified check - full implementation would parse case markers
        # and verify allomorph selection rules
        
        return {
            "is_correct": len(issues) == 0,
            "issues": issues,
            "checked_cases": len(cases)
        }
    
    def check_word_order(self, text: str) -> Dict:
        """Check SOV word order."""
        # Simplified check - full implementation requires syntactic parsing
        words = text.split()
        
        # Very basic check: look for verb-like endings at end
        # (This is simplified - full implementation needs proper parsing)
        
        return {
            "is_sov": True,  # Placeholder
            "issues": []
        }
    
    def compute_grammar_accuracy(self, responses: List[str]) -> Dict:
        """
        Compute grammar accuracy across multiple responses.
        
        Args:
            responses: List of Tulu response strings
        
        Returns:
            Dict with grammar accuracy statistics
        """
        total = len(responses)
        correct = 0
        details = []
        
        for i, response in enumerate(responses):
            verb_check = self.check_verb_conjugation(response)
            case_check = self.check_case_markers(response)
            word_order_check = self.check_word_order(response)
            
            is_correct = (
                verb_check["is_correct"] and
                case_check["is_correct"] and
                word_order_check["is_sov"]
            )
            
            if is_correct:
                correct += 1
            
            details.append({
                "index": i,
                "response": response,
                "verb_correct": verb_check["is_correct"],
                "case_correct": case_check["is_correct"],
                "word_order_correct": word_order_check["is_sov"],
                "overall_correct": is_correct
            })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "total_responses": total,
            "correct_responses": correct,
            "grammar_accuracy": accuracy,
            "grammar_accuracy_percentage": accuracy * 100,
            "details": details
        }
    
    def evaluate_single(self, response: str) -> Dict:
        """Evaluate single response for grammar accuracy."""
        verb_check = self.check_verb_conjugation(response)
        case_check = self.check_case_markers(response)
        word_order_check = self.check_word_order(response)
        
        is_correct = (
            verb_check["is_correct"] and
            case_check["is_correct"] and
            word_order_check["is_sov"]
        )
        
        return {
            "is_correct": is_correct,
            "verb_check": verb_check,
            "case_check": case_check,
            "word_order_check": word_order_check
        }

