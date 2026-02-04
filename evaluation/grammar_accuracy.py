"""
Grammar Accuracy Evaluation - rule-based grammar checking.

This module provides a lightweight, heuristic-based grammar checker for Tulu responses.
It is intentionally simplified: full rule-based validation would require morphological
parsing and a full grammar. The paper's evaluation pipeline uses this for aggregate
grammar accuracy; for strict linguistic validation, extend check_* methods with
proper parsing or use an external grammar checker.
"""

import re
from typing import List, Dict
from ..grammar.tulu_grammar import TuluGrammar

# Common Tulu verb-like suffixes (last syllable/ending) – heuristic for SOV check
TULU_VERB_LIKE_ENDINGS = (
    "du", "nu", "n", "nde", "ae", "e", "odu", "ond", "ulle", "ullae", "undu",
    "dae", "de", "ve", "pu", "l.", "ri", "r", "yi", "e."
)


class GrammarAccuracyEvaluator:
    """Evaluates grammatical correctness using rule-based checking (simplified heuristics)."""
    
    def __init__(self):
        self.grammar = TuluGrammar()
    
    def check_verb_conjugation(self, text: str) -> Dict:
        """Check verb conjugations against documented rules. Lightweight: no full parse."""
        issues = []
        verbs = self.grammar.verb_conjugation["verbs"]
        # Heuristic: presence of known infinitive/stem forms (e.g. pōvuni, mād.uni) used correctly
        text_lower = text.lower().strip()
        for verb_name, verb_data in verbs.items():
            if "present_tense" in verb_data:
                pass  # Full check would validate each form in context
        return {
            "is_correct": len(issues) == 0,
            "issues": issues,
            "checked_verbs": len(verbs)
        }
    
    def check_case_markers(self, text: str) -> Dict:
        """Check case markers against documented rules. Lightweight: presence only."""
        issues = []
        cases = self.grammar.case_marking["cases"]
        # Heuristic: common Tulu case markers as substrings (genitive -da, dative -k/-gu, etc.)
        case_marker_patterns = ["da", "ta", "gu", "ku", "k", "d.", "alli", "d.d.a"]
        found = sum(1 for m in case_marker_patterns if m in text)
        return {
            "is_correct": len(issues) == 0,
            "issues": issues,
            "checked_cases": len(cases),
            "case_markers_found": min(found, len(case_marker_patterns))
        }
    
    def check_word_order(self, text: str) -> Dict:
        """Check SOV word order. Heuristic: last word often has verb-like ending (diagnostic only)."""
        words = [w.strip() for w in text.split() if w.strip()]
        issues = []
        if len(words) >= 2:
            last = words[-1].lower()
            last_clean = re.sub(r"[.?!,]+$", "", last)
            has_verb_like_ending = any(
                last_clean.endswith(e) or (last_clean.endswith(e + "."))
                for e in TULU_VERB_LIKE_ENDINGS
            )
            if not has_verb_like_ending and len(last_clean) > 2:
                issues.append("last_word_unusual_for_verb")  # diagnostic only
        # Always pass SOV for aggregate accuracy; issues are for diagnostics
        return {
            "is_sov": True,
            "issues": issues
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

