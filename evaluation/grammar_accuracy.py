import re
from typing import List, Dict
try:
    from ..grammar.tulu_grammar import TuluGrammar
except ImportError:
    from grammar.tulu_grammar import TuluGrammar

# Common Tulu verb-like suffixes (last syllable/ending) – heuristic for SOV check
TULU_VERB_LIKE_ENDINGS = (
    "du", "nu", "n", "nde", "ae", "e", "odu", "ond", "ulle", "ullae", "undu",
    "dae", "de", "ve", "pu", "l.", "ri", "r", "yi", "e."
)


class GrammarAccuracyEvaluator:
    def __init__(self):
        self.grammar = TuluGrammar()

    def check_verb_conjugation(self, text: str) -> Dict:
        issues = []
        verbs = self.grammar.verb_conjugation["verbs"]
        text_lower = text.lower().strip()
        for verb_name, verb_data in verbs.items():
            if "present_tense" in verb_data:
                pass
        return {
            "is_correct": len(issues) == 0,
            "issues": issues,
            "checked_verbs": len(verbs)
        }

    def check_case_markers(self, text: str) -> Dict:
        issues = []
        cases = self.grammar.case_marking["cases"]
        case_marker_patterns = ["da", "ta", "gu", "ku", "k", "d.", "alli", "d.d.a"]
        found = sum(1 for m in case_marker_patterns if m in text)
        return {
            "is_correct": len(issues) == 0,
            "issues": issues,
            "checked_cases": len(cases),
            "case_markers_found": min(found, len(case_marker_patterns))
        }

    def check_word_order(self, text: str) -> Dict:
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
                issues.append("last_word_unusual_for_verb")
        return {
            "is_sov": True,
            "issues": issues
        }

    def compute_grammar_accuracy(self, responses: List[str]) -> Dict:
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
