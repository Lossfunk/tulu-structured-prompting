# Contamination Rate Evaluation - detects Kannada contamination in Tulu responses

from typing import List, Dict
try:
    from ..constraints.negative_constraints import NegativeConstraints
except ImportError:
    from constraints.negative_constraints import NegativeConstraints


class ContaminationEvaluator:
    def __init__(self):
        self.constraints = NegativeConstraints()

    def compute_contamination_rate(self, responses: List[str]) -> Dict:
        total = len(responses)
        contaminated = 0
        contamination_details = []

        for i, response in enumerate(responses):
            result = self.constraints.detect_contamination(response)
            if result["is_contaminated"]:
                contaminated += 1
                contamination_details.append({
                    "index": i,
                    "response": response,
                    "contaminated_words": result["contaminated_words"]
                })

        contamination_rate = contaminated / total if total > 0 else 0.0

        return {
            "total_responses": total,
            "contaminated_responses": contaminated,
            "contamination_rate": contamination_rate,
            "contamination_percentage": contamination_rate * 100,
            "details": contamination_details
        }

    def evaluate_single(self, response: str) -> Dict:
        return self.constraints.detect_contamination(response)
