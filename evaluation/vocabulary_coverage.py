# Vocabulary Coverage Evaluation - analyzes unique Tulu tokens and lexical diversity

from typing import List, Dict, Set
from collections import Counter


class VocabularyCoverageEvaluator:
    def compute_vocabulary_coverage(self, corpus: List[str]) -> Dict:
        all_tokens = []
        for text in corpus:
            tokens = text.lower().split()
            all_tokens.extend(tokens)

        unique_tokens = set(all_tokens)
        token_freq = Counter(all_tokens)

        ttr = len(unique_tokens) / len(all_tokens) if all_tokens else 0.0
        most_frequent = token_freq.most_common(20)

        return {
            "total_tokens": len(all_tokens),
            "unique_tokens": len(unique_tokens),
            "vocabulary_size": len(unique_tokens),
            "type_token_ratio": ttr,
            "average_tokens_per_response": len(all_tokens) / len(corpus) if corpus else 0.0,
            "most_frequent_tokens": most_frequent,
            "lexical_diversity": ttr
        }

    def compare_vocabularies(self, corpus1: List[str], corpus2: List[str]) -> Dict:
        vocab1 = self.compute_vocabulary_coverage(corpus1)
        vocab2 = self.compute_vocabulary_coverage(corpus2)

        tokens1 = set()
        for text in corpus1:
            tokens1.update(text.lower().split())

        tokens2 = set()
        for text in corpus2:
            tokens2.update(text.lower().split())

        overlap = tokens1.intersection(tokens2)
        unique_to_1 = tokens1 - tokens2
        unique_to_2 = tokens2 - tokens1

        return {
            "corpus1_vocab_size": vocab1["vocabulary_size"],
            "corpus2_vocab_size": vocab2["vocabulary_size"],
            "overlap_size": len(overlap),
            "unique_to_corpus1": len(unique_to_1),
            "unique_to_corpus2": len(unique_to_2),
            "overlap_percentage": len(overlap) / len(tokens1.union(tokens2)) if tokens1.union(tokens2) else 0.0
        }
