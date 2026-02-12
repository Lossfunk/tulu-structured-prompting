# Standardized romanization for Tulu with diacritics

import re
from typing import Dict, Tuple, Optional


class RomanizationSystem:
    # Standardized romanization for Tulu

    def __init__(self):
        self.diacritics = self._load_diacritics()
        self.conversion_rules = self._load_conversion_rules()

    def _load_diacritics(self) -> Dict[str, str]:
        return {
            "retroflex_consonants": {
                "l.": "retroflex l",
                "n.": "retroflex n",
                "t.": "retroflex t",
                "d.": "retroflex d"
            },
            "vowel_length": {
                "ā": "long a",
                "ī": "long i",
                "ū": "long u",
                "ē": "long e",
                "ō": "long o"
            },
            "nasal_distinction": {
                "ṅ": "velar nasal",
                "n": "alveolar n"
            }
        }

    def _load_conversion_rules(self) -> Dict:
        return {
            "retroflex_mapping": {
                "ಳ": "l.",
                "ಣ": "n.",
                "ಟ": "t.",
                "ಡ": "d."
            },
            "vowel_length_mapping": {
                "ಾ": "ā",
                "ೀ": "ī",
                "ೂ": "ū",
                "ೇ": "ē",
                "ോ": "ō"
            },
            "nasal_mapping": {
                "ಂ": "ṅ",
                "ನ": "n"
            }
        }

    def validate_romanization(self, text: str) -> Dict:
        issues = []

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "has_retroflex": bool(re.search(r'[lntd]\.', text)),
            "has_vowel_length": bool(re.search(r'[āīūēō]', text)),
            "has_velar_nasal": bool(re.search(r'ṅ', text))
        }

    def calculate_tokenization_efficiency(self, text: str, tokenizer) -> Dict:
        words = text.split()
        tokens = tokenizer(text)
        token_count = len(tokens) if isinstance(tokens, list) else tokens

        return {
            "word_count": len(words),
            "token_count": token_count,
            "tokens_per_word": token_count / len(words) if words else 0.0,
            "compression_ratio": len(words) / token_count if token_count > 0 else 0.0
        }

    def convert_kannada_to_roman(self, kannada_text: str) -> str:
        result = kannada_text

        for kannada, roman in self.conversion_rules["retroflex_mapping"].items():
            result = result.replace(kannada, roman)

        for kannada, roman in self.conversion_rules["vowel_length_mapping"].items():
            result = result.replace(kannada, roman)

        for kannada, roman in self.conversion_rules["nasal_mapping"].items():
            result = result.replace(kannada, roman)

        return result

    def get_romanization_spec(self) -> str:
        return ("Use diacritics for retroflex consonants (l., n., t., d.), "
                "mark vowel length (ā, ī, ū, ē, ō), and distinguish velar nasal ṅ from alveolar n.")
