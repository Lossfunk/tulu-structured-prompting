# Negative constraints: 50 high-frequency Kannada→Tulu mappings for contamination detection

from typing import List, Dict


class NegativeConstraints:
    # Manages 50+ high-frequency Kannada→Tulu word mappings

    def __init__(self):
        self.constraints = self._load_constraints()

    def _load_constraints(self) -> List[Dict[str, str]]:
        mappings = [
            {"kannada_word": "naanu", "tulu_alternative": "yān", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "nīnu", "tulu_alternative": "ī", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "nīvu", "tulu_alternative": "īr", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "avau", "tulu_alternative": "avu", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "aval.u", "tulu_alternative": "aval.", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "yenu", "tulu_alternative": "yena", "pos": "interrogative", "category": "interrogative"},
            {"kannada_word": "yāva", "tulu_alternative": "yāva", "pos": "interrogative", "category": "interrogative"},
            {"kannada_word": "elli", "tulu_alternative": "ill", "pos": "interrogative", "category": "interrogative"},
            {"kannada_word": "yāke", "tulu_alternative": "yēke", "pos": "interrogative", "category": "interrogative"},
            {"kannada_word": "hēge", "tulu_alternative": "eñji", "pos": "interrogative", "category": "interrogative"},
            {"kannada_word": "illa", "tulu_alternative": "ijji", "pos": "negation", "category": "negation"},
            {"kannada_word": "idu", "tulu_alternative": "ī", "pos": "demonstrative", "category": "demonstrative"},
            {"kannada_word": "adu", "tulu_alternative": "a", "pos": "demonstrative", "category": "demonstrative"},
            {"kannada_word": "mādu", "tulu_alternative": "mād.", "pos": "verb", "category": "verb"},
            {"kannada_word": "bā", "tulu_alternative": "bar", "pos": "verb", "category": "verb"},
            {"kannada_word": "hōgu", "tulu_alternative": "pō", "pos": "verb", "category": "verb"},
            {"kannada_word": "tinnu", "tulu_alternative": "tin", "pos": "verb", "category": "verb"},
            {"kannada_word": "kudi", "tulu_alternative": "kudi", "pos": "verb", "category": "verb"},
            {"kannada_word": "nōdu", "tulu_alternative": "kan.d.", "pos": "verb", "category": "verb"},
            {"kannada_word": "kēlu", "tulu_alternative": "kēl", "pos": "verb", "category": "verb"},
            {"kannada_word": "hēlu", "tulu_alternative": "par", "pos": "verb", "category": "verb"},
            {"kannada_word": "kodu", "tulu_alternative": "kalp", "pos": "verb", "category": "verb"},
            {"kannada_word": "tagol.", "tulu_alternative": "teggol", "pos": "verb", "category": "verb"},
            {"kannada_word": "nimage", "tulu_alternative": "niklegu", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "nimma", "tulu_alternative": "niklena", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "namma", "tulu_alternative": "nama", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "namage", "tulu_alternative": "namakku", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "avaru", "tulu_alternative": "ayere", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "ivaru", "tulu_alternative": "ivere", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "yavaga", "tulu_alternative": "yetu", "pos": "temporal", "category": "temporal"},
            {"kannada_word": "innu", "tulu_alternative": "indu", "pos": "temporal", "category": "temporal"},
            {"kannada_word": "naale", "tulu_alternative": "elle", "pos": "temporal", "category": "temporal"},
            {"kannada_word": "ninne", "tulu_alternative": "ninne", "pos": "temporal", "category": "temporal"},
            {"kannada_word": "aagutte", "tulu_alternative": "aagodu", "pos": "verb", "category": "verb"},
            {"kannada_word": "aagalla", "tulu_alternative": "aagandu", "pos": "verb", "category": "verb"},
            {"kannada_word": "beku", "tulu_alternative": "beeku", "pos": "modal", "category": "modal"},
            {"kannada_word": "ide", "tulu_alternative": "undu", "pos": "verb", "category": "verb"},
            {"kannada_word": "barutte", "tulu_alternative": "barodu", "pos": "verb", "category": "verb"},
            {"kannada_word": "hogutte", "tulu_alternative": "povodu", "pos": "verb", "category": "verb"},
            {"kannada_word": "maadutte", "tulu_alternative": "maadodu", "pos": "verb", "category": "verb"},
            {"kannada_word": "tinnutte", "tulu_alternative": "tindodu", "pos": "verb", "category": "verb"},
            {"kannada_word": "yestu", "tulu_alternative": "yestu", "pos": "interrogative", "category": "interrogative"},
            {"kannada_word": "yaake", "tulu_alternative": "yeke", "pos": "interrogative", "category": "interrogative"},
            {"kannada_word": "hege", "tulu_alternative": "encha", "pos": "interrogative", "category": "interrogative"},
            {"kannada_word": "nanna", "tulu_alternative": "yenna", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "ninna", "tulu_alternative": "ninna", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "avana", "tulu_alternative": "avana", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "avala", "tulu_alternative": "avala", "pos": "pronoun", "category": "pronoun"},
            {"kannada_word": "kannu", "tulu_alternative": "kan.d.", "pos": "verb", "category": "verb"},
            {"kannada_word": "kelu", "tulu_alternative": "kel", "pos": "verb", "category": "verb"},
            {"kannada_word": "helu", "tulu_alternative": "par", "pos": "verb", "category": "verb"},
            {"kannada_word": "yenu", "tulu_alternative": "yena", "pos": "interrogative", "category": "interrogative"},
            {"kannada_word": "yavaga", "tulu_alternative": "yetu", "pos": "temporal", "category": "temporal"},
        ]

        if len(mappings) < 50:
            raise ValueError(f"Must have at least 50 mappings, found {len(mappings)}")

        return mappings

    def get_all_constraints(self) -> List[Dict[str, str]]:
        return self.constraints

    def get_kannada_words(self) -> List[str]:
        return [c["kannada_word"] for c in self.constraints]

    def get_tulu_alternatives(self) -> Dict[str, str]:
        return {c["kannada_word"]: c["tulu_alternative"] for c in self.constraints}

    def format_for_prompt(self) -> str:
        lines = ["Kannada → Tulu"]
        for constraint in self.constraints:
            kannada = constraint["kannada_word"]
            tulu = constraint["tulu_alternative"]
            lines.append(f"{kannada} → {tulu}")
        return "\n".join(lines)

    def detect_contamination(self, text: str) -> Dict:
        # Exact string matching for Kannada contamination
        text_lower = text.lower()
        kannada_words = self.get_kannada_words()

        detected = []
        for kannada_word in kannada_words:
            if kannada_word.lower() in text_lower:
                detected.append(kannada_word)

        return {
            "is_contaminated": len(detected) > 0,
            "contaminated_words": detected,
            "contamination_count": len(detected),
            "contamination_rate": len(detected) / len(kannada_words) if kannada_words else 0.0
        }
