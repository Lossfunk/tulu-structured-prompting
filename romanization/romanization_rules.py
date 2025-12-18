"""
Romanization System for Tulu

Implements Section 3.1 and Appendix A.1 of:
'Prompting LLMs for Extremely Low-Resource Languages: A Case Study on Tulu'

Paper Reference:
- Section 3.1: Romanization system design (3 requirements)
- Table 1: Script comparison (Kannada 3.2 tok/word, Our Roman 1.4 tok/word)
- Appendix A.1: Diacritics specification

Implementation Notes:
- Diacritics for retroflex consonants: l., n., t., d.
- Vowel length marking: ā, ī, ū, ē, ō
- Velar nasal distinction: ṅ vs n
- Tokenization efficiency: 1.4 tokens/word (vs 3.2 for Kannada)
- Correlation with contamination: r=0.78
"""

import re
from typing import Dict, Tuple, Optional


class RomanizationSystem:
    """
    Standardized romanization system for Tulu.
    
    Paper Section: 3.1, Table 1
    """
    
    def __init__(self):
        self.diacritics = self._load_diacritics()
        self.conversion_rules = self._load_conversion_rules()
    
    def _load_diacritics(self) -> Dict[str, str]:
        """
        Load diacritic specifications from paper Appendix A.1.
        
        Paper Reference: Appendix A.1
        """
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
        """
        Load conversion rules for Kannada script → Romanization.
        
        Paper Reference: Section 3.1 - designed for disambiguation from Kannada
        """
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
                "ೋ": "ō"
            },
            "nasal_mapping": {
                "ಂ": "ṅ",
                "ನ": "n"
            }
        }
    
    def validate_romanization(self, text: str) -> Dict:
        """
        Validate that text follows romanization standards.
        
        Paper Reference: Section 3.1 - standardization requirements
        """
        issues = []
        
        # Check for retroflex diacritics where needed
        # (Simplified check - full implementation would require linguistic analysis)
        
        # Check for vowel length marking
        short_vowels = re.findall(r'\b\w*[aeiou]\w*\b', text.lower())
        # Note: This is simplified - full validation requires linguistic rules
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "has_retroflex": bool(re.search(r'[lntd]\.', text)),
            "has_vowel_length": bool(re.search(r'[āīūēō]', text)),
            "has_velar_nasal": bool(re.search(r'ṅ', text))
        }
    
    def calculate_tokenization_efficiency(self, text: str, tokenizer) -> Dict:
        """
        Calculate tokenization efficiency (tokens per word).
        
        Paper Reference: Table 1
        - Kannada script: 3.2 tokens/word
        - Naive Roman: 1.8 tokens/word
        - Our Roman: 1.4 tokens/word
        
        Args:
            text: Text to analyze
            tokenizer: Tokenizer function (e.g., tiktoken)
        
        Returns:
            Dict with tokenization metrics
        """
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
        """
        Convert Kannada script to standardized romanization.
        
        Paper Reference: Section 3.1 - bidirectional conversion support
        
        Note: This is a simplified implementation. Full conversion requires
        comprehensive mapping tables and linguistic rules.
        """
        # Simplified conversion - full implementation would require
        # comprehensive Unicode mapping and linguistic rules
        result = kannada_text
        
        # Apply retroflex mappings
        for kannada, roman in self.conversion_rules["retroflex_mapping"].items():
            result = result.replace(kannada, roman)
        
        # Apply vowel length mappings
        for kannada, roman in self.conversion_rules["vowel_length_mapping"].items():
            result = result.replace(kannada, roman)
        
        # Apply nasal mappings
        for kannada, roman in self.conversion_rules["nasal_mapping"].items():
            result = result.replace(kannada, roman)
        
        return result
    
    def get_romanization_spec(self) -> str:
        """
        Get romanization specification for prompt Layer 1.
        
        Paper Reference: Appendix A.1
        """
        return """Use diacritics for retroflex consonants (l., n., t., d.), mark vowel length (ā, ī, ū, ē, ō), and distinguish velar nasal ṅ from alveolar n."""


if __name__ == "__main__":
    roman = RomanizationSystem()
    print("Romanization System:")
    print(f"  Retroflex consonants: {list(roman.diacritics['retroflex_consonants'].keys())}")
    print(f"  Vowel length marks: {list(roman.diacritics['vowel_length'].keys())}")
    print(f"\nSpecification:")
    print(f"  {roman.get_romanization_spec()}")

