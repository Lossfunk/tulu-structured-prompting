# 5-layer prompt builder for Tulu language learning

import tiktoken
from typing import List, Dict, Optional
try:
    from ..config import Config
except ImportError:
    import config as Config


class TuluPromptBuilder:
    # Builds the 5-layer prompt structure for Tulu language learning

    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.token_count = 0

    def build_full_prompt(
        self,
        user_input: str,
        negative_constraints: List[Dict[str, str]],
        grammar_rules: Dict,
        few_shot_examples: List[Dict[str, str]],
        include_self_verification: bool = True
    ) -> str:
        prompt_parts = []

        prompt_parts.append(self._build_identity_layer())
        prompt_parts.append(self._build_negative_constraints_layer(negative_constraints))
        prompt_parts.append(self._build_grammar_layer(grammar_rules))
        prompt_parts.append(self._build_few_shot_layer(few_shot_examples))
        if include_self_verification:
            prompt_parts.append(self._build_self_verification_layer())

        prompt_parts.append(f"\n## Current User Input\n\nUser said: \"{user_input}\"\n\nRespond in Tulu (remember: respond EXCLUSIVELY in Tulu, no English):")

        full_prompt = "\n\n".join(prompt_parts)

        self.token_count = len(self.encoding.encode(full_prompt))
        target_length = Config.PROMPT_TARGET_LENGTH if hasattr(Config, 'PROMPT_TARGET_LENGTH') else 2800
        if self.token_count > target_length * 1.2:
            print(f"Warning: Prompt length ({self.token_count} tokens) exceeds target ({target_length})")

        return full_prompt

    def _build_identity_layer(self) -> str:
        return (
            "You are a native Tulu speaker from coastal Karnataka.\n"
            "You will respond ONLY in Tulu using standardized romanization. Use diacritics for retroflex consonants (l., n., t., d.), mark vowel length (ā, ī, ū, ē, ō), and distinguish velar nasal ṅ from alveolar n.\n"
            "CRITICAL: Every response must be in Tulu. Do not use Kannada, English, or any other language except for modern technical terms without Tulu equivalents."
        )

    def _build_negative_constraints_layer(self, constraints: List[Dict[str, str]]) -> str:
        parts = ["## CRITICAL VOCABULARY RESTRICTIONS (600 tokens)\n"]
        parts.append("NEVER use these Kannada words. Use Tulu equivalents. NON-NEGOTIABLE.\n")
        parts.append("Kannada → Tulu\n")

        for constraint in constraints:
            kannada = constraint.get("kannada_word", "")
            tulu = constraint.get("tulu_alternative", "")
            parts.append(f"{kannada} → {tulu}")

        parts.append("\nBefore generating ANY response, verify you are using Tulu words, NOT Kannada words. This is CRITICAL and NON-NEGOTIABLE.")

        return "\n".join(parts)

    def _build_grammar_layer(self, grammar_rules: Dict) -> str:
        parts = ["## Tulu Grammar Rules (1200 tokens)\n"]

        if grammar_rules.get("verb_conjugation"):
            parts.append("**Verb Conjugation Rules:**")
            parts.append("Tulu verbs conjugate for gender, tense, person, number, and formality. Key patterns:\n")

            if "povuni" in str(grammar_rules.get("verb_conjugation", {})).lower():
                parts.append("Present tense (using pōvuni 'to go'):")
                parts.append("1sg: pōvn (I go)")
                parts.append("2sg informal: pōv (you go)")
                parts.append("2sg formal: pōpri (you go)")
                parts.append("3sg masc: pōvnu (he goes)")
                parts.append("3sg fem: pōvl. (she goes)")
                parts.append("3sg neut: pōvn (it goes)")
                parts.append("1pl: pōvn (we go)")
                parts.append("2pl: pōpri (you all go)")
                parts.append("3pl: pōvd. (they go)")
                parts.append("Past tense: Replace -v- with -y-, adjust endings")
                parts.append("Future: Use -p- instead of -v-")
                parts.append("Perfect: Use -ina/-yina suffix\n")

        if grammar_rules.get("case_marking"):
            parts.append("**Case Marking System:**")
            parts.append("Eight grammatical cases:")
            parts.append("1. Nominative (unmarked): akk (sister)")
            parts.append("2. Accusative (-ṅ/-aṅ): akkṅ (sister-OBJ)")
            parts.append("3. Dative (-k/-g): akkk (to sister)")
            parts.append("4. Genitive (-da/-ta): akkda (sister's)")
            parts.append("5. Locative (-/-alli): manen (in house)")
            parts.append("6. Ablative (-d.d.a): maned.d.a (from house)")
            parts.append("7. Instrumental (-d.d.a): kat.t.id.d.a (with knife)")
            parts.append("8. Vocative (-ē/-ayē): akkayē (O sister!)\n")
            parts.append("Allomorph selection depends on final phoneme:")
            parts.append("- After vowels: use -ṅ, -k, -")
            parts.append("- After consonants: use -aṅ, -g, -alli\n")

        if grammar_rules.get("pronoun_paradigms"):
            parts.append("**Pronoun System:**")
            parts.append("NOM ACC DAT GEN")
            parts.append("1sg yān yannaṅ yannak yanda")
            parts.append("2sg-inf ī ninnaṅ ninnak ninna")
            parts.append("2sg-form īr nikl nikk ninna")
            parts.append("3sg-masc avu avaṅ avak avanda")
            parts.append("3sg-fem aval. aval.ṅ aval.k aval.da")
            parts.append("1pl-incl namma nammaṅ nammak namma")
            parts.append("1pl-excl eṅga eṅgal eṅgalk eṅga\n")

        if grammar_rules.get("syntax"):
            parts.append("**Syntax:**")
            parts.append("- SOV word order: yān pustakaṅ ōdiyn (I book-ACC read) = 'I read the book'")
            parts.append("- Postpositions: mējen (table-on)")
            parts.append("- Adjective-Noun: periya mane (big house)")
            parts.append("- Genitive-Head: aval.da pustaka (her book)")
            parts.append("- Relative clauses precede head noun")

        return "\n".join(parts)

    def _build_few_shot_layer(self, examples: List[Dict[str, str]]) -> str:
        parts = ["## Few-Shot Examples (600 tokens)\n"]
        parts.append("Learn from these examples:\n")

        num_examples = min(len(examples), 15)

        for i, ex in enumerate(examples[:num_examples], 1):
            if "question" in ex or "q" in str(ex).lower():
                question = ex.get("question", ex.get("q", ""))
                answer = ex.get("answer", ex.get("a", ""))
                parts.append(f"Q: {question}")
                parts.append(f"A: {answer}")
            else:
                english = ex.get("english", "")
                tulu = ex.get("tulu", "")
                parts.append(f"Q: {english}")
                parts.append(f"A: {tulu}")
            parts.append("")

        return "\n".join(parts)

    def _build_self_verification_layer(self) -> str:
        return (
            "## Self-Verification (200 tokens)\n\n"
            "Before responding, mentally verify:\n"
            "1. Am I using Tulu vocabulary (not Kannada)?\n"
            "2. Are verb conjugations correct for person/gender/tense?\n"
            "3. Are case markers appropriate for grammatical role?\n"
            "4. Is word order SOV?\n"
            "5. Are pronouns correct for formality level?\n\n"
            "If unsure about any Tulu word, construct it using the grammar rules provided rather than substituting a Kannada word.\n\n"
            "Now respond naturally in Tulu to the user's question."
        )
