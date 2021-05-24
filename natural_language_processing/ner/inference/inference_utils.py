from typing import List

from src.ner.common.utils import clean_text
from src.ner.common.structures import Document


class SentenceExtractor:
    MAX_LENGTH = 700
    KEYWORDS = (
        'data',
        'dataset',
        'code',
        'survey',
        'we use',
        'we used',
        'study',
        'census',
    )
    @classmethod
    def get_all_valid_sentences(cls, doc: Document) -> List[str]:
        return [
            sentence for sentence in doc.get_all_sentences() if
            SentenceExtractor.is_sentence_valid(sentence)
        ]

    @classmethod
    def is_sentence_valid(cls, sentence: str) -> bool:
        if not SentenceExtractor._has_correct_length(sentence):
            return False

        if SentenceExtractor._has_keyword(sentence):
            return True

        if SentenceExtractor._uppers_in_row(sentence):
            return True

        return False

    @classmethod
    def _uppers_in_row(cls, sentence: str, n: int = 2) -> bool:
        s = clean_text(sentence)
        tokens = s.split()

        i = 0
        for t in tokens[1:]:
            if t[0].isupper():
                i += 1
            else:
                i = 0

            if i == n:
                return True
        return False

    @classmethod
    def _has_correct_length(cls, sentence: str) -> bool:
        if len(sentence) > SentenceExtractor.MAX_LENGTH:
            return False
        return True

    @classmethod
    def _has_keyword(cls, sentence: str):
        for keyword in SentenceExtractor.KEYWORDS:
            if keyword in sentence.lower():
                return True

        return False
