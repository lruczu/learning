from typing import List

from transformers import AutoTokenizer

from api.exceptions import InvalidNumberOfTokens
from biobert.config import CHECKPOINT


class BioBertTokenizer:
    def __init__(
        self,
        max_n_tokens_in_question: int,
        max_n_tokens_in_context: int,
        min_n_tokens_in_question: int = 1,
        min_n_tokens_in_context: int = 1,
    ):
        self.max_n_tokens_in_question = max_n_tokens_in_question
        self.max_n_tokens_in_context = max_n_tokens_in_context

        self.min_n_tokens_in_question = min_n_tokens_in_question
        self.min_n_tokens_in_context = min_n_tokens_in_context

        self.biobert_tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize(self, question: List[str], context: List[str]):

        # TODO: different strategy of controlling number of tokens. Use only once tokenization
        self.validate_n_tokens(question,
                               self.min_n_tokens_in_question,
                               self.max_n_tokens_in_question)
        self.validate_n_tokens(context,
                               self.min_n_tokens_in_context,
                               self.max_n_tokens_in_context)

        return self.biobert_tokenizer(
            question,
            context,
            return_token_type_ids=True,
            padding=True,
            return_tensors="pt",
        )

    def validate_n_tokens(self, texts: List[str], min_n: int, max_n: int):
        for text in texts:
            tokens = self.biobert_tokenizer.tokenize(text)
            if not min_n <= len(tokens) <= max_n:
                raise InvalidNumberOfTokens(text)
