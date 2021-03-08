import pytest

from api.exceptions import InvalidNumberOfTokensInQuestion
from processing.biobert_tokenizer import BioBertTokenizer


biobert_tokenizer = BioBertTokenizer(
    max_n_tokens_in_question=3,
    max_n_tokens_in_context=5
)


@pytest.mark.parametrize(
    "question",
    [
        ["a"],
        ["a b"],
        ["a b c"],
        ["a b c d"],
        ["a b c d e"]
    ]
)
def test_(question):
    if len(question[0]) <= biobert_tokenizer.max_n_tokens_in_question:
        biobert_tokenizer._validate_n_tokens(question[0], 3)
    else:
        with pytest.raises(InvalidNumberOfTokensInQuestion):
            biobert_tokenizer._validate_n_tokens()
