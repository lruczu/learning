from collections import Counter
from typing import List


def exact_match(answers: List[str], model_answer: str) -> int:
    if len(answers) == 0:
        return 1 * (model_answer == '')

    for answer in answers:
        if answer == model_answer:
            return 1
    return 0


def f1(answers: List[str], model_answer: str, tokenizer) -> float:
    if len(answers) == 0:
        if model_answer.strip() == '':
            return 1.
        return 0.

    f1_scores = []
    model_answer_tokenized = tokenizer.tokenize(model_answer)
    for answer in answers:
        answer_tokenized = tokenizer.tokenize(answer)
        common = Counter(model_answer_tokenized) & Counter(answer_tokenized)
        num_same = sum(common.values())

        if len(model_answer_tokenized) == 0 or len(answer_tokenized) == 0:
            f1_scores.append(1. * (model_answer_tokenized == answer_tokenized))
            continue

        if num_same == 0:
            f1_scores.append(0.)
            continue

        precision = 1.0 * num_same / len(model_answer_tokenized)
        recall = 1.0 * num_same / len(answer_tokenized)
        f1_score = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1_score)

    return max(f1_scores)
