import numpy as np

from config import PreprocessingConfig


def get_best_answer(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offset_mapping_start: np.ndarray,
    offset_mapping_end: np.ndarray,
    context: str,
):
    start = np.where(offset_mapping_start == -1, -10000.0, start_logits)
    end = np.where(offset_mapping_start == -1, -10000.0, end_logits)
    start[0] = start_logits[0]
    end[0] = end_logits[0]

    start = np.exp(start - np.log(np.sum(np.exp(start), axis=-1, keepdims=True)))
    end = np.exp(end - np.log(np.sum(np.exp(end), axis=-1, keepdims=True)))

    min_null_score = min(1000000, (start[0] * end[0]).item())

    start[0] = end[0] = 0.0

    starts, ends, scores = _decode(start, end, topk=1, max_answer_len=PreprocessingConfig.MAX_ANSWER_LENGTH)
    start_index = starts[0]
    end_index = ends[0]
    score = scores[0]

    if score > min_null_score:
        start_char_index = offset_mapping_start[start_index]
        end_char_index = offset_mapping_end[end_index]
        return {
            'answer': context[start_char_index:end_char_index],
            'score': score,
        }
    return {
        'answer': '',
        'score': min_null_score,
    }


#  https://huggingface.co/transformers/_modules/transformers/pipelines/
#  question_answering.html#QuestionAnsweringPipeline.__call__
def _decode(start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int):
    """
    Take the output of any :obj:`ModelForQuestionAnswering` and will generate probabilities for each span to be the
    actual answer.

    In addition, it filters out some unwanted/impossible cases like answer len being greater than max_answer_len or
    answer end position being before the starting position. The method supports output the k-best answer through
    the topk argument.

    Args:
        start (:obj:`np.ndarray`): Individual start probabilities for each token.
        end (:obj:`np.ndarray`): Individual end probabilities for each token.
        topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
        max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
    """
    # Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]

    # Compute the score of each tuple(start, end) to be the real answer
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
    return start, end, candidates[0, start, end]
