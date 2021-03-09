from typing import Optional, Tuple

import numpy as np


# for the time being the most basic approach
def find_best(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    token_type_ids: np.ndarray,
    offsets: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """
    Args:
        start_logits: of shape (# tokens,)
        end_logits: of shape (# tokens,)
        token_type_ids: binary array of shape (# tokens,), 1 means a token belongs to context,
            if 0, it belongs to the question
        offsets: (# tokens, 2), mapping token index into character indices (start and end of
        the context)
    """
    start_index = np.argmax(start_logits)
    end_index = np.argmax(end_logits)

    # the answer begins at a special character
    if offsets[start_index].sum() == 0:
        return
    # the answer ends at a special character
    if offsets[end_index].sum() == 0:
        return

    # unrealizable answer
    if end_index < start_index:
        return

    # start token belong to the question
    if token_type_ids[start_index] == 0:
        return

    # end token belong to the question
    if token_type_ids[end_index] == 0:
        return

    score = start_logits[start_index] + end_logits[end_index]

    start_char_index = offsets[start_index][0]
    end_char_index = offsets[end_index][1]

    return start_char_index, end_char_index
