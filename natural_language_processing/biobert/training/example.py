from typing import Optional


class Example:
    def __init__(
        self,
        question: str,
        context: str,
        answer: Optional[str] = None,
        answer_start_index: Optional[int] = None,
        category: Optional[str] = 'general',
        is_impossible: bool = False
    ):
        """
        If is_impossible flag is on, answer and answer_start_indx fields are ignored.
        """
        self.question = question
        self.context = context
        self.answer = answer if not is_impossible else ''
        self.answer_start_index = answer_start_index if not is_impossible else -1
        self.is_impossible = is_impossible
        self.category = category

        self._valid_args()

    def to_dict(self):
        return {
            'question': self.question,
            'context': self.context,
            'answer': self.answer,
            'answer_start_index': self.answer_start_index,
            'is_impossible': self.is_impossible,
            'category': self.category,
        }

    def _valid_args(self):
        if self.is_impossible:
            return

        if self.answer is None or self.answer_start_index is None:
            raise ValueError('Answer and start index cannot be None for possible answer.')

        answer_end_index = self.answer_start_index + len(self.answer)
        if self.context[self.answer_start_index:answer_end_index] != self.answer:
            raise ValueError('Answer and provided start index misaligned.')

    def __repr__(self) -> str:
        return str(self.__dict__)

    __str__ = __repr__
