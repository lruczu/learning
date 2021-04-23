from typing import List, Optional


class TestExample:
    def __init__(
        self,
        question: str,
        context: str,
        answers: List[str],
        category: Optional[str],
    ):
        self.question = question
        self.context = context
        self.answers = answers
        self.category = category

    def to_dict(self):
        return {
            'question': self.question,
            'context': self.context,
            'answers': self.answers,
            'category': self.category
        }

    def _valid_args(self):
        for answer in self.answers:
            if answer not in self.context:
                raise ValueError('Answer must be in the context.')

    def __repr__(self) -> str:
        return str(self.__dict__)

    __str__ = __repr__
