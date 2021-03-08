class QAExample:
    def __init__(
        self,
        question: str,
        context: str,
        answer: str,
        start_index: str,
        end_index: str,
    ):
        self.question = question
        self.context = context
        self.answer = answer
        self.start_index = start_index
        self.end_index = end_index

        self._valid_args()

    def to_dict(self):
        return self.__dict__

    def _valid_args(self):
        if self.context[self.start_index:self.end_index] != self.answer:
            raise ValueError('Indices not aligned with the answer.')
