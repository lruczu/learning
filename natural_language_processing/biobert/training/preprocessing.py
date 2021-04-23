from typing import List, Optional

import numpy as np
from sentence_splitter import SentenceSplitter

from training.example import Example


class Step:
    def apply(self, example: Example) -> Optional[Example]:
        ...


class WithNoAnswer(Step):
    def __init__(self, with_no_answer: bool):
        self.with_no_answer = with_no_answer

    def apply(self, example: Example):
        if not self.with_no_answer:
            return None if example.is_impossible else example  # has answer
        return example


class NumberOfSentencesInContext(Step):
    SPLITTER = SentenceSplitter(language='en')

    def __init__(
        self,
        n_sentences: int,
    ):
        self.n_sentences = n_sentences

    def apply(self, example: Example):
        sentences = NumberOfSentencesInContext.SPLITTER.split(example.context)
        answer_location = self._locate_answer(sentences, example.answer, example.answer_start_index)

        sentences_indices_to_include = self._draw_sentences(
            len(sentences),
            answer_location,
            example.is_impossible,
        )

        new_context = ' '.join(
            [
                sentences[i] for i in sentences_indices_to_include
            ]
        )

        if example.is_impossible:
            return Example(
                question=example.question,
                context=new_context,
                category=example.category,
                is_impossible=True,
            )

        try:
            return Example(
                question=example.question,
                context=new_context,
                answer=example.answer,
                category=example.category,
                answer_start_index=new_context.index(example.answer)
            )

        except ValueError:
            return  # todo: handle strange edge cases

    def _locate_answer(self, sentences: List[str], answer: str, answer_start_index: int) -> int:
        """Returns index of sentence in which answer is (0 if not exists)."""
        candidates = []
        i = 0
        for s_index, s in enumerate(sentences):
            if answer in s:
                a_index = s.index(answer) + i
                candidates.append((abs(a_index - answer_start_index), s_index))

            i += len(s)
        if len(candidates) == 0:
            return 0
        return sorted(candidates, key=lambda x: x[0])[0][1]

    def _draw_sentences(
        self,
        n_sentences_in_context: int,
        answer_location: int,
        is_impossible: bool,
    ):
        """
        In real application we don't know where the answer can be. So a-priori
        the location of the answer has the uniform distribution.
        We can imaging a window (let's say of 3 sentences) with equal probability
        covering the answer:
                a
        [x, x, x]
            [x, x, x]
                [x, x, x]
        Attention: In the above example not always a list of three sentences will be returned,
        If the window start at the answer and the answer is located at the last sentence, only
        one is returned.
        """
        if is_impossible:
            n_possible = min(self.n_sentences, n_sentences_in_context)
            window_start_min = np.random.choice(n_sentences_in_context - n_possible + 1)
            return np.arange(
                window_start_min,
                window_start_min + n_possible,
            ).tolist()

        if self.n_sentences == 1:
            return [answer_location]

        window_start_min = max(answer_location - self.n_sentences + 1, 0)
        window_start_max = answer_location

        window_start = np.random.choice(np.arange(window_start_min, window_start_max + 1))

        return np.arange(window_start,
                         min(window_start + self.n_sentences, n_sentences_in_context)
                         ).tolist()


class Preprocessor:
    def __init__(
        self,
        steps: List[Step]
    ):
        self.steps = steps

    def apply(self, example: Example) -> Optional[Example]:
        for step in self.steps:
            example = step.apply(example)
            if example is None:
                return
        return example
