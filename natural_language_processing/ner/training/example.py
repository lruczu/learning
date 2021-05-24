from typing import List


class Example:
    def __init__(self, tokens: List[str], tags: List[str]):
        assert len(tokens) == len(tags)
        self.tokens = tokens
        self.tags = tags


class TestExample:
    def __init__(self, id_: int, sentence: str):
        self.id_ = id_
        self.sentence = sentence
