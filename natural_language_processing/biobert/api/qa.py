from typing import List

from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    sentence: str


class QA(BaseModel):
    question: str
    results: List[SearchResult]
