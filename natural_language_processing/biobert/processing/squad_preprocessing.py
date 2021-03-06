"""
Here there is the code that transforms original squad files:
- train-v2.0.json
- dev-v2.0.json
into the form we can load and fine tune on.
"""
import json
from pathlib import Path

import pandas as pd

from processing.io import SaveTool


def process_sqaud(path: str) -> pd.DataFrame:
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    start_indices = []

    squad_data = squad_dict['data']
    for chunk in squad_data:
        for paragraph in chunk['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:  # here we implicitly ignore questions with an answer
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer['text'])
                    start_indices.append(answer['answer_start'])

    return pd.DataFrame({
        SaveTool.CONTEXT: contexts,
        SaveTool.QUESTION: questions,
        SaveTool.ANSWER: answers,
        SaveTool.START_INDEX: start_indices
    })
