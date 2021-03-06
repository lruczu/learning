from pathlib import Path
from typing import List, Union, Tuple

import datasets
from datasets import load_dataset
import jsonlines
import pandas as pd


class IO:
    QUESTION = "question"
    CONTEXT = "context"
    START_INDEX = "start_index"
    ANSWER = "answer"
    END_INDEX = "end_index"  # it will be added during saving

    @staticmethod
    def save(df: pd.DataFrame, path: Union[str, Path]):
        """Save DF as "json object per line" file.
        Args:
            df: requires columns
                - question: str
                - context: str
                - start_index: str
                - answer: str
            path: where to save json file
        """
        IO._valid_columns(df)

        with jsonlines.open(path, 'w') as file:
            for row in df.to_dict(orient="records"):
                valid_start_index, valid_end_index = IO._get_valid_start_end_indices(
                    row[IO.ANSWER], row[IO.START_INDEX], row[IO.CONTEXT],
                )
                file.write(
                    {
                        IO.QUESTION: row[IO.QUESTION],
                        IO.CONTEXT: row[IO.CONTEXT],
                        IO.START_INDEX: valid_start_index,
                        IO.ANSWER: row[IO.ANSWER],
                        IO.END_INDEX: valid_end_index,
                    }
                )

    @staticmethod
    def load(paths: Union[str,Path, List[str], List[Path]]) -> datasets.Dataset:
        if isinstance(paths, list):
            paths = [str(path) for path in paths]
        return load_dataset("json", data_files=paths, split="train")  # ignore 'split' argument

    @staticmethod
    def _valid_columns(df: pd.DataFrame):
        IO._valid_column(df, IO.QUESTION)
        IO._valid_column(df, IO.CONTEXT)
        IO._valid_column(df, IO.START_INDEX)
        IO._valid_column(df, IO.ANSWER)

    @staticmethod
    def _valid_column(df: pd.DataFrame, col: str):
        if col not in df.columns:
            raise ValueError(f"Column: {col} is required.")

    @staticmethod
    def _get_valid_start_end_indices(answer: str, start_index: int, context: str) -> Tuple[int, int]:
        """Get aligned answer indices with context
        """
        end_index = start_index + len(answer)

        if context[start_index:end_index] == answer:
            return start_index, end_index

        if context[start_index - 1:end_index - 1] == answer:
            return start_index - 1, end_index - 1

        if context[start_index + 1:end_index + 1] == answer:
            return start_index + 1, end_index + 1

        raise ValueError(f'Answer "{answer}" is misaligned with the context.')
