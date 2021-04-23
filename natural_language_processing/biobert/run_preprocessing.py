from argparse import ArgumentParser
import glob
import os
import random
from typing import Optional

import jsonlines

from training.example import Example
from training.preprocessing import NumberOfSentencesInContext, WithNoAnswer, Preprocessor


def main(args):
    preprocessor = Preprocessor([
        WithNoAnswer(args.with_no_answer),
        NumberOfSentencesInContext(args.n_sentences),
    ])

    process_single_file(
        args,
        preprocessor,
        drop_duplicates=args.drop_duplicate_questions,
    )


def process_single_file(
    args,
    preprocessor,
    drop_duplicates: bool = False,
):
    seen_questions = set()

    with jsonlines.open(args.input_path) as reader:
        for i, line in enumerate(reader):
            example = Example(
                question=line['question'],
                context=line['context'],
                answer=line['answer'],
                answer_start_index=line['answer_start_index'],
                is_impossible=line['is_impossible'],
                category=line['category'],
            )
            example = preprocessor.apply(example)
            if not example:
                continue

            if drop_duplicates:
                if example.question in seen_questions:
                    continue
                else:
                    seen_questions.add(example.question)

            with jsonlines.open(args.output_path, mode='a') as writer:
                writer.write(example.to_dict())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', default='preprocessed.json', type=str)
    parser.add_argument('--with_no_answer', default=True, type=bool)
    parser.add_argument('--n_sentences', default=1, type=int)
    parser.add_argument('--drop_duplicate_questions', action='store_true')

    args = parser.parse_args()

    main(args)
