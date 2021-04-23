from argparse import ArgumentParser
import random

import numpy as np

from training.example import Example
from training.io import load_examples, save_examples


def main(args):
    examples = load_examples(args.input_path)
    random.seed(args.seed)

    n_all = len(examples)
    n_no_answer = len([e for e in examples if e.is_impossible])
    init_prop = n_no_answer / n_all
    no_answer_prop = float(args.no_answer_prop)

    if init_prop >= no_answer_prop:
        raise ValueError(f'Dataset has already proportion: {init_prop} of examples without answer.')

    # (n_no_answer + x) / (n_all + x) = proportion
    # n_no_answer + x = proportion * n_all + proportion * x
    # x = (proportion * n_all - n_no_answer) / (1 - proportion)

    x = int((no_answer_prop * n_all - n_no_answer) / (1 - no_answer_prop))

    counter = 0
    while counter < x:
        i1 = np.random.choice(n_all)
        i2 = np.random.choice(n_all)

        if examples[i1].question == examples[i2].question:
            continue

        if examples[i1].answer == examples[i2].answer:
            continue

        e = Example(
            question=examples[i1].question,
            context=examples[i2].context,
            category=examples[i1].category,
            is_impossible=True
        )
        examples.append(e)
        counter += 1

    save_examples(examples, args.output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_path', help='From where to load a list of Examples')
    parser.add_argument('--output_path')
    parser.add_argument('--no_answer_prop', default=0.33)
    parser.add_argument('--seed', default=123)
    args = parser.parse_args()

    main(args)
