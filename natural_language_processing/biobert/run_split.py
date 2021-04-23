from argparse import ArgumentParser
import glob
import random

from training.io import load_examples, save_examples


def main(args):
    files = glob.glob(args.input_data_path_pattern)
    examples = load_examples(files)
    random.seed(args.seed)
    train_examples = []
    valid_examples = []
    for e in examples:
        if random.random() < float(args.valid_prop):
            valid_examples.append(e)
        else:
            train_examples.append(e)

    save_examples(train_examples, args.output_path_train)
    save_examples(valid_examples, args.output_path_valid)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_data_path_pattern', help='From where to load a list of Examples')
    parser.add_argument('--output_path_train', default='train.json')
    parser.add_argument('--output_path_valid', default='valid.json')
    parser.add_argument('--valid_prop', default=0.1)
    parser.add_argument('--seed', default=123)
    args = parser.parse_args()

    main(args)
