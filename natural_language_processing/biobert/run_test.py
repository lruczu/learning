from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

from config import Inference, PreprocessingConfig
from inference.loading import load_model, load_tokenizer
from inference.utils import get_best_answer
from training.io import load_test_examples
from training.qa_test_dataset import QATestDataset
from training.metrics import exact_match, f1


def main(args):
    model = load_model(Inference.MODEL_PATH)
    tokenizer = load_tokenizer(Inference.TOKENIZER_PATH)
    examples = load_test_examples(args.test_path)

    qa_dataset = QATestDataset(
        examples,
        tokenizer,
        PreprocessingConfig.MAX_LENGTH,
        PreprocessingConfig.DOC_STRIDE,
    )
    data_loader = DataLoader(qa_dataset, batch_size=args.batch_size)

    exact_match_scores = defaultdict(list)
    f1_scores = defaultdict(list)

    model = model.to('cuda')
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            n = batch['example_index'].shape[0]
            output = model(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                token_type_ids=batch['token_type_ids'].to('cuda'),
            )
            for i in range(n):
                example_index = batch['example_index'][i].item()
                answer = get_best_answer(
                    start_logits=output['start_logits'][i].cpu().numpy().tolist(),
                    end_logits=output['end_logits'][i].cpu().numpy().tolist(),
                    offset_mapping_start=batch['offset_mapping_start'][i].detach().numpy().tolist(),
                    offset_mapping_end=batch['offset_mapping_end'][i].detach().numpy().tolist(),
                    context=qa_dataset.examples[example_index].context,
                )
                exact_match_score = exact_match(
                    qa_dataset.examples[example_index].answers,
                    answer['answer'],
                )
                f1_score = f1(
                    qa_dataset.examples[example_index].answers,
                    answer['answer'],
                    tokenizer,
                )
                exact_match_scores[qa_dataset.examples[example_index].category].append(exact_match_score)
                f1_scores[qa_dataset.examples[example_index].category].append(f1_score)

    for category in exact_match_scores:
        print(f'*** {category} ***')
        print(f'Exact match: {round(np.mean(exact_match_scores[category]), 3)}')
        print(f'F1: {round(np.mean(f1_scores[category]), 3)}')
        print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_path')
    parser.add_argument('--batch_size', default=16)
    args = parser.parse_args()

    main(args)
