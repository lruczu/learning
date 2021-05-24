from typing import List

from src.ner.training.ner_test_dataset import NERTestDataset
from src.ner.training.example import TestExample
from src.ner.training.train_utils import extract_candidates, extract_datasets


def get_datasets_from_sentence(model, tokenizer, sentence: str) -> List[str]:
    dataset = NERTestDataset([TestExample(0, sentence)], tokenizer)
    i = dataset[0]
    last_valid = i['attention_mask'].sum().item()
    o = model.bert(
        input_ids=i['input_ids'].view(1, -1).to('cuda'),
        attention_mask=i['attention_mask'].view(1, -1).to('cuda'),
        token_type_ids=i['token_type_ids'].view(1, -1).to('cuda'),
    )
    o = model.last_layer(o['last_hidden_state'])

    o = o.detach()

    candidates = extract_candidates(
        o[0].argmax(axis=1).detach(),
        last_valid,
    )

    datasets = extract_datasets(
        candidates,
        i['input_ids'].numpy(),
        tokenizer,
    )

    return datasets
