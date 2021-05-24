from collections import defaultdict
import glob

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.ner.training.example import TestExample
import src.ner.config as config
from src.ner.training.model import NERModel
from src.ner.training.train_utils import LabelEncoding
from src.ner.common.structures import Document
from src.ner.training.ner_test_dataset import NERTestDataset


def main(test_dir_pattern):
    test_files = glob.glob(test_dir_pattern)
    tokenizer = AutoTokenizer(config.CHECKPOINT)
    ner_model = NERModel()
    ner_model.load(config.MODEL_CHECKPOINT)

    bert = ner_model.bert.eval().cuda()
    last_layer = ner_model.last_layer.eval().cuda()

    mapping = {}
    submission_mapping = defaultdict(list)
    doc_index = 0
    n_files = len(test_files)

    for i in range(0, n_files, 50):
        files_to_processed = test_files[i:i+50]
        test_examples = []
        for file in files_to_processed:
            doc = Document.read_from_path(file)
            sentences = doc.get_all_sentences()
            mapping[doc_index] = doc.document_id
            for sentence in sentences:
                te = TestExample(
                    doc_index,
                    sentence
                )
                test_examples.append(te)
            doc_index += 1

        dataset = NERTestDataset(test_examples, tokenizer)
        data_loader = DataLoader(dataset, batch_size=config.TEST_BATCH_SIZE)

        for batch in data_loader:
            o = bert(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                token_type_ids=batch['token_type_ids'].to('cuda'),
            )
            o = last_layer(o['last_hidden_state'])
            o = o.detach()
            o = o.argmax(axis=-1)
            probabilities = o.max(axis=-1)
            for j, sentence_prediction in enumerate(o):
                last_valid = batch['attention_mask'][j].argmin().item()
                candidates = LabelEncoding.extract_candidates(
                    sentence_prediction,
                    probabilities[j],
                    last_valid,
                )
                if len(candidates) == 0:
                    continue

                id_ = batch['id'][j].item()
                ids = batch['input_ids'][j]
                for start_index, end_index in candidates:
                    submission_mapping[
                        mapping[id_]
                    ].append(
                        tokenizer.decode(ids[start_index:end_index])
                    )

    return submission_mapping


if __name__ == '__main__':
    main('test_dir/*.json')
