from typing import List, Optional

from transformers import AutoTokenizer


class BioBertTokenizer:
    def __init__(
        self,
        checkpoint: str,
        max_length: int,
        doc_stride: int,
        cached_dir: str = Optional[str],
    ):
        self.max_length = max_length
        self.doc_stride = doc_stride

        self.biobert_tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            cached_dir=cached_dir
        )

    def tokenize(self, question: List[str], context: List[str]):
        """
        Returns:
             A dict-like object (str: torch.Tensor) with keys:
            'input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping'
        """
        return self.biobert_tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation='longest_first',
            return_offsets_mapping=True,
            return_token_type_ids=True,
            padding='max_length',
            return_tensors="pt",
        )
        """
        For later
        return self.biobert_tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,  # check if it might be removed
            padding="max_length",
            return_tensors="pt",
        )
        """

    def save(self, save_dir: str):
        self.biobert_tokenizer.save_pretrained(save_dir)
