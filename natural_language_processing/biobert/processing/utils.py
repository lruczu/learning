class Example:
  def __init__(
      self,
      question: str,
      context: str,
      answer: str,
      answer_start_index: int,
      is_impossible: bool = False
  ):
    self.question = question
    self.context = context
    self.answer = answer
    self.answer_start_index = answer_start_index
    self.is_impossible = is_impossible


from typing import List


class Tokenizer:
    def __init__(
            self,
            tokenizer,
            max_length: int,
            stride: int,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

    def prepare_train_features(self, examples: List[Example]):
        tokenizer_output = self.tokenizer(
            [example.question for example in examples],
            [example.context for example in examples],
            truncation='only_second',
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        return tokenizer_output
        sample_mapping = tokenizer_output.pop("overflow_to_sample_mapping")
        offset_mapping = tokenizer_output.pop("offset_mapping")

        tokenizer_output["start_positions"] = []
        tokenizer_output["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenizer_output["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenizer_output.sequence_ids(i)

            sample_index = sample_mapping[i]

            if examples[sample_index].is_impossible:
                tokenizer_output["start_positions"].append(cls_index)
                tokenizer_output["end_positions"].append(cls_index)
            else:
                start_char = examples[sample_index].answer_start_index
                end_char = start_char + len(examples[sample_index].answer)

                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenizer_output["start_positions"].append(cls_index)
                tokenizer_output["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenizer_output["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenizer_output["end_positions"].append(token_end_index + 1)

        return tokenizer_output
