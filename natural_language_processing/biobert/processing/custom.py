class Tokenizer:
    MAX_N_TOKENS = 512

    def __init__(
        self,
        min_question_length: int,
        max_question_length: int,
        min_context_length: int,
        max_context_length: int,
        checkpoint: str,
    ):
        self.min_question_length = min_question_length
        self.max_question_length = max_question_length

        self.min_context_length = min_context_length
        self.max_context_length = max_context_length

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer()

    def tokenize(self, question: str, context: str):
        if not self.min_question_length <= len(question) <= self.max_question_length:
            raise Exception

        if not self.min_context_length <= len(context) <= self.max_context_length:
            raise Exception

        char_start_index = 0
        char_end_index = 0

        tokenized_qa = tokenizer(
            question,
            context,
            return_attention_mask=True,
            return_token_type_ids=True,  # check if necessary
            return_offsets_mapping=True,
        )

        sequence_ids = tokenized_qa.sequence_ids()

        if len(sequence_ids) > Tokenizer.MAX_N_TOKENS:
            raise Exception

        offsets = tokenized_qa['offset_mapping']

        token_start_index = sequence_ids.index(1)
        token_end_index = len(sequence_ids) - 1

        start_position = 0
        end_position = 0

        if (
                offsets[token_start_index][0] <= char_start_index and
                offsets[token_end_index][1] >= char_end_index
        ):
            while token_start_index < len(sequence_ids) and offsets[token_start_index][0] <= char_start_index:
                token_start_index += 1

            while offsets[token_end_index][1] >= char_end_index:
                token_end_index -= 1

            start_position = token_start_index
            end_position = token_end_index + 1

        return {
            'token_ids': tokenized_qa['input_ids'],
            'attention_mask': tokenized_qa['attention_mask'],
            'return_token_type_ids': tokenized_qa['return_token_type_ids'],
            'start_position': start_position,
            'end_position': end_position
        }
