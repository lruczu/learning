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
            max_length=10,
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


inputs = tokenizer.encode_plus(example.question_text, example.context_text, return_tensors='pt')
start_logits, end_logits = model(**inputs)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


start_logits = to_list(start_logits)[0]
end_logits = to_list(end_logits)[0]

start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)

print(start_idx_and_logit[:5])
print(end_idx_and_logit[:5])

start_indexes = [idx for idx, logit in start_idx_and_logit[:5]]
end_indexes = [idx for idx, logit in end_idx_and_logit[:5]]


tokens = to_list(inputs['input_ids'])[0]

question_indexes = [i+1 for i, token in enumerate(tokens[1:tokens.index(102)])]
question_indexes

import collections

# keep track of all preliminary predictions
PrelimPrediction = collections.namedtuple(
    "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
)

prelim_preds = []
for start_index in start_indexes:
    for end_index in end_indexes:
        # throw out invalid predictions
        if start_index in question_indexes:
            continue
        if end_index in question_indexes:
            continue
        if end_index < start_index:
            continue
        prelim_preds.append(
            PrelimPrediction(
                start_index = start_index,
                end_index = end_index,
                start_logit = start_logits[start_index],
                end_logit = end_logits[end_index]
            )
        )

prelim_preds = sorted(prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
pprint(prelim_preds[:5])

BestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "BestPrediction", ["text", "start_logit", "end_logit"]
)

nbest = []
seen_predictions = []
for pred in prelim_preds:

    # for now we only care about the top 5 best predictions
    if len(nbest) >= 5:
        break

    # loop through predictions according to their start index
    if pred.start_index > 0:  # non-null answers have start_index > 0

        text = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                tokens[pred.start_index:pred.end_index + 1]
            )
        )
        # clean whitespace
        text = text.strip()
        text = " ".join(text.split())

        if text in seen_predictions:
            continue

            # flag this text as being seen -- if we see it again, don't add it to the nbest list
            seen_predictions.append(text)

            # add this text prediction to a pruned list of the top 5 best predictions
            nbest.append(BestPrediction(text=text, start_logit=pred.start_logit, end_logit=pred.end_logit))

        # and don't forget -- include the null answer!
        nbest.append(BestPrediction(text="", start_logit=start_logits[0], end_logit=end_logits[0]))

"""
The null answer is scored as the sum of the start_logit and end_logit associated with the [CLS] token.
"""

"""
https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Evaluating-a-model-on-the-SQuAD2.0-dev-set-with-HF
"""