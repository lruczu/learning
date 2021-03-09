import datasets
import torch


@torch.no_grad()
def evaluate(
    dataset: datasets.Dataset,
    model,
    tokenizer,
    batch_size: int = 32,
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    for batch in dataloader:
        tensor_dict = tokenizer.tokenize(batch['question'], batch['context'])

        output = model(
            input_ids=tensor_dict['input_ids'],
            token_type_ids=tensor_dict['token_type_ids'],
            attention_mask=tensor_dict['attention_mask'],
            start_positions=torch.tensor(batch['start_index']),
            end_positions=torch.tensor(batch['end_index'])
        )
        start_logits = output['start_logits']
        end_logits = output['end_logits']

