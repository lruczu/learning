CHECKPOINT = 'bert-base-cased'

MODEL_CHECKPOINT = 'ner-model'

Parameters = {
    'MAX_LEN': 128,
    'VAL_PROP': 0.15,
    'N_EPOCHS': 10,
    'TRAIN_BATCH_SIZE': 16,
    'TEST_BATCH_SIZE': 16,
    'WEIGHT_DECAY': 0.01,
    'lr': 2e-5,
    'WARM_UP_PROP': 0.15,
    'TRAINING_PATH': 'data/training_data.json',
    'submission_paths': [
        'data/2f392438-e215-4169-bebf-21ac4ff253e1.json',
        'data/3f316b38-1a24-45a9-8d8c-4e05a42257c6.json',
        'data/8e6996b4-ca08-4c0b-bed2-aaf07a4c6a60.json',
        'data/2100032a-7c33-4bff-97ef-690822c43466.json'
    ]
}

EXAMPLES_TO_DISPLAY = [
    "We used data from the Adult Changes in Thought (ACT) study, the Alzheimer's Disease Neuroimaging Initiative (ADNI), the Rush Memory and Aging Project (MAP) and Religious Orders Study (ROS), and the University of Pittsburgh Alzheimer Disease Research Center (PITT). Each study has published widely, and their genetic data are included in large analyses of late-onset Alzheimer's disease [6, 7] ."
]
