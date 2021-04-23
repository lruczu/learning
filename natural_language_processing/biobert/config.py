import os


HUGGINGFACE_BIOBERT_CHECKPOINT = "dmis-lab/biobert-v1.1"


# directory in which to keep models from Hugging Face
CACHE_DIR = 'cached'


class TrainingConfig:
    """
    parameter EXPERIMENT_NAME requires some explanation.
    Under the catalog EXPERIMENT_NAME/
    There will be
        - model/ (model will be saved at the end of each epoch)
        - tokenizer/ (similarly as model)
        - logs/ (here you can find some training metrics)

    If you train your model on squad data and call your experiment 'squad',
    under the catalog squad/model you will have fine-tuned model.
    Furthermore, you can train model on some custom dataset having trained on squad one
    as a starting point. To do that you need to change the ...
    And of course paths to data and experiment name.
    By default we start training raw BioBert from Hugging Face.
    """
    WEIGHT_DECAY = 0.  # check sth from [0.01-0.1] as an educated guess
    LR = 5e-5  # check sth from [1e-5-5e-5] as an educated guess
    WARM_UP_PROP = 0.  # check sth from [0-0.2] as an educated guess
    N_EPOCHS = 3
    BATCH_SIZE = 16
    TRAIN_DATA_PATH = 'example_train.json'
    VALID_DATA_PATH = 'example_valid.json'
    EXPERIMENT_NAME = 'tuning_squad'
    MODEL_CHECKPOINT = HUGGINGFACE_BIOBERT_CHECKPOINT
    TOKENIZER_CHECKPOINT = HUGGINGFACE_BIOBERT_CHECKPOINT  # probably won't be changed


class PreprocessingConfig:
    MAX_LENGTH = 384
    DOC_STRIDE = 128
    MAX_QUERY_LENGTH = 64
    MAX_ANSWER_LENGTH = 15
    TRAINING_PATH = os.path.join('training_data', 'training.json')
    VALIDATION_PATH = os.path.join('training_data', 'validation.json')


class Inference:
    MODEL_PATH = HUGGINGFACE_BIOBERT_CHECKPOINT  # change to trained one in a custom way
    TOKENIZER_PATH = HUGGINGFACE_BIOBERT_CHECKPOINT  # probably won't be changed
    CUDA = True
    BATCH_SIZE = 16
