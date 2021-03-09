import os

CHECKPOINT = "dmis-lab/biobert-v1.1"

# Where to save pretrained model/tokenizer from huggingface repository
CACHED_MODEL_DIR = os.path.join('cached', 'model')
CACHED_TOKENIZER_DIR = os.path.join('cached', 'tokenizer')

# optimizer params
WEIGHT_DECAY = 0.01
LR = 5e-5

# learning rate scheduler param
WARM_UP_PROP = 0.1

# training params
N_EPOCHS = 3
BATCH_SIZE = 32
LOG_DIR = os.path.join('experiment', 'logs')

# location of trained model
OUTPUT_DIRECTORY = os.path.join('experiment', 'model')

# number of steps to save to latest model
MODEL_CHECKPOINT_SAVE = 500

# data
TRAIN_DATA_PATH = 'example_train.json'  # or list of
VALID_DATA_PATH = 'example_valid.json'

# tokenizer params
DOC_STRIDE = 128
MAX_LENGTH = 384
