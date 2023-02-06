from .tasks import currentTask

import torch

# Device

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Seed

SEED = 151836

# Folders

ADVERSARIAL_DIR = f"./adversarialSamples/{currentTask}"
DATASETS_DIR = f"./datasets/{currentTask}"
MODELS_DIR = f"./models/{currentTask}"
HISTORY_DIR = f"./results/attacks/history/{currentTask}"
EVALUATIONS_DIR = f"./results/attacks/evaluation/{currentTask}"

# Dataframes

MODEL_PREDICTIONS_PATH = f"./results/models/predictions/predictions_{currentTask}.csv"
BASELINE_PATH = f"./results/models/baseline/baseline_{currentTask}.csv"
SIMILARITY_PATH = f"./results/models/similarity/similarity_{currentTask}.csv"

# Models

NORMALIZATION_PARAMS = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
INPUT_SIZE = 224

# Dataset

DATASET_SIZE = 150
