# config.py

import os

# Dataset configurations
DATASET_NAME = "math_dataset"
DATASET_CONFIG_NAME = "algebra__linear_1d"
TEST_SIZE = 0.2
SHUFFLE = True
SEED = 42

# Model configurations
MODEL_NAME = "EleutherAI/pythia-70m"

# Training configurations
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# Paths
OUTPUT_DIR = "./results"
LOGGING_DIR = "./logs"
INSTRUCT_OUTPUT_DIR = "./instruct_results"
INSTRUCT_LOGGING_DIR = "./instruct_logs"
FINE_TUNED_MODEL_DIR = "./fine_tuned_pythia_70m"
INSTRUCT_FINETUNED_MODEL_DIR = "./instruct_finetuned_pythia_70m"
