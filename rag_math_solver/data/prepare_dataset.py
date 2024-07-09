from datasets import load_dataset
import os
import sys
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import rag_math_solver.config as config

def extract_dataset():
    # Load the dataset
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG_NAME, trust_remote_code=True)
    return dataset['train']

def transform_dataset(dataset):
    # Perform any necessary transformations here
    cleaned_dataset = dataset.map(clean_example)
    return cleaned_dataset

def load_dataset(test_size=0.2, seed=42):
    dataset = extract_dataset()
    transformed_dataset = transform_dataset(dataset)
    train_data, test_data = train_test_split(transformed_dataset, test_size=test_size, shuffle=True, random_state=seed)
    return train_data, test_data

def clean_example(example):
    # Example cleaning function (customize as per your dataset's needs)
    cleaned_question = example['question'].strip()
    cleaned_answer = example['answer'].strip()
    return {'question': cleaned_question, 'answer': cleaned_answer}

if __name__ == "__main__":
    train_dataset, eval_dataset = load_dataset()
    print(f"Train set size: {len(train_dataset)}")
    print(f"Evaluation set size: {len(eval_dataset)}")