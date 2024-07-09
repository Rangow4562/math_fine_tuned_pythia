import random
import sys
import os
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import rag_math_solver.config as config
from transformers import TrainingArguments, Trainer

def prepare_instruct_dataset(dataset):
    instructions = [
        "Solve the following mathematical problem:",
        "Calculate the answer to this equation:",
        "Find the solution to this math question:",
    ]

    def format_instruct(example):
        instruction = random.choice(instructions)
        return {
            "input": f"{instruction}\n{example['question']}",
            "output": example['answer']
        }

    return dataset.map(format_instruct)

def instruct_finetune(model, tokenizer, train_dataset, eval_dataset):
    # Prepare instructional prompts for training and evaluation datasets
    instruct_train = prepare_instruct_dataset(train_dataset)
    instruct_eval = prepare_instruct_dataset(eval_dataset)

    # Define training arguments for instructive fine-tuning
    training_args = TrainingArguments(
        output_dir=config.INSTRUCT_OUTPUT_DIR,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.INSTRUCT_LOGGING_DIR,
    )
    # Initialize Trainer for instructive fine-tuning
    instruct_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=instruct_train,
        eval_dataset=instruct_eval,
    )

    # Start training
    instruct_trainer.train()

    # Save fine-tuned model and tokenizer
    model.save_pretrained(config.INSTRUCT_FINETUNED_MODEL_DIR)
    tokenizer.save_pretrained(config.INSTRUCT_FINETUNED_MODEL_DIR)
