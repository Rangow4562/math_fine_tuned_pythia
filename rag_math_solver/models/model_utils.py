from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import rag_math_solver.config as config

def setup_model_and_tokenizer(model_name):
    # Load model and tokenizer from Hugging Face Transformers library
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ensure padding token is set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

def preprocess_function(examples, tokenizer):
    # Prepare inputs and labels for fine-tuning
    inputs = [f"Problem: {q}\nSolution:" for q in examples["question"]]
    targets = [f" {a}" for a in examples["answer"]]
    
    # Tokenize inputs and labels
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def fine_tune_model(model, tokenizer, train_dataset, eval_dataset):
    # Tokenize training and evaluation datasets
    tokenized_train = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    tokenized_eval = eval_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    # Define training arguments for fine-tuning
    
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.LOGGING_DIR,
    )

    # Initialize Trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    # Start fine-tuning
    trainer.train()

    # Save fine-tuned model and tokenizer
    model.save_pretrained(config.FINE_TUNED_MODEL_DIR)
    tokenizer.save_pretrained(config.FINE_TUNED_MODEL_DIR)
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# # import multiprocessing
# from functools import partial

# def preprocess_function_parallel(examples, tokenizer, batch_size=8):
#     # Define a function to preprocess a batch
#     def preprocess_batch(batch_examples):
#         return [preprocess_function(example, tokenizer) for example in batch_examples]

#     # Split examples into batches
#     batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]

#     # Parallelize preprocessing
#     with multiprocessing.Pool() as pool:
#         processed_batches = pool.map(preprocess_batch, batches)

#     # Flatten processed batches
#     tokenized_examples = [example for batch in processed_batches for example in batch]
#     return tokenized_examples

# def fine_tune_model_parallel(model, tokenizer, train_dataset, eval_dataset):
#     tokenized_train = preprocess_function_parallel(train_dataset, tokenizer)
#     tokenized_eval = preprocess_function_parallel(eval_dataset, tokenizer)

#     training_args = TrainingArguments(
#         output_dir="./results",
#         num_train_epochs=3,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir="./logs",
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train,
#         eval_dataset=tokenized_eval,
#     )

#     trainer.train()
#     model.save_pretrained("./fine_tuned_pythia_70m")
#     tokenizer.save_pretrained("./fine_tuned_pythia_70m")
