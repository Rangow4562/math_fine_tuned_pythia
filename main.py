import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import rag_math_solver.config as config
from rag_math_solver.data.prepare_dataset import extract_dataset, transform_dataset, load_dataset, clean_example
from rag_math_solver.models.model_utils import setup_model_and_tokenizer, fine_tune_model
from rag_math_solver.rag.rag_pipeline import setup_rag, rag_generate
from rag_math_solver.evaluation.evaluate import evaluate_rag
from rag_math_solver.fine_tuning.instruct_finetune import instruct_finetune

def main():    
    # Load and split dataset
    train_dataset, eval_dataset = load_dataset(test_size=0.2, seed=42)
    print(f"Train set size: {len(train_dataset)}")
    print(f"Evaluation set size: {len(eval_dataset)}")

    # Setup model and tokenizer
    model_name = (config.MODEL_NAME)
    model, tokenizer = setup_model_and_tokenizer(model_name)

    # Fine-tune the model
    fine_tune_model(model, tokenizer, train_dataset, eval_dataset)

    # Setup RAG pipeline
    generator, index, get_embedding = setup_rag(train_dataset, tokenizer)

    # Evaluate RAG pipeline
    evaluate_rag(eval_dataset, generator, index, get_embedding, train_dataset)

    # Instruct fine-tuning
    instruct_finetune(model, tokenizer, train_dataset, eval_dataset)

    # Example usage of RAG pipeline
    query = "Solve the equation: 3x + 5 = 14"
    answer = rag_generate(query, generator, index, get_embedding, train_dataset)
    print(f"Query: {query}")
    print(f"Generated Answer: {answer}")

if __name__ == "__main__":
    main()
