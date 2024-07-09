from nltk.translate.bleu_score import sentence_bleu
import nltk
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import rag_math_solver.config as config
from rag_math_solver.rag.rag_pipeline import setup_rag, rag_generate

nltk.download('punkt', quiet=True)

def evaluate_rag(eval_dataset, generator, index, get_embedding, train_dataset, num_samples=100):
    correct = 0
    total_bleu = 0

    for i in range(min(num_samples, len(eval_dataset))):
        example = eval_dataset[i]
        query = example['question']
        true_answer = example['answer']
        
        # Generate answer using RAG pipeline
        generated_answer = rag_generate(query, generator, index, get_embedding, train_dataset)

        # Check correctness
        if generated_answer.strip() == true_answer.strip():
            correct += 1

        # Calculate BLEU score
        reference = [true_answer.split()]
        candidate = generated_answer.split()
        bleu_score = sentence_bleu(reference, candidate)
        total_bleu += bleu_score

    # Compute accuracy and average BLEU score
    accuracy = correct / num_samples
    avg_bleu = total_bleu / num_samples

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Average BLEU score: {avg_bleu:.4f}")
