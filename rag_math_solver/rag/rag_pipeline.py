import numpy as np
import faiss
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from transformers import pipeline
import rag_math_solver.config as config

def setup_rag(train_dataset, tokenizer):
    # Setup text generation pipeline using Hugging Face Transformers
    generator = pipeline("text-generation", model=config.FINE_TUNED_MODEL_DIR)
    
    # Setup Faiss index for efficient nearest neighbor search
    index = faiss.IndexFlatL2(768)  # 768 is the dimension of the embeddings

    # Define function to get embeddings
    def get_embedding(text):
        return np.mean(tokenizer.encode(text, return_tensors="np"), axis=1)

    # Add embeddings of training dataset to Faiss index
    for example in train_dataset:
        embedding = get_embedding(example["question"])
        index.add(embedding)

    return generator, index, get_embedding

def rag_generate(query, generator, index, get_embedding, train_dataset, k=5):
    # Generate response using RAG pipeline
    query_embedding = get_embedding(query)
    _, indices = index.search(query_embedding, k)
    
    context = ""
    for idx in indices[0]:
        context += f"Q: {train_dataset[idx]['question']}\nA: {train_dataset[idx]['answer']}\n\n"
    
    prompt = f"{context}Q: {query}\nA:"
    response = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    
    return response.split("A:")[-1].strip()
