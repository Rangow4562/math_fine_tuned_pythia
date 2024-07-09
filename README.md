# Math Solver with Retrieval-Augmented Generation (RAG)

### Math Problem Solver with Retrieval-Augmented Generation (RAG)

The Math Problem Solver with Retrieval-Augmented Generation (RAG) is a Python-based project designed to automate the solving of mathematical problems using state-of-the-art natural language processing (NLP) techniques. It leverages the power of Transformers models, fine-tuned on mathematical datasets, to provide accurate solutions to algebraic and linear equations.

### The project integrates several key components:

Data Preparation: Includes scripts for loading, cleaning, and preparing mathematical datasets.
Model Fine-Tuning: Utilizes Hugging Face's Transformers library to fine-tune pre-trained language models for specific mathematical problem-solving tasks.
Retrieval-Augmented Generation (RAG): Implements a pipeline that combines information retrieval with generative language models to enhance answer accuracy and relevance.
Evaluation: Provides mechanisms for evaluating the performance of the RAG pipeline using metrics like accuracy and BLEU score.
Instructive Fine-Tuning: Allows for further refinement of models through instructive fine-tuning using instructional prompts.
The project is structured to support both research and practical applications in automated mathematical problem-solving, aiming to streamline educational tools, tutoring systems, and more.

## Overview

### Goals

1. Automate Mathematical Problem Solving:
Reduce the need for manual calculation by automating the solving of algebraic and linear equations.
Provide accurate and reliable solutions for educational and practical applications.
2. Enhance Educational Tools:
Support educational platforms and tools by offering an automated math-solving capability.
Assist students in understanding and verifying solutions to mathematical problems.
3. Explore Advanced NLP Techniques:
Explore the capabilities of retrieval-augmented generation (RAG) techniques in solving specific domains like mathematics.
Fine-tune and optimize transformer models for mathematical problem-solving tasks.

### Key Features
1. Data Preparation:
Dataset Loading and Cleaning: Scripts to load mathematical datasets and preprocess them for model training and evaluation.
2. Model Fine-Tuning:
Hugging Face Transformers: Utilizes pre-trained transformer models fine-tuned on mathematical datasets to understand and generate solutions.
3. Retrieval-Augmented Generation (RAG):
Contextual Information Retrieval: Integrates information retrieval to enhance answer accuracy by retrieving relevant contexts.
Generative Answer Generation: Generates answers based on retrieved contexts and input queries.
4. Evaluation:
Performance Metrics: Evaluates the accuracy and relevance of generated answers using metrics like accuracy and BLEU score.
5. Instructive Fine-Tuning:
Interactive Learning: Supports instructive fine-tuning of models through instructional prompts to improve performance and accuracy.

#### Applications
The project finds applications in:

1. Education: Supporting educational platforms with automated math-solving capabilities.
2. Technology: Enhancing tutoring systems and educational tools.
3. Research: Exploring the intersection of NLP and mathematical problem-solving.


## Installation

### Prerequisites

- Python 3.x
- Virtual environment (optional but recommended)

### Installation Steps

1. Clone the repository:
```
   git clone https://github.com/your_username/your_project.git
   cd your_project
```

2. Install dependencies using pip:
```
   pip install -r requirements.txt
```
### Usage

1. Data Preparation
To prepare the dataset:
```
python -m rag_math_solver.data.prepare_dataset
```

2. Model Fine-Tuning
To fine-tune the model:
```
python -m rag_math_solver.models.fine_tune_model
```

3. Setting up RAG Pipeline
To set up the RAG pipeline:

```
python -m rag_math_solver.rag.setup_rag_pipeline
```

4. Evaluation
To evaluate the RAG pipeline:

```
python -m rag_math_solver.evaluation.evaluate_rag
```

5. Instruct Fine-Tuning
To perform instructive fine-tuning:

```
python -m rag_math_solver.fine_tuning.instruct_finetune
```

6. Example Usage
To run the main pipeline and generate answers:

```
python -m main
```

7. Testing
Describe how to run tests for the project:

```
python -m unittest discover -s tests -p "*.py"
```

8. Model's Link

#1 : fine_tuned_pythia_70m

```
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("entity2260/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("entity2260/pythia-70m")
```

#2 : instruct_finetuned_pythia_70m


```
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("entity2260/algebra_linear_1d")
model = AutoModelForSeq2SeqLM.from_pretrained("entity2260/algebra_linear_1d")
```


### To run with docker (with containerised csv outputs) use
```
#1 : 
docker build -f Dockerfile -t rag-math-solver . --no-cache  

#2 :
docker run -it --rm rag-math-solver
```

```
rag_math_solver/
│
├── rag_math_solver/
│ ├── data/
│ │ ├── prepare_dataset.py
│ │ └── ...
│ ├── evaluation/
│ │ ├── evaluate.py
│ │ └── ...
│ ├── fine_tuning/
│ │ ├── instruct_finetune.py
│ │ └── ...
│ ├── models/
│ │ ├── model_utils.py
│ │ └── ...
│ ├── rag/
│ │ ├── rag_pipeline.py
│ │ └── ...
│ ├── main.py
│ └── ...
│
├── test/
│ ├── test_prepare_dataset.py
│ ├── test_main.py
│ └── ...
│
├── requirements.txt
├── README.md
```