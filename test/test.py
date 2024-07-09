from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file as load_safetensors
import torch

# Define the paths to your files
model_path = "fine_tuned_pythia_70m"
config_path = f"{model_path}/config.json"
generation_config_path = f"{model_path}/generation_config.json"
model_file = f"{model_path}/model.safetensors"
tokenizer_files = {
    "special_tokens_map_file": f"{model_path}/special_tokens_map.json",
    "tokenizer_file": f"{model_path}/tokenizer.json",
    "tokenizer_config_file": f"{model_path}/tokenizer_config.json"
}

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_files)
print("Tokenizer loaded successfully")

# Load the model weights from safetensors
model_state_dict = load_safetensors(model_file)

# Load the model with the state dict
model = AutoModelForCausalLM.from_pretrained(model_path, state_dict=model_state_dict)
print("Model loaded successfully")

# Check if the model and tokenizer are correctly loaded
if tokenizer and model:
    print("Model and tokenizer are ready for use")

# Now you can use the model and tokenizer for text generation
input_text = "Solve 3807 = -328*a + 20863"
inputs = tokenizer(input_text, return_tensors="pt")
print("Input tokens:", inputs)

# Generate the output with modified parameters
outputs = model.generate(
    **inputs, 
    max_new_tokens=50,  # Adjust max_new_tokens as needed
    temperature=0.7,  # Adjust temperature for sampling
    top_k=50,  # Top-k sampling
    top_p=0.95,  # Top-p (nucleus) sampling
    do_sample=True  # Enable sampling
)
print("Generated tokens:", outputs)

# Decode the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract and print the computed value of a
import re
match = re.search(r'-?\d+', generated_text)
if match:
    computed_value_a = int(match.group())
    print("Computed value of a:", computed_value_a)
else:
    print("Unable to extract computed value of a from generated text.")

print("Generated text:", generated_text)
