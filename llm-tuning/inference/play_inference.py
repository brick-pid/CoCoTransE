"""Goal:
1. support multiple languages inference;
2. output into a jsonl file; same as cangjie humaneval output;
3. format as  {"src": ..., "pred": ...} 
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model and tokenizer
model_path = "/root/ljb/output/2024-05-26_20-58_julia_it_10000_lr1e-05_ef-bs10"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define your input prompt
input_text = """Translate this from Python to Julia:
Python: from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: for idx, elem in enumerate(numbers): for idx2, elem2 in enumerate(numbers): if idx != idx2: distance = abs(elem - elem2) if distance < threshold: return True return False
Julia:
"""

# Tokenize the input
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text with a stop token
generated_text = model.generate(
    input_ids=input_ids,
    max_length=512,  # adjust the max length as needed,
    eos_token_id=49152,  # stop token
)

# Decode the generated text
input_len = len(input_ids[0])
output_text = tokenizer.decode(generated_text[0][input_len:], skip_special_tokens=True)
print(output_text)
print(generated_text[0])
