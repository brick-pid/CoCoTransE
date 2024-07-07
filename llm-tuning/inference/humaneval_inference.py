"""Goal:
1. support multiple languages inference;
2. output into a jsonl file; same as cangjie humaneval output;
3. format as  {"src": ..., "pred": ...} 
"""

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the model")
args = parser.parse_args()

model_path = args.model_path
model_name = model_path.split("/")[-1]
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)

if "cangjie" in model_name:
    pl = "cangjie"
elif "julia" in model_name:
    pl = "julia"
elif "ocaml" in model_name:
    pl = "ocaml"
else:
    raise ValueError("Unknown programming language in ", model_name)

if pl == "cangjie":
    humaneval_prompt_path = "/root/ljb/evaluate/humaneval-x-java.jsonl"
else:
    humaneval_prompt_path = "/root/ljb/evaluate/humaneval-x-py.jsonl"
output_path = f"/root/ljb/inference/results/{model_name}.jsonl"

def get_src(example):
    if pl == "cangjie":
        return example['java']
    else:
        return example['declaration'] + example['canonical_solution']

# load humaneval-py data
result = [] # each line is a dict of {'src': ..., 'pred': ...}
with open(humaneval_prompt_path, 'r') as f:
    lines = f.readlines()
    total_lines = len(lines)
    for line in tqdm(lines, total=total_lines, desc="Translating"):
        example = json.loads(line)
        src = get_src(example)
        instruct = f"Translate this from Python to {pl.capitalize()}:\nPython: {src}\n{pl.capitalize()}:\n"
        input_ids = tokenizer.encode(instruct, return_tensors="pt").to("cuda")
        pred = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>"),  # stop token
        )
        input_len = len(input_ids[0])
        pred_text = tokenizer.decode(pred[0][input_len:], skip_special_tokens=True)
        result.append({'src': src, 'pred': pred_text})


# Save the result
with open(output_path, 'w') as f:
    for line in result:
        f.write(json.dumps(line) + '\n')

