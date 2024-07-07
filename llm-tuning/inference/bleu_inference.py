import sys
sys.path.append("/root/ljb")

from dataset.dataset_manager import create_it_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric

from tqdm import tqdm
import json

model_path = "/root/ljb/output/2024-06-02_15-18_starcoder2-3b_julia_it_2200_lr1e-05_ebs32_augTrue"
model_name = model_path.split("/")[-1]
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
output_path = f"/root/ljb/inference/bleu/{model_name}.jsonl"

type = "it"
if "julia" in model_name:
    pl = "julia"
elif "ocaml" in model_name:
    pl = "ocaml"
elif "cangjie" in model_name:
    pl = "cangjie"
else:
    raise ValueError("Unknown programming language in ", model_name)

sample_size = 2200
dataset = create_it_dataset(pl, sample_size=sample_size)
eval_split = dataset['test']
# for debug
# eval_split = eval_split.select(range(5))




pred_list = []
tgt_list = []
for example in tqdm(eval_split):
    it = example['instruction']
    tgt = example['response']
    
    input_ids = tokenizer.encode(it, return_tensors="pt").to("cuda")
    if len(input_ids) > 512:
        print("Skipping example as input length is greater than 512")
        continue
    pred = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,  # adjust the max length as needed,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>"),  # stop token
    )
    input_len = len(input_ids[0])
    pred = tokenizer.decode(pred[0][input_len:], skip_special_tokens=True)

    pred_list.append([pred])
    tgt_list.append([tgt])

bleu = load_metric("sacrebleu")
score = bleu.compute(predictions=pred_list, references=tgt_list)

# outptu to file
with open(output_path, "w") as f:
    f.write(f"BLEU Score: {score['score']}\n")
    f.write(f"BLEU Details: {score}\n")

    # write the predictions and targets also
    f.write("\n\n")

    for i in range(len(pred_list)):
        d = {"pred": pred_list[i][0], "gt": tgt_list[i][0]}
        f.write(json.dumps(d) + "\n")





