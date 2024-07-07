import os
import datetime
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from dataset.dataset_manager import create_it_dataset

# Hyperparameters
os.environ["WANDB_PROJECT"] = "starcoder2-3b-ft"  # name your W&B project
model_name_or_path = "bigcode/starcoder2-3b"
model_name = model_name_or_path.split("/")[-1]

if "checkpoint" in model_name:
    model_name = model_name_or_path.split("/")[-2] + "_" + model_name_or_path.split("/")[-1]

if "julia" in model_name:
    pl = "julia"
elif "ocaml" in model_name:
    pl = "ocaml"
elif "cangjie" in model_name:
    pl = "cangjie"
elif model_name == "starcoder2-3b":
    pl = "julia"
else:
    raise ValueError("Unknown programming language in ", model_name)

type = "it"
aug_flag = True
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

sample_size = 2200 #2200
lr = 1e-5
lr_scheduler_type = "cosine"
batch_size =1
gradient_accumulation_steps = 32
num_train_epochs = 4
effective_batch_size = batch_size * gradient_accumulation_steps
run_name = f"{now}_{model_name}_{pl}_{type}_{sample_size}_lr{lr}_ebs{effective_batch_size}_aug{aug_flag}"
output_dir = f"./output/{run_name}"


# Utilities
def format_text(example):
    """
    Formats the prompt for the model.
    """
    output_texts = []
    for i in range(len(example['instruction'])):
        text = example['instruction'][i] + example['response'][i] + "<|end|>"
        output_texts.append(text)
    return output_texts

# Load Resources
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

if "<|end|>" not in tokenizer.special_tokens_map.values():
    current_additional_special_tokens = tokenizer.additional_special_tokens
    new_tokens_to_add = ['<|end|>']
    current_additional_special_tokens.extend(new_tokens_to_add)
    tokenizer.add_special_tokens({"additional_special_tokens": current_additional_special_tokens})
    print("current additional special tokens: ", tokenizer.special_tokens_map)
else:
    print("Special token '<|end|>' already exists.")

# Data collator for completion
response_template = "### Translate"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# resize the token embeddings
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))


dataset = create_it_dataset(pl, sample_size=sample_size, aug=aug_flag)
train_split = dataset['train']
eval_split = dataset['test']

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    learning_rate=lr,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=0.05,
    num_train_epochs=num_train_epochs,
    gradient_accumulation_steps=gradient_accumulation_steps,
    bf16=True,
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=10,
    report_to="wandb",
    run_name=run_name
)


# Train the model
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_split,
    eval_dataset=eval_split,
    args=training_args,
    max_seq_length=1280,
    formatting_func=format_text,
    data_collator=collator,
)
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(output_dir)
