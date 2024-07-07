import os
import datetime
import torch
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from dataset.dataset_manager import create_cpt_dataset

# Hyperparameters
os.environ["WANDB_PROJECT"] = "starcoder2-3b-cpt"  # name your W&B project
model_name_or_path = "bigcode/starcoder2-3b"
pl = "cangjie"
type = "cpt"
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

lr = 1e-4
lr_scheduler_type = "cosine"
batch_size = 4
gradient_accumulation_steps = 16

max_seq_len = 512

num_train_epochs = 3
effective_batch_size = batch_size * gradient_accumulation_steps
run_name = f"{now}_{pl}_{type}_lr{lr}_ebs{effective_batch_size}"
output_dir = f"./output_cpt/{run_name}"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)


dataset = create_cpt_dataset(pl)
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
    max_seq_length=max_seq_len,
    dataset_text_field="pl",
    packing=True,
)
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(output_dir)
