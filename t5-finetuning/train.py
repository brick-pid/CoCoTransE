import os
import hydra
import datetime
from dataset.dataset_manager import create_it_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from utils import tokenize_function

os.environ["WANDB_PROJECT"] = "codet5p-translation-finetuning"  # name your W&B project

@hydra.main(config_path="config", config_name="config")
def main(cfg):

    print("config file: ", cfg)
    model_name_or_path = cfg.data.model_name_or_path

    model_name = model_name_or_path.split("/")[-1]
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    effective_batch_size = cfg.optim.per_device_train_batch_size * cfg.optim.grad_accu
    run_name = f"{now}_{model_name}_{cfg.data.src}_to_{cfg.data.tgt}_{cfg.data.sample_size}_lr{cfg.optim.lr}_ebs{effective_batch_size}_aug{cfg.aug}_nl{cfg.data.nl}_patch{cfg.data.patch}"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    dataset = create_it_dataset(cfg.data.tgt, cfg.data.sample_size, cfg.aug)

    for split in ['train', 'test']:
        dataset[split] = dataset[split].map(lambda x: tokenize_function(x, tokenizer), batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir = '.',
        num_train_epochs=3,
        per_device_train_batch_size=cfg.optim.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.optim.per_device_train_batch_size * 8,
        gradient_accumulation_steps=cfg.optim.grad_accu,
        learning_rate=cfg.optim.lr,
        lr_scheduler_type=cfg.optim.lr_scheduler_type,
        warmup_ratio=cfg.optim.warmup_ratio,
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=10,
        report_to="wandb",
        predict_with_generate=True,
        generation_max_length=512,
        run_name=run_name
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(run_name)  # save_model() -> _save(), will save model, including tokenizer and config(if not null)

if __name__ == "__main__":
    main()