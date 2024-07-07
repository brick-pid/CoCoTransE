def tokenize_function(examples, tokenizer):
    inputs = tokenizer(examples["instruction"], truncation=True, max_length=512)
    targets = tokenizer(examples["response"], truncation=True, max_length=512)
    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": targets.input_ids,
    }