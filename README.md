# CoCoTransE
## Dataset Access
We currently provide a subset of Cangjie pre-trained and fine-tuning datasets in the `data_demo` folder. A complete dataset will be provided in the future.

## Training
Training the model is divided into two phases: continuous pretraining and instruction finetuning. The `llm-tuning` folder includes code for continuous pretraining and instruction fine-tuning of the starcoder model; the `t5-finetuning` folder includes code for instruction fine-tuning of the code-t5p model; and the `t5-pretraining` folder includes code for continuous pretraining of the code-t5p model.