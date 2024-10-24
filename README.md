# CoCoTransE

## Overview
![CoCoTransE](./assets/cocotranse.png)

## Dataset Access
We have open-sourced both our Cangjie([仓颉](https://developer.huawei.com/consumer/cn/doc/cangjie-guides-V5/cj-wp-abstract-V5)) monolingual dataset and the Cangjie-Java parallel corpus. These datasets can be found in the `data` directory.

## Training
Model training is divided into two phases: continuous pretraining and instruction fine-tuning. 
- The `llm-tuning` folder contains the code for continuous pretraining and instruction fine-tuning of the StarCoder model.
- The `t5-finetuning` folder includes the code for instruction fine-tuning of the Code-T5p model.
- The `t5-pretraining` folder contains the code for continuous pretraining of the Code-T5p model.

## Evaluation
We evaluate translation results using the [BLEU](https://aclanthology.org/P02-1040.pdf) automated metric and [Function Equivalence](https://openreview.net/pdf?id=fVxIEHGnVT). 

### How to run?
1. Open the `bash-test/call_bash.ipynb` file;
2. Change the model_id to the model translation result you want to evaluate. For example, you can use the demo result `starcoder2-3b_cangjie_it_2200_lr1e-05_ebs32` provided for easy testing.
3. run the notebook.

### Notice
the model translation result should be a jsonl file with the following format:
```jsonl
{"src": "...", "pred": "..."}
{"src": "...", "pred": "..."}
{"src": "...", "pred": "..."}
```
