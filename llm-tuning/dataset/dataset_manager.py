from datasets import load_from_disk, load_dataset, concatenate_datasets
from transformers import AutoTokenizer


def create_cpt_dataset(pl: str) -> dict:
    """
    return dataset dict of train and test split
    """
    if pl != 'cangjie':
        ds_dir = "dataset/multipl_parallel"
        dataset = load_from_disk(ds_dir)[pl]
        dataset = calculate_tokens(dataset)
        dataset = dataset.remove_columns(['func_name', 'py', 'id'])
    elif pl == 'cangjie':
        cj_func_path = "dataset/cj_functions.jsonl"
        dataset = load_dataset('json', data_files={'train': cj_func_path}, split='train')
        dataset = dataset.rename_column('cj', 'pl')
        # dataset = calculate_tokens(dataset)
    else:
        raise NotImplementedError(f"{pl} dataset not implemented")
    
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_split = dataset['train']
    eval_split = dataset['test']

    return {
        "train": train_split,
        "test": eval_split
    }

# method to create dataset based on the given dataset type
def create_it_dataset(pl: str, sample_size: int = None, aug: bool = False) -> dict:
    """
    return dataset dict of train and test split
    """
    if pl != 'cangjie':
        ds_dir = "dataset/multipl_parallel"
        dataset = load_from_disk(ds_dir)[pl]
        dataset = dataset.map(lambda x: create_it_prompt("python", x["py"], pl, x["pl"]), keep_in_memory=True)
        dataset = dataset.filter(lambda x: len(x["instruction"] + x["response"]) < 1024)
        if sample_size:
            dataset = dataset.select(range(sample_size))
        if aug:
            print("before augmentation: ", dataset)

            # back augment
            back_aug = dataset.map(lambda x: create_it_prompt(pl, x["pl"], "python", x["py"]), keep_in_memory=True)

            # bonus pl aug
            if pl == 'julia':
                r = load_from_disk(ds_dir)['r']
                if sample_size:
                    r = r.select(range(sample_size))
                r_to_py = r.map(lambda x: create_it_prompt("r", x["pl"], "python", x["py"]), keep_in_memory=True)
                py_to_r = r.map(lambda x: create_it_prompt("python", x["py"], "r", x["pl"]), keep_in_memory=True)
                dataset = concatenate_datasets([dataset, back_aug, r_to_py, py_to_r])
            elif pl == 'ocaml':
                rkt_py = load_from_disk(ds_dir)['racket']
                if sample_size:
                    rkt_py = rkt_py.select(range(sample_size))
                py_to_rkt = rkt_py.map(lambda x: create_it_prompt("python", x["py"], "racket", x["pl"]), keep_in_memory=True)
                rkt_to_py = rkt_py.map(lambda x: create_it_prompt("racket", x["pl"], "py", x["py"]), keep_in_memory=True)
                
                dataset = concatenate_datasets([dataset, back_aug, py_to_rkt, rkt_to_py])
            
            print("after augmentation: ", dataset)


    elif pl == 'cangjie':
        cj_parallel_dir = "/root/ljb/dataset/patched_cj_parallel"
        datasets = load_from_disk(cj_parallel_dir)
        for split in datasets:
            datasets[split] = datasets[split].map(lambda x: create_it_prompt("java", x["src"], "cangjie", x["gt"]), keep_in_memory=True)
            # datasets[split] = datasets[split].filter(lambda x: len(x["instruction"] + x["response"]) < 1024)
        
        if aug:
            print("before augmentation: ", datasets)

            # back augment
            back_aug = datasets['train'].map(lambda x: create_it_prompt("cangjie", x["gt"], "java", x["src"]), keep_in_memory=True)
            
            # java-python augment
            java_py_path = "/root/ljb/dataset/java-python-parallel.jsonl"
            java_py = load_dataset('json', data_files={'train': java_py_path}, split='train')
            j_to_py = java_py.map(lambda x: create_it_prompt("java", x["java"], "python", x["python"]), keep_in_memory=True)
            py_to_j = java_py.map(lambda x: create_it_prompt("python", x["python"], "java", x["java"]), keep_in_memory=True)

            # merge augmented datasets
            datasets['train'] = datasets['train'].remove_columns(list(set(datasets['train'].column_names) - set(['instruction', 'response'])))
            back_aug = back_aug.remove_columns(list(set(back_aug.column_names) - set(['instruction', 'response'])))
            j_to_py = j_to_py.remove_columns(list(set(j_to_py.column_names) - set(['instruction', 'response'])))
            py_to_j = py_to_j.remove_columns(list(set(py_to_j.column_names) - set(['instruction', 'response'])))

            datasets['train'] = concatenate_datasets([datasets['train'], back_aug, j_to_py, py_to_j])

            print("after augmentation: ", datasets)

    else:
        raise NotImplementedError(f"{pl} dataset not implemented")
    
    if pl == 'cangjie':
        train_split = datasets['train']
        eval_split = datasets['test']
    else:
        datasets = dataset.train_test_split(test_size=0.05, seed=42)
        train_split = datasets['train']
        eval_split = datasets['test']

    return {
        "train": train_split,
        "test": eval_split
    }


# general it input creation
def create_it_prompt(src_pl, src, tgt_pl, tgt):
    it = f"### Translate this from {src_pl.capitalize()} to {tgt_pl.capitalize()}:"
    it = it + '\n' + f"{src_pl.capitalize()}: " + src + '\n' + f"### Translate to {tgt_pl.capitalize()}: "
    return {
        "instruction": it,
        "response": tgt
    
    }


def calculate_tokens(dataset, token_num = 5300000):
    """
    Calculate number of tokens of text column in dataset
    """

    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
    
    token_count = 0
    seq_count = 0
    for example in dataset:
        token_count += len(tokenizer.encode(example["pl"]))
        seq_count += 1
        if token_count > token_num:
            break
    
    print(f"token_count for cpt: {token_count}")
    return dataset.select(range(seq_count))

