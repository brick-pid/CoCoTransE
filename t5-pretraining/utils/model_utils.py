import torch
import datasets
from torch.utils.data import DataLoader
from omegaconf import open_dict
from datasets.iterable_dataset import IterableDataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoConfig,
)

from utils.copied_utils import (
    compute_input_and_target_lengths,
    DataCollatorForT5MLM,
    tokenize_function,
    DataCollatorForNI,
)
from utils.t5_model import MyT5

from safetensors.torch import load_model

def get_model(args, config):
    # 这里klass定义了一个字典，有两个键值对，分别是'hf_t5'和'local_t5'，对应的值分别是T5ForConditionalGeneration和MyT5
    # 然后根据args.model.klass的值来选择对应的模型
    klass = {
        'hf_t5': T5ForConditionalGeneration,
        'local_t5': MyT5,
    }[args.model.klass]

    # 初始化模型参数: 1. 从checkpoint_path加载模型参数; 2. 随机初始化模型参数; 3. 从HuggingFace加载模型参数
    if args.model.checkpoint_path:
        model = klass(config)
        load_model(model, args.model.checkpoint_path) # load safetensors checkpoint
    elif args.model.random_init:
        model = klass(config)
    else:
        assert klass == T5ForConditionalGeneration, 'To load HFs weights you need to use HF model'
        model = klass.from_pretrained(
            args.model.name,
            config=config,
        )

    with open_dict(args):
        args.n_all_param = sum([p.nelement() for p in model.parameters()])
    
    return model


def get_config(args):
    config = AutoConfig.from_pretrained(
        args.model.name,
    )

    if hasattr(args.model, 'overwrite'):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f'config does not have attribute {k}'
            setattr(config, k, v)

    if hasattr(args.model, 'add_config'):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f'config already has attribute {k}'
            setattr(config, k, v)

    return config


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model.name,
        use_fast=True
    )
    tokenizer.model_max_length = int(1e9)

    return tokenizer


def load_dataset_splits(args):
    if args.mode == 'pt':
        if args.data.data_type == 'function' and args.data.pl in ['ocaml', 'julia', 'racket', 'r', 'lua']:
            dataset = datasets.load_from_disk("/home/sjw/ljb/multipl-parallel/multipl_parallel")[args.data.pl]
            dataset = dataset.rename_column('pl', 'text')
            dataset = dataset.remove_columns(['py', 'id', 'func_name'])
            dataset_splits = dataset.train_test_split(test_size=0.05, seed=args.seed)

        elif args.data.pl == 'cj' and args.data.data_type == 'function':
            path_to_jsonl = '/home/sjw/ljb/cangjie_data/cj_functions.jsonl'
            dataset = datasets.load_dataset('json', data_files={'train': path_to_jsonl}, split='train')
            # rename 'cj' to 'text' for common process
            dataset = dataset.rename_column('cj', 'text')
            dataset_splits = dataset.train_test_split(test_size=0.05, seed=args.seed)

        elif args.data.pl == 'cj' and args.data.data_type == 'file':
            dataset = datasets.load_dataset('text', data_dir='/home/sjw/ljb/cangjie_data/cangjie/', sample_by="document")
            dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
            dataset_splits = dataset.train_test_split(test_size=0.05, seed=args.seed)
            
        # elif args.data.corpus == 'cj_java_mix':
        #     dataset = datasets.load_dataset('text', data_dir='/home/sjw/ljb/cangjie_data/cangjie/', sample_by="document")
        #     dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        #     dataset_splits = dataset.train_test_split(test_size=0.05, seed=args.seed)
            
        #     java = datasets.load_dataset('code_search_net', 'java', split=f'train[:{args.data.mix_ratio}%]')
        #     java = java.select_columns(['func_code_string'])
        #     java = java.rename_column('func_code_string', 'text')

        #     dataset_splits['train'] = datasets.concatenate_datasets([dataset_splits['train'], java]).shuffle(seed=args.seed)

        #     # # for debug, set train_dataset to 1% of original size
        #     # dataset_splits['train'] = dataset_splits['train'].shard(num_shards=100, index=0)
        else:
            raise NotImplementedError(f"dataset type not support for {args.data.pl}-{args.data.data_type}!")
    elif args.mode == 'ft':
        dataset_splits = datasets.load_dataset(
            args.data.exec_file_path,
            data_dir=args.data.data_dir,
            task_dir=args.data.task_dir,
            max_num_instances_per_task=args.data.max_num_instances_per_task,
            max_num_instances_per_eval_task=args.data.max_num_instances_per_task
        )
    else:
        raise NotImplementedError

    return dataset_splits


def process_dataset(dataset_splits, args, tokenizer):
    """
    We increase the input_length, because instead of masking tokens T5 replaces
    masked spans with a single token, therefore to avoid padding we need to have
    longer sequences at the start, before masking
    """
    if args.mode == 'pt':
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():
            before_mask_input_length, target_length = compute_input_and_target_lengths(
                inputs_length=args.data.input_length,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
            )

            # 把 before_mask_input_length 和 target_length 两个新计算出来的参数加入到 args.data 中
            # {'input_length': 512, 'mlm_probability': 0.15, 'mean_noise_span_length': 3.0, 'num_workers': 8} --->
            # {'input_length': 512, 'mlm_probability': 0.15, 'mean_noise_span_length': 3.0, 'num_workers': 8, 'before_mask_input_length': 568, 'target_length': 114}
            with open_dict(args):
                args.data.before_mask_input_length = before_mask_input_length
                args.data.target_length = target_length

            dataset_split = dataset_split.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'in_length': before_mask_input_length,
                },
                remove_columns=['text'],
            )

            if isinstance(dataset_split, IterableDataset):
                dataset_split = dataset_split.shuffle(seed=args.seed, buffer_size=10_000)
            else:
                dataset_split = dataset_split.shuffle(seed=args.seed)
            final_datasets[split] = dataset_split
    elif args.mode == 'ft':
        final_datasets = dataset_splits
    else:
        raise NotImplementedError

    return final_datasets


def get_data_collator(tokenizer, config, args):
    if args.mode == 'pt':
        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=args.data.mlm_probability,
            mean_noise_span_length=args.data.mean_noise_span_length,
            input_length=args.data.input_length,
            target_length=args.data.target_length,
            pad_token_id=config.pad_token_id,
        )
    elif args.mode == 'ft':
        data_collator = DataCollatorForNI(
            tokenizer,
            padding="longest",
            max_source_length=args.data.max_seq_len,
            max_target_length=args.data.max_target_len,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
            add_task_name=args.data.add_task_name,
            add_task_definition=args.data.add_task_definition,
            num_pos_examples=args.data.num_pos_examples,
            num_neg_examples=args.data.num_neg_examples,
            add_explanation=args.data.add_explanation,
            tk_instruct=args.data.tk_instruct
        )
    else:
        raise NotImplementedError

    return data_collator


def get_dataloaders(tokenizer, config, args):
    dataset_splits = load_dataset_splits(args)
    dataset = process_dataset(dataset_splits=dataset_splits, args=args, tokenizer=tokenizer)
    data_collator = get_data_collator(tokenizer=tokenizer, config=config,
                                      args=args)

    is_iterable = isinstance(dataset['train'], IterableDataset)

    dataloaders = {}

    for split in ['train', 'test']:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        # shuffle = (split == 'train') and not is_iterable

        # if args.mode == 'ft' and split == 'train':
        #     assert shuffle is True
        # else:
        #     assert shuffle is False

        dataloaders[split] = DataLoader(
            dataset[split],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # Add & Check args about data loaders
    with open_dict(args):
        if not is_iterable:
            args.data.train_batches = len(dataloaders['train'])
            args.data.test_batches = len(dataloaders['test'])

        if args.optim.epochs > 0:
            assert not is_iterable
            args.optim.total_steps = (len(dataloaders['train']) // args.optim.grad_acc) * args.optim.epochs 

        args.eval.corrected_steps = args.eval.steps

    return dataloaders['train'], dataloaders['test']


def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.optim.name == 'adamw':
        from transformers import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adamwscale':
        from .copied_utils import AdamWScale
        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
            relative_step=False,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_lr_scheduler(optimizer, args, logger):
    if args.optim.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            CosineAnnealingLR,
        )

        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.optim.warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=args.optim.total_steps - args.optim.warmup_steps,
            eta_min=args.optim.final_cosine,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.optim.warmup_steps]
        )
    elif args.optim.lr_scheduler == 'legacy':
        import math
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            LambdaLR,
        )

        msg = "You are using T5 legacy LR Schedule, it's independent from the optim.base_lr"
        logger.log_message(msg)

        num_steps_optimizer1 = math.ceil(args.optim.total_steps * 0.9)
        iters_left_for_optimizer2 = args.optim.total_steps - num_steps_optimizer1

        scheduler1 = LambdaLR(
            optimizer,
            lambda step: min(
                1e-2, 1.0 / math.sqrt(step)
            ) / args.optim.base_lr if step else 1e-2 / args.optim.base_lr
        )

        scheduler2 = LinearLR(
            optimizer,
            start_factor=(
                min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / args.optim.base_lr
            ),
            end_factor=0,
            total_iters=iters_left_for_optimizer2,
            last_epoch=-1,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1]
        )
    elif args.optim.lr_scheduler == 'constant':
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=args.optim.lr_scheduler,
            optimizer=optimizer,
        )
    else:
        raise NotImplementedError

    return lr_scheduler
