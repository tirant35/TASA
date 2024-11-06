import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import os
import json
import torch
import transformers
import utils
import random
from pathlib import Path
from data_construct import update_data_from_options
from torch.utils.data import Dataset
from transformers import Trainer
from peft import PeftModel
import time
from utils import load_data_to_list, smart_tokenizer_and_embedding_resize, save_data_to_json
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as LORA_TARGET_MAP
from constants import (
    IGNORE_INDEX,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    SELECT_PROMPT_START,
    SELECT_PROMPT_END
)
from peft import (
    LoraConfig,
    get_peft_model,
)

from swanlab.integration.huggingface import SwanLabCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/data/yimin/models/base/meta-llama/Meta-Llama-3-8B")
    checkpoint_path: Optional[str] = field(default=None)
    peft_type: Optional[str] = field(default="lora")



@dataclass
class DataArguments:
    update_config_path: str = field(default=None)
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    distance: str = field(default="COS", metadata={"help": "The distance metric used to calculate the distance between two vectors."})
    n_clusters: int = field(default=500, metadata={"help": "The number of representative data extracted from each dataset."})
    encoder: str = field(default="embedding")
    embedding_path: Optional[str] = field(default=None, metadata={"help": "Path to the embedding model file."})



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    option: str = field(default="add")
    option_type: str = field(default="new")
    reduce_ratio: Optional[float] = field(default=0.5)
    cache_dir: Optional[str] = field(default="./cache")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_deepspeed: bool = field(default=False)
    use_swanlab: bool = field(default=False)
    use_instruction: bool = field(default=False)
    use_output: bool = field(default=False)
    
    
def get_target(model_type, named_modules) -> List[str]:
    target_modules = LORA_TARGET_MAP.get(model_type, [])
    if not target_modules:
        cls = torch.nn.Linear
        lora_module_names = {name.split('.')[-1] for name, module in named_modules if isinstance(module, cls)}
        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")
        return list(lora_module_names)
    return target_modules

def get_model_path(checkpoint_path: str):
    with open(Path(checkpoint_path) / 'adapter_config.json', 'r') as f:
        adapter_config = json.load(f)
        return adapter_config['base_model_name_or_path']


def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments) -> tuple:
    if model_args.checkpoint_path is not None:
        model_path = get_model_path(model_args.checkpoint_path)
    
    model_kwargs = {
        "cache_dir": training_args.cache_dir,
        "torch_dtype": 'auto',
        "trust_remote_code": True
    }
    if not training_args.use_deepspeed:
        model_kwargs["device_map"] = "auto"
    else:
        logger.warning("Using DeepSpeed")

    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    lora_model = PeftModel.from_pretrained(model, model_args.checkpoint_path, is_trainable=True)

    lora_model.print_trainable_parameters()

    # model.is_parallelizable = True
    # model.model_parallel = True
    # torch.cuda.empty_cache()
    if torch.cuda.device_count() > 1:
        lora_model.is_parallelizable = True
        lora_model.model_parallel = True

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    return lora_model, tokenizer


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class MixedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                checkpoint_path: str,
                update_config: dict,
                option: str,
                option_type: str,
                tokenizer: transformers.PreTrainedTokenizer, 
                save_path: str,
                **kwargs
                ):
        super(MixedDataset, self).__init__()
        logger.warning("Loading data...")

        list_data_dict, prompt = update_data_from_options(option, option_type, checkpoint_path, update_config, **kwargs)
        random.shuffle(list_data_dict)
        save_data_to_json(list_data_dict, save_path)
        
        logger.warning(f"Loaded {len(list_data_dict)} examples...")
        
        logger.warning("Formatting inputs...")
        
        sources = [prompt.format_map(example) for example in list_data_dict]
        # targets = [f"[\"domain\":{example['domain']}\n\"task\":{example['task']}\n\"skill\":{example['skill']}\n]{tokenizer.eos_token}" for example in list_data_dict]
        targets = [f"{example['domain']}\n{example['task']}{tokenizer.eos_token}" for example in list_data_dict]

        logger.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        
        self.prompt = prompt

    def __len__(self):
        return len(self.input_ids)
    
    def get_prompt(self):
        return self.prompt

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    def get_domain_task_skill(self):
        return self.domain_list, self.task_list, self.skill_list


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
    model_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    param = {}
    with open(data_args.update_config_path, 'r') as f:
        config = json.load(f)
    if training_args.option == 'balance':
        if training_args.option_type == 'mix':
            param['reduce'] = training_args.reduce_ratio
    if training_args.option == 'add':
        param['model_path'] = get_model_path(model_args.checkpoint_path)
        param['embedding_path'] = data_args.embedding_path
        param['n_clusters'] = data_args.n_clusters
        param['distance'] = data_args.distance
        if training_args.option_type == 'mix':
            param['reduce'] = training_args.reduce_ratio
    if training_args.option == 'delete':
        param['reduce'] = training_args.reduce_ratio
    train_dataset = MixedDataset(
        tokenizer=tokenizer,
        checkpoint_path=model_args.checkpoint_path,
        update_config=config,
        option=training_args.option,
        option_type=training_args.option_type,
        save_path=training_args.output_dir,
        **param)
    prompt = train_dataset.get_prompt()
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator), prompt

    

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)
    
 
    
    data_module, prompt = make_supervised_data_module(
        tokenizer=tokenizer, 
        data_args=data_args, 
        training_args=training_args, 
        model_args=model_args)
    
    train_prompt = test_prompt = prompt
    
    if training_args.use_swanlab:
        swanlab_callback = SwanLabCallback(project="train_selector", logdir='./logs', mode="local")
    logger.warning("Creating trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        callbacks=[swanlab_callback] if training_args.use_swanlab else None,
        args=training_args,
        **data_module)
    # Training
    if training_args.do_train:
        logger.info("Training...")
        train_result = trainer.train()
        trainer.save_model(training_args.output_dir)
        logger.info("Save selector successfully")
        metrics = train_result.metrics

        metrics["train_prompt"] = train_prompt
        metrics["test_prompt"] = test_prompt

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    train()
