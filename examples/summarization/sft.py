from dataclasses import dataclass, field

import torch
from accelerate import PartialState
from datasets import load_dataset
from tqdm.rich import tqdm
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import ModelConfig, SFTTrainer
from trl.trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config


tqdm.pandas()


def hh_combine(examples):
    if isinstance(examples["chosen"], str):
        return examples["prompt"] + examples["chosen"]
    elif isinstance(examples["chosen"], list):
        return list(map(str.__add__, examples["prompt"], examples["chosen"]))
    else:
        raise Exception(f"weird input examples of type {type(examples)}")


@dataclass
class ScriptArguments:
    task_type: str = field(default="hh")
    dataset_name: str = field(default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"})
    dataset_train_name: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_test_name: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    output_model_name: str = field(default="", metadata={"help": "model name to upload"})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    packing: bool = field(default=False, metadata={"help": "Whether to apply data packing or not during training"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False, metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    ################
    # Dataset
    ################
    train_dataset = load_dataset(args.dataset_name, split=args.dataset_train_name)
    eval_dataset = load_dataset(args.dataset_name, split=args.dataset_test_name)

    # train_dataset = train_dataset.map(lambda ex: {"text": ex['prompt'] + ex['chosen']})
    # eval_dataset = eval_dataset.map(lambda ex: {"text": ex['prompt'] + ex['chosen']})

    if args.task_type == "tldr":
        formatting_func = None
        dataset_text_field = "query_reference_response"
    elif args.task_type == "hh":
        formatting_func = hh_combine
        dataset_text_field = None

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=args.packing,
        formatting_func=formatting_func,
        dataset_text_field=dataset_text_field,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)

    if PartialState().is_main_process:
        model = trainer.model.merge_and_unload()
        model.push_to_hub(args.output_model_name)
        tokenizer.push_to_hub(args.output_model_name)
