# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/reward_modeling.py \
    --model_name_or_path=facebook/opt-350m \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=64 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=512 \
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, RewardConfig, RewardTrainer


tqdm.pandas()


@dataclass
class RewardScriptArguments:
    mode: str = field(default="train", metadata={"help": "the dataset name"})
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_eval_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    sanity_check: bool = field(default=False, metadata={"help": "only train on 1000 samples"})
    output_dataset_name: str = field(default=None, metadata={"help": "the dataset name"})


def get_peft_config(model_config: ModelConfig):
    if model_config.use_peft is False:
        return None

    target_modules = model_config.lora_target_modules if model_config.lora_target_modules is not None else "all-linear"

    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        task_type=model_config.lora_task_type,
        target_modules=target_modules,
        modules_to_save=model_config.lora_modules_to_save,
    )

    return peft_config


def tldr_preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(prompt + chosen)
        tokenized_rejected = tokenizer(prompt + rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


def tldr_relabel_dataset_fn(batch: Dict[str, List]):
    relabel_batch = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "pred_chosen": [],
        "pred_rejected": [],
    }
    for prompt, chosen, rejected, pred_chosen, pred_rejected in zip(
        batch["prompt"],
        batch["chosen"],
        batch["rejected"],
        batch["pred_chosen"],
        batch["pred_rejected"],
    ):
        relabel_batch["prompt"].append(prompt)
        if pred_chosen >= pred_rejected:
            relabel_batch["chosen"].append(chosen)
            relabel_batch["rejected"].append(rejected)
            relabel_batch["pred_chosen"].append(pred_chosen)
            relabel_batch["pred_rejected"].append(pred_rejected)
        else:
            relabel_batch["chosen"].append(rejected)
            relabel_batch["rejected"].append(chosen)
            relabel_batch["pred_chosen"].append(pred_rejected)
            relabel_batch["pred_rejected"].append(pred_chosen)

    return relabel_batch


def tldr_relabel_dataset(dataset, pred_chosen, pred_rejected):
    if "pred_chosen" in dataset.column_names:
        dataset = dataset.remove_columns(["pred_chosen"])
    if "pred_rejected" in dataset.column_names:
        dataset = dataset.remove_columns(["pred_rejected"])

    dataset = dataset.add_column("pred_chosen", pred_chosen)
    dataset = dataset.add_column("pred_rejected", pred_rejected)
    dataset = dataset.map(tldr_relabel_dataset_fn, batched=True, remove_columns=dataset.column_names)
    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser((RewardScriptArguments, RewardConfig, ModelConfig))
    script_args, reward_config, model_config = parser.parse_args_into_dataclasses()
    # reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer_name = (
        script_args.tokenizer_name if script_args.tokenizer_name is not None else model_config.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    )

    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    model.config.pad_token_id = tokenizer.pad_token_id

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(script_args.dataset_name)

    if script_args.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(100))

        reward_config.push_to_hub = False
        reward_config.report_to = ""

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    raw_datasets = raw_datasets.map(
        tldr_preprocess_function,
        batched=True,
    )
    raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
        and len(x["input_ids_rejected"]) <= reward_config.max_length
    )
    train_dataset = raw_datasets[script_args.dataset_train_split]
    eval_dataset = raw_datasets[script_args.dataset_eval_split] if script_args.dataset_eval_split else None

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )

    if script_args.mode == "train":
        trainer.train()
        trainer.save_model(reward_config.output_dir)
    elif script_args.mode == "eval":
        results = trainer.evaluate()
        print(results)
    elif script_args.mode == "relabel":
        relabel_dataset = DatasetDict()

        preds = trainer.predict(train_dataset).predictions
        relabel_dataset[script_args.dataset_train_split] = tldr_relabel_dataset(
            raw_datasets[script_args.dataset_train_split], preds[:, 0], preds[:, 1]
        )

        if script_args.dataset_eval_split:
            preds = trainer.predict(eval_dataset).predictions
            relabel_dataset[script_args.dataset_eval_split] = tldr_relabel_dataset(
                raw_datasets[script_args.dataset_eval_split], preds[:, 0], preds[:, 1]
            )

        if trainer.accelerator.is_local_main_process and not script_args.sanity_check:
            print("Pushing")
            relabel_dataset.push_to_hub(script_args.output_dataset_name)
    else:
        raise NotImplementedError(f"mode {script_args.mode} is not valid")
