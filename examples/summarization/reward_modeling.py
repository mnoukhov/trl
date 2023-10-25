# coding=utf-8
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
import os
from dataclasses import dataclass, field
from typing import Optional

import bitsandbytes as bnb
import torch
from accelerate import Accelerator
from datasets import builder, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from trl import RewardTrainer


tqdm.pandas()
builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True
# torch.autograd.set_detect_anomaly(True)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with RewardTrainer
    """

    model_name: Optional[str] = field(
        default="/home/toolkit/huggingface/tldr_sft_pythia7b", metadata={"help": "the model name"}
    )
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="CarperAI/openai_summarize_comparisons", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="prompt", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of update steps between two logs"})
    train_split: Optional[str] = field(
        default="train", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    eval_split: Optional[str] = field(
        default="test[:5000]", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    weight_decay: Optional[float] = field(default=0.001)
    num_warmup_steps: Optional[int] = field(default=100)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    seq_length: Optional[int] = field(default=550, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_all_linear: Optional[bool] = field(default=False, metadata={"help": "lora adapter on all linear layers"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="results", metadata={"help": "the output directory"})
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )

    just_eval: Optional[bool] = field(default=False)
    carper_ckpt: Optional[str] = field(default=None)


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.load_in_4bit else (bnb.nn.Linear8bitLt if args.load_in_8bit else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    if "score" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("score")

    return list(lora_module_names)


def create_and_prepare_model(args):
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
        device_map = {"": Accelerator().local_process_index}
    else:
        device_map = None
        quantization_config = None

    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        num_labels=1,
        torch_dtype=dtype,
    )

    model.config.torch_dtype = dtype

    if args.carper_ckpt is not None:
        state_dict = torch.load(args.carper_ckpt)
        state_dict["score.weight"] = state_dict.pop("v_head.weight")
        # the state dict contains some old unused params, so strict=False
        model.load_state_dict(state_dict, strict=False)

    model.config.use_cache = not args.gradient_checkpointing
    # if script_args.ignore_bias_buffers:
    # torch distributed hack
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        args.gradient_checkpointing = False

    # we add `score` to the list of modules to save to
    # correctly save the score head.
    # set target modules to be query_key_value for Pythia
    if args.lora_all_linear:
        modules = find_all_linear_names(args, model)
    else:
        modules = None

    if args.use_peft:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=modules,
            modules_to_save=["score"],
        )

        model = get_peft_model(model, peft_config)

        modules_to_save = ["score"]
        for key, _ in model.named_modules():
            target_module_found = any(key.endswith(target_key) for target_key in modules_to_save)
            if target_module_found:
                model.get_submodule(key + ".original_module").requires_grad_(False)

    if args.bf16:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "score" in name or "embed_tokens" in name:
                if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    tokenizer_name = script_args.model_name if script_args.tokenizer_name is None else script_args.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.truncation_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def create_and_prepare_dataset(args, tokenizer, split, num_proc=2):
    dataset = load_dataset(args.dataset_name, split=split)

    def summary_filter(example):
        return (example["chosen"] != example["rejected"]) and (
            len(example["chosen"].split()) >= 5 or len(example["rejected"].split()) >= 5
        )

    pre_filter = len(dataset)
    dataset = dataset.filter(summary_filter)
    print(f"filtered {pre_filter - len(dataset)} samples from {split}")

    original_columns = dataset.column_names

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(
                prompt + "\n" + chosen, padding="max_length", truncation=True, max_length=script_args.seq_length
            )
            tokenized_rejected = tokenizer(
                prompt + "\n" + rejected, padding="max_length", truncation=True, max_length=script_args.seq_length
            )

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    dataset = dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)

    return dataset


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model, tokenizer = create_and_prepare_model(script_args)
train_dataset = create_and_prepare_dataset(script_args, tokenizer, script_args.train_split)
eval_dataset = create_and_prepare_dataset(script_args, tokenizer, script_args.eval_split)

# don't include gradient_checkpointing here, see trl#728
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    bf16=script_args.bf16,
    num_train_epochs=script_args.num_train_epochs,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    report_to=script_args.log_with,
    remove_unused_columns=False,
    lr_scheduler_type=script_args.lr_scheduler_type,
    weight_decay=script_args.weight_decay,
    optim=script_args.optimizer_type,
    warmup_steps=script_args.num_warmup_steps,
    logging_steps=script_args.logging_steps,
    evaluation_strategy="epoch" if script_args.eval_split != "none" else "no",
    gradient_checkpointing=script_args.gradient_checkpointing,
)

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_length=script_args.seq_length,
)

if script_args.just_eval:
    print("Evaluating")
    results = trainer.evaluate()
    print(results)
else:
    last_checkpoint = get_last_checkpoint(script_args.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.evaluate()

    print("Saving last checkpoint of the model")
    trainer.save_model(script_args.output_dir)

    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)