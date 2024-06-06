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
from typing import Dict, List, Optional, Union

import torch
from datasets import DatasetDict, builder, load_dataset
from peft import LoraConfig
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from trl import ModelConfig, RewardTrainer


tqdm.pandas()
builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


# torch.autograd.set_detect_anomaly(True)
@dataclass
class RewardScriptArguments:
    mode: str = field(default="train", metadata={"help": "the dataset name"})
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_eval_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    sanity_check: bool = field(default=False, metadata={"help": "only train on 1000 samples"})
    output_dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    max_length: int = field(default=512)


### fix from https://github.com/huggingface/trl/issues/274


class GPTRewardTrainer(RewardTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_chosen = rewards[jidx]
        rewards_rejected = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss


@dataclass
class GPTRewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        # features_chosen = []
        # features_rejected = []
        merged_features = []
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch


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


if __name__ == "__main__":
    parser = HfArgumentParser((RewardScriptArguments, TrainingArguments, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    )

    model_name = model_config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_datasets = load_dataset(script_args.dataset_name)
    if script_args.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(100))
    raw_datasets = raw_datasets.map(
        tldr_preprocess_function,
        batched=True,
    )

    train_dataset = raw_datasets[script_args.dataset_train_split] if script_args.mode != "eval" else None
    eval_dataset = raw_datasets[script_args.dataset_eval_split] if script_args.dataset_eval_split else None

    data_collator = GPTRewardDataCollatorWithPadding(tokenizer, max_length=script_args.max_length)

    trainer = GPTRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_length=script_args.max_length,
        data_collator=data_collator,
        peft_config=get_peft_config(model_config),
    )

    if script_args.mode == "train":
        print("Training")
        trainer.train()
        trainer.evaluate()

        print("Saving last checkpoint of the model")
        trainer.save_model(script_args.output_dir)

        output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
        trainer.model.save_pretrained(output_dir)
    elif script_args.mode == "eval":
        print("Evaluating")
        # results = trainer.evaluate()
        results = trainer.evaluate()
        print(results)
    elif script_args.mode == "relabel":

        def relabel_with_preds(batch: Dict[str, List]):
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

        relabel_dataset = DatasetDict()
        for split, pred_dataset in [("train", train_dataset), ("test", eval_dataset)]:
            if pred_dataset is None:
                continue
            trainer.accelerator.print(f"Prediction {split}")
            preds, _, metrics = trainer.predict(pred_dataset)
            trainer.accelerator.print(f"metrics {metrics}")

            if trainer.accelerator.is_local_main_process:
                print("Relabelling Dataset and Saving")
                ds_split = script_args.train_split if split == "train" else script_args.eval_split
                dataset = load_dataset(script_args.dataset_name, split=ds_split)
                dataset = dataset.add_column("pred_chosen", preds[:, 0])
                dataset = dataset.add_column("pred_rejected", preds[:, 1])

                dataset = dataset.map(relabel_with_preds, batched=True)

                dataset._info.description = f"{script_args.dataset_name} relabelled with {model_name}"
                relabel_dataset[split] = dataset

        if trainer.accelerator.is_local_main_process:
            print("Saving")
            relabel_dataset.save_to_disk(script_args.output_dir)
            print("Pushing")
            relabel_dataset.push_to_hub(os.path.basename(script_args.output_dir))
    elif script_args.mode == "predict":
        relabel_dataset = DatasetDict()
        for split, pred_dataset in [("train", train_dataset), ("test", eval_dataset)]:
            if pred_dataset is None:
                continue
            trainer.accelerator.print(f"Prediction {split}")
            preds, _, metrics = trainer.predict(pred_dataset)
            trainer.accelerator.print(f"metrics {metrics}")

            if trainer.accelerator.is_local_main_process:
                print("Relabelling Dataset and Saving")
                ds_split = script_args.train_split if split == "train" else script_args.eval_split
                dataset = load_dataset(script_args.dataset_name, split=ds_split)
                model_basename = model_name.rsplit("/", 1)[-1]
                dataset = dataset.add_column(f"pred_chosen_{model_basename}", preds[:, 0])
                dataset = dataset.add_column(f"pred_rejected_{model_basename}", preds[:, 1])

                dataset._info.description = f"{script_args.dataset_name} relabelled with {model_name}"
                relabel_dataset[split] = dataset

        if trainer.accelerator.is_local_main_process:
            print("Saving")
            relabel_dataset.save_to_disk(script_args.output_dir)
            print("Pushing")
            relabel_dataset.push_to_hub(os.path.basename(script_args.output_dir))
    else:
        raise Exception(f"incorrect mode {script_args.mode}")
        # TODO this freezes for some reason
        # for split, dataset in relabel_dataset.items():
        #     if trainer.accelerator.is_local_main_process:
        #         eval_dataset = prepare_dataset(script_args, dataset, tokenizer)
        #     trainer.accelerator.print(f"Re-evaluating relabel {split} dataset of size {len(dataset)}")
        #     trainer.accelerator.wait_for_everyone()
        #     results = trainer.evaluate(eval_dataset)
        #     trainer.accelerator.print(results)
