import gc
import os
import random
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, builder, load_dataset
from huggingface_hub import list_repo_refs
from peft import PeftModelForCausalLM, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments, AutoModel
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

import wandb

builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class GenerateScriptArguments:
    output_dir: Optional[str] = field(
        default="/home/toolkit/trl_results",
        metadata={"help": "output folder"},
    )
    num_gpus: Optional[int] = field(default=1)
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    base_model_revision: Optional[str] = field(default=None)
    model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    model_revisions: Optional[List[str]] = field(default_factory=list)
    # base_model_revision: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
    )
    split: Optional[str] = field(default="validation", metadata={"help": "the dataset name"})
    batch_size: Optional[int] = field(default=4)
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})

    temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(default=48, metadata={"help": "max new tokens"})
    gen_dtype: Optional[str] = field(default="auto")


@dataclass
class EvalScriptArguments:
    wandb_log_id: Optional[str] = field(default=None)
    gold_model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    gold_model_revision: Optional[str] = field(default=None)
    eval_dtype: Optional[str] = field(default="auto")
    eval_batch_size: Optional[int] = field(default=16)
    max_length: Optional[int] = field(default=512)
    gold_tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    flash_attention: Optional[bool] = field(default=False)


def prepare_ultrafeedback_dataset(args, dataset, tokenizer, num_proc=2):
    original_columns = dataset.column_names

    def preprocess_func(examples):
        return_batch = {"prompt": [], "chosen": [], "rejected": [], "query_reference_response": []}
        for i in range(len(examples["prompt"])):

            prompt_message = examples["chosen"][i][:-1]
            chosen_messages = examples["chosen"][i][-1:]
            rejected_messages = examples["rejected"][i][-1:]
            return_batch["chosen"].append(tokenizer.apply_chat_template(chosen_messages, tokenize=False))
            return_batch["rejected"].append(tokenizer.apply_chat_template(rejected_messages, tokenize=False))
            return_batch["prompt"].append(tokenizer.apply_chat_template(prompt_message, tokenize=False))
            return_batch["query_reference_response"].append(return_batch["prompt"][-1] + return_batch["chosen"][-1])
        return return_batch

    dataset = dataset.map(preprocess_func, batched=True, num_proc=num_proc, remove_columns=original_columns)
    for index in random.sample(range(len(dataset)), 3):
        print(f"Sample {index} of the processed dataset:\n\n{dataset[index]}")

    return dataset


def generate(script_args):
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"

    dataset = load_dataset(script_args.dataset_name, split=script_args.split)
    dataset = prepare_ultrafeedback_dataset(script_args, dataset, tokenizer)
    prompts = dataset["prompt"]

    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        top_p=script_args.top_p,
        n=1,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )

    gens = {}

    model = AutoPeftModelForCausalLM.from_pretrained(script_args.model_name)
    merged = model.merge_and_unload()
    model_save_path = f"{script_args.output_dir}/merged_model"
    merged.save_pretrained(model_save_path)
    del model
    del merged
    model_name = model_save_path

    llm = LLM(
        model=model_name,
        tokenizer=script_args.tokenizer_name,
        dtype=script_args.gen_dtype,
        max_model_len=script_args.seq_length,
        tensor_parallel_size=script_args.num_gpus,
        trust_remote_code=True,
    )

    llm.set_tokenizer(tokenizer)
    revision_name = "default"
    generations = llm.generate(prompts, sampling_params)

    texts = [output.prompt + output.outputs[0].text for output in generations]
    print("prompt + gen:")
    for i in range(3):
        print(f"text {i}: {texts[i]}")
    gens[revision_name] = texts

    dataset = dataset.add_column(f"generations_{revision_name}", texts)

    # delete old model
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()

    if script_args.output_dir is not None:
        # TODO add hash to dataset path
        # sampling_str = str(sampling_params)
        # sampling_hash = hashlib.sha256(sampling_str.encode()).hexdigest()[:10]
        dataset_path = os.path.join(
            script_args.output_dir,
            script_args.dataset_name.replace("/", "_"),
            script_args.model_name.replace("/", "_"),
        )
        os.makedirs(dataset_path, exist_ok=True)
        dataset.save_to_disk(dataset_path)
        with open(f"{dataset_path}_sampling_params.txt", "w") as f:
            print(sampling_params, file=f)

    print(f"generated {len(gens)} steps")
    reference = dataset["query_reference_response"]

    return reference, gens


def evaluate(args, reference, generations, model_name=None):
    if args.wandb_log_id is not None:
        # don't overwrite the wandb name of the original run
        if args.wandb_log_id == "model_name":
            # model name = config_wandblogid
            wandb_log_id = model_name.split("_")[-1]
        else:
            wandb_log_id = args.wandb_log_id

        os.environ.pop("WANDB_NAME")
        # original_name = wandb_name.removeprefix("geneval_")
        wandb.init(id=wandb_log_id, resume="allow")
        log_to_wandb = True
        print(f"Logging to WandB {wandb_log_id}")
    else:
        log_to_wandb = False

    torch_dtype = args.eval_dtype if args.eval_dtype in ["auto", None] else getattr(torch, args.eval_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.gold_tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.max_length = args.max_length

    model = AutoModel.from_pretrained(args.gold_model_name, trust_remote_code=True)
    model.to("cuda")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = args.max_length

    dataset = {"chosen": [], "rejected": []}
    print(reference)
    for gen, ref in zip(generations["default"], reference):
        dataset["chosen"].append(gen)
        dataset["rejected"].append(ref)
    dataset = Dataset.from_dict(dataset)
    print(f"len dataset: {len(dataset)}")

    gen_rewards = []
    ref_rewards = []

    num_batches = math.ceil(len(dataset) / args.eval_batch_size)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.eval_batch_size
            end_idx = min((batch_idx + 1) * args.eval_batch_size, len(dataset))
            batch = dataset[start_idx:end_idx]
            inputs = tokenizer(batch["chosen"], return_tensors="pt", padding=True, truncation=True).to("cuda")

            chosen_reward = model(**inputs).item()
            inputs = tokenizer(batch["rejected"], return_tensors="pt", padding=True, truncation=True).to("cuda")
            rejected_reward = model(**inputs).item()
            gen_rewards.extend(chosen_reward)
            ref_rewards.extend(rejected_reward)

        print(f"some gen rewards: {gen_rewards[:3]}")
        print(f"some ref rewards: {ref_rewards[:3]}")
        print(f"len gen rewards: {len(gen_rewards)}")
        gen_rewards = torch.tensor(gen_rewards)
        ref_rewards = torch.tensor(ref_rewards)

        win_rate = torch.tensor((gen_rewards > ref_rewards), dtype=gen_rewards.dtype).mean().item()
        norm_reward = torch.tensor((gen_rewards - ref_rewards), dtype=gen_rewards.dtype).mean().item()

    print(f"win_rate: {win_rate}")
    print(f"norm_reward: {norm_reward}")
    step = 0  # TODO change this
    if log_to_wandb:
        wandb.log(
            {
                "gold/win_rate": win_rate,
                "gold/norm_reward": norm_reward,
                "train/global_step": step,
            }
        )


def main_args_dict(args_dict):
    parser = HfArgumentParser([GenerateScriptArguments, EvalScriptArguments])
    generate_args, eval_args = parser.parse_dict(args_dict)
    if eval_args.gold_tokenizer_name is None:
        eval_args.gold_tokenizer_name = generate_args.tokenizer_name

    print("GENERATING")
    reference, generations = generate(generate_args)
    # dataset = load_dataset(generate_args.dataset_name, split=generate_args.split)
    # generations = {"step0": dataset["query_reference_response"]}
    # reference = dataset["query_reference_response"]
    print("EVALUATING")
    evaluate(eval_args, reference, generations, generate_args.model_name)


if __name__ == "__main__":
    parser = HfArgumentParser([GenerateScriptArguments, EvalScriptArguments])
    generate_args, eval_args = parser.parse_args_into_dataclasses()
    if eval_args.gold_tokenizer_name is None:
        eval_args.gold_tokenizer_name = generate_args.tokenizer_name

    print("GENERATING")
    reference, generations = generate(generate_args)
    # dataset = load_dataset(generate_args.dataset_name, split=generate_args.train_split)
    # generations = {"step0": dataset["query_reference_response"]}
    # reference = dataset["query_reference_response"]
    print("EVALUATING")
    evaluate(eval_args, reference, generations)
