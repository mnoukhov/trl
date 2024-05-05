import gc
import os
import random
import math
from dataclasses import dataclass, field
from typing import List, Optional

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, builder, load_dataset
from huggingface_hub import list_repo_refs
from peft import PeftModelForCausalLM, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments, AutoModel
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

import wandb

from handbook_data import apply_chat_template, setup_chat_format_simple

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
    eval_batch_size: Optional[int] = field(default=4)
    max_length: Optional[int] = field(default=512)
    gold_tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    flash_attention: Optional[bool] = field(default=False)


def prepare_ultrafeedback_dataset(args, dataset, tokenizer, num_proc=2):
    original_columns = dataset.column_names

    def preprocess_func(examples):
        return_batch = {"prompt": [], "chosen": [], "rejected": [], "raw_prompt": [], "query_reference_response": []}
        for i in range(len(examples["prompt"])):

            prompt_message = examples["chosen"][i][0]
            return_batch["prompt"].append(f"{prompt_message['role']}\n{prompt_message['content']}\nassistant\n")
            chosen_messages = examples["chosen"][i][-1:]
            rejected_messages = examples["rejected"][i][-1:]
            return_batch["chosen"].append(tokenizer.apply_chat_template(chosen_messages, tokenize=False))
            return_batch["rejected"].append(tokenizer.apply_chat_template(rejected_messages, tokenize=False))
            return_batch["query_reference_response"].append(
                f"[INST] {prompt_message['content']} [\INST] " + chosen_messages[0]["content"]
            )
            return_batch["raw_prompt"].append(prompt_message["content"])
        return return_batch

    dataset = dataset.map(preprocess_func, batched=True, num_proc=num_proc, remove_columns=original_columns)
    for index in random.sample(range(len(dataset)), 3):
        print(f"Sample {index} of the processed dataset:\n\n{dataset[index]}")

    return dataset


def generate(script_args):
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        # tokenizer.pad_token = tokenizer.eos_token
        print("\n\nNo pad token found in tokenizer, setting it to <|padding|>")
        tokenizer.pad_token = "<|padding|>"
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        print(f"Pad token found in tokenizer: {tokenizer.pad_token}")

    tokenizer.padding_side = "left"

    _, tokenizer = setup_chat_format_simple(None, tokenizer)
    dataset = load_dataset(script_args.dataset_name, split=script_args.split)

    # dataset = prepare_ultrafeedback_dataset(script_args, dataset, tokenizer) #uncomment to not use ztemplate
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "rm_eval",
            "auto_insert_empty_system_msg": False,
        },
        desc="Applying chat template on Gen Eval",
    )

    # print train samples
    for index in random.sample(range(len(dataset)), 3):
        print(f"Sample {index} of the processed dataset:")
        for key in dataset.column_names:
            print(f"{key}: {dataset[index][key]}")

    prompts = dataset["prompt"]

    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        top_p=script_args.top_p,
        n=1,
        include_stop_str_in_output=True,
        skip_special_tokens=True,
    )

    gens = {}
    dtype = torch.bfloat16 if script_args.gen_dtype == "bf16" else torch.float32
    if os.path.exists(os.path.join(script_args.model_name, "adapter_config.json")):
        model = AutoPeftModelForCausalLM.from_pretrained(script_args.model_name)
        merged = model.merge_and_unload()
        model_save_path = f"{script_args.output_dir}/merged_model"
        merged.save_pretrained(model_save_path)
        del model
        del merged
        model_name = model_save_path
    else:
        model_name = script_args.model_name

    llm = LLM(
        model=model_name,
        tokenizer=script_args.tokenizer_name,
        dtype=dtype,
        max_model_len=script_args.seq_length,
        tensor_parallel_size=script_args.num_gpus,
        trust_remote_code=True,
    )

    llm.set_tokenizer(tokenizer)
    revision_name = "default"
    generations = llm.generate(prompts, sampling_params)

    raw_prompts = dataset["raw_prompt"]
    rm_formatted_text = [
        f"[INST] {prompt} [\INST] {txt.outputs[0].text}" for prompt, txt in zip(raw_prompts, generations)
    ]
    # texts = [output.prompt + output.outputs[0].text for output in generations]
    print("prompt + gen:")
    for i in range(3):
        print("=========================== " + str(i) + " ============================")
        print(f"PROMPT: {raw_prompts[i]}\n")
        print("---------------------------\n")
        print(f"GEN: {rm_formatted_text[i]}\n")
        print("---------------------------\n")
        print(f"REF: {dataset['query_reference_response'][i]}\n")

    gens[revision_name] = rm_formatted_text

    dataset = dataset.add_column(f"generations_{revision_name}", rm_formatted_text)

    # delete old model
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()

    if script_args.output_dir is not None:
        dataset_path = os.path.join(
            script_args.output_dir,
            script_args.dataset_name.replace("/", "_"),
            script_args.model_name.replace("/", "_"),
        )
        os.makedirs(dataset_path, exist_ok=True)
        dataset.save_to_disk(dataset_path)
        with open(f"{dataset_path}_sampling_params.txt", "w") as f:
            print(sampling_params, file=f)

    reference = dataset["query_reference_response"]

    return reference, gens


def evaluate(args, reference, generations):

    tokenizer = AutoTokenizer.from_pretrained(args.gold_tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.max_length = args.max_length
    tokenizer.model_max_length = args.max_length

    model = AutoModel.from_pretrained(args.gold_model_name, trust_remote_code=True)
    model.to("cuda")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = args.max_length

    dataset = {"chosen": [], "rejected": []}
    for gen, ref in zip(generations["default"], reference):
        dataset["chosen"].append(gen)
        dataset["rejected"].append(ref)
    dataset = Dataset.from_dict(dataset)
    print(f"len dataset: {len(dataset)}")

    gen_rewards = []

    num_batches = math.ceil(len(dataset) / args.eval_batch_size)

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), "evaluating"):
            start_idx = batch_idx * args.eval_batch_size
            end_idx = min((batch_idx + 1) * args.eval_batch_size, len(dataset))
            batch = dataset[start_idx:end_idx]
            inputs = tokenizer(batch["chosen"], return_tensors="pt", padding=True, truncation=True).to("cuda")
            chosen_reward = model(**inputs).cpu().detach().numpy()
            gen_rewards.extend(chosen_reward)

        gen_rewards = torch.tensor(gen_rewards)
        assert gen_rewards.shape[0] == len(dataset)

    avg_gen_reward = gen_rewards.mean().item()
    print(f"avg_gen_reward: {avg_gen_reward}")


def main_args_dict(args_dict):
    parser = HfArgumentParser([GenerateScriptArguments, EvalScriptArguments])
    generate_args, eval_args = parser.parse_dict(args_dict)
    if eval_args.gold_tokenizer_name is None:
        eval_args.gold_tokenizer_name = generate_args.tokenizer_name

    print("GENERATING")
    reference, generations = generate(generate_args)

    print("EVALUATING")
    evaluate(eval_args, reference, generations, generate_args.model_name)


if __name__ == "__main__":
    parser = HfArgumentParser([GenerateScriptArguments, EvalScriptArguments])
    generate_args, eval_args = parser.parse_args_into_dataclasses()
    if eval_args.gold_tokenizer_name is None:
        eval_args.gold_tokenizer_name = generate_args.tokenizer_name

    print("GENERATING")
    reference, generations = generate(generate_args)

    print("EVALUATING")
    evaluate(eval_args, reference, generations)
