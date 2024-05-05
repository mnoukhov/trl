import os
from dataclasses import dataclass, field
from typing import Optional
import random

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset
from peft import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams

from handbook_data import setup_chat_format_simple, apply_chat_template


@dataclass
class ScriptArguments:
    output_dir: Optional[str] = field(
        default="compare_results",
        metadata={"help": "output folder"},
    )
    model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    merged_model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
    )
    train_split: Optional[str] = field(default="train[:20]", metadata={"help": "the dataset name"})
    eval_split: Optional[str] = field(default="test[:20]", metadata={"help": "the dataset name"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    better_transformer: Optional[bool] = field(default=False)
    flash_attention: Optional[bool] = field(default=False)
    batch_size: Optional[int] = field(default=4)
    bf16: Optional[bool] = field(default=False)
    fp16: Optional[bool] = field(default=False)
    fp16_model: Optional[bool] = field(default=False)
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})

    temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    do_sample: Optional[bool] = field(default=True)
    max_new_tokens: Optional[int] = field(default=48, metadata={"help": "max new tokens"})
    num_gen: Optional[int] = field(default=8, metadata={"help": "max new tokens"})

    cache_dir: Optional[str] = field(default=None, metadata={"help": "cache dir"})


def prepare_vllm_model(script_args, model_name, tokenizer):
    if script_args.bf16:
        torch_dtype = torch.bfloat16
    elif script_args.fp16_model:
        torch_dtype = torch.float16
    else:
        torch_dtype = "auto"

    llm = LLM(
        model=model_name,
        dtype=torch_dtype,
        tokenizer=script_args.tokenizer_name,
        max_model_len=script_args.seq_length,
    )
    llm.set_tokenizer(tokenizer)
    print(llm)
    print(tokenizer)
    print(len(tokenizer))

    return llm


def prepare_ultrafeedback_dataset(args, dataset, tokenizer, num_proc=2):
    original_columns = dataset.column_names

    def preprocess_func(examples):
        return_batch = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }
        for i in range(len(examples["prompt"])):

            prompt_message = examples["chosen"][i][0]
            return_batch["prompt"].append(f"{prompt_message['role']}\n{prompt_message['content']}\nassistant\n")
            chosen_messages = examples["chosen"][i][-1:]
            rejected_messages = examples["rejected"][i][-1:]
            return_batch["chosen"].append(tokenizer.apply_chat_template(chosen_messages, tokenize=False))
            return_batch["rejected"].append(tokenizer.apply_chat_template(rejected_messages, tokenize=False))
            # return_batch["prompt"].append(tokenizer.apply_chat_template(prompt_message, tokenize=False))
        return return_batch

    dataset = dataset.map(preprocess_func, batched=True, num_proc=num_proc, remove_columns=original_columns)
    for index in random.sample(range(len(dataset)), 3):
        print(f"Sample {index} of the processed dataset:\n\n{dataset[index]}")

    return dataset


def create_and_prepare_model(args, generation=False):
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
        device_map = {"": Accelerator().local_process_index}
    else:
        device_map = None
        quantization_config = None

    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16_model:
        torch_dtype = torch.float16
    else:
        torch_dtype = None

    if os.path.exists(os.path.join(args.merged_model_name, "adapter_config.json")):
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.merged_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            cache_dir=args.cache_dir,
        )
        merged = model.merge_and_unload()
        model_save_path = os.path.join(args.merged_model_name, "merged_model")
        merged.save_pretrained(model_save_path)
        del model
        del merged
        model_name = model_save_path
    else:
        model_name = args.merged_model_name

    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     args.model_name,
    #     quantization_config=quantization_config,
    #     device_map=device_map,
    #     torch_dtype=torch_dtype,
    #     cache_dir=args.cache_dir,
    # )

    # if args.better_transformer:
    #     model.to_bettertransformer()

    if script_args.tokenizer_name is not None:
        tokenizer_name = script_args.tokenizer_name
    else:
        tokenizer_name = script_args.model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        # tokenizer.pad_token = tokenizer.eos_token
        print("\n\nNo pad token found in tokenizer, setting it to <|padding|>")
        tokenizer.pad_token = "<|padding|>"
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        print(f"Pad token found in tokenizer: {tokenizer.pad_token}")

    # if getattr(tokenizer, "pad_token", None) is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # if getattr(model.config, "pad_token_id", None) is None:
    #     model.config.pad_token_id = model.config.eos_token_id

    tokenizer.padding_side = "left"
    return model_name, tokenizer


def generate_from_prompt(prompt_ids, prompts_attention_mask, model, args):
    generation_config = GenerationConfig(
        temperature=args.temperature,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_gen,
        pad_token_id=model.config.pad_token_id,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompts_attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
        )
        sequences = accelerator.gather(generation_output.sequences)

    return sequences


def generate_with_llm(llm, prompt, script_args):
    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        n=script_args.num_gen,
        top_p=0.95,
    )

    outputs = llm.generate(prompts, sampling_params)

    return outputs


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

accelerator = Accelerator()

splits_names = {
    "train": script_args.train_split,
}
relabel_dataset = DatasetDict()

model_name, tokenizer = create_and_prepare_model(script_args, generation=True)
if "ultra" in script_args.dataset_name:
    _, tokenizer = setup_chat_format_simple(None, tokenizer)
llm = prepare_vllm_model(script_args, model_name, tokenizer)


for split, split_name in splits_names.items():
    dataset = load_dataset(script_args.dataset_name, split=split_name)
    if "ultra" in script_args.dataset_name:
        dataset = dataset.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "dpo_gen",
                "auto_insert_empty_system_msg": False,
            },
            desc="Applying chat template",
        )
        # dataset = prepare_ultrafeedback_dataset(script_args, dataset, tokenizer, num_proc=2)

    # print train samples
    for index in random.sample(range(len(dataset)), 3):
        print(f"Sample {index} of the processed dataset:")
        for key in dataset.column_names:
            print(f"{key}: {dataset[index][key]}")

    prompts = dataset["prompt"]
    raw_prompts = dataset["raw_prompt"]
    print(f"Split: {split}, Number of prompts: {len(prompts)}")
    print(prompts[:3])
    print(f"raw prompt: {raw_prompts[:3]}")

    generations = generate_with_llm(llm, prompts, script_args)

    del llm
    torch.cuda.empty_cache()

    dataloader = DataLoader(dataset, batch_size=script_args.batch_size)

    output_dataset = {"prompt": [], "completions": []}

    for prompt_id, prompt in enumerate(raw_prompts):
        outputs = generations[prompt_id].outputs  # [i].text
        completions = [outputs[j].text for j in range(len(outputs))]
        output_dataset["prompt"].append(prompt)
        output_dataset["completions"].append(completions)

    ds_info = DatasetInfo(f"{script_args.dataset_name} {script_args.num_gen} generations per prompt.")
    relabel_dataset[split] = Dataset.from_dict(output_dataset, split=split, info=ds_info)

relabel_dataset.save_to_disk(script_args.output_dir)
# relabel_dataset.push_to_hub(os.path.basename(script_args.output_dir))
