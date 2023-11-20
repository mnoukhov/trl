import shutil
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    GenerationConfig,
)


shutil.disk_usage = lambda x: shutil._ntuple_diskusage(1, 1, 1)


@dataclass
class ScriptArguments:
    output_dir: Optional[str] = field(
        default="compare_results",
        metadata={"help": "output folder"},
    )
    model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
<<<<<<< HEAD
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
=======
        default="CarperAI/openai_summarize_comparisons", metadata={"help": "the dataset name"}
>>>>>>> b09d03aadde19ca4e58f5a6530834e4302411483
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
    seq_length: Optional[int] = field(default=560, metadata={"help": "Input sequence length"})

    temperature: Optional[int] = field(default=0.7, metadata={"help": "Gen temperature"})
    do_sample: Optional[bool] = field(default=True)
    max_new_tokens: Optional[int] = field(default=128, metadata={"help": "max new tokens"})


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

    if generation:
        cls = AutoModelForCausalLM
    else:
        cls = AutoModelForSequenceClassification

    model = cls.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        num_labels=1,
        torch_dtype=torch_dtype,
    )

    if args.better_transformer:
        model.to_bettertransformer()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    tokenizer.padding_side = "left"
    return model, tokenizer


def generate_from_prompt(prompt_ids, prompts_attention_mask, model, args):
    generation_config = GenerationConfig(
        temperature=args.temperature,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
<<<<<<< HEAD
        num_return_sequences=2,
=======
        num_return_sequences=1,
>>>>>>> b09d03aadde19ca4e58f5a6530834e4302411483
        pad_token_id=model.config.pad_token_id,
    )

    input_ids = prompt_ids
    attention_mask = prompts_attention_mask
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
        )
        sequences = accelerator.gather(generation_output.sequences)
<<<<<<< HEAD
    
=======

>>>>>>> b09d03aadde19ca4e58f5a6530834e4302411483
    return sequences


def preprocess_function(examples):
    str_chosen = []
    str_rejected = []
<<<<<<< HEAD
    prompts = []

    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        
        prompts.append(prompt + "\nTL;DR:")
        str_chosen.append(prompt + "\nTL;DR:" + chosen)
        str_rejected.append(prompt + "\nTL;DR:" + rejected)
=======
    prompt = []

    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        str_chosen.append(prompt + "\n" + chosen)
        str_rejected.append(prompt + "\n" + rejected)
>>>>>>> b09d03aadde19ca4e58f5a6530834e4302411483

    tokenized_chosen = tokenizer(
        str_chosen, padding="max_length", truncation=True, max_length=script_args.seq_length, return_tensors="pt"
    )
    tokenized_rejected = tokenizer(
        str_rejected, padding="max_length", truncation=True, max_length=script_args.seq_length, return_tensors="pt"
    )

    tokenized_prompt = tokenizer(
<<<<<<< HEAD
        prompts,
=======
        examples["prompt"],
>>>>>>> b09d03aadde19ca4e58f5a6530834e4302411483
        padding="max_length",
        truncation=True,
        max_length=script_args.seq_length,
        return_tensors="pt",
    )

    return {
        "input_ids_prompt": tokenized_prompt["input_ids"],
        "attention_mask_prompt": tokenized_prompt["attention_mask"],
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

accelerator = Accelerator()

<<<<<<< HEAD
data_splits = [split for split in [script_args.train_split] if split is not None]
relabel_dataset = DatasetDict()

reward_model , _ = create_and_prepare_model(script_args, generation=False)
model, tokenizer = create_and_prepare_model(script_args, generation=True)

for split in data_splits:

    dataset = load_dataset(script_args.dataset_name, split=split)
    dataloader = DataLoader(dataset, batch_size=script_args.batch_size)

    model, reward_model, dataloader = accelerator.prepare(model, reward_model, dataloader)
    model.eval()
    reward_model.eval()
=======
data_splits = [split for split in [script_args.train_split, script_args.eval_split] if split is not None]
relabel_dataset = DatasetDict()

for split in data_splits:
    model, tokenizer = create_and_prepare_model(script_args, generation=True)
    dataset = load_dataset(script_args.dataset_name, split=split)

    dataloader = DataLoader(dataset, batch_size=script_args.batch_size)

    model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()
>>>>>>> b09d03aadde19ca4e58f5a6530834e4302411483

    output_dataset = {"prompt": [], "chosen": [], "rejected": []}

    generated_sequences = []
    for examples in tqdm(dataloader):
        inputs = preprocess_function(examples)

        with torch.no_grad():
            sequences = generate_from_prompt(
                inputs["input_ids_prompt"].to(accelerator.device),
                inputs["attention_mask_prompt"].to(accelerator.device),
                model,
                script_args,
            )
<<<<<<< HEAD
        
            generated_sequences = sequences
            generated_attention_mask = torch.ones_like(generated_sequences)
            generated_attention_mask[generated_sequences == tokenizer.pad_token_id] = 0
            rewards_generated = reward_model(
                input_ids=generated_sequences.to(accelerator.device),
                attention_mask=generated_attention_mask.to(accelerator.device),
            )[0]
    
            generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            rewards_generated = rewards_generated.view(-1, 2, rewards_generated.shape[-1])
            rewards_generated_even = rewards_generated[:, 0, :]
            rewards_generated_odd = rewards_generated[:, 1, :]
        
            pseudolabels = torch.sign(rewards_generated_even - rewards_generated_odd)
            pseudolabels = accelerator.gather(pseudolabels).cpu().numpy()

            #loop through each two generated texts
            for gen_text_even, gen_text_odd, label in zip(generated_texts[::2], generated_texts[1::2], pseudolabels):
                prompt = gen_text_even.split("\nTL;DR:")[0]
                gen_text_even = gen_text_even.split("\nTL;DR:")[1]
                gen_text_odd = gen_text_odd.split("\nTL;DR:")[1]

                output_dataset["prompt"].append(prompt)
                if label >= 0:
                    output_dataset["chosen"].append(gen_text_even)
                    output_dataset["rejected"].append(gen_text_odd)
                else:
                    output_dataset["chosen"].append(gen_text_odd)
                    output_dataset["rejected"].append(gen_text_even)
            
    ds_info = DatasetInfo("CarperAI/openai_summarize_unlabelled relabeled with a DPO finetuned Pythia 410m")
=======
            generated_sequences.append(sequences)

    generated_sequences = torch.cat(generated_sequences, dim=0)
    model, tokenizer = create_and_prepare_model(script_args, generation=False)
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    generated_sequences_data_loader = DataLoader(generated_sequences, batch_size=script_args.batch_size)
    model, generated_sequences_data_loader = accelerator.prepare(model, generated_sequences_data_loader)

    for examples, generated_sequences in tqdm(zip(dataloader, generated_sequences_data_loader)):
        inputs = preprocess_function(examples)
        with torch.no_grad():
            generated_attention_mask = torch.ones_like(generated_sequences)
            generated_attention_mask[generated_sequences == tokenizer.pad_token_id] = 0
            rewards_generated = model(
                input_ids=generated_sequences.to(accelerator.device),
                attention_mask=generated_attention_mask.to(accelerator.device),
            )[0]
            rewards_chosen = model(
                input_ids=inputs["input_ids_chosen"].to(accelerator.device),
                attention_mask=inputs["attention_mask_chosen"].to(accelerator.device),
            )[0]
            rewards_rejected = model(
                input_ids=inputs["input_ids_rejected"].to(accelerator.device),
                attention_mask=inputs["attention_mask_rejected"].to(accelerator.device),
            )[0]
          
            generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            pseudolabels = torch.sign(rewards_generated - rewards_chosen)

            pseudolabels = accelerator.gather(pseudolabels).cpu().numpy()

            for prompt, init_chosen, init_rejected, generated, label in zip(
                examples["prompt"], examples["chosen"], examples["rejected"], generated_texts, pseudolabels
            ):
                output_dataset["prompt"].append(prompt)
                # TODO decide which one to save as chosen/rejected
                if label >= 0:
                    output_dataset["chosen"].append(generated)
                    output_dataset["rejected"].append(init_chosen)
                else:
                    output_dataset["chosen"].append(init_chosen)
                    output_dataset["rejected"].append(generated)

    ds_info = DatasetInfo("CarperAI/openai_summarize_comparisons relabelled with a DPO finetuned Pythia 410m")
>>>>>>> b09d03aadde19ca4e58f5a6530834e4302411483
    relabel_dataset[split] = Dataset.from_dict(output_dataset, split=split, info=ds_info)

relabel_dataset.save_to_disk(script_args.output_dir)
