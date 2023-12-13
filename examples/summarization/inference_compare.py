import os
from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class ScriptArguments:
    output_dir: Optional[str] = field(
        default="compare_results",
        metadata={"help": "output folder"},
    )
    model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
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

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        num_labels=1,
        torch_dtype=torch_dtype,
    )

    if args.better_transformer:
        model.to_bettertransformer()

    if script_args.tokenizer_name is not None:
        tokenizer_name = script_args.tokenizer_name
    else:
        tokenizer_name = script_args.model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
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
        num_return_sequences=2,
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


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id=-100,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    # if not self.is_encoder_decoder:
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def reward_model(accelerator, model, inputs):
    with torch.no_grad():
        policy_logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits.to(torch.float32)
        policy_logps = get_batch_logps(policy_logits, inputs["labels"], average_log_prob=False)

        with accelerator.unwrap_model(model).disable_adapter():
            ref_logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits.to(torch.float32)
            ref_logps = get_batch_logps(ref_logits, inputs["labels"], average_log_prob=False)

    return policy_logps - ref_logps


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

accelerator = Accelerator()

splits_names = {
    "train": script_args.train_split,
}
relabel_dataset = DatasetDict()

model, tokenizer = create_and_prepare_model(script_args, generation=True)


for split, split_name in splits_names.items():
    dataset = load_dataset(script_args.dataset_name, split=split_name)
    dataloader = DataLoader(dataset, batch_size=script_args.batch_size)

    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    output_dataset = {"prompt": [], "chosen": [], "rejected": []}

    for examples in tqdm(dataloader):
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            [prompt.strip() for prompt in examples["prompt"]],
            padding=True,
            truncation=True,
            max_length=script_args.seq_length,
            pad_to_multiple_of=(8 if script_args.fp16 else None),
            return_tensors="pt",
        )

        # accelerator.print("generate")
        with torch.no_grad():
            generated_sequences = generate_from_prompt(
                inputs["input_ids"].to(accelerator.device),
                inputs["attention_mask"].to(accelerator.device),
                model,
                script_args,
            )

            # accelerator.print("decode")
            generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

            generated_labels = generated_sequences.clone()
            prompt_lens = inputs["attention_mask"].sum(-1)
            generated_encoding_attn_mask = torch.ones_like(generated_labels)

            for i, prompt_len in enumerate(prompt_lens):
                padding_len = (generated_sequences[i * 2] == tokenizer.pad_token_id).sum().item()
                generated_sequences[i * 2] = torch.roll(generated_sequences[i * 2], -padding_len)
                generated_sequences[i * 2 + 1] = torch.roll(generated_sequences[i * 2 + 1], -padding_len)
                generated_labels[i * 2] = torch.roll(generated_labels[i * 2], -padding_len)
                generated_labels[i * 2 + 1] = torch.roll(generated_labels[i * 2 + 1], -padding_len)
                generated_labels[i * 2 : i * 2 + 2, :prompt_len] = -100

            generated_labels[generated_labels == tokenizer.pad_token_id] = -100
            generated_encoding_attn_mask[generated_sequences == tokenizer.pad_token_id] = 0

            reward_model_inputs = {
                "input_ids": generated_sequences.to(accelerator.device),
                "attention_mask": generated_encoding_attn_mask.to(accelerator.device),
                "labels": generated_labels.to(accelerator.device),
            }

            # accelerator.print("rm")
            rewards_generated = reward_model(accelerator, model, reward_model_inputs)  # batch size

            rewards_generated_even = rewards_generated[::2]
            rewards_generated_odd = rewards_generated[1::2]

            pseudolabels = torch.sign(rewards_generated_even - rewards_generated_odd)
            # accelerator.print("gather")
            pseudolabels = accelerator.gather(pseudolabels).cpu().numpy()

            # accelerator.print("iterate")
            for gen_text_even, gen_text_odd, label, prompt in zip(
                generated_texts[::2], generated_texts[1::2], pseudolabels, examples["prompt"]
            ):
                gen_text_even = gen_text_even[len(prompt) :].strip()
                gen_text_odd = gen_text_odd[len(prompt) :].strip()

                output_dataset["prompt"].append(prompt)
                if label >= 0:
                    output_dataset["chosen"].append(gen_text_even)
                    output_dataset["rejected"].append(gen_text_odd)
                else:
                    output_dataset["chosen"].append(gen_text_odd)
                    output_dataset["rejected"].append(gen_text_even)

    ds_info = DatasetInfo("CarperAI/openai_summarize_unlabelled relabeled with a DPO finetuned Pythia 410m")
    relabel_dataset[split] = Dataset.from_dict(output_dataset, split=split, info=ds_info)

relabel_dataset.save_to_disk(script_args.output_dir)
relabel_dataset.push_to_hub(os.path.basename(script_args.output_dir))
