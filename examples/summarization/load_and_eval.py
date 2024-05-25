import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import gather_object, tqdm
from datasets import builder, load_from_disk

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    pipeline,
)

import wandb


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class EvalScriptArguments:
    output_dir: Optional[str] = field(
        default="/home/toolkit/trl_results",
        metadata={"help": "output folder"},
    )
    dataset_name: Optional[str] = field(
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
    )
    wandb_log_id: Optional[str] = field(default=None)
    gold_model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    gold_model_revision: Optional[str] = field(default=None)
    eval_dtype: Optional[str] = field(default="auto")
    eval_batch_size: Optional[int] = field(default=16)
    gold_tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    flash_attention: Optional[bool] = field(default=False)
    sanity_check: Optional[bool] = field(default=False)
    ensure_eos: Optional[bool] = field(default=None)


def evaluate(args, prompts, reference, generations, model_name=None):
    if args.wandb_log_id is not None:
        # don't overwrite the wandb name of the original run
        if args.wandb_log_id == "model_name":
            # model name = config_wandblogid
            wandb_log_id = model_name.split("_")[-1]
        elif args.wandb_log_id == "model_path":
            # model path = /home/.../wandb_log_id/output
            wandb_log_id = model_name.split("/")[-2]
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
    gold_tokenizer_name = args.gold_tokenizer_name if args.gold_tokenizer_name is not None else args.gold_model_name
    tokenizer = AutoTokenizer.from_pretrained(gold_tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    distributed_state = PartialState()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.gold_model_name,
        revision=args.gold_model_revision,
        torch_dtype=torch_dtype,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    reward_pipeline = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=distributed_state.device,
        function_to_apply="none",
        # batch_size=args.eval_batch_size,
    )

    ref_outputs = []
    with distributed_state.split_between_processes(reference) as text:
        for out in tqdm(
            reward_pipeline(text, batch_size=args.eval_batch_size),
        ):
            if isinstance(out, dict):
                out = [out]
            ref_outputs.extend([o["score"] for o in out])

    ref_outputs = gather_object(ref_outputs)

    if distributed_state.is_main_process:
        ref_rewards = np.array(ref_outputs)

        step = 0
        for step_str, query_response in generations.items():
            gen_outputs = reward_pipeline(query_response)
            gen_rewards = np.array([out["score"] for out in gen_outputs])

            win_rate = (gen_rewards > ref_rewards).mean().item()
            norm_reward = (gen_rewards - ref_rewards).mean().item()
            mean_reward = gen_rewards.mean().item()

            if step_str.startswith("checkpoint-"):
                step_str = step_str.removeprefix("checkpoint-")

            if step_str.isdigit():
                step = int(step_str)
            else:
                print(f"Warning step name {step_str} is not an integer")
                step = step + 1

            if log_to_wandb:
                num_samples = 32
                # sample_generations = wandb.Table(
                #     columns=["Prompt", "Policy", "Reference"],
                #     rows=[
                #         [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                #         for prompt, pol, ref in zip(
                #             prompts[:num_samples], query_response[:num_samples], reference[:num_samples]
                #         )
                #     ],
                # )
                wandb.log(
                    {
                        "gold/hf_win_rate": win_rate,
                        "gold/hf_norm_reward": norm_reward,
                        "gold/hf_reward": mean_reward,
                        # "gold/hf_samples": sample_generations,
                        "train/global_step": step,
                    },
                )

            print(f"step {step}: reward {mean_reward} win-rate {win_rate} norm-reward {norm_reward}")


def main(args):
    print("LOADING GENERATED")
    dataset = load_from_disk(args.dataset_name)

    # TODO
    # pieces = args.dataset_name.split("/")
    # assert pieces[4] == "results"
    # args.wandb_log_id = pieces[5]

    if args.sanity_check:
        dataset = dataset.select(range(100))
        args.wandb_log_id = None

    generated_col = dataset.column_names[-1]

    if args.ensure_eos is not None:
        eos_token = "<|endoftext|>"

        def ensure_eos_or_not(example):
            for column_name in ["query_reference_response", generated_col]:
                if args.ensure_eos and not example[column_name].endswith(eos_token):
                    example[column_name] = example[column_name] + eos_token
                elif not args.ensure_os:
                    example[column_name] = example[column_name].removesuffix(eos_token)

            return example

        dataset = dataset.map(ensure_eos_or_not)

    prompts = dataset["query"]
    reference = dataset["query_reference_response"]

    ckpt_str = generated_col.split("_")[1]
    generations = {ckpt_str: dataset[generated_col]}

    print("EVALUATING")
    evaluate(args, prompts, reference, generations, model_name=generated_col)


def main_args_dict(args_dict):
    parser = HfArgumentParser([EvalScriptArguments])
    args = parser.parse_dict(args_dict, allow_extra_keys=True)
    main(args)


if __name__ == "__main__":
    parser = HfArgumentParser([EvalScriptArguments])
    args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    main(args)
