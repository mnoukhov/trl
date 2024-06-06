import gc
import os
import random
from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import pandas as pd
import torch
from datasets import builder, load_dataset, load_from_disk
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, HfArgumentParser
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

import wandb


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class LoadArguments:
    output_dir: Optional[str] = field(
        default="/home/toolkit/trl_results",
        metadata={"help": "output folder"},
    )
    dataset_name: Optional[str] = field(
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
    )
    dataset_prompt_field: str = field(
        default="prompt", metadata={"help": "name of the prompt field in the dataset, e.g. 'query' in summarization"}
    )
    dataset_chosen_field: str = field(
        default="chosen",
        metadata={"help": "name of the chosen field in the dataset, e.g. 'reference_response' in summarization"},
    )
    sanity_check: Optional[bool] = field(default=False)


@dataclass
class LLMJudgeArguments:
    wandb_log_id: Optional[str] = field(default=None)
    llm_judge_model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    llm_judge_model_revision: Optional[str] = field(default=None)
    llm_judge_dtype: Optional[str] = field(default="auto")
    llm_judge_temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    llm_judge_top_p: Optional[float] = field(default=0.9, metadata={"help": "Gen temperature"})
    llm_judge_max_new_tokens: Optional[int] = field(default=None, metadata={"help": "max new tokens"})
    template: Literal["tldr", "hh"] = field(default="tldr", metadata={"help": "the template, e.g. summarization"})
    seed: Optional[int] = field(default=0)


OPTIONS = ["A", "B"]

Template = namedtuple("Template", ["judge_prompt", "comparison_key", "output_key"])

tldr_prompt = """Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

### Post:
{prompt}

### Summary A:
{response0}

### Summary B:
{response1}

### Instructions:
FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">"""

TLDR_TEMPLATE = Template(judge_prompt=tldr_prompt, comparison_key="Comparison:", output_key="Preferred:")


hh_prompt = """For the following query to a chatbot, which response is more helpful?
Query: {prompt}

Response A:
{response0}

Response B:
{response1}

FIRST provide a one-sentence comparison of the two responses and explain which you feel is more helpful. \
SECOND, on a new line, state only "A" or "B" to indicate which response is more helpful. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
More helpful: <"A" or "B">"""

HH_TEMPLATE = Template(judge_prompt=hh_prompt, comparison_key="Comparison:", output_key="More helpful:")


def create_llm_judge_prompts(tokenizer, prompts, reference, generated, seed, prompt_template):
    llm_judge_prompts = []
    generated_indices = []
    random.seed(seed)
    for prompt, ref, gen in zip(prompts, reference, generated):
        generated_idx = random.randint(0, 1)
        if generated_idx == 0:
            response0 = gen.strip()
            response1 = ref.strip()
        else:
            response0 = ref.strip()
            response1 = gen.strip()

        query = prompt_template.format(prompt=prompt, response0=response0, response1=response1)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        llm_judge_prompts.append(formatted_prompt)
        generated_indices.append(generated_idx)

    return llm_judge_prompts, generated_indices


def llm_as_a_judge(args, prompts, reference, generations, model_name=None):
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

    llm = LLM(
        model=args.llm_judge_model_name,
        revision=args.llm_judge_model_revision,
        dtype=args.llm_judge_dtype,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.llm_judge_temperature,
        max_tokens=args.llm_judge_max_new_tokens,
        top_p=args.llm_judge_top_p,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    )

    if args.template == "tldr":
        llm_judge_template = TLDR_TEMPLATE
    elif args.template == "hh":
        llm_judge_template = HH_TEMPLATE
    else:
        raise NotImplementedError("not a valid template")

    ## get reference continuation rewards
    step = 0
    for step_str, generated in generations.items():
        print(f"Evaluating {step_str}")
        llm_judge_prompts, generated_indices = create_llm_judge_prompts(
            tokenizer,
            prompts,
            reference,
            generated,
            args.seed,
            llm_judge_template.judge_prompt,
        )
        llm_judge_output = llm.generate(llm_judge_prompts, sampling_params)
        llm_judge_texts = [output.outputs[0].text for output in llm_judge_output]

        comparisons, preferred = [], []
        for llm_judge_completion in llm_judge_texts:
            if llm_judge_template.comparison_key in llm_judge_completion:
                comparisons.append(
                    llm_judge_completion.split(llm_judge_template.comparison_key)[1]
                    .split(llm_judge_template.output_key)[0]
                    .strip()
                )
            else:
                comparisons.append("")

            if llm_judge_template.output_key in llm_judge_completion:
                preferred.append(llm_judge_completion.split(llm_judge_template.output_key)[1].strip())
            else:
                preferred.append("X")

        full_convo = [prompt + text for prompt, text in zip(llm_judge_prompts, llm_judge_texts)]

        winner = []
        win_sum = 0
        num_fails = 0
        for pref, gen_idx in zip(preferred, generated_indices):
            if pref == OPTIONS[gen_idx]:
                winner.append("ours")
                win_sum += 1
            elif pref == OPTIONS[1 - gen_idx]:
                winner.append("reference")
            else:
                winner.append("fail")
                num_fails += 1

        win_rate = win_sum / (len(preferred) - num_fails)
        if num_fails > 0:
            print(f"Failed to get preference from {num_fails} examples out of {len(preferred)}")

        if step_str.startswith("checkpoint-"):
            step_str = step_str.removeprefix("checkpoint-")

        if step_str.isdigit():
            step = int(step_str)
        else:
            print(f"Warning step name {step_str} is not an integer")
            step = step + 1

        if log_to_wandb:
            wandb.log(
                {
                    "llm_judge/win_rate": win_rate,
                    "train/global_step": step,
                }
            )

        print(f"step {step}: win-rate {win_rate}")

        if args.output_dir is not None:
            df = pd.DataFrame(
                {
                    "prompt": prompts,
                    "reference": reference,
                    "generated": generated,
                    "winner": winner,
                    "llm_prompt": llm_judge_prompts,
                    "full_conov": full_convo,
                    "generated_idx": generated_indices,
                }
            )
            df.to_csv(os.path.join(args.output_dir, f"step{step}.csv"))


def main(load_args, eval_args):
    eval_args.num_gpus = generate_args.num_gpus
    eval_args.output_dir = generate_args.output_dir

    if generate_args.sanity_check:
        eval_args.wandb_log_id = None

    print("LOADING")
    dataset = load_from_disk(load_args.dataset_name)

    if load_args.sanity_check:
        dataset = dataset.select(range(100))
        load_args.wandb_log_id = None

    prompts = dataset["query"]
    reference = dataset["query_reference_response"]
    generated_columns = [name for name in dataset.column_names if "checkpoint" in name]

    eos_token = "<|endoftext|>"

    def remove_eos(example):
        for column_name in ["query_reference_response"] + generated_columns:
            example[column_name] = example[column_name].removesuffix(eos_token)

        return example

    dataset = dataset.map(remove_eos)

    generations = {}
    for column_name in generated_columns:
        model_name, checkpoint_name = column_name.split("_")
        generations[checkpoint_name] = dataset[column_name]

    print("EVALUATING")
    llm_as_a_judge(eval_args, prompts, reference, generations, model_name)


def main_args_dict(args_dict):
    parser = HfArgumentParser([LoadArguments, LLMJudgeArguments])
    generate_args, eval_args = parser.parse_dict(args_dict)
    main(generate_args, eval_args)


if __name__ == "__main__":
    parser = HfArgumentParser([LoadArguments, LLMJudgeArguments])
    generate_args, eval_args = parser.parse_args_into_dataclasses()
    main(generate_args, eval_args)
