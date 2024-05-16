import gc
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import torch
from datasets import builder, load_dataset
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, HfArgumentParser
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

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
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    model_revision: Optional[str] = field(default=None, metadata={"help": "the model name"})
    model_paths: Optional[List[str]] = field(default_factory=list)
    # base_model_revision: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
    )
    split: Optional[str] = field(default="validation", metadata={"help": "the dataset name"})
    batch_size: Optional[int] = field(default=4)

    temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(default=48, metadata={"help": "max new tokens"})
    gen_dtype: Optional[str] = field(default="auto")
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
    seed: Optional[int] = field(default=0)


OPTIONS = ["A", "B"]

TEMPLATE = """Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

### Post:
{post}

### Summary A:
{response0}

### Summary B:
{response1}

### Instructions:
FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
"""


def generate(script_args):
    dataset = load_dataset(script_args.dataset_name, split=script_args.split)
    if script_args.sanity_check:
        dataset = dataset.select(range(100))

    prompts = dataset["query"]

    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        top_p=script_args.top_p,
        n=1,
    )

    gens = {}

    model_paths = [script_args.model_name_or_path]
    # path with possible checkpoint subfolders
    if os.path.exists(script_args.model_name_or_path):
        checkpoint_subfolders = [
            path
            for path in os.listdir(script_args.model_name_or_path)
            if path.startswith("checkpoint") and (not script_args.model_paths or path in script_args.model_paths)
        ]

        # if there are checkpoint subfolders, use those instead of model_path
        if checkpoint_subfolders:
            model_paths = [
                os.path.join(script_args.model_name_or_path, subfolder) for subfolder in checkpoint_subfolders
            ]

    for model_name_or_path in model_paths:
        # model_name_or_path = os.path.join(script_args.model_name_or_path, checkpoint_name)

        print(f"generating {model_name_or_path}")

        if script_args.base_model_name is not None:
            assert script_args.model_revision is None
            # peft model that needs to be merged
            base_model = AutoModelForCausalLM.from_pretrained(
                script_args.base_model_name, revision=script_args.base_model_revision
            )
            # merge the model and save
            model = PeftModelForCausalLM.from_pretrained(base_model, model_name_or_path, device_map="cpu")
            merged = model.merge_and_unload()
            model_save_path = os.path.join(model_name_or_path, "_merged")
            merged.save_pretrained(model_save_path)
            del model
            del merged
            model_name_or_path = model_save_path

        llm = LLM(
            model=model_name_or_path,
            revision=script_args.model_revision,
            tokenizer=script_args.tokenizer_name,
            dtype=script_args.gen_dtype,
            tensor_parallel_size=script_args.num_gpus,
            trust_remote_code=True,
        )

        generations = llm.generate(prompts, sampling_params)

        texts = [output.outputs[0].text for output in generations]

        model_or_checkpoint_name = os.path.basename(model_name_or_path)
        gens[model_or_checkpoint_name] = texts

        dataset = dataset.add_column(f"generations_{model_or_checkpoint_name}", texts)

        # delete old model
        destroy_model_parallel()
        del llm.llm_engine.model_executor.driver_worker
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()

    dataset_path = os.path.join(script_args.model_name_or_path, "_generations")
    os.makedirs(dataset_path, exist_ok=True)
    dataset.save_to_disk(os.path.join(dataset_path, "dataset"))
    with open(os.path.join(dataset_path, "sampling_params.txt"), "w") as f:
        print(sampling_params, file=f)

    print(f"generated {len(gens)} steps")
    reference = []
    for ref_response in dataset["reference_response"]:
        if ref_response.endswith("<|endoftext|>"):
            ref_response = ref_response.split("<|endoftext|>")[0]

        reference.append(ref_response.strip())

    return prompts, reference, gens


def create_llm_judge_prompts(tokenizer, prompts, reference, generated, seed):
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

        query = TEMPLATE.format(post=prompt, response0=response0, response1=response1)
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

    ## get reference continuation rewards
    step = 0
    for step_str, generated in generations.items():
        print(f"Evaluating {step_str}")
        llm_judge_prompts, generated_indices = create_llm_judge_prompts(
            tokenizer, prompts, reference, generated, args.seed
        )
        llm_judge_output = llm.generate(llm_judge_prompts, sampling_params)
        llm_judge_texts = [output.outputs[0].text for output in llm_judge_output]

        comparisons, preferred = [], []
        for llm_judge_completion in llm_judge_texts:
            if "Comparison:" in llm_judge_completion:
                comparisons.append(llm_judge_completion.split("Comparison:")[1].split("Preferred:")[0].strip())
            else:
                comparisons.append("")

            if "Preferred:" in llm_judge_completion:
                preferred.append(llm_judge_completion.split("Preferred:")[1].strip())
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


def main(generate_args, eval_args):
    eval_args.num_gpus = generate_args.num_gpus
    eval_args.output_dir = generate_args.output_dir

    if generate_args.sanity_check:
        eval_args.wandb_log_id = None

    print("GENERATING")
    prompts, reference, generations = generate(generate_args)
    # dataset = load_dataset(generate_args.dataset_name, split=generate_args.split)
    # generations = {"step0": dataset["query_reference_response"]}
    # prompts = dataset["query"]
    # reference = dataset["reference_response"]
    # generations = {"step0": dataset["reference_response"]}
    print("EVALUATING")
    llm_as_a_judge(eval_args, prompts, reference, generations, generate_args.model_name_or_path)


def main_args_dict(args_dict):
    parser = HfArgumentParser([GenerateScriptArguments, LLMJudgeArguments])
    generate_args, eval_args = parser.parse_dict(args_dict)
    main(generate_args, eval_args)


if __name__ == "__main__":
    parser = HfArgumentParser([GenerateScriptArguments, LLMJudgeArguments])
    generate_args, eval_args = parser.parse_args_into_dataclasses()
    main(generate_args, eval_args)
