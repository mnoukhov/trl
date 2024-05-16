import gc
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from datasets import builder, load_dataset
from peft import PeftModelForCausalLM
from scalar_rm_model import ScalarModel, ScalarModelConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    pipeline,
)
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
    model_paths: Optional[List[str]] = field(default_factory=list)
    # base_model_revision: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
    )
    split: Optional[str] = field(default="validation", metadata={"help": "the dataset name"})
    generate_batch_size: Optional[int] = field(default=4)

    temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(default=48, metadata={"help": "max new tokens"})
    gen_dtype: Optional[str] = field(default="auto")
    sanity_check: Optional[bool] = field(default=False)


@dataclass
class EvalScriptArguments:
    wandb_log_id: Optional[str] = field(default=None)
    gold_model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    gold_model_revision: Optional[str] = field(default=None)
    eval_dtype: Optional[str] = field(default="auto")
    eval_batch_size: Optional[int] = field(default=16)
    gold_tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    flash_attention: Optional[bool] = field(default=False)


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
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )

    gens = {}
    checkpoint_subfolders = [
        path
        for path in os.listdir(script_args.model_name_or_path)
        if path.startswith("checkpoint") and (not script_args.model_paths or path in script_args.model_paths)
    ]
    for checkpoint_name in checkpoint_subfolders:
        print(f"generating step {checkpoint_name}")

        model_name_or_path = os.path.join(script_args.model_name_or_path, checkpoint_name)
        if script_args.base_model_name is not None:
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
            tokenizer=script_args.tokenizer_name,
            dtype=script_args.gen_dtype,
            tensor_parallel_size=script_args.num_gpus,
            trust_remote_code=True,
        )

        generations = llm.generate(prompts, sampling_params)

        texts = [output.prompt + output.outputs[0].text for output in generations]

        gens[checkpoint_name] = texts

        dataset = dataset.add_column(f"generations_{checkpoint_name}", texts)

        # delete old model
        destroy_model_parallel()
        del llm.llm_engine.model_executor.driver_worker
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()

    if script_args.output_dir is not None:
        # TODO add hash to dataset path
        # sampling_str = str(sampling_params)
        # sampling_hash = hashlib.sha256(sampling_str.encode()).hexdigest()[:10]

        # TODO fix model name or path string
        dataset_path = os.path.join(
            script_args.output_dir,
            script_args.dataset_name.replace("/", "_"),
            script_args.model_name_or_path.replace("/", "_"),
        )
        os.makedirs(dataset_path, exist_ok=True)
        dataset.save_to_disk(dataset_path)
        with open(f"{dataset_path}_sampling_params.txt", "w") as f:
            print(sampling_params, file=f)

    print(f"generated {len(gens)} steps")
    reference = dataset["query_reference_response"]
    prompts = dataset["query"]

    return prompts, reference, gens

    # ds_info = DatasetInfo(
    #     f"{script_args.dataset_name} split {script_args.train_split} prompts used to generate with {script_args.model_name}"
    #     f" temp {script_args.temperature} top_p {script_args.top_p} "
    # )
    # generated_dataset = Dataset.from_generator(dataset_generator, info=ds_info)
    # generated_dataset.push_to_hub(os.path.basename(script_args.output_dir), split="train")


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
    tokenizer = AutoTokenizer.from_pretrained(args.gold_tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if args.gold_model_name.startswith("vwxyzjn"):
        # ScalarModel
        scalar_model_config = ScalarModelConfig.from_pretrained(
            args.gold_model_name,
            revision=args.gold_model_revision,
        )
        # hack to remove the path
        # models/EleutherAI/pythia-6.9b-deduped/sft_model_55513 -> EleutherAI/pythia-6.9b-deduped
        if scalar_model_config.base_model.startswith("models/"):
            original_model = scalar_model_config.base_config["_name_or_path"].split("/")[2]
            sft_model = f"vwxyzjn/EleutherAI_{original_model}__sft__tldr"
            scalar_model_config.base_config["_name_or_path"] = sft_model
            scalar_model_config.base_model = sft_model
            _, seed, _ = args.gold_model_revision.split("__")
            scalar_model_config.base_model_revision = f"sft__{seed}__1708611267"

        # quantization_config = get_quantization_config(model_config)
        model = ScalarModel.from_pretrained(
            args.gold_model_name,
            revision=args.gold_model_revision,
            config=scalar_model_config,
            torch_dtype=torch_dtype,
            use_flash_attention_2=args.flash_attention,
            device_map="auto",
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.gold_model_name,
            revision=args.gold_model_revision,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    reward_pipeline = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        function_to_apply="none",
        batch_size=args.eval_batch_size,
    )

    ref_outputs = reward_pipeline(reference)
    ref_rewards = np.array([out["score"] for out in ref_outputs])

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
            sample_generations = wandb.Table(
                columns=["Prompt", "Policy", "Reference"],
                rows=[
                    [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                    for prompt, pol, ref in zip(
                        prompts[:num_samples], query_response[:num_samples], reference[:num_samples]
                    )
                ],
            )
            wandb.log(
                {
                    "gold/win_rate": win_rate,
                    "gold/norm_reward": norm_reward,
                    "gold/reward": mean_reward,
                    "gold/samples": sample_generations,
                    "train/global_step": step,
                },
                step=step,
            )

        print(f"step {step}: win-rate {win_rate} norm-reward {norm_reward}")


def main(generate_args, eval_args):
    if eval_args.gold_tokenizer_name is None:
        eval_args.gold_tokenizer_name = generate_args.tokenizer_name

    if generate_args.sanity_check:
        eval_args.wandb_log_id = None

    print("GENERATING")
    prompts, reference, generations = generate(generate_args)
    #
    # dataset = load_dataset(generate_args.dataset_name, split=generate_args.split)
    # dataset = dataset.select(range(100))
    # generations = {"step0": dataset["query_reference_response"]}
    # reference = dataset["query_reference_response"]
    print("EVALUATING")
    evaluate(eval_args, prompts, reference, generations, generate_args.model_name_or_path)


def main_args_dict(args_dict):
    parser = HfArgumentParser([GenerateScriptArguments, EvalScriptArguments])
    generate_args, eval_args = parser.parse_dict(args_dict)
    main(generate_args, eval_args)


if __name__ == "__main__":
    parser = HfArgumentParser([GenerateScriptArguments, EvalScriptArguments])
    generate_args, eval_args = parser.parse_args_into_dataclasses()
    main(generate_args, eval_args)
