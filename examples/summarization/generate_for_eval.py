import gc
import os
from dataclasses import dataclass, field
from typing import List, Optional

import ray
import torch
from datasets import builder, load_dataset
from peft import PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class GenerateScriptArguments:
    output_dir: Optional[str] = field(
        default="/home/toolkit/trl_results",
        metadata={"help": "output folder"},
    )
    generated_output_name: Optional[str] = field(
        default="_dataset",
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
        print(f"generating {model_name_or_path}")
        model_or_checkpoint_name = os.path.basename(model_name_or_path)

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

        gens[model_or_checkpoint_name] = texts

        dataset = dataset.add_column(f"generations_{model_or_checkpoint_name}", texts)

        # delete old model
        destroy_model_parallel()
        del llm.llm_engine.model_executor.driver_worker
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
        # torch.distributed.destroy_process_group()

    dataset_path = os.path.join(script_args.output_dir, script_args.generated_output_name)
    os.makedirs(dataset_path, exist_ok=True)
    print("saving dataset to")
    print(dataset_path)
    dataset.save_to_disk(dataset_path)
    with open(f"{dataset_path}_sampling_params.txt", "w") as f:
        print(sampling_params, file=f)

    print(f"generated {len(gens)} steps")


def main(generate_args):
    print("GENERATING")
    generate(generate_args)


def main_args_dict(args_dict):
    parser = HfArgumentParser([GenerateScriptArguments])
    generate_args = parser.parse_dict(args_dict)
    main(generate_args)


if __name__ == "__main__":
    parser = HfArgumentParser([GenerateScriptArguments])
    generate_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    main(generate_args)
