from dataclasses import dataclass, field

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from trl import DPOTrainer, ModelConfig
from trl.trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config


@dataclass
class DPOScriptArguments:
    output_model_name: str = field(default="", metadata={"help": "model name to upload"})
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_train_name: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_test_name: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    max_length: int = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: int = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: int = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    sanity_check: bool = field(default=False, metadata={"help": "only train on 1000 samples"})
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "debug argument for distributed training;"
            "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    generate_during_eval: bool = field(default=False, metadata={"help": "Generate during evaluation"})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False, metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser((DPOScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model_ref = None

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)

    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset_name)
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    train_dataset = ds[args.dataset_train_name]
    eval_dataset = ds[args.dataset_test_name]

    ################
    # Training
    ################
    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        tokenizer=tokenizer,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_length=args.max_length,
        max_target_length=args.max_target_length,
        max_prompt_length=args.max_prompt_length,
        generate_during_eval=args.generate_during_eval,
        peft_config=get_peft_config(model_config),
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print("DONE")

    trainer.save_model(training_args.output_dir)

    if PartialState().is_main_process:
        # model = trainer.model.merge_and_unload()
        trainer.push_to_hub(args.output_model_name)
        tokenizer.push_to_hub(args.output_model_name)
