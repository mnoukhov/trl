import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import PartialState

# from callbacks import PerplexityCallback
from datasets import builder, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from trl import DPOTrainer, ModelConfig
from trl.trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class DPOScriptArguments:
    task_type: str = field(default="hh")
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_eval_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    eval_dataset_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
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

    ema_decay: float = field(default=0.999)


class EMAUpdateCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer.ref_model.update_parameters(self.trainer.model)


if __name__ == "__main__":
    parser = HfArgumentParser((DPOScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

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

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token_id is None:
        assert args.task_type != "tldr"
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    train_dataset = load_dataset(args.dataset_name, split=args.dataset_train_split)
    eval_dataset_name = args.eval_dataset_name if args.eval_dataset_name is not None else args.dataset_name
    eval_dataset = load_dataset(eval_dataset_name, split=args.dataset_eval_split)

    if args.sanity_check:
        train_dataset = train_dataset.select(range(128))
        eval_dataset = eval_dataset.select(range(128))
        training_args.push_to_hub = False
        # training_args.hub_model_id = None

    ref_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay)
    )
    ################
    # Training
    ################
    trainer = DPOTrainer(
        model,
        ref_model=ref_model,
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

    ema_update = EMAUpdateCallback(trainer)
    trainer.add_callback(ema_update)

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if PartialState().is_main_process and training_args.push_to_hub:
        trainer.push_to_hub(training_args.hub_model_id)
        tokenizer.push_to_hub(training_args.hub_model_id)

    if model_config.use_peft:
        merged_path = os.path.join(training_args.output_dir, "_merged")
        model = trainer.model.merge_and_unload()
        model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
