# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import torch
from datasets import load_dataset
from evaluate import evaluator
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    pipeline,
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(
        default="lvwerra/gpt2-imdb", metadata={"help": "the model name"}
    )
    log_with: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    mini_batch_size: Optional[int] = field(
        default=16, metadata={"help": "the PPO minibatch size"}
    )
    batch_size: Optional[int] = field(default=256, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"}
    )
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "kl target for early stopping"}
    )


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    config,
    dataset_name="imdb",
    input_min_text_length=2,
    input_max_text_length=8,
    split="train",
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split=split)
    ds = ds.rename_columns({"text": "review"})
    if split == "train":
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)
    elif split == "test":
        ds = ds.select(range(5000))

    if input_max_text_length is not None:
        input_size = LengthSampler(input_min_text_length, input_max_text_length)
    else:
        input_size = lambda: None

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def train(config):
    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
    }
    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(config)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    tokenizer.pad_token = tokenizer.eos_token

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator
    )

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    sentiment_pipe = pipeline(
        "sentiment-analysis", model="lvwerra/distilbert-imdb", device=device
    )

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for i, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        # Get response from gpt2
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        # double check sentiment output
        # return dict?
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # Add validation here
        ppo_trainer.log_stats(stats, batch, rewards)

    test_generation_kwargs = {
        **generation_kwargs,
        "max_new_tokens": 16,
    }
    test_dataset = build_dataset(
        config, split="test", input_min_text_length=None, input_max_text_length=None
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=script_args.batch_size,
        collate_fn=collator,
        shuffle=False,
    )

    test_dataloader = ppo_trainer.accelerator.prepare_data_loader(test_dataloader)

    rewards = []
    for test_batch in tqdm(test_dataloader):
        ## reward
        # max length of input is 8
        query_tensors = test_batch["input_ids"][:, 8]
        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **test_generation_kwargs
        )
        test_batch["response"] = tokenizer.batch_decode(response_tensors)
        texts = [q + r for q, r in zip(test_batch["query"], test_batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        batch_rewards = torch.tensor([output[1]["score"] for output in pipe_outputs])

        batch_rewards = ppo_trainer.accelerator.gather_for_metrics(batch_rewards)

        rewards.extend(batch_rewards.tolist())

        ## perplexity

    print(f"avg reward {sum(rewards) / len(rewards)}")


def perplexity(input_ids, model):
    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 1)
        ), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
            ).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    attn_mask,
                ],
                dim=1,
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def test(config):
    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
    }
    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    tokenizer.pad_token = tokenizer.eos_token
    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 16,
    }

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    test_dataset = build_dataset(
        config, split="test", input_min_text_length=8, input_max_text_length=8
    )

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=script_args.batch_size,
        collate_fn=collator,
        shuffle=False,
    )
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=test_dataset)

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    sentiment_pipe = pipeline(
        "sentiment-analysis", model="lvwerra/distilbert-imdb", device=device
    )
    print("test")
    reward_sum = 0.0
    for test_batch in tqdm(test_dataloader):
        query_tensors = test_batch["input_ids"]
        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **generation_kwargs
        )
        test_batch["response"] = tokenizer.batch_decode(response_tensors)
        texts = [q + r for q, r in zip(test_batch["query"], test_batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        reward_sum += sum(rewards)

    print(f"avg reward {reward_sum / len(test_dataset)}")


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    config = PPOConfig(
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        mini_batch_size=script_args.mini_batch_size,
        batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
    )

    test(config)
