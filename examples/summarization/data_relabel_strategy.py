import torch


def min_max(examples):
    """
    takes batches from hf dataset with columns "prompt", "completions" and "scores"
    raturns "prompt", "chosen" and "rejected" columns by taking max as chosen and min as rejected
    """
    new_batch = {"prompt": [], "chosen": [], "rejected": []}
    scores = torch.Tensor(examples["scores"])
    max_indices = torch.argmax(scores, dim=1)
    min_indices = torch.argmin(scores, dim=1)

    for i in range(len(examples["prompt"])):
        prompt = examples["prompt"][i]
        chosen = examples["completions"][i][max_indices[i]]
        rejected = examples["completions"][i][min_indices[i]]
        new_batch["prompt"].append(prompt)
        new_batch["chosen"].append(chosen)
        new_batch["rejected"].append(rejected)

    return new_batch


def top_two(examples):
    """
    takes batches from hf dataset with columns "prompt", "completions" and "scores"
    raturns "prompt", "chosen" and "rejected" columns by taking top two completions
    """
    new_batch = {"prompt": [], "chosen": [], "rejected": []}
    scores = torch.Tensor(examples["scores"])
    max_indices = torch.topk(scores, 2, dim=1).indices
    for i in range(len(examples["prompt"])):
        prompt = examples["prompt"][i]
        chosen = examples["completions"][i][max_indices[i][0]]
        rejected = examples["completions"][i][max_indices[i][1]]
        new_batch["prompt"].append(prompt)
        new_batch["chosen"].append(chosen)
        new_batch["rejected"].append(rejected)

    return new_batch
