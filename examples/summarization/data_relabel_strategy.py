import torch


def min_max(examples):
    """
    takes batches from hf dataset with columns "prompt", "completions" and "scores"
    raturns "prompt", "chosen" and "rejected" columns by taking max as chosen and min as rejected
    """
    return_batch = {"prompt": [], "chosen": [], "rejected": []}
    scores = torch.Tensor(examples["scores"])
    max_indices = torch.argmax(scores, dim=1)
    min_indices = torch.argmin(scores, dim=1)

    for i in range(len(examples["prompt"])):
        prompt = examples["prompt"][i]
        chosen = examples["completions"][i][max_indices[i]]
        rejected = examples["completions"][i][min_indices[i]]
        examples["prompt"].append(prompt)
        examples["chosen"].append(chosen)
        examples["rejected"].append(rejected)

    return return_batch
