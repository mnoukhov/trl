import argparse
import copy
import glob
import os

import evaluate_ppl
import yaml
from haven import haven_utils as hu
from haven import haven_wizard as hw


def run_exp(exp_dict, savedir, args):
    exp_name = exp_dict.pop("name")

    if not args.no_wandb:
        os.environ["WANDB_RUN_ID"] = os.path.basename(savedir)

    if exp_name.startswith("ppl"):
        evaluate_ppl.main_dict(exp_dict)

    # main(
    #     config_path_or_config=exp_dict,
    #     project_name="rl4lms",
    #     experiment_name=experiment_name,
    #     base_path_to_store_results=savedir,
    #     entity_name="mila-language-drift",
    #     log_to_wandb=(not args.no_wandb),
    # )


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="/home/toolkit/trl/results",
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-r",
        "--reset",
        type=int,
        default=0,
        help="If true, reset the experiment. Else, resume.",
    )
    parser.add_argument(
        "-j",
        "--job_scheduler",
        default=None,
        type=str,
        help="Run the experiments as jobs in the cluster.",
    )
    parser.add_argument(
        "-p",
        "--python_binary",
        default="/home/toolkit/.conda/envs/trl/bin/python",
        help="path to your python executable",
    )
    parser.add_argument(
        "-n", "--gpus", default=1, type=int, help="number of gpus to use for experiment"
    )
    parser.add_argument(
        "--gpu-mem", default=80, type=int, help="mem of gpus to use for experiment"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="disable wandb", default=False
    )
    # parser.add_argument(
    #     "--exp-id", default=None, help="id used to resume an experiment"
    # )

    args, _ = parser.parse_known_args()

    with open(args.exp_group, "r") as fp:
        exp_dict = yaml.safe_load(fp)

    exp_dict["name"] = os.path.basename(args.exp_group)

    exp_list = [exp_dict]

    if args.job_scheduler == "toolkit":
        with open("/home/toolkit/wandb_api_key", "r") as f:
            wandb_api_key = f.read().rstrip()

        job_config = {
            "account_id": os.environ["EAI_ACCOUNT_ID"],
            # "image": "registry.console.elementai.com/snow.colab/cuda",
            # "image": "registry.console.elementai.com/snow.colab_public/ssh",
            # "image": "registry.console.elementai.com/snow.mnoukhov/rl4lms",
            "image": "registry.console.elementai.com/snow.interactive_toolkit/default",
            "data": [
                "snow.mnoukhov.home:/home/toolkit",
                "snow.colab.public:/mnt/public",
            ],
            "environment_vars": [
                "HOME=/home/toolkit",
                f"HF_HOME=/home/toolkit/huggingface/",
                f"WANDB_API_KEY={wandb_api_key}",
                "WANDB_RESUME=allow",
                "WANDB__SERVICE_WAIT=300",
            ],
            "restartable": True,
            "resources": {
                "cpu": 4 * args.gpus,
                "mem": 48 * args.gpus,
                "gpu_mem": args.gpu_mem,
                "gpu": args.gpus,
            },
            "interactive": False,
            "bid": 9999,
        }
        job_scheduler = "toolkit"
    else:
        job_config = None
        job_scheduler = None

    # Run experiments and create results file
    hw.run_wizard(
        func=run_exp,
        exp_list=exp_list,
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        job_scheduler=job_scheduler,
        results_fname="results/notebook.ipynb",
        python_binary_path=args.python_binary,
        args=args,
        use_threads=True,
        save_logs=False,
        # exp_id=args.exp_id,
    )
