#!/bin/bash
#SBATCH --partition=long                          # Ask for unkillable job
#SBATCH --cpus-per-task=24                                # Ask for 2 CPU
#SBATCH --gres=gpu:v100:4                                  # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:59:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/s/sophie.xhonneux/michael-trl/slurm-%j.out  # Write the log on scratch


module load python/3.10

source /home/mila/s/sophie.xhonneux/scratch/envs/michael_trl/bin/activate

cd /home/mila/s/sophie.xhonneux/projects/trl/examples/summarization

export WANDB_PROJECT=trl
export WANDB_ENTITY=mila-language-drift
export WANDB_NAME=pythia1b_sft_hh_rlhf
export WANDB_MODE=online

python run_on_mila.py -e configs/dpo_pythia1b_hh_rlhf.yml --wandb \
    --savedir_base /home/mila/s/sophie.xhonneux/scratch/michael-trl/hh_rlhf/${SLURM_JOB_ID} \
    -n 4

