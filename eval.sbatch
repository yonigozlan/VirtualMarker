#!/usr/bin/bash
#SBATCH --job-name=VM_RICH
#SBATCH --output=VM_RICH.out
#SBATCH --error=VM_RICH.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1

python3 main/eval.py


