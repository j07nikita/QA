#!/bin/bash
#SBATCH --account=NLP
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --time=40-00:00:00
#SBATCH --mail-type=ALL

module add cuda/8.0
module add cudnn/7-cuda-8.0

python3 scripts/reader/preprocess.py data/datasets data/datasets --split train-v1.1 --workers 2


