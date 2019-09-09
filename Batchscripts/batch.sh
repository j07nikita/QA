#!/bin/bash
#SBATCH --account=NLP
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --time=40-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem=50G
module add cuda/8.0
module add cudnn/7-cuda-8.0

python3 scripts/reader/train.py --embedding-file glove.840B.300d.txt --tune-partial 1000 --gpu 2


