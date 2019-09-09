#!/bin/bash
#SBATCH --account=NLP
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=14
#SBATCH --time=40-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem=63G
module add cuda/8.0
module add cudnn/7-cuda-8.0

python3 scripts/reader/predict.py --model models/model1.mdl data/datasets/dev-v1.1.json --out-dir . --gpu 2


