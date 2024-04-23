#!/bin/bash

#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 8-00:00:00
#SBATCH --mem=50g
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o jobout
#SBATCH -e joberr

module load anaconda/5.2.0
module load cuda/11.3
source activate pytorchDL_1_12
python mask_learning.py

