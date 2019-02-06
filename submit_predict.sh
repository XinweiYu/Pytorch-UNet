#!/bin/bash

#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1

#SBATCH -t 0:59:59
# sends mail when process begins, and
# when it ends. Make sure you define your email

#SBATCH --mail-type=end
#SBATCH --mail-user=xinweiy@princeton.edu
# Load anaconda3 environment
cd /home/xinweiy/github/Pytorch-UNet
module load anaconda3
python3 ./predict.py -i /scratch/network/xinweiy/data/train/106.jpg -o /scratch/network/xinweiy/output/106_mask.jpg -m /home/xinweiy/github/checkpoints/direction.pth