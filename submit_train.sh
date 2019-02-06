#!/bin/bash


#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1

#SBATCH -t 23:59:59
# sends mail when process begins, and
# when it ends. Make sure you define your email

#SBATCH --mail-type=end
#SBATCH --mail-user=xinweiy@princeton.edu
# Load anaconda3 environment
cd /home/xinweiy/github/Pytorch-UNet
module load anaconda3
python3 ./train.py -e 40 -m 1