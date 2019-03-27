#!/bin/bash


#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1

#SBATCH -t 71:59:59
# sends mail when process begins, and
# when it ends. Make sure you define your email

#SBATCH --mail-type=end
#SBATCH --mail-user=xinweiy@princeton.edu
# Load anaconda3 environment
cd /home/xinweiy/github/Pytorch-UNet
module load anaconda3
python3 ./train.py -e 20 -m 1  -l 1e-2 -w 1 -c /home/xinweiy/github/checkpoints/Channel1086.pth