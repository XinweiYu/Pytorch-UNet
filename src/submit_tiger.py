#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:52:07 2019
this code is to submit job in tigergpu.
@author: yxw
"""
import os

job_directory = './job_submit'
# input_list = list()

job_file = os.path.join(job_directory, 'submit.job')
with open(job_file, "w") as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines("#SBATCH -N 1 # node count\n")
    fh.writelines("#SBATCH --ntasks-per-node=1\n")
    fh.writelines("#SBATCH --ntasks-per-socket=1\n")
    fh.writelines("#SBATCH --cpus-per-task=2\n")
    fh.writelines("#SBATCH --gres=gpu:1\n")
    fh.writelines("#SBATCH -t 23:59:59\n")
    fh.writelines("# sends mail when process begins, and\n")
    fh.writelines("#when it ends. Make sure you define your email\n")
    fh.writelines("#SBATCH --mail-type=end\n")
    fh.writelines("#SBATCH --mail-user=xinweiy@princeton.edu\n")
    fh.writelines("# Load anaconda3 environment\n")
    fh.writelines("source ./venv/bin/activate\n")
    #fh.writelines("python ./N2S_Multiscale.py\n")
    fh.writelines("python ./NoisyGAN.py\n")
    #fh.writelines("python ./Noisy_Prob.py\n")
    fh.writelines("deactivate\n")
    fh.close()
os.system("sbatch %s" % job_file)
