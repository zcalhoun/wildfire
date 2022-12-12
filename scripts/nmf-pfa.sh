#!/bin/bash

#SBATCH --output=nmf-pfa.out
#SBATCH --err=err-nmf-pfa.err
#SBATCH --mem=64G
#SBATCH -p gpu-common --gres=gpu:1
#SBATCH -c 6

singularity exec -B /hpc/group/carlsonlab/zdc6/ ../wildfire-tweets.sif python NMF-PFA.py
