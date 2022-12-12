#!/bin/bash
#SBATCH -p carlsonlab-gpu 
#SBATCH --account=carlsonlab
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --output=vae.out

singularity exec ../wildfire-tweets.sif python vae.py
