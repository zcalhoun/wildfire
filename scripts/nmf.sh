#!/bin/bash

#SBATCH --output=nmf.out

singularity exec -B /hpc/group/carlsonlab/zdc6/ ../wildfire-tweets.sif python nmf.py
