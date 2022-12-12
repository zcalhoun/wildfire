#!/bin/bash
#SBATCH --output=lemmatize_portland.out

singularity exec -B /hpc/group/carlsonlab/zdc6/ ../wildfire-tweets.sif ./lemmatize_portland.sh
