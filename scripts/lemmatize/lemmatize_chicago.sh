#!/bin/bash

#SBATCH --output=lemmatize_dallas.out

singularity exec -B /hpc/group/carlsonlab/zdc6/ ../wildfire-tweets.sif python lemmatize.py \
	-id /hpc/group/carlsonlab/zdc6/wildfire/data/tweets/chicago/ \
	-td /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
	-ad /hpc/group/carlsonlab/zdc6/wildfire/data/aqi/ \
	-state Illinois \
	-city Chicago \
	-county Cook
