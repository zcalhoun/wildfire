#!/bin/bash

#SBATCH --output=lemmatize_los_angeles.out

singularity exec -B /hpc/group/carlsonlab/zdc6/ ../wildfire-tweets.sif python lemmatize.py \
	-id /hpc/group/carlsonlab/zdc6/wildfire/data/tweets/los_angeles/ \
	-td /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
	-ad /hpc/group/carlsonlab/zdc6/wildfire/data/aqi/ \
	-state California \
	-city Los\ Angeles \
	-county Los\ Angeles

