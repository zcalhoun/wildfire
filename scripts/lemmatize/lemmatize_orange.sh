#!/bin/bash

#SBATCH --output=lemmatize_orange.out

singularity exec -B /hpc/group/carlsonlab/zdc6/ ../wildfire-tweets.sif python lemmatize.py \
	-id /hpc/group/carlsonlab/zdc6/wildfire/data/tweets/orange/ \
	-td /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
	-ad /hpc/group/carlsonlab/zdc6/wildfire/data/aqi/ \
	-state California \
	-city Orange \
	-county Orange 
