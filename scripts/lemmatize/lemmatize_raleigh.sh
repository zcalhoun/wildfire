#!/bin/bash

#SBATCH --output=lemmatize_raleigh.out

singularity exec -B /hpc/group/carlsonlab/zdc6/ ../wildfire-tweets.sif python lemmatize.py \
	-id /hpc/group/carlsonlab/zdc6/wildfire/data/tweets/raleigh/ \
	-td /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
	-ad /hpc/group/carlsonlab/zdc6/wildfire/data/aqi/ \
	-state North\ Carolina \
	-city Raleigh \
	-county Wake 
