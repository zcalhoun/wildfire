#!/bin/bash

#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=lemmatize_dallas.out
#SBATCH -p common
#SBATCH --mem-per-cpu=4GB
#SBATCH --array 0-500

singularity exec -B /hpc/group/carlsonlab/zdc6/ ../../wildfire-tweets.sif python lemmatize_array.py \
	-id /hpc/group/carlsonlab/zdc6/wildfire/data/tweets/dallas/ \
	-td /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
	-ad /hpc/group/carlsonlab/zdc6/wildfire/data/aqi/ \
	-state Texas \
	-city Dallas \
	-county Dallas 
