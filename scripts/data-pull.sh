#!/bin/bash

#SBATCH --output=data-pull-slurm.out

singularity exec -B /hpc/group/carlsonlab/zdc6/ ../wildfire-tweets.sif python NMF-PFA.py \
	--lemmatized_path /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized \
	--target_path /hpc/group/carlsonlab/zdc6/wildfire/data/count_vectorized/1000_10_by_file \
	--sample_method by_file \
	--tweet_agg_num 1000 \
	--tweet_sample_count 10 \
	--min_df 300 \
	--max_df 0.01 \
	--train_cities portland los\ angeles phoenix san\ francisco raleigh dallas chicago \
	--test_cities seattle orange new\ york
