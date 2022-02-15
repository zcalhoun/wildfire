#!/bin/bash
singularity run ../wildfire-tweets.sif
python3 lda.py
