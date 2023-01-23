#!/bin/bash
#SBATCH --time=3-12:00:00
#SBATCH --account=def-kharches
#SBATCH --mem=1024G
#SBATCH --cpus-per-task=22

# This is for Narval.
module load python/3.9
module load scipy-stack/2022a
module load StdEnv/2020
module load scipy-stack
# module load pandas/1.4.0 # pandas is already in python/3.9, I think. Else you may need to spider to find out.
#
parallel -j 22 < jobfile
