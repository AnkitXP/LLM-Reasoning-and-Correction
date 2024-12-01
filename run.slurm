#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Run_Main       #Set the job name
#SBATCH --time=10:00:00             #Set the wall clock limit
#SBATCH --nodes=1                 #Request 1 node
#SBATCH --ntasks-per-node=4       #Request 8 cores per node
#SBATCH --mem=64Gb                 #Request 8GB per node 
#SBATCH --output=logs/main_out.%j      #Send stdout/err
#SBATCH --gres=gpu:a100:1              #Request 1 GPU per node
#SBATCH --partition=gpu           #Request the GPU queue

ml purge
ml Miniconda3

cd $SCRATCH/CSCE642/LLM-Reasoning-and-Correction/code

source activate llm-self-correct

python main.py --task train