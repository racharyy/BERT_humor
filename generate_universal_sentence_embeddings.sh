#!/bin/bash
#SBATCH -o /scratch/mhasan8/output/humor_out.%j.txt -t 1-00:00:00
#SBATCH -c 2
#SBATCH --mem=28gb
#SBATCH -J kamrul
#SBATCH -p standard
#SBATCH -a 201-332

module load anaconda3/5.3.0b
module load git
module load tensorflow/2.0.0b 


python /scratch/mhasan8/TedEX/BERT_humor/universal_sentence_encoder.py $SLURM_ARRAY_TASK_ID
