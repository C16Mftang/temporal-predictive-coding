#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=4:00:00
# set name of job
#SBATCH --job-name=strf
# partition
#SBATCH --partition=short
# set gpu 
#SBATCH --gres=gpu:1
# qos
#SBATCH --qos=basic
#SBATCH --account=ndcn-computational-neuroscience
# change the location of the .out file
#SBATCH --output=/data/ndcn-computational-neuroscience/scro4155/temporal-pc/output/%j.out
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=mufeng.tang@bndu.ox.ac.uk

# Specifying virtual envs
module load Anaconda3
source activate $DATA/temporalenv

# run the application
python scripts/strf.py --datapath 'nat_data' --train-size 20000 --batch-size 2000 --hidden-size 320 --sparsez 0.5 --learn-lr 5e-3 --learn-iters 400 --inf-lr 5e-2 --inf-iters 50 --sparseWout 1.5 --sparseWr 1.0 --lr-decay-step 10 --lr-decay-rate 0.8