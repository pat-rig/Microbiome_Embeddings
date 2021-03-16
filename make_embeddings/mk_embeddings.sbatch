#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=MakeGloVeEmbeds

#resources:

#SBATCH --cpus-per-task=16
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=5G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:0
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=04:40:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=pakoehle@gmail.com
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####

cp /home/koehler/data/seqtab_filter.07.txt /scratch/$SLURM_JOB_ID/

####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####
date +%r

singularity exec /common/singularityImages/legacy/TCML-Cuda10_0Tensorflow1_13.simg python3 ~/make_embeddings/draw_embeddings.py $SLURM_JOB_ID 

date +%r

echo DONE!
