#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DRAW MULTIPLE RANDOM SUBSETS OF seqtab_filter.07.txt AND FIT GLOVE EMBEDDINGS

SAVE FOLDER WITH
    (1) TEST-SAMPLE_IDs
    (2) EMBEDDING MATRICES
    (3) SEEDS USED TO SPLIT TRAIN AND TEST DATA
"""
import os
import sys
import random
# self coded:
from make_glove_input import mk_glove_input 
from make_shell_script import shell_script, mk_shell_script


# Read jobID from .sbatch file for execution on TCML (cluster)
jobID = sys.argv[1]

# Define number of replications
R = 3

# Draw seeds
drawn_seeds = [random.randint(0, 1e6) for i in range(R)]
  
     
for seed in drawn_seeds:
    
    # save relatively large GloVe-inputs to (temporary) scratch folder
    # in order to avoid high network traffic on TCML
    scratch_dir = "/scratch/" + str(jobID) + "/"
    mk_glove_input(s=seed, destin_dir=scratch_dir, data_dir='../data',
                   job = jobID)
    # now we have the glove input
    
    # 1. create shell script for each subsample
    filename = scratch_dir + "glove_input_" + str(seed)
    file_content = shell_script(filename)
    script_name = "runGlove_" + str(seed) + ".sh"
    script_path = scratch_dir + script_name
    mk_shell_script(script_path, file_content)
    
    # 2. call shell script via bash runGlove_$SEED
    shell_command = "bash " + script_path
    os.system(shell_command)
   
