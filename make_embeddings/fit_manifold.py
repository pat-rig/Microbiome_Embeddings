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
import umap.umap_ as umap
import pickle
# self coded:
from make_glove_input import mk_glove_input 
from make_shell_script import shell_script, mk_shell_script


# Read jobID from .sbatch file for execution on TCML (cluster)
jobID = sys.argv[1]
# manually for local
jobID = 42

# define seeds
with open('../data/seeds.obj', mode='rb') as seed_file:
    seeds = pickle.load(seed_file)
    seed_file.close()

drawn_seeds = [0]
    
for seed in drawn_seeds:
    
    # save relatively large GloVe-inputs to (temporary) scratch folder
    # in order to avoid high network traffic on TCML
    scratch_dir = "/scratch/" + str(jobID) + "/"
    mk_glove_input(s=seed, destin_dir=scratch_dir, data_dir='../data',
                   job = jobID)
    # now we have the glove input
    
    # load into DataFrame
    filename = scratch_dir + "glove_input_" + str(seed)
    emb_train = pd.read_csv(filename, )

    # fit UMAP
    reducer = umap.UMAP(n_components = 15)