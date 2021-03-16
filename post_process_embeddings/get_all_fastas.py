#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Get all .fasta s
"""
import os
import pickle

# Load all seeds
with open('../data/seeds.obj', mode='rb') as seedfile:
    seeds = pickle.load(seedfile)
    seedfile.close()

# Formulate generalization of fasta generation
# specify path to data directory
path2data = '../data'
for seed in seeds:
    command = 'Rscript label_qualvec_transform_mat_generic.R embeddings/glove_input_' + str(seed) + '_emb.txt renamed_embeddings/renamed_emb_' + str(seed) + '.txt fasta/asvs_' + str(seed) + '.fasta ' + path2data
    os.system(command)