#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    EVALUATE PREDICTION EXPERIMENTS
    
    1. Inspect Feature Importance Scores
    
    import os
    os.chdir('/Users/patrick/Drive/13_sem/research_project/TEMP/prediction_experiments')
"""
import numpy as np
import pickle
import helper_predict as hf
import matplotlib as plt
import pandas as pd

# =============================================================================
# Prepare Data
# =============================================================================


# supply directory structure with relative paths
data_dir = '../data/' 
emb_dir = data_dir + 'embeddings/'
fasta_dir = data_dir + 'fasta/'
otu_dir = data_dir + 'otu_objs/'


# perform analysis with or without meta data as predictors?
meta = False

# load results from prediction experiments
results_obj = 'prediction_results_meta=' + str(meta) + '.obj'
with open(results_obj, mode='rb') as results_file:
    result = pickle.load(results_file)
    results_file.close()
    
# retreive RandomForest Objects
forests = result[1]
# separate per algorithm: [0]: glove, [1]: PCA, [2]: raw data
forests_by_algo = [forests[i::3] for i in range(3)]

# retreive Feature Importances
feat_imps = [forests_by_algo[i][j].feature_importances_ for i in range(3) for j in range(len(forests_by_algo[i]))]
# has 81 entries. First 27 for glove, second for PCA, third for raw data

# retreive seeds
seeds = result[0].loc[:,'seed'].unique()
# Collect seqs of ASVs present in subsample corresponding to seed

# =============================================================================
# Only execute on cluster
# =============================================================================
wd_id = os.getcwd().split('/')[1] # identifier to on which machine we are

if wd_id != 'Users':
    seqs_per_subsample = []
    # Collect from otu_train_$SEED.objs
    for seed in seeds:
        # load training data corresponding to that seed
        otu_file = otu_dir + 'otu_train_' + str(seed) + '.obj'
        with open(otu_file, mode='rb') as training_data:
            df = pickle.load(training_data)
            seqs_per_subsample.append(df.columns.values) # already in proper order
            training_data.close()
    
    # save object
    with open('seqs_per_subsample.obj', mode='wb') as seqfile:
        pickle.dump(seqs_per_subsample, seqfile)     
        seqfile.close()
else:
    # load if on local machine and object already exists
    with open('seqs_per_subsample.obj', mode='rb') as seqfile:
        seqs_per_subsample = pickle.load(seqfile)
        seqfile.close()
    
        
# older version:
# =============================================================================
# # define paths to renamed embeddings
# path_to_ids = '../data/embeddings/'
# # collects sequences from glove output
# for seed in seeds:
#     # load renamed glove embedding matrix
#     path_to_file = path_to_ids + 'glove_input_' + str(seed) + '_emb.txt'
#     odd_df = pd.read_csv(path_to_file) 
#     # not appropriately formatted but sufficient
#     
#     # save ids for the current subsample here
#     seqs_current = []
#     for i in range(odd_df.shape[0]):
#         # extract and store string marking the sequence 
#         seq = np.array(odd_df.iloc[i, :])[0].split(' ')[0]
#         seqs_current.append(seq)
#     
#     # save all ASVs appearing in one subsample in one list entry
#     seqs_per_subsample.append(seqs_current)
#     # clean up
#     del(odd_df)
# 
# =============================================================================

# =============================================================================
# Unit Test
# =============================================================================
# Get importances and and embedding matrices per run
pca_imp = feat_imps[27:54:1]
pca_embedding_matrices = result[3]

# observe that the number of ASVs per subsample do not vary!
no_asvs_per_run = [len(x) for x in seqs_per_subsample]
any(np.array(no_asvs_per_run) != no_asvs_per_run[0])
# False

# check if they contain different sequences at the same index
len_a = len(seqs_per_subsample[0])
unequal = 0
for a, b in zip(seqs_per_subsample[0], seqs_per_subsample[1][:len_a:]):
    unequal += int(a != b)
print(unequal)
# 0
# 
# => use one entry of seqs_per_subsample for indexing
seqs = seqs_per_subsample[0]

# =============================================================================
# Analyze PCA
# =============================================================================
seq_table_raw_data = []

# treat subsamples individually and aggregate interpretation in the end
for imp, j in zip(pca_imp, range(len(pca_imp))):
    
    # how many features to inspect? fix arbitrarily for now.
    # justify through histograms of PC scores later
    n_imp = 10
    # idx of most important features
    pca_most_imp_idx = np.argsort(imp)[:0:-1][:n_imp:1]
    
    # select embedding matrix for current subsample
    E = pca_embedding_matrices[j] # 100 rows
    
    # inspect ASV-weights in most important features (rows of E)
    imp_pcs = E[pca_most_imp_idx, :]
    sorted_weights = np.argsort(np.abs(imp_pcs), axis=1)[:0:-1]
    # Pick top10 ASVs per important principal axis
    n_asv = 10
    imp_asv_per_pc_idx = [imp_pc[:n_asv] for imp_pc in sorted_weights] 
    # <-- contains indices of 10 highest scoring asvs for 10 most important PCs
    # for the currently investigated seed!
    
    # unlist and tabulate
    high_scoring_asv_idx = np.array(imp_asv_per_pc_idx).flatten()
    # count sequences instead of indices, since the latter is not comparable
    # across subsamples
    for idx in high_scoring_asv_idx:
        seq_table_raw_data.append(seqs[idx])
    
    
pca_table = np.unique(seq_table_raw_data, return_counts=True)
# First glance at the results 
np.sort(pca_table[1])[:0:-1][:5]
# array([99, 11, 10, 10, 10])

# problematic: '<unk>' sequence(s) achieve most often a high score
pca_table[0][0]
# '<unk>'


    
        
# =============================================================================
# GloVe
# =============================================================================

# glove from /data/embeddings
# save embedding matrices here
# =============================================================================
# glove_embedding_matrices = []
# 
# 
# # load seeds (identifiers for individual files)
# with open(data_dir + 'seeds.obj', mode='rb') as seedfile:
#     seeds = pickle.load(seedfile)
#     seedfile.close()
# 
# for seed in seeds:
#     # define filenames
#     emb_txt = emb_dir + 'glove_input_' + str(seed) + '_emb.txt'
#     emb_fasta = fasta_dir + 'asvs_' + str(seed) + '.fasta'
#     
#     # load embedding matrix and corresponding nucleotide sequences of ASVs
#     qual_vecs, embed_ids, embed_seqs = hf.getQualVecs(data_dir=data_dir,
#                                                           embedding_txt=emb_txt,
#                                                           embedding_fasta=emb_fasta)
#     
#     glove_embedding_matrices.append(qual_vecs)
# 
# # Compare ASVs associated with most important features
# # how many features to inspect
# n_imp = 5
# =============================================================================

