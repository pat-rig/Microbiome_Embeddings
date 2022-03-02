#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    EVALUATE PREDICTION EXPERIMENTS 

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
forests = result['forests']
# separate per algorithm: [0]: glove, [1]: PCA, [2]: raw data
forests_by_algo = [forests[i::3] for i in range(3)]

# retreive Feature Importances
feat_imps = [forests_by_algo[i][j].feature_importances_ for i in range(3) for j in range(len(forests_by_algo[i]))]
# has 81 entries. First 27 for glove, second for PCA, third for raw data

# retreive seeds
seeds = result['performance'].loc[:,'seed'].unique()
# Collect seqs of ASVs present in subsample corresponding to seed


# =============================================================================
# Unit Test
# =============================================================================
# Get importances and and embedding matrices per run
pca_imp = feat_imps[27:54:1]
pca_embedding_matrices = result['pca_embeddings']
seqs_per_subsample = result['emb_seqs']

# observe that the number of ASVs per subsample vary!
no_asvs_per_run = [len(x) for x in seqs_per_subsample]
any(np.array(no_asvs_per_run) != no_asvs_per_run[0])
# True

# check if they contain different sequences at the same index
len_a = len(seqs_per_subsample[0])
unequal = 0
for a, b in zip(seqs_per_subsample[0], seqs_per_subsample[1][:len_a:]):
    unequal += int(a != b)
print(unequal)
# 26446

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
    pca_most_imp_idx = np.argsort(imp)[:0:-1][:n_imp+1:1]
    
    # select embedding matrix for current subsample
    E = pca_embedding_matrices[j] # 100 rows
    
    # inspect ASV-weights in most important features (rows of E)
    imp_pcs = E[pca_most_imp_idx, :]
    sorted_weights = np.argsort(np.abs(imp_pcs), axis=1)

    # Pick top10 ASVs per important principal axis
    n_asv = 10
    # reverse order in sorted_weights!
    l = sorted_weights.shape[1]
    imp_asv_per_pc_idx = [imp_pc[:l-n_asv-1:-1] for imp_pc in sorted_weights] 
    # <-- contains indices of 10 highest scoring asvs for 10 most important PCs
    # for the currently investigated seed!
    
    # unlist and tabulate
    high_scoring_asv_idx = np.array(imp_asv_per_pc_idx).flatten()
    # count sequences instead of indices, since the latter is not comparable
    # across subsamples
    for idx in high_scoring_asv_idx:
        seq_table_raw_data.append(seqs_per_subsample[j][idx])
    
    
pca_table = np.unique(seq_table_raw_data, return_counts=True)
# First glance at the results 
# Plot distribution of frequencies <-- what is a reasonable threshold?
pd.DataFrame(pca_table[1]).hist()
# Many only appear in one or two important features
# only a few appear 17+ times and some more around 7+times
# how many?
sorted_frequencies = np.sort(pca_table[1])[:0:-1]
no_sign_asvs_pca = np.sum(pca_table[1] > 60) #being conservative here with 7
# 9

# look at top ten sequences explicitly
pca_table[0][np.argsort(pca_table[1])[:0:-1][:10:]]

# =============================================================================
# Raw Data
# =============================================================================
# Guiding Question: Are there very important ASVs for the raw forests which
# do not appear among the most important ASVs in PCA --> why?
raw_imp = feat_imps[54::1]
# explore at distribution of importance scores
pd.DataFrame(raw_imp[0]).hist()
raw_table_raw = []

for imp, j in zip(raw_imp, range(len(raw_imp))):
    # get indices of most important features
    n_imp = 100
    # idx of most important features
    raw_most_imp_idx = np.argsort(imp)[:0:-1][:n_imp:1]
    # extract sequences
    imp_seqs = np.array(seqs_per_subsample[j])[raw_most_imp_idx]
    # append to raw data for table
    raw_table_raw.append(imp_seqs)

raw_table = np.unique(raw_table_raw, return_counts=True)
# explore results
pd.DataFrame(raw_table[1]).hist()
# qualitatively equal to pca plot
# how often 

# !!! Not comparable quantitatively, since one ASV could be important for multiple PCs
# how many above 17?
no_sign_asvs_raw = np.sum(raw_table[1] > 17)
# 13


# Interpretaion of those numbers comparing the tables
# Find ASVs important for raw but not important for pca

# sort frequencies decreasingly
sort_raw_idx = np.argsort(raw_table[1])[:0:-1]
sort_pca_idx = np.argsort(pca_table[1])[:0:-1]
# collect important sequences 
sign_asvs_raw = raw_table[0][sort_raw_idx[:no_sign_asvs_raw:1]]
sign_asvs_pca = pca_table[0][sort_pca_idx[:no_sign_asvs_pca:1]]
# how many seqs important for raw not in pca?
np.sum([sign_asvs_raw[i] in sign_asvs_pca for i in range(len(sign_asvs_raw))])
# 0
# ASVs that are most important for the raw-forests are not considered important
# for the PC forests (even if no_sign_asvs_raw threshold is lowered to 7).

# collect abundances of respective asvs.
# save asvs in separate object and continue on machine with large memory
# in backtrace_signal.py
table_dict = {'pca': pca_table, 'raw': raw_table,
              'sign_asvs': {'pca': sign_asvs_pca, 'raw': sign_asvs_raw}}
with open('imp_asv_tables.obj', mode='wb') as asv_file:
    pickle.dump(table_dict, asv_file)
    asv_file.close()

