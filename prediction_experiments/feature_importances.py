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

# define paths to renamed embeddings
path_to_ids = '../data/embeddings/'


# perform analysis with or without meta data as predictors?
meta = False


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
# Retreive ids of ASVs present in subsample corresponding to seed
seqs_per_subsample = []
for seed in seeds:
    # load renamed glove embedding matrix
    path_to_file = path_to_ids + 'glove_input_' + str(seed) + '_emb.txt'
    odd_df = pd.read_csv(path_to_file) 
    # not appropriately formatted but sufficient
    
    # save ids for the current subsample here
    seqs_current = []
    for i in range(odd_df.shape[0]):
        # extract and store string marking the sequence 
        seq = np.array(odd_df.iloc[i, :])[0].split(' ')[0]
        seqs_current.append(seq)
    
    # save all ASVs appearing in one subsample in one list entry
    seqs_per_subsample.append(seqs_current)
    # clean up
    del(odd_df)


# =============================================================================
# Unit Test
# =============================================================================
# observe that the number of ASVs in one subsample vary!
[len(x) for x in seqs_per_subsample]
# check if they contain different sequences at the same index
len_a = len(seqs_per_subsample[0])
unequal = 0
for a, b in zip(seqs_per_subsample[0], seqs_per_subsample[1][:len_a:]):
    unequal += int(a != b)
print(unequal)
# 26447 


# =============================================================================
# Analyze PCA
# =============================================================================
pca_imp = feat_imps[27:54:1]
# load embedding matrices and ids of ASVs involved
pca_embedding_matrices = result[3]
seq_table_raw_data = []

# treat subsamples individually and aggregate interpretation in the end
for imp, j in zip(pca_imp, range(len(pca_imp))):
    
    # how many features to inspect? fix arbitrarily for now.
    n_imp = 10
    # idx of most important features
    pca_most_imp_idx = np.argsort(imp)[:0:-1][:n_imp:1]
    
    # select embedding matrix and corresponing sequences for current subsample
    E = pca_embedding_matrices[j] # 100 rows
    seqs = seqs_per_subsample[j]
    
    # inspect ASV-weights in most important features (rows of the trafo matrix)
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


    
        




    
    
    frpca_feature_importances = feat_imps[27]
    frpca_imp_feats_idx = np.argsort(frpca_feature_importances)[:0:-1][:5]
    # look at respective columns
    frpca_emb_matrix = pca_embedding_matrices[0]
    imp_axes = frpca_emb_matrix[frpca_imp_feats_idx]
    
    # look at histograms of scores --> anything standing out?
    imp_axes_df = pd.DataFrame(imp_axes).transpose()
    imp_axes_df.hist(bins=100, range=(-0.0001, 0.0001))
    # all scores very close to zero.
    # !vectors are normalized to length one in 27k dimensional space
    
    # retreive highest scoring ASVs from the most important columns
    sorted_imp_df = np.argsort(imp_axes_df, axis=0)
    highest_scoring_asvs_per_dim = np.argsort(imp_axes_df, axis = 0).iloc[0:5,:]
    highest_scoring_asvs = np.array(highest_scoring_asvs_per_dim).flatten()
    # these asvs are among the five asvs which have the highest score within the 
    # five most important features to determine IBD status
    
    # count frequencies
    np.unique(highest_scoring_asvs, return_counts=True)
    # (array([   1,    4,    8,   13,   27,   64,   78,   99,  188,  332,  380, 433,  557,  564,  658,  890, 1473, 1606]),
    # array([    1,    2,    4,    1,     2,   1,    1,    1,    1,    1,    1,   2,    2,    1,    1,    1,    1,   1]))
    
    # generalize to all coordinate systems
    # individual vs. mean approach <-- requires ID matching





mean_pca_embedding_matrix = np.mean(pca_embedding_tensor, axis=2)
# inspect shape!
print(mean_pca_embedding_matrix.shape)

# extract the means of the most important columns
pca_most_imp_axes = mean_pca_embedding_matrix[:, pca_most_imp_idx]

# =============================================================================
# GloVe
# =============================================================================

# glove from /data/embeddings
# save embedding matrices here
glove_embedding_matrices = []

# supply directory structure with relative paths
data_dir = '../data/' 
emb_dir = data_dir + 'embeddings/'
fasta_dir = data_dir + 'fasta/'
otu_dir = data_dir + 'otu_objs/'

# load seeds (identifiers for individual files)
with open(data_dir + 'seeds.obj', mode='rb') as seedfile:
    seeds = pickle.load(seedfile)
    seedfile.close()

for seed in seeds:
    # define filenames
    emb_txt = emb_dir + 'glove_input_' + str(seed) + '_emb.txt'
    emb_fasta = fasta_dir + 'asvs_' + str(seed) + '.fasta'
    
    # load embedding matrix and corresponding nucleotide sequences of ASVs
    qual_vecs, embed_ids, embed_seqs = hf.getQualVecs(data_dir=data_dir,
                                                          embedding_txt=emb_txt,
                                                          embedding_fasta=emb_fasta)
    
    glove_embedding_matrices.append(qual_vecs)

# Compare ASVs associated with most important features
# how many features to inspect
n_imp = 5





# =============================================================================
# First Approach
# =============================================================================
# compute mean feature importance for PCA
pca_imp = np.mean(feat_imps[27:54:1], axis=0)
# more elegantly?
# [np.nanmean(feat_imps[0 + (i-1) * no_reps : i * no_reps : 1], axis = 0) for i in range(3)]

    # average over all subsamples
pca_embedding_tensor = np.dstack(pca_embedding_matrices)
    # DOES NOT WORK SINCE TRANSFORMATION MATRICES HAVE DIFFERENT DIMENSIONS
    # POSTPROCESSING STEP NECESSARY
    # 1. init large matrix containing all possible microbes
    # 2. get ids of microbes invovled in this subsample
    # 3. insert embedding matrix elements into full matrix 
    # 4. add counter st we know how often a certain microbe appeared
    # 5. sum up tensor along axes 2 and divide by matrix from (4.)
# equivalent not reasonable for glove since columns are interchangeable!
#
# require further postprocessing for raw
# raw_imp = np.mean(feat_imps[54::1], axis=0)
# ValueError: operands could not be broadcast together with shapes (26627,) (26640,) 
# guess on cause: 
# Since forests are trained on different subsets of the data, some forests
# use variables (ASVs) which are not used by others.

# Fixes
# Per Forest: Map variable to ASV name
# (1)
# --> create new array with appropriate encoding for non-existing features (not zero!!)
#       IMPORTANT TO CHECK BEFOREHAND WHETHER THE ESITMATES ARE NA!
# (2)
#     analyze individually and summarize analysis in the end


