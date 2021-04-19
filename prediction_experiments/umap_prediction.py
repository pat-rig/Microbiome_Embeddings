#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Fit Manifold Embedding and Predict IBD Status
    
    Code Scaffold from PredictIBD.py
    
    import os
    os.chdir('/Users/patrick/Drive/13_sem/research_project/TEMP/prediction_experiments')
"""
import pickle
import importlib
import umap.umap_ as umap
import helper_predict as hf
import pandas as pd


# Only for one seed as opposed to all in PredictIBD.py
seed = 0

# supply directory structure with relative paths
data_dir = '../data/' # contains initial files from Tataru and David (2020)
emb_dir = data_dir + 'embeddings/'
test_dir = data_dir + 'test_samples/'
fasta_dir = data_dir + 'fasta/' 
otu_dir = data_dir + 'otu_objs/'


# define names of required files
emb_txt = emb_dir + 'glove_input_' + str(seed) + '_emb.txt'
emb_fasta = fasta_dir + 'asvs_' + str(seed) + '.fasta'
      
# load embedding matrix and corresponding nucleotide sequences of ASVs
# drops unknown sequences!
qual_vecs, embed_ids, embed_seqs = hf.getQualVecs(data_dir=data_dir,
                                                  embedding_txt=emb_txt,
                                                  embedding_fasta=emb_fasta)    

# load sample by ASV matrix and meta data (from saved python objects)
# separated by train and test
f = open(data_dir + otu_dir + "otu_train_" + str(seed) + ".obj", "rb")
otu_train = pickle.load(f)
f.close()

f = open(data_dir + otu_dir + "otu_test_" + str(seed) + ".obj", "rb")
otu_test = pickle.load(f)
f.close()

f = open(data_dir + otu_dir + "map_train_" + str(seed) + ".obj", "rb")
map_train = pickle.load(f)
f.close()

f = open(data_dir + otu_dir + "map_test_" + str(seed) + ".obj", "rb")
map_test = pickle.load(f)
f.close()

# relabel columns with sequence ids
otu_train = hf.matchOtuQual(otu_train, embed_ids, embed_seqs)
otu_test = hf.matchOtuQual(otu_test, embed_ids, embed_seqs)

# Reformat input for Random Forests
importlib.reload(hf)
target = "IBD"

# =============================================================================
# Input From Tataru and David (2020)
# =============================================================================
# load data
X_train_asin, X_val_asin, X_test_asin, y_train_asin, y_val_asin, y_test_asin, axes\
    = hf.getMlInput(otu_train, otu_test, map_train, map_test, target = target,
                    asinNormalized = True)
# X_train_asin = pd.concat([X_train_asin, X_val_asin], axis = 0)
# y_train_asin = y_train_asin + y_val_asin

# remove meta variables
# specifiy column labels to be dropped
drop_cols = X_train_asin.columns.values[:-14:-1]
# drop meta data
X_train_asin = X_train_asin.drop(drop_cols, axis=1)
# X_val_asin = X_val_asin.drop(drop_cols, axis=1)
X_test_asin = X_test_asin.drop(drop_cols, axis=1)

# Fit UMAP on asin-transformed data
# init Manifold object
reducer = umap.UMAP(n_components = 15, n_neighbors = 25)
embedded_points = reducer.fit_transform(X_train_asin)
embedded_points = pd.DataFrame(embedded_points)
# embed test points
X_test_asin_emb = reducer.transform(X_test_asin)
X_test_asin_emb = pd.DataFrame(X_test_asin_emb)
    
m_asin, auc_asin, auc_train_asin, fpr_asin, tpr_asin, prec_asin, f1_asin, f2_asin, _\
    = hf.predictIBD(embedded_points, y_train_asin, X_test_asin_emb, y_test_asin,
                    graph_title = "Normalized asinh Taxa Abundances in UMAP",
                    max_depth = 5, n_estimators = 170, weight = 20,
                    plot = False, plot_pr = False)


# =============================================================================
# Proper Input
# =============================================================================
# Load Full Training Data Frame and Test ids for this seed
test_file = '../data/test_samples/test_samples_0.txt'
test_ids = pd.read_csv(test_file, header = None, sep = '\t')

abundance_data = pd.read_csv('../data/seqtab_filter.07.txt', sep='\t',
                                 index_col=0)

# Split manually into train and test
train_points = abundance_data.drop(labels = list(test_ids.iloc[0, :]),
                                   axis = 'index')
test_points = abundance_data.loc[test_ids.iloc[0, :], :]

# fit UMAP
print(datetime.datetime.now())
reducer = umap.UMAP(n_components = 100, n_neighbors = 20)
embedded_points = pd.DataFrame(reducer.fit_transform(train_points))
print(datetime.datetime.now())
# 6mins

# embed test points
test_points = reducer.transform(test_points)

# prepare Prediction Experiment
# match ids from otu_train.index.values against train_points
# transform to err-notation
err2qid = pd.read_csv('../data/err-to-qid.txt', sep = '\t',
                      index_col = 1,header=
rf_train_ids = err2qid.loc[otu_train.index.values, :]

rf_train_ids.index.values

rf_train_points = train_points.loc[rf_train_ids.loc[:, 'run_accession'], :]
