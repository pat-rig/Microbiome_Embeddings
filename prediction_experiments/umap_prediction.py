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
import datetime as dt
print('Start:' + str(dt.datetime.now()))
reducer = umap.UMAP(n_components = 100)
embedded_points = reducer.fit_transform(X_train_asin)
print('End:' + str(dt.datetime.now()))

# embed test points

    
m_asin, auc_asin, auc_train_asin, fpr_asin, tpr_asin, prec_asin, f1_asin, f2_asin, _\
    = hf.predictIBD(X_train_asin, y_train_asin, X_test_asin, y_test_asin,
                    graph_title = "Normalized asinh Taxa Abundances " + str(X_train.shape[1]) + " features",
                    max_depth = 5, n_estimators = 170, weight = 20,
                    plot = False, plot_pr = False)

    