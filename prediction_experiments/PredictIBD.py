#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    COMPARE VARIANCE IN PREDICTION EXPERIMENTS BY TATARU AND DAVID (2020)
    
    INPUTS (FOR EACH SEED):
        1. GloVe Embedding Matrix
        2. Test Sample File
        3. Raw Data (For PCA, Raw Data and UMAP)
    
    OUTPUT:
        pd.DataFrame with cols: algorithm, seed, AUROC, AUPRC, F1, ...
                          rows: different subsamples
        
    WORKFLOW:
        1. Assess only Variance in GloVe 
        
    SCAFFOLD:
        
    NOTES:
        
        working directory:
    '/Users/patrick/Drive/13_sem/research_project/TEMP/prediction_experiments'
        
"""
import sys
import os
import pandas as pd 
import helper_predict as hf
import pickle
import importlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn

# translate argument if executed from command line
meta_argument = sys.argv[1].split('=')[-1]
if meta_argument == 'True':
    meta = True
else:
    meta = False
    
# save performance metrics here
performance = pd.DataFrame(columns = ['algo', 'seed', 'auc', 'precision',
                                      'f1', 'f2'])

# save models here
forests = []
# save false positive and true positive rates here
confusion = []
# save principal axes here
pca_embeddings = []

# supply directory structure with relative paths
data_dir = '../data/' # contains initial files from Tataru and David (2020)
emb_dir = data_dir + 'embeddings/'
test_dir = data_dir + 'test_samples/'
fasta_dir = data_dir + 'fasta/' 
otu_dir = data_dir + 'otu_objs/'

# get names of files to be used
# load seeds (identifiers for individual files)
with open(data_dir + 'seeds.obj', mode='rb') as seedfile:
    seeds = pickle.load(seedfile)
    seedfile.close()
  
# Iterate over all resamplings (identified by seed)
for seed in seeds:    
    
    # ensure that necessary otu-file exists
    otu_test_file = 'otu_test_' + str(seed) + '.obj'
    files_in_otu_dir = os.listdir(otu_dir)
    # skip this seed if it doesn't exist
    if not (otu_test_file in files_in_otu_dir):
        continue
    
    # define names of required files
    emb_txt = emb_dir + 'glove_input_' + str(seed) + '_emb.txt'
    emb_fasta = fasta_dir + 'asvs_' + str(seed) + '.fasta'
          
    # load embedding matrix and corresponding nucleotide sequences of ASVs
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
    
    # GloVe Embedding
    X_train, X_val, X_test, y_train, y_val, y_test, axes =  hf.getMlInput(otu_train,
                                                                   otu_test,
                                                                   map_train,
                                                                   map_test,
                                                                   target = target,
                                                                   embed = True,
                                                                   qual_vecs = qual_vecs)
    # remove meta variable if specified by argument
    if not meta:
        # specifiy column labels to be dropped
        drop_cols = X_train.columns.values[100::1]
        # drop meta data
        X_train = X_train.drop(drop_cols, axis=1)
        X_val = X_val.drop(drop_cols, axis=1)
        X_test = X_test.drop(drop_cols, axis=1)
        
    
    X_train = pd.concat([X_train, X_val], axis = 0)
    y_train = y_train + y_val
    
    # todo:
    # optimize hyper params over validation set, if not meta
    # GridSearch RF: Train on X_train and Predict on X_val
    #       => choose hyper pars with highest AUC/AUPR on X_val
    
    
    m_embed, auc_embed, auc_train_embed, fpr_embed, tpr_embed, prec_embed, f1_embed, f2_embed, _ = hf.predictIBD(X_train, y_train,
                                                                  X_test, y_test,
                                                                  graph_title = "Embedding weighted by averaging taxa "+ str(X_train.shape[1]) + " features",
                                                                  max_depth = 5,
                                                                  n_estimators = 95,
                                                                  weight = 20,
                                                                  plot = False,
                                                                  plot_pr = False)
    # monitor progress on console
    print(auc_embed)
    print(prec_embed)
    
    row_embed = pd.DataFrame({'algo': ['glove'], 'seed': [seed], 'auc': [auc_embed],
                        'precision': [prec_embed], 'f1': [f1_embed], 
                        'f2': [f2_embed]})
    
    # save performance metrics in data frame
    performance = performance.append(row_embed)
    # save forest
    forests.append(m_embed)
    # save fpr and tpr
    confusion.append([fpr_embed, tpr_embed])


    
    # PCA Embedding
    X_train_pca, X_val_pca, X_test_pca, y_train_pca, y_val_pca, y_test_pca, principal_axes  = hf.getMlInput(otu_train, otu_test, map_train, map_test, 
                                                                target = target, pca_reduced = True, numComponents = 100)
    X_train_pca = pd.concat([X_train_pca, X_val_pca], axis = 0)
    y_train_pca = y_train_pca + y_val_pca
    
    # remove meta variable if specified by argument
    if not meta:
        # specifiy column labels to be dropped
        drop_cols = X_train_pca.columns.values[100::1]
        # drop meta data
        X_train_pca = X_train_pca.drop(drop_cols, axis=1)
        X_val_pca = X_val_pca.drop(drop_cols, axis=1)
        X_test_pca = X_test_pca.drop(drop_cols, axis=1)
    
    m_pca, auc_pca, auc_train_pca, fpr_pca, tpr_pca, prec_pca, f1_pca, f2_pca, _  = hf.predictIBD(X_train_pca, y_train_pca, X_test_pca, y_test_pca, graph_title = "PCA dimensionality reduced " + str(X_train_pca.shape[1]) + " features", 
                  max_depth = 5, n_estimators = 50, weight = 20, plot = False, plot_pr = False)
    
    row_pca = pd.DataFrame({'algo': ['pca'], 'seed': [seed], 'auc': [auc_pca],
                        'precision': [prec_pca], 'f1': [f1_pca], 
                        'f2': [f2_pca]})
    
    # save performance metrics in data frame
    performance = performance.append(row_pca)
    # save forest
    forests.append(m_pca)
    # save fpr and tpr
    confusion.append([fpr_pca, tpr_pca])
    # save principal axes
    pca_embeddings.append(principal_axes)
    
    
    # Normalized Raw Count Data
    X_train_asin, X_val_asin, X_test_asin, y_train_asin, y_val_asin, y_test_asin, axes = hf.getMlInput(otu_train, otu_test, map_train, map_test, 
                                                                target = target, asinNormalized = True)
    X_train_asin = pd.concat([X_train_asin, X_val_asin], axis = 0)
    y_train_asin = y_train_asin + y_val_asin
    
    # remove meta variable if specified by argument
    if not meta:
        # specifiy column labels to be dropped
        drop_cols = X_train_pca.columns.values[:-14:-1]
        # drop meta data
        X_train_asin = X_train_asin.drop(drop_cols, axis=1)
        X_val_asin = X_val_asin.drop(drop_cols, axis=1)
        X_test_asin = X_test_asin.drop(drop_cols, axis=1)
    
    m_asin, auc_asin, auc_train_asin, fpr_asin, tpr_asin, prec_asin, f1_asin, f2_asin, _ = hf.predictIBD(X_train_asin, y_train_asin, X_test_asin, y_test_asin, graph_title = "Normalized asinh Taxa Abundances " + str(X_train.shape[1]) + " features",
                  max_depth = 5, n_estimators = 170, weight = 20, plot = False, plot_pr = False)

    

    
    row_asin = pd.DataFrame({'algo': ['norm_raw'], 'seed': [seed], 'auc': [auc_asin],
                        'precision': [prec_asin], 'f1': [f1_asin], 
                        'f2': [f2_asin]})
    
    # save performance metrics in data frame
    performance = performance.append(row_asin)
    # save forest
    forests.append(m_asin)
    # save fpr and tpr
    confusion.append([fpr_asin, tpr_asin])

    # save everything in list in order to save all objects as one file
    result_objects = [performance, forests, confusion, pca_embeddings]

# save computation results
with open('prediction_results_meta=' + str(meta) + '.obj', mode='wb') as results_file:
    pickle.dump(result_objects, results_file)
    results_file.close()
    
# plot one boxplot per performance metric grouped by algorithm
perf_melted = pd.melt(performance,
                      id_vars = ['algo'], 
                      value_vars = ['auc', 'precision', 'f1', 'f2'],
                      var_name = 'metric')

seaborn_boxplot = seaborn.boxplot(x='algo', y='value', data=perf_melted, hue='metric')
fig = seaborn_boxplot.get_figure()
fig.savefig("../fig/boxplots_performances_all_algos_meta=" + str(meta) + ".pdf")


