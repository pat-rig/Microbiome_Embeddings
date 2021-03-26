#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Helper Functions for PredictIBD.py
    Partially Adopted from helper_functions.py from TATARU and DAVID
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pickle
import random

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
import math
import copy
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from inspect import signature
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import fbeta_score
import sklearn




def getQualVecs(data_dir, embedding_txt, embedding_fasta):
   
    qual_vecs = pd.read_csv(embedding_txt, sep = " ", index_col = 0,
                            dtype = {0:str}, header = None)
    qual_repseqs = pd.read_csv(embedding_fasta, sep = "\t", header = None)
    
    import re
    ids = qual_repseqs.iloc[range(0, qual_repseqs.shape[0], 2), 0]
    ids = [re.sub(">", "", i) for i in ids.values]

    seqs = qual_repseqs.iloc[range(1, qual_repseqs.shape[0], 2), 0]
    seqs = [str(i) for i in seqs.values]

    #Drop <unk> character
    ids = ids[0: len(ids)-1]
    seqs = seqs[0: len(seqs)-1]
    qual_vecs = qual_vecs.iloc[0: len(seqs), :]

    print(len(ids))
    print(len(seqs))
    print(qual_vecs.shape)
    return(qual_vecs, ids, seqs)


def asinh(otu):
    return(np.arcsinh(otu))


def matchOtuQual(otu, embed_ids, embed_seqs):
    # select only columns present in embedding matrix
    otu_reorder = otu.loc[:, embed_seqs]
    # unit test for latter procedure
    if np.sum([i==j for i,j in zip(otu_reorder.columns.values, embed_seqs)]) == len(embed_seqs):
        print("all good")
    else:
        print("There's a problem, stop")
    # relabel columns with sequence ids
    otu_reorder.columns = embed_ids
    return(otu_reorder)



def splitTrainTest(otu_keep, map_keep, test_samples):
    test_samples = test_samples[[i in otu_keep.index.values for i in test_samples]] #Only include ids that we haven't dropped in previous steps
    otu_train = otu_keep.loc[[not(i in np.array(test_samples)) for i in otu_keep.index.values], :]
    otu_test = otu_keep.loc[test_samples, :]

    map_train = map_keep.loc[[not(i in np.array(test_samples)) for i in map_keep.index.values], :]
    map_test = map_keep.loc[test_samples, :]

    print("OTU TRAIN: " + str(otu_train.shape))
    print("MAP TRAIN: " + str(map_train.shape))
    print("OTU TEST: " + str(otu_test.shape))
    print("MAP TEST: " + str(map_test.shape))
    return(otu_train, otu_test, map_train, map_test)


def normalize(otu):
    #Normalize
    sample_sums = otu.sum(axis=1)
    otu_norm = otu.div(sample_sums, axis=0)
    return(otu_norm)


def embed_wo_average(otu, qual_vecs):
    qual_vecs_use = qual_vecs.loc[list(otu.columns.values)]
    df = pd.DataFrame(np.dot(otu, qual_vecs_use), index = otu.index.values)
    return(df)

def getFeatureImportance(m, data, y):
    feat_imp = m.feature_importances_
    feat_imp_labeled = zip(data.columns.values, feat_imp)
    feat_imp_sort = sorted(feat_imp_labeled, key = lambda t: t[1], reverse = True)
    return(feat_imp_sort)



def predictIBD(X_train, y_train, X_test, y_test, graph_title = "",
               max_depth = 12, n_estimators = 140, plot = False,
               plot_pr = False, weight = 20, feat_imp = False, flipped = False):
    """
    Fit RF on X_train, y_train and predict on X_test
    Compute and Return various performance measures and feature importance
    """
    weights = {0:1, 1:weight}
    m = RandomForestClassifier(max_depth= max_depth, random_state=0, n_estimators= n_estimators, class_weight = weights)
    m.fit(X_train, y_train)
    probs = m.predict_proba(X_test)
    probs_train = m.predict_proba(X_train)
    roc_auc, fpr, tpr, precision, f1, f2 = computeMLstats(m, data = X_test,
                                                          y = y_test, plot = plot,
                                                          plot_pr = plot_pr,
                                                          graph_title = graph_title, 
                                                          flipped = flipped)
    

    feat_imp_sort = getFeatureImportance(m, data = X_train, y = y_train)
    
    return(m, roc_auc, None, fpr, tpr, precision, f1, f2, feat_imp_sort)


def computeMLstats(m, data, y, plot = False, plot_pr = False, graph_title = None, flipped = False):
    """
    Compute ROC and AUC
    """
    
    probs = m.predict_proba(data)
    
    #Flip for opposite class imbalance
    if flipped:
        y = [1 - i for i in y]
        probs = 1 - probs
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y, probs[:, 1])
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    
    #Compute precision-recall
    precision, recall, _ = precision_recall_curve(y, probs[:,1])

    #avg_pr = average_precision_score(precision, recall)
    average_precision = average_precision_score(y, probs[:,1])
    
    f1 = f1_score(y, np.argmax(probs, axis = 1))
    f2 = fbeta_score(y, np.argmax(probs, axis = 1), beta = 2)
    
    if plot:
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='AUC ROC = %0.2f' %  roc_auc)
        #'AUC PR = %0.2f' % pr_avg_pr
        
        plt.legend(loc="lower right")
        x = np.linspace(0, 1, 10)
        plt.plot(x, x)
        plt.title(graph_title)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        
    if plot_pr:
        plt.subplot(1,2,2)
        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post', label='AUC PR = %0.2f' %  average_precision)
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.legend(loc="lower right")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
    
    return(roc_auc, fpr, tpr, average_precision, f1, f2)



#Concat otu abundances and metadata for final RF Input
def combineData(microbe_data, mapping_data, names = []):
    micro_norm = preprocessing.scale(microbe_data)
    map_norm = preprocessing.scale(mapping_data)
    data = pd.concat([pd.DataFrame(micro_norm), pd.DataFrame(map_norm)], axis = 1)
    if not names == []:
        data.columns = np.concatenate((names, [i for i in mapping_data.columns.values]))
    return(data)

def setTarget(mapping, target = ""):
    y = [float(i) for i in mapping[target]]
    mapping = mapping.drop(target, axis = 1)
    return(mapping, y)

def getMlInput(otu_train, otu_test, map_train, map_test, target, 
               embed = False, pca_reduced = False, asinNormalized = False,
               percNormalized = False, pathwayEmbed = False,
               qual_vecs = None, numComponents = 250, names = []):
    # require combineData, setTarget, embed_average
    
    #split training set again to get some validation data for training hyperparameters
    otu_train_train = otu_train.sample(frac = 0.9, random_state = 10)
    otu_val = otu_train.drop(otu_train_train.index.values)
    map_train_train = map_train.loc[otu_train_train.index.values]
    map_val = map_train.drop(otu_train_train.index.values)
    
    map_train_train, y_train = setTarget(map_train_train, target = target)
    map_val, y_val = setTarget(map_val, target = target)
    map_test, y_test = setTarget(map_test, target = target)
    axes = None
 
    if embed:
        X_train = combineData(embed_average(otu_train_train, qual_vecs), map_train_train, names = qual_vecs.columns.values)
        X_val = combineData(embed_average(otu_val, qual_vecs), map_val, names = qual_vecs.columns.values)
        X_test = combineData(embed_average(otu_test, qual_vecs), map_test, names = qual_vecs.columns.values)
    elif pca_reduced:
        pca_train, pca_val, pca_test, axes = getPCAReduced(otu_train_train, otu_val, otu_test, components = numComponents)
        X_train = combineData(pca_train, map_train_train, names = names)
        X_val = combineData(pca_val, map_val, names = names)
        X_test = combineData(pca_test, map_test, names = names)
    elif asinNormalized:
        X_train = combineData(asinh(otu_train_train), map_train_train, names = names)
        X_val = combineData(asinh(otu_val), map_val, names = names)
        X_test = combineData(asinh(otu_test), map_test, names = names)
    elif percNormalized: 
        X_train = combineData(otu_train_train.div(otu_train_train.sum(axis=1), axis=0), map_train_train, naming = naming)
        X_val = combineData(otu_val.div(otu_val.sum(axis=1), axis=0), map_val, names = names)
        X_test = combineData(otu_test.div(otu_test.sum(axis=1), axis=0), map_test, names = names)
    elif pathwayEmbed:
        X_train = combineData(embed_average(otu_train_train, pathway_table), map_train_train, names = names)
        X_val = combineData(embed_average(otu_val, pathway_table), map_val, names = names)
        X_test = combineData(embed_average(otu_test, pathway_table), map_test, names = names)
    
    return(X_train, X_val, X_test, y_train, y_val, y_test, axes)  



def getCrossValMlInput(otu_train, otu_test, map_train, map_test, target, 
               embed = False, pca_reduced = False, asinNormalized = False, percNormalized = False, pathwayEmbed = False,
               qual_vecs = None, numComponents = 250, names = [], folds = 10):
    
    
    map_test, y_test = setTarget(map_test, target = target)
    
    #split training set again to get some validation data for training hyperparameters
    random.seed(1)
    kf = KFold(n_splits = folds)
    kf.get_n_splits(otu_train)
    X_train_list = []
    X_val_list = []

    y_train_list = []
    y_val_list = []

    i = 0
    for train_index, val_index in kf.split(otu_train):
        otu_train_train = otu_train.iloc[train_index, :]
        otu_val = otu_train.iloc[val_index, :]
        map_train_train = map_train.iloc[train_index, :]
        map_val = map_train.iloc[val_index, :]
    
        map_train_train, y_train = setTarget(map_train_train, target = target)
        map_val, y_val = setTarget(map_val, target = target)
        

        if embed:
            X_train = combineData(embed_average(otu_train_train, qual_vecs),
                                  map_train_train, names = names)
            X_val = combineData(embed_average(otu_val, qual_vecs), map_val, names = names)
            if i == 0:
                X_test = combineData(embed_average(otu_test, qual_vecs), map_test, names = names)
        elif pca_reduced:
            # Perform PCA
            pca_train, pca_val, pca_test = getPCAReduced(otu_train_train, otu_val, otu_test, components = numComponents)
            X_train = combineData(pca_train, map_train_train, names = names)
            X_val = combineData(pca_val, map_val, names = names)
            if i == 0:
                X_test = combineData(pca_test, map_test, names = names)
        elif asinNormalized:
            X_train = combineData(asinh(otu_train_train), map_train_train, names = names)
            X_val = combineData(asinh(otu_val), map_val, names = names)
            if i == 0:
                X_test = combineData(asinh(otu_test), map_test, names = names)
        elif percNormalized: 
            X_train = combineData(otu_train_train.div(otu_train_train.sum(axis=1), axis=0), map_train_train, names = names)
            X_val = combineData(otu_val.div(otu_val.sum(axis=1), axis=0), map_val, names = names)
            if i == 0:
                X_test = combineData(otu_test.div(otu_test.sum(axis=1), axis=0), map_test, names = names)
        elif pathwayEmbed:
            X_train = combineData(embed_average(otu_train_train, pathway_table), map_train_train, names = names)
            X_val = combineData(embed_average(otu_val, pathway_table), map_val, names = names)
            if i == 0:
                X_test = combineData(embed_average(otu_test, pathway_table), map_test, names = names)
        X_train_list.append(X_train)
        X_val_list.append(X_val)
       
        y_train_list.append(y_train)
        y_val_list.append( y_val)
        i = i + 1
      
    return(X_train_list, X_val_list, X_test, y_train_list, y_val_list, y_test) 



def embed_average(otu, qual_vecs):
    # does the number of identical(sample by asv columns, embed_matrix rows)
    # names
    # match the number of all asvs?
    if(np.sum([i == j for i,j in zip(otu.columns.values, qual_vecs.index.values)]) == otu.shape[1]):
        print("all good")
    else:
        print("There's a problem")
    df = pd.DataFrame(np.dot(asinh(otu), qual_vecs), index = otu.index.values)
    return(df)

from sklearn.model_selection import StratifiedShuffleSplit
def crossValPrediction(otu_use, y, max_depth = 10, n_estimators = 65,
                       weight = 5, plot = False, plot_pr = False, folds = 5):
    """
    
    """
    
    # stratify with respect to y 
    # is y IBD status?
    kf = StratifiedShuffleSplit(n_splits = folds)
    kf.get_n_splits(otu_use, y)
    
    auc_crossVal = []
    auc_prec_crossVal = []
    f1_crossVal = []
    feat_imp_crossVal = []
    i = 0
    for train_index, val_index in kf.split(otu_use, y):
        otu_train = otu_use.iloc[train_index, :]
        otu_val = otu_use.iloc[val_index, :]
        y_train = np.array(y)[train_index]
        y_val = np.array(y)[val_index]
        
        # plt.subplot(1, 2, 1)
        m, auc, auc_train, fpr, tpr, prec, f1, f2, feat_imp = predictIBD(otu_train, y_train, otu_val, y_val,
                  max_depth = max_depth, n_estimators = n_estimators, weight = weight, plot = plot, plot_pr = plot_pr, feat_imp = True)
        auc_crossVal.append(auc)
        auc_prec_crossVal.append(prec)
        f1_crossVal.append(f1)
        feat_imp_crossVal.append(feat_imp)
        
        i = i + 1
    return(auc_crossVal, auc_prec_crossVal, f1_crossVal, feat_imp_crossVal)

def getPCAReduced(X_train, X_val, X_test, components = 500):
    pca = PCA(n_components= components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    # retreive TRANSPOSED embedding matrix
    principal_axes = pca.components_ # shape: (n_components, n_features)
    return(X_train_pca, X_val_pca, X_test_pca, principal_axes)

def plotPCA(table, otu_raw, components):
    pca = PCA(n_components= components)
    pca = pca.fit(table)
    table_pca = pca.transform(table)
    table_pca = table_pca / np.max(table_pca)
    df = pd.DataFrame(table_pca, index = table.index.values)
    sample_sums_table = otu_raw.sum(axis = 1)
    plt.scatter(df.iloc[:,0], df.iloc[:,1], c = sample_sums_table, cmap='viridis')
    plt.colorbar()
    plt.xlabel(pca.explained_variance_ratio_[0])
    plt.ylabel(pca.explained_variance_ratio_[1])
