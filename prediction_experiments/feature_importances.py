#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    EVALUATE PREDICTION EXPERIMENTS
    
    1. Inspect Feature Importance Scores
    
    import os
    os.chdir('/Users/patrick/Drive/13_sem/research_project/TEMP/prediction_experiments')
"""
import numpy as np

# load results from prediction experiments
with open('prediction_results.obj', mode='rb') as results_file:
    result = pickle.load(results_file)
    results_file.close()
    
# retreive RandomForest Objects
forests = result[1]
# separate per algorithm: [0]: glove, [1]: PCA, [2]: raw data
forests_by_algo = [forests[i::3] for i in range(3)]

# retreive Feature Importances
feat_imps = [forests_by_algo[i][j].feature_importances_ for i in range(3) for j in range(len(forests_by_algo[i]))]
# has 81 entries. First 27 for glove, second for PCA, third for raw data


# compute means per algorithm over feature importances
no_reps = len(forests_by_algo[1])
glove_imp = np.mean(feat_imps[:27:1], axis=0)
pca_imp = np.mean(feat_imps[27:54:1], axis=0)
raw_imp = np.mean(feat_imps[54::1], axis=0)
# more elegantly?
# [np.nanmean(feat_imps[0 + (i-1) * no_reps : i * no_reps : 1], axis = 0) for i in range(3)]

# sort decreasing 
np.argsort(glove_imp)
np.argsort(pca_imp)
np.argsort(raw_imp)