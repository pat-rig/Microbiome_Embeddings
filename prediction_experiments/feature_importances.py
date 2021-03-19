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


# compute means per algorithm over feature importances
glove_imp = np.mean(feat_imps[:27:1], axis=0)
pca_imp = np.mean(feat_imps[27:54:1], axis=0)
raw_imp = np.mean(feat_imps[54::1], axis=0)
# ValueError: operands could not be broadcast together with shapes (26627,) (26640,) 
# guess on cause: 
# Since forests are trained on different subsets of the data, some forests
# use variables (ASVs) which are not used by others.

# Fix
# Per Forest: Map variable to ASV name
# --> create new array with appropriate encoding for non-existing features (not zero!!)
#       IMPORTANT TO CHECK BEFOREHAND WHETHER THE ESITMATES ARE NA!

# more elegantly?
# [np.nanmean(feat_imps[0 + (i-1) * no_reps : i * no_reps : 1], axis = 0) for i in range(3)]

# sort decreasingly 
np.argsort(glove_imp)
np.argsort(pca_imp)
# indices 104 and 109 are associated with the highest gini importance
# what do they correspond to?
# extract variable 'names' from input to RFC


# Compare ASVs associated with most important features
# how many features to inspect
n_imp = 5






np.argsort(raw_imp)


