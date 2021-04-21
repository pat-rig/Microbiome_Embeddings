#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Optimize Hyper Parameters for Preprocessing and Prediction 
    
    import os
    os.chdir('/Users/patrick/drive/13_sem/research_project/TEMP/prediction_experiments')
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import helper_predict as hf
import umap.umap_ as umap
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

# perform optimization only for one seed
seed = 0

# =============================================================================
# Prepare Data
# =============================================================================
# load abundance data
abundance_data = pd.read_csv('../data/seqtab_filter.07.txt', sep='\t',
                                 index_col=0)
# glove not fit on these obs
test_file = '../data/test_samples/test_samples_' + str(seed) + '.txt'
test_ids_err = pd.read_csv(test_file, header = None, sep = '\t')

# load meta data and convert index dtype to string for compatibility
meta_data = pd.read_csv('../data/AG_mapping.txt', sep='\t', index_col=0) 
meta_data = meta_data.set_index(pd.Index(meta_data.index.values, dtype='str',
                                         name='sample_title'))
# for mapping different types of ids
err2qid = pd.read_csv('../data/err-to-qid.txt', sep='\t', index_col=0)
qid2err = pd.read_csv('../data/err-to-qid.txt', sep='\t', index_col=1)
# change index of test_ids to qid for convenient indexing
test_ids = err2qid.loc[test_ids_err.iloc[0, :]]
# some test_ids do not exist in meta_data
# ==> simply indexing with test_ids won't work.

# select subset of feces probes
feces = meta_data['HMP_SITE'].value_counts().index.values[0]
feces_idx = meta_data['HMP_SITE'] == feces
meta_feces = meta_data[feces_idx]

# select subset of corresponding abundance data
# replace err ids with qids
# cant convert abundance_data.index to qids since not all in err2qid

# create list for indexing abundance data
present_in_abundance = []
for qid in meta_feces.index.values:
    print('Check if abundance data is available for fecal sample' + str(qid))
    # check if more than one err-ids correspond to this qid
    if qid in qid2err.index.values:
        no_errs = qid2err.loc[qid].shape[0]
    else:
        continue
    if no_errs == 1:
        # check if we have abundance data for this errid
        err_string = qid2err.loc[qid].values[0]
        if err_string in abundance_data.index.values:
            present_in_abundance.append(err_string)
    if no_errs > 1:
        # check for all potential matches if we have abundance data for one
        for err_string in qid2err.loc[qid]['run_accession']:
            if err_string in abundance_data.index.values:
                present_in_abundance.append(err_string)

len(present_in_abundance)
# 12171 sequenced feces probes 
abundance_feces = abundance_data.loc[present_in_abundance]
# reindex qith qids
qid_keys = err2qid.loc[abundance_feces.index.values]['sample_title'].values
abundance_feces = abundance_feces.set_index(qid_keys)

# select observations for which IBD information is present,
# i.e. is explicitly positive or negative
meta_feces['IBD'].value_counts()
# 13.5k negative 0.7k positive
# create column with ibd info \in {NA, True, False}
abundance_feces_labels_str = meta_feces['IBD'][abundance_feces.index.values]
# transform labels to binary and NA
label_map = pd.DataFrame([0, None, 1, None, 1, 1],
                         index=[abundance_feces_labels_str.value_counts().index])
ibd_info = [label_map.loc[long_string].values for long_string in abundance_feces_labels_str.values]
# remove np array layer
ibd_info = [array[0][0] for array in ibd_info]
abundance_feces['ibd_info'] = ibd_info

# remove test_samples
# those still contain observations which might have no ibd information.
# will only be used for fitting the embedding later

# there exist some test_ids which already have been discarded because they are
# not feces probes
test_ids_not_in_feces_bool = [not (qid in abundance_feces.index.values) for qid in test_ids.sample_title.values]
test_ids_not_feces = test_ids.loc[test_ids_not_in_feces_bool]
# 959 samples not in feces but also not used for fitting the embedding
test_ids_in_feces = test_ids.loc[np.logical_not(test_ids_not_in_feces_bool)]

X_emb_train = abundance_feces.drop(test_ids_in_feces.sample_title.values,
                                   axis='index').drop('ibd_info',
                                                      axis='columns')
X_test = abundance_feces.loc[test_ids_in_feces.sample_title].drop('ibd_info', axis='columns')

# select obs with labels --> use for optimizing prediction algorithms
unlabelled_obs_bool = abundance_feces['ibd_info'].loc[X_emb_train.index.values].isna() 
unlabelled_obs = X_emb_train.index.values[unlabelled_obs_bool]
# 745 from 10k
X_CV = X_emb_train.drop(unlabelled_obs, axis='index')
# 9k rows 27k cols
y_CV = abundance_feces['ibd_info'].loc[X_emb_train.index.values].drop(unlabelled_obs, axis='index')
# 506 positive labels

# unit test that labels and observations correspond
any(np.logical_not(X_CV.index == y_CV.index))
# False

# select obs with labels in test set. others can be discarded
unlabelled_test_obs_bool = abundance_feces['ibd_info'].loc[X_test.index.values].isna() 
unlabelled_test_obs = X_test.index.values[unlabelled_test_obs_bool]

X_test = X_test.drop(unlabelled_test_obs, axis='index')
# now reduced to 1668 obs from 1814 obs
y_test = abundance_feces['ibd_info'].loc[X_test.index.values]

# unit test 
any(np.logical_not(X_test.index == y_test.index))
# False

# Note, this intermediate step of preprocessing (dividing into train and test,
# before selecting labelled and unlabelled data was necessary due to work done
# before setting up the prediction task [iot not to discard long computations])


# =============================================================================
# Prediction Pipeline
# =============================================================================
# supply directory structure with relative paths
data_dir = '../data/' # contains initial files from Tataru and David (2020)
emb_dir = data_dir + 'embeddings/'
test_dir = data_dir + 'test_samples/'
fasta_dir = data_dir + 'fasta/' 
# otu_dir = data_dir + 'otu_objs/'
 
# 0. Project X_CV onto embeddings

# load GloVe embedding    
emb_txt = emb_dir + 'glove_input_' + str(seed) + '_emb.txt'
emb_fasta = fasta_dir + 'asvs_' + str(seed) + '.fasta'

qual_vecs, embed_ids, embed_seqs = hf.getQualVecs(data_dir=data_dir,
                                                      embedding_txt=emb_txt,
                                                      embedding_fasta=emb_fasta)

# discard columns which could not be used for the embedding due to algorithmic
# constraint in glove implementation
seqs_not_in_glove_bool = [not(cv_seq in qual_vecs.index.values) for cv_seq in X_CV.columns.values]
seqs_not_in_glove = X_CV.columns.values[seqs_not_in_glove_bool]
X_CV_glove = X_CV.drop(seqs_not_in_glove, axis='columns')
# not scaled, since input to glove wasnt scaled either

# project onto glove space
X_CV_glove = np.dot(X_CV_glove, qual_vecs)

# Find Optimal random forest for GloVe
rf = RandomForestClassifier()
# specify grid
rf_params = {'n_estimators': [50, 500], 'max_depth': [5, None]}
# specify metrics. do not require refitting the best model
gridsearch_glove_rf = GridSearchCV(estimator=rf, param_grid=rf_params, cv=5,
                          scoring = ['average_precision', 'roc_auc', 'f1'],
                          refit=False)
gridsearch_glove_rf.fit(X_CV_glove, y_CV)


# Final Preprocessing for UMAP compatibility
X_UMAP = abundance_data.drop(test_ids.index, axis='index')
cv_errIDs = qid2err.loc[X_CV.index]
# <-- grabs both entries for duplicates! i.e. len(cv_errIDs) > len(X_CV.index)
# --> select those err-ids appearing in X_CV
cv_err_bool = [err in X_UMAP.index.values for err in cv_errIDs.run_accession.values]
err_in_cv = cv_errIDs.loc[cv_err_bool]
# X_CV_umap = X_UMAP.loc[err_in_cv.run_accession]

# Fit UMAP on same data as GloVe was fit on
# educated guess for n_neighbors for now
manifold = umap.UMAP(n_neighbors=25, n_components=100)
X_UMAP_projected = manifold.fit_transform(X_UMAP)
X_UMAP_projected = pd.DataFrame(X_UMAP_projected,
                                index=X_UMAP.index)
# select CV observations
X_CV_UMAP = X_UMAP_projected.loc[err_in_cv.run_accession]

# Find Optimal k-NN Classifier
kNN = KNeighborsClassifier()
kNN_params = {'n_neighbors': [5, 10, 50]}

gs_umap_knn = GridSearchCV(estimator=kNN, param_grid=kNN_params, cv=5,
                           scoring = ['average_precision', 'roc_auc', 'f1'],
                           refit=False)
gs_umap_knn.fit(X_CV_UMAP, y_CV)

# kNN also suited for GloVe?
kNN_glove = KNeighborsClassifier()
# specify metrics. do not require refitting the best model
gridsearch_glove_kNN = GridSearchCV(estimator=kNN_glove, param_grid=kNN_params,
                                   cv=5, scoring = ['average_precision', 'roc_auc', 'f1'],
                                   refit=False)
gridsearch_glove_kNN.fit(X_CV_glove, y_CV)
# nope! performance worse (s. gridsearch_glove_kNN.cv_results_)

# =============================================================================
# Results GloVeRF vs UMAP-kNN
# =============================================================================

## Cross Validation
# f1 score substantially higher for umap: 5 folds
#   0.20 +/- 0.11 (glove)
#   0.34 +/- 0.14 (UMAP)    

#   beware that that it does not use probabilities but hard classifications!
#   i.e. thresholds not taken into account

# average precision slightly higher (0.28 +/- 0.10) for umap than for glove (0.26 +/- 0.12)

## Inspect Performance on held out test set: Precision Recall Curve 
# retreive best params
rf_glove_best_params_idx = np.where(gridsearch_glove_rf.cv_results_['rank_test_f1'] == 1)[0][0]
rf_params_glove = gridsearch_glove_rf.cv_results_['params'][rf_glove_best_params_idx]

umap_best_idx = np.where(gs_umap_knn.cv_results_['rank_test_f1'] == 1)[0][0]
umap_best_k = gs_umap_knn.cv_results_['params'][umap_best_idx]['n_neighbors']

# refit models on whole CV data set
glove_rf = RandomForestClassifier(n_estimators=rf_params_glove['n_estimators'],
                                  max_depth=rf_params_glove['max_depth'])
glove_rf.fit(X_CV_glove, y_CV)

umap_knn = KNeighborsClassifier(umap_best_k)
umap_knn.fit(X_CV_UMAP, y_CV)

X_test_UMAP = manifold.transform(X_test)

plot_precision_recall_curve(glove_rf, X_test, y_test)
plot_precision_recall_curve(umap_knn, X_test, y_test)











kNN = KNeighborsClassifier(n_neighbors=10)
gsCV_obj = GridSearchCV(estimator=kNN, param_grid=param, cv=10)
gsCV_obj.fit(X_train, y_train)