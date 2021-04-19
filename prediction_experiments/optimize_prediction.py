#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Optimize Hyper Parameters for Preprocessing and Prediction 
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


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
test_ids = pd.read_csv(test_file, header = None, sep = '\t')

# load meta data
meta_data = pd.read_csv('../data/AG_mapping.txt', sep='\t', index_col=0)

# select subset of relevant meta data: feces with IBD information
meta_data['IBD'].value_counts()
not_ibd = meta_data['IBD'].value_counts().index.values[[0, 1, 3]]
ibd_idx = np.logical_not([ibd_status in not_ibd for ibd_status in meta_data['IBD']]) 

feces = meta_data['BODY_SITE'].value_counts().index.values[0]
feces_idx = meta_data['BODY_SITE'] == feces

select_idx = np.logical_and(feces_idx, ibd_idx)
# could be potentially changed to IBS or both or even subset of IBD manually
meta_of_interest = meta_data[select_idx]

# select subset of relevant abundance data
# load index mappings
err2qid = pd.read_csv('../data/err-to-qid.txt', sep='\t')
select_qid = select_idx.index.values[select_idx]
select_qid_bool = [qid in select_qid for qid in err2qid['sample_title']]
select_err = err2qid['run_accession'][select_qid_bool]

X_train = abundance_data[select_err] # subset by index

# retreive labels







# split train and test
X_train = abundance_data.drop(labels = list(test_ids.iloc[0, :]),
                                   axis = 'index')
X_test = abundance_data.loc[test_ids.iloc[0, :], :]
# obtain y_train manually

# require preprocessor in pipeline to discard non-labelled obs after embed-fit


# =============================================================================
# Build Pipeline
# =============================================================================

kNN = KNeighborsClassifier(n_neighbors=10)
gsCV_obj = GridSearchCV(estimator=kNN, param_grid=param, cv=10)
gsCV_obj.fit(X_train, y_train)