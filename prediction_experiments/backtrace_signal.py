"""
    WHY ARE ASVs WITH MOST SIGNAL FOR RAW DATA NOT IMPORTANT FOR PCA?
    
    import os
    os.chdir('/Users/patrick/Drive/13_sem/research_project/TEMP/prediction_experiments')
"""
import pickle
import pandas as pd

# load tables of most important asvs
with open('imp_asv_tables.obj', mode='rb') as table_file:
    tables = pickle.load(table_file)
    table_file.close()
    
# only execute on large memory machine (file requires ~1GB)
# load abundance data 
abundance_data = pd.read_csv('../data/seqtab_filter.07.txt', sep='\t', index_col=0)

# pick asvs to inspect
sign_asvs_raw = tables['sign_asvs']['raw']
sign_asvs_pca = tables['sign_asvs']['pca']

# extract and save abundance vectors
sign_abundances = {'pca': abundance_data.loc[:, sign_asvs_pca],
                   'raw': abundance_data.loc[:, sign_asvs_raw]}

with open('sign_abundances.obj', mode='wb') as abundance_file:
    pickle.dump(sign_abundances, abundance_file)
    abundance_file.close()
    
    
