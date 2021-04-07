"""
    WHY ARE ASVs WITH MOST SIGNAL FOR RAW DATA NOT IMPORTANT FOR PCA?
    
    import os
    os.chdir('/Users/patrick/Drive/13_sem/research_project/TEMP/prediction_experiments')
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import boxplot_stats


# load tables of most important asvs
with open('imp_asv_tables.obj', mode='rb') as table_file:
    tables = pickle.load(table_file)
    table_file.close()
    
    # Only execute on machine with large memory
# only execute on large memory machine (file requires ~1GB)
# load abundance data 
abundance_data = pd.read_csv('../data/seqtab_filter.07.txt', sep='\t',
                             index_col=0)

# pick asvs to inspect
sign_asvs_raw = tables['sign_asvs']['raw']
sign_asvs_pca = tables['sign_asvs']['pca']
com_asvs_bool = [raw_asv in sign_asvs_pca for raw_asv in sign_asvs_raw]
com_asvs = sign_asvs_raw[com_asvs_bool]

# extract and save abundance vectors
sign_abundances = {'pca': abundance_data.loc[:, sign_asvs_pca],
                   'raw': abundance_data.loc[:, sign_asvs_raw]}

with open('sign_abundances.obj', mode='wb') as abundance_file:
    pickle.dump(sign_abundances, abundance_file)
    abundance_file.close()
    
# if on local machine: collect abundances

with open('sign_abundances.obj', mode='rb') as abundance_file:
    sign_abundances = pickle.load(abundance_file)
    abundance_file.close()

# =============================================================================
# Variance in important ASVs
# =============================================================================
# first look compute variances per column
raw_variances = sign_abundances['raw'].var(axis=0).values
pca_variances = sign_abundances['pca'].var(axis=0).values    

# how to visualize that variances are higher for pca?

# =============================================================================
# Association between important ASVs and IBD status
# =============================================================================
# Collect abundance and meta data from IBD-relevant patients
with open("../data/otu_objs/fully_postprocessed.obj", mode='rb') as both_file:
    fully_postprocessed = pickle.load(both_file)
    both_file.close()
    
meta = fully_postprocessed['meta']
abundance = fully_postprocessed['abundance']
# inspect different values
IBD_levels = np.unique(meta['IBD'])

negative = meta['IBD'] == IBD_levels[-2]

# collect abundances and IBD status for important ASVs
def collect_abundances(algo='raw'):
    
    raw_abundance = abundance.loc[:, sign_abundances[algo].columns.values]
    raw_abundance['IBD'] = np.logical_not(negative)
    
    # melt into long format for seaborn compatibility
    asv_names = np.delete(raw_abundance.columns.values, -1)
    raw_abundance_mld = pd.melt(raw_abundance, id_vars = 'IBD',
                                value_vars = asv_names)
    
    return raw_abundance, raw_abundance_mld

raw_abundance, raw_abundance_mld = collect_abundances('raw')
pca_abundance, pca_abundance_mld = collect_abundances('pca')
# put together iot generalize plotting method
plotting_dict = {'raw': {'wide': raw_abundance, 'long': raw_abundance_mld},
                 'pca': {'wide': pca_abundance, 'long': pca_abundance_mld}}

# BOXPLOTS

def plot_imp_asv_abundance_dists(algo = 'raw'):
    
    raw_abundance = plotting_dict[algo]['wide']
    raw_abundance_mld = plotting_dict[algo]['long']
    
    # write percentage of outliers above whiskers 
    # compute outliers per ASV per IBD status
    # divide into two DataFrames respective to IBD status
    raw_grouped_df = raw_abundance.groupby(by= 'IBD')
    # compute boxplot stats for all columns for both groups
    # one pd.Series per df. Eah Series contains stats for one column
    all_box_stats = [df.apply(boxplot_stats) for name, df in raw_grouped_df] 
    # total number of outliers 
    no_outliers = [len(asv[0]['fliers']) for status in all_box_stats for asv in status]
    # how many entries belong to status 0?
    no_asvs = int((len(no_outliers) - 2)/2) #ignore ibd column
    # compute percentage on all obs per status
    no_obs_per_status = [status.shape[0] for name, status in raw_grouped_df]
    outlier_perc_healthy = [out / no_obs_per_status[0] for out in no_outliers[:no_asvs:1]]
    outlier_perc_sick = [out / no_obs_per_status[1] for out in no_outliers[no_asvs+1:-1:1]]
    
    # sort by absolute difference in IQR
    iqr_diffs = [np.abs(all_box_stats[0][i][0]['iqr'] -
                        all_box_stats[1][i][0]['iqr']) for i in range(no_asvs + 1)]
    # create list of strings indicating the order. ignore ibd col after sorting.
    plot_order_idx = np.flip(np.argsort(iqr_diffs[:-1:]))
    plot_order_names = raw_abundance.columns.values[plot_order_idx]
    # list of booleans whether it is a common asv
    com_bool_list = [asv in com_asvs for asv in plot_order_names]
    
    # more outliers for healthy patients
    outlier_ratio = np.array(outlier_perc_healthy) / np.array(outlier_perc_sick)
    outlier_ratio_strings = [str(np.round(ratio, 2)) for ratio in outlier_ratio[plot_order_idx]]
    # add ** if ASV appears for both algorithms
    outlier_ratio_strings = [('#' + ratio if both_bool else ratio) for ratio, both_bool in zip(outlier_ratio_strings, com_bool_list)]
    
    
    # increase font size
    sns.set(font_scale=2.5)
    sns.set_style('white')
    # ylim param by hand
    yend = 450
    if algo == 'pca':
        yend = 850
        
    # change data frame for legend names
    names = {False: 'Healthy', True: 'Sick'}
    raw_abundance_mld_plt = raw_abundance_mld.replace({'IBD': names})
    boxes_imp_raw = sns.catplot(x = 'variable', y = 'value', hue = 'IBD',
                                data = raw_abundance_mld_plt,
                                kind = 'box', height=9, aspect=3,
                                showfliers=False, order=list(plot_order_names))
    boxes_imp_raw.set(title='Most Important ASVs for ' + algo + ' Forest',
                      xlabel = 'Ratio of Outliers (healthy/sick)',
                      ylabel = 'Abundance', ylim = (0,yend),
                      xticklabels = outlier_ratio_strings)    
    # save figure
    boxes_imp_raw.savefig('../fig/imp_asvs_' + algo + '_boxplots.pdf')
    
    # return ASV names in plotting order for further analyses            
    return plot_order_names

[plot_imp_asv_abundance_dists(algo = alg) for alg in ['raw', 'pca']]

# now inspect why three very indicative asvs from raw (first three in plot)
# were not 'found' by PCA

# get names of ASVs to inspect
not_captured_by_pca = plot_imp_asv_abundance_dists('raw')[:3:]
all_pca_asvs = plot_imp_asv_abundance_dists('pca')

names_of_interest = np.concatenate((not_captured_by_pca, all_pca_asvs))
# remember! first three are the ones not_caputred_by_pca

# compute variances of all ASVs
all_variances = abundance_data.var(axis=0)
# select variances of ASVs of interest
vars_of_interest = all_variances.loc[names_of_interest]
# compute quantiles
var_quantiles = np.round([np.sum(var < all_variances)/len(all_variances)
                          for var in vars_of_interest], 4)

# not_captured_by_pca:
# array([0.0077, 0.0059, 0.0065])  variances of these ASVs are really small!

# are the ones for PCA relevant ASVs substantially bigger?
# no they are all smaller?!
log_vars = np.log10(np.array(all_variances))
tickpos = [log_vars.min()]
for i in range(8):
    tickpos.append(i)
    
ecdf = sns.displot(log_vars, kind='ecdf', height=9,
                   aspect=2)
ecdf.set(title = 'Cumulative Distribution Function of ASV Abundance Variances',
         xlabel = 'Variance',
         xticks = tickpos)
xlabels = [str(np.round(10**log_vars.min(), 2)), "1", "10"]
for i in range(2,8):
    xlabels.append("$10^" + str(i) + "$")
ecdf.set_xticklabels(xlabels)

ecdf.savefig('../fig/ecdf_abundance_variances.pdf')


# HISTOGRAMS
hist_data = raw_abundance.iloc[:, [0, -1]]
sns.histplot(x = hist_data.columns.values[0],
             hue = 'IBD', data = hist_data,).set(xlabel = None)
                                       #ylim=(0,0.015), xlim =(0,500))
ax.set(xlabel = None, ylim=(0,500), xlim =(0,100))




# draw histograms separately since seaborn normalizes area not per group
pos_idx = np.where(raw_abundance['IBD'])[0]
sns.histplot(raw_abundance.iloc[pos_idx, 5], stat = 'density',
             bins = [0, 50, 100, 150, 200, 1000]).set(xlabel = None, title = 'IBD pos',
                                                                   xlim=(0,1000))
sns.histplot(raw_abundance.iloc[np.array(negative), 5], stat = 'density', bins = [0, 50, 100, 150, 200, 1000]).set(xlabel = None,
                                                                         xlim=(0,1000))

# look at interactions
# two dimensional patterns?

# GloVe vs PCA
# Project onto most important components
# onedimensional

# two dimensional: contourlines
