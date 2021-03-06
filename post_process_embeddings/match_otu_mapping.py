import pandas as pd
import numpy as np
import helper_functions as hf
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
import importlib
import math
import copy

# load sample x ASV_frequency table
data_dir = "../data"
otu_file = data_dir + "/seqtab_filter.07.txt"
otu = pd.read_csv(otu_file, sep = "\t", index_col=0)
print("Samples: " + str(otu.shape[0]) + "  Taxa: " + str(otu.shape[1]))
otu_dir = data_dir + "/otu_objs"

# load mapping_file that contains meta data
mapping_file = data_dir + "/AG_mapping.txt"
mapping = pd.read_csv(mapping_file, sep = "\t", index_col=0)

err_qid = pd.read_csv(data_dir + "/err-to-qid.txt", sep = "\t", index_col = 0)

convert_sample_ids = err_qid.loc[otu.index.values, :]
otu = otu.set_index(convert_sample_ids.sample_title)

# issue (1) was fixed here
otu_clean, map_clean = hf.match_otu_map(otu, mapping)

number_criteria = []
cat_criteria = ["IBD", "EXERCISE_FREQUENCY", "SEX", "ONE_LITER_OF_WATER_A_DAY_FREQUENCY", 
        "SEAFOOD_FREQUENCY", "PROBIOTIC_FREQUENCY", "OLIVE_OIL", "FRUIT_FREQUENCY", 
         "SLEEP_DURATION", "SUGAR_SWEETENED_DRINK_FREQUENCY", "MILK_CHEESE_FREQUENCY",
         "RED_MEAT_FREQUENCY","MEAT_EGGS_FREQUENCY", "VEGETABLE_FREQUENCY", "BODY_SITE"]

otu_clean, map_clean = hf.filterForMetadata(otu_clean, map_clean, number_criteria, cat_criteria)

fully_post_processed = {'abundance': otu_clean, 'meta': map_clean}
with open(otu_dir + "/fully_postprocessed.obj", mode='wb') as both_file:
    pickle.dump(fully_post_processed, both_file)
    both_file.close()
    
# Iterate over all seeds and save objects
# load seeds
with open('../data/seeds.obj', mode = 'rb') as seedfile:
    seeds = pickle.load(seedfile)
    
for seed in seeds:
    
    #Make train/test set
    test_samples_file = data_dir + "/test_samples/" + "/test_samples_" + str(seed) + ".txt"
    with open(test_samples_file) as f:
        test_samples = f.read().split()
    
    # preprocess names of test samples
    err_qid = pd.read_csv(data_dir + "/err-to-qid.txt", sep = "\t", index_col = 0)
    test_samples = err_qid.loc[test_samples, "sample_title"]
    test_samples = test_samples[test_samples == test_samples] #delete Nan values
    test_samples = test_samples[[test_samples[i] in otu_clean.index.values for i in range(len(test_samples))]]
    
    
    otu_train, otu_test, map_train, map_test = hf.splitTrainTest(otu_clean,
                                                                 map_clean,
                                                                 test_samples)
    
    map_train = map_train.drop('BODY_SITE', axis = 1)
    map_test = map_test.drop('BODY_SITE', axis = 1)
    map_train, map_test = hf.makeMappingNumeric(map_train, map_test,
                                                number_criteria, cat_criteria)
    
    
    f = open(otu_dir + "/otu_train_" + str(seed) + ".obj", "wb")
    pickle.dump(otu_train, f)
    f.close()
    
    f = open(otu_dir + "/otu_test_" + str(seed) + ".obj", "wb")
    pickle.dump(otu_test, f)
    f.close()
    
    f = open(otu_dir + "/map_train_" + str(seed) + ".obj", "wb")
    pickle.dump(map_train, f)
    f.close()
    
    f = open(otu_dir + "/map_test_" + str(seed) +  ".obj", "wb")
    pickle.dump(map_test, f)
    f.close()
