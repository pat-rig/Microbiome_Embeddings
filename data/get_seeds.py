#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Retrieve all Seeds from filenames in data/test_samples and save as python
    object.
    
    Execute from shell and set first argument to path to data/test_samples
"""
import os
import sys
import pickle

# retrieve path to data/test_samples
test_samples_path = sys.argv[1]

files = os.listdir(test_samples_path)
seeds = []

for file in files:
    seed = file.rstrip('.txt').split('_')[-1]
    seeds.append(seed)
        
# write to .obj    
with open('seeds.obj', mode = 'wb') as file:
    pickle.dump(seeds, file)
    file.close()
    