#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    POSTPROCESS CLUSTER RUNS FOR GLOVE REFITS
    
    Some Glove inits fail completely
     => Check Cluster log files job.$JOBID.err/out for loss = nan.
     => Move all associated data to separate folder

    MUST BE GIVEN THE PATH WHERE ALL THE FILES ARE LOCATED
    REQUIRES A FOLDER ../data/nan_associated WHERE FILES WILL BE MOVED TO
"""

import os
import sys

# set working directory according to given argument
path = sys.argv[1]
os.chdir(str(path))
print(os.getcwd())
# init list of seeds and whether it's associated with a failed fit
seeds = []
nan_indicator = []

# extract all job ids from job.ID.err files
all_files = os.listdir('.')

job_files_logical = ['job.' in file and '.err' in file for file in all_files]
job_files_idx = [idx for idx, value in enumerate(job_files_logical) if value]
job_files = [all_files[idx] for idx in job_files_idx]

job_ids = [file.split('.')[1] for file in job_files]


for jobid in job_ids:
    
    # retrieve name of runGlove_$SEED.sh and corresponding loss @it 001
    
    # look in .out for seed
    with open('job.' + str(jobid) + '.out', mode='r') as logfile:
        for line in logfile:
            # search for line where the scriptname, including the seed, appears
            if '.sh' in line:
                shell_script = line.split('/')[-1]
                stripped_seed = shell_script.rstrip('.sh\n').lstrip('runGlove_')
                seeds.append(stripped_seed)
    
    # look in .err for loss
    with open('job.' + str(jobid) + '.err', mode='r') as errfile:
        for line in errfile:
            if 'iter: 001' in line:
                if 'cost: -nan' in line:
                    # indicate that the seed at the respective index caused nan
                    	nan_indicator.append(True)
                else:
                    nan_indicator.append(False)
                    
bad_seeds = [s for i, s in enumerate(seeds) if nan_indicator[i]]
good_seeds = [s for i, s in enumerate(seeds) if not nan_indicator[i]]


# move embedding matrix and test_sample indices into ../data/nan_associated
for seed in bad_seeds:
    # create filenames
    embedding_file = 'glove_input_' + str(seed) + '_emb.txt'
    test_sample_file = 'test_samples_' + str(seed) + '.txt'
    # create moving commands
    mv_emb = 'mv ' + embedding_file + ' ../data/nan_associated'
    mv_test = 'mv ../data/' + test_sample_file + ' ../data/nan_associated'
    # execute moving
    os.system(mv_emb)
    os.system(mv_test)

                    
                    
                    
