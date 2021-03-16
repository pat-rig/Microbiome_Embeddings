import csv
import numpy as np
import random


def mk_glove_input(s, destin_dir, data_dir, job):
    
    """
    TRANSFORM sample x ASV-frequency TO sample x occurrence
    
    Resulting table contains all hosts as rows.
    
    Each row contains the SEQUENCES of the phyla which were observed at least 
    once.
    
    PARAMS:
        
    s: 
        seed for init of random number generator
    
    destin_dir:
        path to directory where sample x occ matrix is saved

    job:
	jobID of the current .sbatch submission for execution on TCML cluster as int

    """

    # Load Data
    # Set directories and file names accordingly if organized differently
    input_file = "/scratch/" + str(job) + "/seqtab_filter.07.txt"
    f = open(input_file, 'r')
       
    # Specify names of output 
    outfile = open(destin_dir + 'glove_input_' + str(s) + '.txt', mode = 'w')
    test_samples_file = open(data_dir + '/test_samples_' + str(s) + '.txt', 'w')
    print("Processing: " + input_file)
    
    # Build file-writer
    writer = csv.writer(outfile, delimiter = "\t", quoting = csv.QUOTE_NONE,
                        escapechar = '')
    
    # Retrieve column names for sample by ASV-occurrence matrix
    taxa_names = f.readline()
    taxa_names = taxa_names.strip().split("\t")
    
    
    i = 0
    j = 0
    test_samples = []
    random.seed(s)
    for line in f:
        vec = line.split("\t")
        sample_id = vec[0]
        if random.random() > 0.15:
            # from original script Tataru and David (2020)
            if i == 3261 or i == 3260: # line 3261/0 -- what are special about these?
                print([i for i in vec[1:]]) # print occurrence frequencies -- why?
                
            # create logical index vector: which ASVs occur?
            present = [float(i) > 0 for i in vec[1:]] 
            # write corresponding sequences to one line
            writer.writerow(np.array(taxa_names)[present])
            print(i, end = '\t')
            i += 1
        else:
            test_samples.append(sample_id)
        j += 1
        
    test_samples_file.write("\t".join(test_samples))
    print("Finished")
    	
    f.close()
    outfile.close()







