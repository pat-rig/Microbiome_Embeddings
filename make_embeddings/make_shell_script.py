#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv


def shell_script(glove_input_file):
    """
        RETURN THE CONTENT OF runGloVe_$SEED.sh AS STRING WHERE THE INPUT FILE
        IS FELXIBLY DEFINED BY input_file
    """
    
    # init script content as list (of lists) that contain rows
    script_content = []
    
    # create strings for naming the weight file and specifying directories
    emb_prefix = glove_input_file.split("/")[-1]
    scratch_prefix = glove_input_file.split("g")[0]
    
    # use template script: Read each line and substitute the file name if 
    # necessary.
    with open('runGlove.sh', mode='r') as f:
        for line in f:
            
            # remove \n at the end of line
            clean_line = line.rstrip('\n')
            # empty lines require special treatment
            # use whitespace as empty line, since writer won't write  
            # empty fields
            
            if not clean_line == '':
                # replace names of files and names of directories appropriately
                r = clean_line.replace("glove_input_file", glove_input_file +
                                       ".txt")
                r2 = r.replace("glove_emb_filter", emb_prefix + "_emb")
                r3 = r2.replace("vocab_filter",
                                scratch_prefix + "vocab_filter")
                r4 = r3.replace("overflow_filter",
                                scratch_prefix + "overflow_filter")
                r5 = r4.replace("cooccur_filter",
                                scratch_prefix + "cooccur_filter")
                r6 = r5.replace("gradsq_filter",
                                scratch_prefix + "gradsq_filter")
                script_content.append([r6])
        
    return(script_content)



def mk_shell_script(file_name, content):
    """
        WRITE CONTENT-STRING TO SHELL SCRIPT FILE AND NAME APPROPRIATELY
    
        param. file_name [str]: path including prefix of resulting .sh
        param. content [list]: list of lists containing the rows of the script
    
    """

    # create file
    with open(file_name, mode='w') as file:
        # init writer
        # does dellimiter "" work? want to treat each row as one field
        # if not: quickn dirty: use character that will never appear in 
        # the script
        writer = csv.writer(file, delimiter=";", quoting=csv.QUOTE_NONE,
                            escapechar="\\", quotechar=":", lineterminator='\n')
        # somewhat sketchy to set escapechar and quotechar arbitrarily to non
        # appearing characters
        writer.writerows(content)
    
        file.close()
    
    print("Created " + file_name)

 
